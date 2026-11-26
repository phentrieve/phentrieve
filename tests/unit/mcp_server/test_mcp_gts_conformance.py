"""GTS Core rules that the tool surface and the HTTP transport must meet."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.mcp.facade import create_phentrieve_mcp
from api.mcp.server import build_mcp_asgi, mount_phentrieve_mcp_facade
from api.version import get_api_version

pytestmark = pytest.mark.unit

_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0.0"},
    },
}
_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _tools():
    return asyncio.run(create_phentrieve_mcp().list_tools())


def test_no_tool_name_carries_the_server_prefix():
    """GTS-CORE-NAM-02: the gateway adds the namespace, the server does not."""
    assert not [t.name for t in _tools() if t.name.startswith("phentrieve_")]


def test_every_input_property_is_documented_with_examples():
    """GTS-CORE-SCH-01/-02: descriptions everywhere, examples on required/array."""
    for tool in _tools():
        schema = tool.parameters
        required = set(schema.get("required") or [])
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"{tool.name}.{name}"
            is_array = prop.get("type") == "array" or any(
                b.get("type") == "array" for b in prop.get("anyOf") or []
            )
            if name in required or is_array:
                assert prop.get("examples"), f"{tool.name}.{name}"


def test_output_schemas_describe_the_envelope():
    """GTS-CORE-SCH-04: a published outputSchema names success, result(s), _meta."""
    for tool in _tools():
        props = tool.output_schema["properties"]
        assert {"success", "_meta"} <= props.keys(), tool.name
        assert {"result", "results"} & props.keys(), tool.name


@pytest.fixture
def client():
    app = build_mcp_asgi(create_phentrieve_mcp())
    with TestClient(app, base_url="http://127.0.0.1:8734") as c:
        yield c


def test_mcp_is_stateless_json(client):
    """GTS-CORE-TRN-01/-02: /mcp answers JSON, without redirect or session id."""
    resp = client.post("/mcp", json=_INIT, headers=_HEADERS, follow_redirects=False)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert "mcp-session-id" not in resp.headers


def test_foreign_origin_is_rejected(client):
    """GTS-CORE-TRN-04: DNS-rebinding protection."""
    headers = {**_HEADERS, "Origin": "https://gts-check.invalid"}
    assert client.post("/mcp", json=_INIT, headers=headers).status_code == 403


def test_health_reports_status_version_transport(client):
    """GTS-OPS-HLT-01 and GTS-CORE-VER-02: /health matches serverInfo.version."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == get_api_version()
    assert body["transport"] == "streamable-http-stateless"


def test_mounted_facade_serves_mcp_without_redirect():
    """GTS-CORE-TRN-01 on the API: /mcp itself, not a 307 to /mcp/."""
    app = FastAPI()
    mount_phentrieve_mcp_facade(app)
    mcp_app = app.state.phentrieve_mcp_http_app

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with mcp_app.router.lifespan_context(_app):
            yield

    app.router.lifespan_context = lifespan
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        resp = c.post("/mcp", json=_INIT, headers=_HEADERS, follow_redirects=False)
    assert resp.status_code == 200
    assert resp.json()["result"]["serverInfo"]["name"] == "phentrieve"
