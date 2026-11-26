"""Mount helper for the Phentrieve MCP facade over Streamable HTTP.

The legacy fastapi-mcp OpenAPI-to-MCP path and the stdio entry point have been
removed; Phentrieve exposes MCP only via Streamable HTTP -- mounted at /mcp on
the main API (see api/main.py) or served standalone (api/mcp/http_server.py),
both built on FastMCP v3.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastmcp import FastMCP
    from fastmcp.server.http import StarletteWithLifespan

logger = logging.getLogger("phentrieve.mcp")


def build_mcp_asgi(mcp: FastMCP, path: str = "/mcp") -> StarletteWithLifespan:
    """The Streamable HTTP app: stateless, JSON responses, foreign Origins rejected."""
    from api.mcp.config import settings

    return mcp.http_app(
        path=path,
        stateless_http=True,
        json_response=True,
        host_origin_protection="auto",
        allowed_origins=settings.allowed_origins,
    )


def mount_phentrieve_mcp_facade(app: FastAPI, *, mount_path: str = "/mcp") -> None:
    """Mount the Phentrieve MCP facade as a Streamable HTTP sub-application.

    Builds the FastMCP ASGI app and stores it on ``app.state`` so the host's
    lifespan can enter the MCP session-manager lifespan context (see
    ``api/main.py``). Mounting is idempotent per ``mount_path``.
    """
    from api.mcp.facade import create_phentrieve_mcp

    normalized = mount_path or "/mcp"
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    for route in app.routes:
        if getattr(route, "path", None) == normalized:
            logger.debug("MCP already mounted at '%s'; skipping", normalized)
            return

    from starlette.routing import Route

    mcp = create_phentrieve_mcp()
    mcp_asgi = build_mcp_asgi(mcp, normalized)
    # An exact route, not a Mount: a Mount answers /mcp with a 307 to /mcp/.
    app.router.routes.append(
        Route(normalized, endpoint=mcp_asgi, methods=["GET", "POST", "DELETE"])
    )
    # The host lifespan must enter this ASGI app's lifespan to start the
    # StreamableHTTP session manager.
    app.state.phentrieve_mcp_http_app = mcp_asgi
