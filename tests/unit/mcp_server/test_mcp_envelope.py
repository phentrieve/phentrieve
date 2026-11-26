"""Unit tests for api.mcp.envelope (Family B run_mcp_tool boundary)."""

from __future__ import annotations

import asyncio
import json

from api.llm_quota import QuotaExceededError
from api.mcp.envelope import (
    ERROR_CODES,
    McpToolError,
    error_code_for_exception,
    get_recent_errors,
    recovery_action_for,
    retryable_for,
    run_mcp_tool,
)


def _run(coro):
    return asyncio.run(coro)


def test_error_codes_are_the_closed_envelope_enum():
    assert ERROR_CODES == {
        "invalid_input",
        "not_found",
        "ambiguous_query",
        "upstream_unavailable",
        "rate_limited",
        "internal",
    }


def test_unknown_error_code_is_clamped_to_internal():
    assert McpToolError("validation_failed", "x").error_code == "internal"


def test_recovery_action_and_retryable():
    # D4: a missing/ambiguous identifier should be resolved (via search), not
    # reformulated -- a compare/export call has no free text to reformulate.
    assert recovery_action_for("not_found") == "resolve_identifier"
    assert recovery_action_for("ambiguous_query") == "resolve_identifier"
    assert recovery_action_for("rate_limited") == "retry_backoff"
    assert retryable_for("rate_limited") is True
    assert retryable_for("upstream_unavailable") is True
    assert retryable_for("not_found") is False


def test_error_code_for_exception():
    exc = QuotaExceededError(
        quota_used=5, quota_limit=5, quota_remaining=0, usage_date_utc="2026-06-13"
    )
    assert error_code_for_exception(exc) == "rate_limited"
    assert error_code_for_exception(ValueError("bad")) == "invalid_input"
    assert error_code_for_exception(RuntimeError("x")) == "internal"


def test_run_mcp_tool_success_stamps_meta():
    async def call():
        return {"results": [{"hpo_id": "HP:1"}]}

    out = _run(run_mcp_tool("search_hpo_terms", call, response_mode="compact"))
    assert out["success"] is True
    meta = out["_meta"]
    assert meta["tool"] == "search_hpo_terms"
    assert isinstance(meta["request_id"], str) and meta["request_id"]
    assert isinstance(meta["elapsed_ms"], (int, float))
    assert meta["response_mode"] == "compact"
    assert meta["unsafe_for_clinical_use"] is True
    assert meta["capabilities_version"].startswith("sha256:")


def test_run_mcp_tool_frames_a_record_under_result_with_provenance(monkeypatch):
    from api.mcp import envelope

    monkeypatch.setattr(
        envelope,
        "provenance",
        lambda: {
            "source": "hpo",
            "data_version": "v2026-02-16",
            "retrieved_at": "2026-05-22T20:24:44Z",
        },
    )

    async def call():
        return {"term1_id": "HP:0001250", "similarity_score": 1.0}

    out = _run(run_mcp_tool("compare_hpo_terms", call))
    assert out["result"] == {"term1_id": "HP:0001250", "similarity_score": 1.0}
    assert "term1_id" not in out
    assert out["_meta"]["source"] == "hpo"
    assert out["_meta"]["data_version"] == "v2026-02-16"
    assert out["_meta"]["retrieved_at"] == "2026-05-22T20:24:44Z"


def test_provenance_reads_hpo_database_metadata(tmp_path, monkeypatch):
    from api.mcp import envelope
    from phentrieve.data_processing.hpo_database import HPODatabase

    with HPODatabase(tmp_path / "hpo_data.db") as db:
        db.initialize_schema()
        db.set_metadata("hpo_version", "v2026-02-16")
        db.set_metadata("hpo_download_date", "2026-05-22T20:24:44.746817+00:00")
    monkeypatch.setattr("phentrieve.utils.get_default_data_dir", lambda: tmp_path)
    monkeypatch.setattr(envelope, "_PROVENANCE", {})
    assert envelope.provenance() == {
        "source": "hpo",
        "data_version": "v2026-02-16",
        "retrieved_at": "2026-05-22T20:24:44Z",
    }


def test_run_mcp_tool_merges_body_meta():
    async def call():
        return {
            "results": [],
            "_meta": {"next_commands": [{"tool": "x", "arguments": {}}]},
        }

    out = _run(run_mcp_tool("search_hpo_terms", call))
    assert out["_meta"]["next_commands"] == [{"tool": "x", "arguments": {}}]
    assert out["_meta"]["unsafe_for_clinical_use"] is True


def test_run_mcp_tool_known_error_envelope():
    async def call():
        raise McpToolError(
            "not_found", "No HPO term HP:9999999.", details={"field": "term2_id"}
        )

    res = _run(run_mcp_tool("compare_hpo_terms", call))
    assert res.is_error is True
    out = res.structured_content
    assert json.loads(res.content[0].text) == out
    assert out["success"] is False
    assert out["error_code"] == "not_found"
    assert out["retryable"] is False
    assert out["recovery_action"] == "resolve_identifier"
    assert out["field"] == "term2_id"
    assert out["_meta"]["unsafe_for_clinical_use"] is True
    # D4: the recovery next-command resolves the identifier via search.
    next_tools = {c["tool"] for c in out["_meta"]["next_commands"]}
    assert "search_hpo_terms" in next_tools


def test_run_mcp_tool_internal_error_sanitized_and_recorded():
    async def call():
        raise RuntimeError("kaboom secret path /etc/passwd")

    out = _run(run_mcp_tool("search_hpo_terms", call)).structured_content
    assert out["success"] is False
    assert out["error_code"] == "internal"
    assert "/etc/passwd" not in out["message"]
    recent = get_recent_errors()
    assert recent and recent[-1]["tool"] == "search_hpo_terms"
