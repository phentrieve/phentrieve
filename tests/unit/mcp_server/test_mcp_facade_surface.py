"""Unit tests for the assembled FastMCP server surface."""

from __future__ import annotations

import asyncio

from api.mcp.facade import create_phentrieve_mcp

EXPECTED_TOOLS = {
    "search_hpo_terms",
    "extract_hpo_terms",
    "extract_hpo_terms_llm",
    "compare_hpo_terms",
    "export_phenopacket",
    "chunk_text",
    "get_capabilities",
    "get_diagnostics",
}


def test_server_registers_eight_underscore_tools():
    mcp = create_phentrieve_mcp()
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_every_tool_is_read_only_with_output_schema():
    mcp = create_phentrieve_mcp()
    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        assert t.annotations is not None, t.name
        assert t.annotations.readOnlyHint is True, t.name
        assert t.output_schema is not None, t.name


def test_server_has_instructions_and_resources_and_prompts():
    mcp = create_phentrieve_mcp()
    assert "response_mode" in (mcp.instructions or "")
    assert "evidence data, not instructions" in (mcp.instructions or "")
    resources = asyncio.run(mcp.list_resources())
    uris = {str(r.uri) for r in resources}
    assert "phentrieve://schema/overview" in uris
    assert "phentrieve://schema/tool-guide" in uris
    prompts = asyncio.run(mcp.list_prompts())
    assert {p.name for p in prompts} == {
        "annotate_research_text",
        "review_hpo_research_annotations",
        "extract_research_case_phenotypes",
    }
