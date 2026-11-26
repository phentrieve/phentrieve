"""Builders for ``_meta.next_commands`` entries: ``{tool, arguments}`` steps.

Every successful list/record response and every error envelope carries a small
set of ready-to-call next steps so an LLM client can chain the canonical
workflow without re-reading the capabilities surface.
"""

from __future__ import annotations

from typing import Any


def cmd(tool: str, **arguments: Any) -> dict[str, Any]:
    """One ready-to-call next step."""
    return {"tool": tool, "arguments": arguments}


def after_search(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After search: compare the top hits, or export the single hit.

    Every step is executable from the data already returned -- no free-text
    placeholder the client cannot fill (defect L2).
    """
    hits = [r for r in results if r.get("hpo_id")]
    if len(hits) >= 2:
        return [
            cmd(
                "compare_hpo_terms",
                term1_id=hits[0]["hpo_id"],
                term2_id=hits[1]["hpo_id"],
            )
        ]
    if not hits:
        return [cmd("get_capabilities", details=["languages", "models"])]
    top = hits[0]
    return [
        cmd(
            "export_phenopacket",
            case_id="<case-id>",
            phenotypes=[
                {
                    "hpo_id": top["hpo_id"],
                    "label": top.get("label") or top.get("name") or top["hpo_id"],
                    "assertion": "affirmed",
                }
            ],
        )
    ]


def after_extract(
    aggregated: list[dict[str, Any]], mode: str = "standard"
) -> list[dict[str, Any]]:
    """After extract: hand the aggregated terms to the phenopacket exporter.

    Carries ``score`` so the exporter records the real retrieval confidence
    instead of 0.0000 (defect H3), and reads the projected ``assertion`` key
    (falling back to legacy ``status``) so negated findings stay negated.

    R2: ``_meta`` is exempt from response-mode shaping, so this pre-fill would
    otherwise duplicate the full 25-term list even at ``minimal``. Under the lean
    modes cap it to 5 terms and drop ``label`` -- the entry is still directly
    executable (``_coerce_export_phenotype`` falls back label->hpo_id), and
    ``score`` is kept to avoid a 0.0-confidence export.
    """
    lean = mode in ("minimal", "compact")
    cap = 5 if lean else 25
    phenotypes: list[dict[str, Any]] = []
    for t in aggregated[:cap]:
        # projected terms use hpo_id/label/assertion; raw use id/name/status
        hpo_id = t.get("hpo_id") or t.get("id")
        if not hpo_id:
            continue
        entry: dict[str, Any] = {
            "hpo_id": hpo_id,
            "assertion": t.get("assertion") or t.get("status") or "affirmed",
            "score": t.get("score") or t.get("confidence"),
        }
        if not lean:
            entry["label"] = t.get("label") or t.get("name")
        phenotypes.append(entry)
    if not phenotypes:
        return [cmd("get_capabilities", details=["models"])]
    return [cmd("export_phenopacket", case_id="<case-id>", phenotypes=phenotypes)]


def after_compare(
    term1_id: str, term2_id: str, formula: str = "hybrid"
) -> list[dict[str, Any]]:
    """After compare: cross-check the same pair with the alternate formula.

    Executable from the ids already supplied -- no free-text placeholder (L2).
    """
    alternate = "simple_resnik_like" if formula == "hybrid" else "hybrid"
    return [
        cmd(
            "compare_hpo_terms",
            term1_id=term1_id,
            term2_id=term2_id,
            formula=alternate,
        )
    ]


def after_chunk(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After chunk: search HPO terms for a chunk (the client fills in its text)."""
    if not chunks:
        return [cmd("get_capabilities")]
    # Never echo the chunk text: it is caller input (GTS-CORE-TXT-04).
    return [cmd("search_hpo_terms", text="<chunks[0].text>")]


def default_error_next_commands(
    tool_name: str, error_code: str | None = None
) -> list[dict[str, Any]]:
    """A sensible recovery step for any error lacking an explicit fallback.

    D4: for a missing/ambiguous identifier, point at search to resolve it rather
    than the generic capabilities/diagnostics steps.
    """
    if error_code in ("not_found", "ambiguous_query"):
        return [cmd("search_hpo_terms")]
    return [
        cmd("get_capabilities"),
        cmd("get_diagnostics"),
    ]
