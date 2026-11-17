"""Standard full-text extraction skips terms outside Phenotypic abnormality.

Chunk matches that the ontology guard KNOWS to sit outside the Phenotypic
abnormality subtree (clinical modifier, clinical course, mode of inheritance,
...) are not patient phenotypes and must not reach the chunk results or the
aggregated terms (#468). The guard fails open: unknown terms are kept, and
nothing is skipped when the HPO graph is unavailable.

All IDs and labels are placeholders; no embedding model or HPO data is needed.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.config import PHENOTYPE_ROOT
from phentrieve.text_processing._hpo_extraction_helpers import process_chunk_matches
from phentrieve.text_processing.hpo_extraction_orchestrator import (
    orchestrate_hpo_extraction,
)

ROOT = "HP:0000001"
MODIFIER_BRANCH = "HP:9000900"
MODIFIER = "HP:9000001"
PHENOTYPE = "HP:9000002"
OTHER_PHENOTYPE = "HP:9000003"
UNKNOWN = "HP:9000004"

ANCESTORS = {
    MODIFIER: {ROOT, MODIFIER_BRANCH},
    PHENOTYPE: {ROOT, PHENOTYPE_ROOT},
    OTHER_PHENOTYPE: {ROOT, PHENOTYPE_ROOT},
}


def _guard_with_map(monkeypatch, ancestors: dict[str, set[str]]) -> None:
    monkeypatch.setattr(
        ontology_guard,
        "_ancestors_map",
        lambda: {t: frozenset(a) for t, a in ancestors.items()},
    )


def _raw(*rows: tuple[str, str, float]) -> dict[str, Any]:
    return {
        "ids": [[hpo_id for hpo_id, _, _ in rows]],
        "metadatas": [[{"id": hpo_id, "label": label} for hpo_id, label, _ in rows]],
        "similarities": [[score for _, _, score in rows]],
        "distances": [[1 - score for _, _, score in rows]],
        "documents": [["" for _ in rows]],
    }


RAW = _raw(
    (MODIFIER, "Placeholder modifier", 0.9),
    (PHENOTYPE, "Placeholder phenotype", 0.8),
    (UNKNOWN, "Placeholder unknown", 0.7),
    (OTHER_PHENOTYPE, "Other placeholder phenotype", 0.6),
)


def _process(**overrides: Any) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "text_chunks": ["placeholder finding text"],
        "all_query_results": [RAW],
        "num_results_per_chunk": 10,
        "chunk_retrieval_threshold": 0.3,
        "top_term_per_chunk": False,
        "assertion_statuses": None,
    }
    kwargs.update(overrides)
    return process_chunk_matches(**kwargs)


def _match_ids(chunk_results: list[dict[str, Any]]) -> list[str]:
    return [match["id"] for match in chunk_results[0]["matches"]]


def test_modifier_dropped_phenotype_and_unknown_kept(monkeypatch):
    _guard_with_map(monkeypatch, ANCESTORS)

    assert _match_ids(_process()) == [PHENOTYPE, UNKNOWN, OTHER_PHENOTYPE]


def test_dropped_modifier_does_not_use_a_result_slot(monkeypatch):
    _guard_with_map(monkeypatch, ANCESTORS)

    assert _match_ids(_process(num_results_per_chunk=2)) == [PHENOTYPE, UNKNOWN]


def test_top_term_per_chunk_is_the_best_phenotype(monkeypatch):
    _guard_with_map(monkeypatch, ANCESTORS)

    assert _match_ids(_process(top_term_per_chunk=True)) == [PHENOTYPE]


def test_phenotype_root_itself_is_kept(monkeypatch):
    _guard_with_map(monkeypatch, ANCESTORS)
    raw = _raw((PHENOTYPE_ROOT, "Placeholder root", 0.9))

    assert _match_ids(_process(all_query_results=[raw])) == [PHENOTYPE_ROOT]


def test_nothing_dropped_when_graph_unavailable(monkeypatch):
    _guard_with_map(monkeypatch, {})

    assert _match_ids(_process()) == [MODIFIER, PHENOTYPE, UNKNOWN, OTHER_PHENOTYPE]


def test_skipped_count_is_logged(monkeypatch, caplog):
    _guard_with_map(monkeypatch, ANCESTORS)

    with caplog.at_level(
        logging.INFO, logger="phentrieve.text_processing._hpo_extraction_helpers"
    ):
        _process(text_chunks=["first text", "second text"], all_query_results=[RAW] * 2)

    assert "Skipped 2 chunk matches" in caplog.text


@pytest.mark.parametrize("precomputed", [False, True])
def test_orchestrator_drops_modifier_from_chunks_and_aggregate(
    monkeypatch, precomputed
):
    _guard_with_map(monkeypatch, ANCESTORS)
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    retriever.query_batch.return_value = [RAW]

    result = orchestrate_hpo_extraction(
        text_chunks=["placeholder finding text"],
        retriever=retriever,
        precomputed_query_results=[RAW] if precomputed else None,
    )

    assert _match_ids(result.chunk_results) == [PHENOTYPE, UNKNOWN, OTHER_PHENOTYPE]
    assert [term["id"] for term in result.aggregated_results] == [
        PHENOTYPE,
        UNKNOWN,
        OTHER_PHENOTYPE,
    ]
    # Raw retrieval output is passed through unfiltered.
    assert result.raw_query_results == [RAW]


def test_orchestrator_keeps_everything_when_graph_unavailable(monkeypatch):
    _guard_with_map(monkeypatch, {})
    retriever = MagicMock()
    retriever.detect_index_type.return_value = "single_vector"
    retriever.query_batch.return_value = [RAW]

    result = orchestrate_hpo_extraction(
        text_chunks=["placeholder finding text"], retriever=retriever
    )

    assert {term["id"] for term in result.aggregated_results} == {
        MODIFIER,
        PHENOTYPE,
        UNKNOWN,
        OTHER_PHENOTYPE,
    }
