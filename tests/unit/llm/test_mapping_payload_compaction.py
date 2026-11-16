"""Compaction of the phase-2 mapping payload (placeholder items only)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from phentrieve.llm.pipeline_phase2 import (
    RETRIEVAL_SCORE_DECIMALS,
    compact_mapping_item,
    run_mapping_batch,
)
from phentrieve.llm.prompts.loader import get_mapping_prompt

pytestmark = pytest.mark.unit


def _candidate(hpo_id: str, term: str, score: Any, **extra: Any) -> dict[str, Any]:
    return {"hpo_id": hpo_id, "term_name": term, "score": score, **extra}


def _item(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "phrase": "Placeholder-Phrase",
        "category": "abnormal",
        "grounded_context": {
            "primary_chunk_text": " primary placeholder text ",
            "neighbor_chunk_texts": ["previous placeholder", "", "next placeholder"],
        },
        "candidates": candidates,
    }


def test_shared_retrieval_query_is_lifted_to_item_level() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000001", "Term one", 0.9, retrieval_query="query a"),
                _candidate("HP:0000002", "Term two", 0.8, retrieval_query="query a"),
            ]
        )
    )

    assert compact["retrieval_query"] == "query a"
    assert all("retrieval_query" not in c for c in compact["candidates"])


def test_differing_candidate_query_stays_on_that_candidate() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000001", "Term one", 0.9, retrieval_query="query a"),
                _candidate("HP:0000002", "Term two", 0.8, retrieval_query="query b"),
                _candidate("HP:0000003", "Term three", 0.7, retrieval_query="query a"),
            ]
        )
    )

    assert compact["retrieval_query"] == "query a"
    assert [c.get("retrieval_query") for c in compact["candidates"]] == [
        None,
        "query b",
        None,
    ]
    assert "retrieval_query" not in compact["candidates"][0]


def test_query_tie_lifts_the_earliest_query() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000001", "Term one", 0.9, retrieval_query="query b"),
                _candidate("HP:0000002", "Term two", 0.8, retrieval_query="query a"),
            ]
        )
    )

    assert compact["retrieval_query"] == "query b"
    assert "retrieval_query" not in compact["candidates"][0]
    assert compact["candidates"][1]["retrieval_query"] == "query a"


def test_query_is_not_lifted_when_a_candidate_has_none() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000001", "Term one", 0.9, retrieval_query="query a"),
                _candidate("HP:0000002", "Term two", 0.8),
            ]
        )
    )

    assert "retrieval_query" not in compact
    assert compact["candidates"][0]["retrieval_query"] == "query a"
    assert "retrieval_query" not in compact["candidates"][1]


def test_no_item_level_query_without_candidates_or_queries() -> None:
    assert "retrieval_query" not in compact_mapping_item(_item([]))
    assert "retrieval_query" not in compact_mapping_item(
        _item([_candidate("HP:0000001", "Term one", 0.9)])
    )


def test_matched_text_is_omitted_only_when_exactly_equal_to_term() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000001", "Term one", 0.9, matched_text="Term one"),
                _candidate("HP:0000002", "Term two", 0.8, matched_text="term two"),
                _candidate("HP:0000003", "Term three", 0.7, matched_text="Term three "),
                _candidate("HP:0000004", "Term four", 0.6, matched_text="Other name"),
                _candidate("HP:0000005", "Term five", 0.5),
            ]
        )
    )

    assert [c.get("matched_text") for c in compact["candidates"]] == [
        None,
        "term two",
        "Term three ",
        "Other name",
        None,
    ]
    assert "matched_text" not in compact["candidates"][0]


def test_scores_are_rounded_and_candidate_order_is_kept() -> None:
    assert RETRIEVAL_SCORE_DECIMALS == 4
    compact = compact_mapping_item(
        _item(
            [
                _candidate("HP:0000003", "Term three", 0.8731245398521423),
                _candidate("HP:0000001", "Term one", 0.8731244),
                _candidate("HP:0000002", "Term two", 0.5),
                _candidate("HP:0000004", "Term four", 1),
                _candidate("HP:0000005", "Term five", None),
            ]
        )
    )

    assert [c["id"] for c in compact["candidates"]] == [
        "HP:0000003",
        "HP:0000001",
        "HP:0000002",
        "HP:0000004",
        "HP:0000005",
    ]
    assert [c["retrieval_score"] for c in compact["candidates"]] == [
        0.8731,
        0.8731,
        0.5,
        1.0,
        None,
    ]


def test_rounding_never_inverts_a_descending_score_list() -> None:
    scores = [0.91239, 0.91235, 0.91234, 0.5000049, 0.5, 0.49999]
    compact = compact_mapping_item(
        _item(
            [
                _candidate(f"HP:{index:07d}", f"Term {index}", score)
                for index, score in enumerate(scores)
            ]
        )
    )

    rounded = [c["retrieval_score"] for c in compact["candidates"]]
    assert rounded == sorted(rounded, reverse=True)


def test_nothing_else_is_removed_from_the_payload() -> None:
    compact = compact_mapping_item(
        _item(
            [
                _candidate(
                    "HP:0000001",
                    "Term one",
                    0.9,
                    retrieval_query="query a",
                    matched_text="Term one",
                    matched_component="label",
                ),
                _candidate(
                    "HP:0000002",
                    "Term two",
                    0.8,
                    retrieval_query="query a",
                    matched_text="Other name",
                    matched_component="synonym",
                ),
            ]
        ),
        item_id="item_1",
    )

    assert compact == {
        "primary_chunk_text": "primary placeholder text",
        "neighbor_chunk_texts": ["previous placeholder", "next placeholder"],
        "phrase": "placeholder phrase",
        "category": "abnormal",
        "retrieval_query": "query a",
        "candidates": [
            {
                "id": "HP:0000001",
                "term": "Term one",
                "retrieval_score": 0.9,
                "matched_component": "label",
            },
            {
                "id": "HP:0000002",
                "term": "Term two",
                "retrieval_score": 0.8,
                "matched_text": "Other name",
                "matched_component": "synonym",
            },
        ],
        "item_id": "item_1",
    }
    assert list(compact)[:4] == [
        "primary_chunk_text",
        "neighbor_chunk_texts",
        "phrase",
        "category",
    ]


class _RecordingProvider:
    last_usage: dict[str, int] = {}

    def __init__(self) -> None:
        self.user_prompts: list[str] = []

    def run_structured_prompt(self, *, system_prompt, user_prompt, response_model):
        self.user_prompts.append(user_prompt)
        return None


def _payload(user_prompt: str) -> dict[str, Any]:
    start = user_prompt.index("{")
    end = user_prompt.rindex("}") + 1
    return json.loads(user_prompt[start:end])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_single_and_batch_payloads_are_both_compacted(batch_size: int) -> None:
    provider = _RecordingProvider()
    batch = [
        _item(
            [
                _candidate(
                    "HP:0000001",
                    "Term one",
                    0.123456789,
                    retrieval_query="query a",
                    matched_text="Term one",
                )
            ]
        )
        for _ in range(batch_size)
    ]

    run_mapping_batch(
        provider=provider, batch=batch, mapping_prompt=get_mapping_prompt("en")
    )

    payload = _payload(provider.user_prompts[0])
    items = [payload] if batch_size == 1 else payload["items"]
    assert len(items) == batch_size
    for item in items:
        assert item["retrieval_query"] == "query a"
        assert item["candidates"] == [
            {"id": "HP:0000001", "term": "Term one", "retrieval_score": 0.1235}
        ]
