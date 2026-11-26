"""Unit tests for clinical modifier extraction, resolution, carving, and guarding.

All examples in this file are invented clinical phrases.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.llm.evidence_offsets import EvidenceLocator, carve_modifier_span
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.pipeline_phase1 import expand_combined_phase1_extractions
from phentrieve.llm.pipeline_phase2 import (
    DEFAULT_MAX_MODIFIER_CANDIDATES,
    compact_modifier_mapping_item,
    finding_already_expresses_modifier,
)
from phentrieve.llm.types import (
    LLMBatchMappingSelection,
    LLMBatchMappingSelections,
    LLMGroundedExtractedPhenotypes,
    LLMGroundedExtractedPhenotypesWithModifiers,
    LLMMappingSelection,
    LLMPhenotype,
    LLMPhenotypeEvidence,
    LLMPipelineConfig,
)
from phentrieve.ontology_profiles import (
    HPO_CLINICAL_MODIFIER_ROOT,
    HPO_DEFAULT_PROFILE,
    HPO_FREQUENCY_ROOT,
    HPO_PHENOTYPIC_ABNORMALITY_ROOT,
    HPO_SEVERITY_ROOT,
    OntologyProfile,
    OntologyRootsConfig,
)

pytestmark = pytest.mark.unit


# Synthetic HPO ancestry map covering invented test cases.
TEST_GUARD_MAP: dict[str, frozenset[str]] = {
    # Findings under HP:0000118 (Phenotypic abnormality)
    "HP:0000365": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Hearing impairment
    "HP:0001903": frozenset({"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}),  # Anemia
    "HP:0001873": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Thrombocytopenia
    "HP:0000518": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Ptosis / Bilateral ptosis
    "HP:0002315": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Headache
    "HP:0001250": frozenset({"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}),  # Seizure
    "HP:0001263": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Global developmental delay
    "HP:0002187": frozenset(
        {"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}
    ),  # Intellectual disability, profound
    "HP:0000969": frozenset({"HP:0000001", HPO_PHENOTYPIC_ABNORMALITY_ROOT}),  # Edema
    # Clinical modifiers under HP:0012823
    "HP:0012825": frozenset(
        {"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT, HPO_SEVERITY_ROOT}
    ),  # Mild
    "HP:0012828": frozenset(
        {"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT, HPO_SEVERITY_ROOT}
    ),  # Severe
    "HP:0012833": frozenset({"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT}),  # Unilateral
    "HP:0012832": frozenset({"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT}),  # Bilateral
    "HP:0003676": frozenset({"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT}),  # Progressive
    "HP:0003621": frozenset(
        {"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT}
    ),  # Juvenile onset
    "HP:0003826": frozenset({"HP:0000001", HPO_CLINICAL_MODIFIER_ROOT}),  # Stillbirth
    # Frequency branch (disallowed as individual clinical modifier)
    "HP:0040281": frozenset({"HP:0000001", HPO_FREQUENCY_ROOT}),  # Very frequent
}


def _patch_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ontology_guard, "_ancestors_map", lambda: TEST_GUARD_MAP)


class _FakeProvider:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.last_usage = {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }
        self.last_request_count = 1
        self.token_count_source = "exact"

    def run_structured_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[Any],
        max_output_tokens: int | None = None,
    ) -> Any:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
                "max_output_tokens": max_output_tokens,
            }
        )
        assert self._responses, f"Unexpected LLM call for {response_model.__name__}"
        return self._responses.pop(0)


class _FakeToolExecutor:
    def __init__(self, candidates_by_phrase: dict[str, list[dict[str, Any]]]) -> None:
        self._candidates_by_phrase = {
            key.lower(): value for key, value in candidates_by_phrase.items()
        }
        self.queries: list[dict[str, Any]] = []

    def query_batch_hpo_terms(
        self,
        *,
        phrases: list[str],
        language: str,
        n_results: int,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.queries.append(
            {
                "phrases": list(phrases),
                "language": language,
                "n_results": n_results,
                "allowed_term_ids": (
                    set(allowed_term_ids) if allowed_term_ids is not None else None
                ),
            }
        )
        results: list[dict[str, Any]] = []
        for phrase in phrases:
            cands = list(self._candidates_by_phrase.get(phrase.lower(), []))
            if allowed_term_ids is not None:
                cands = [c for c in cands if c.get("hpo_id") in allowed_term_ids]
            results.append({"phrase": phrase, "candidates": cands[:n_results]})
        return results


def _single_chunk(text: str) -> list[dict[str, Any]]:
    return [{"chunk_id": 1, "text": text, "start_char": 0, "end_char": len(text)}]


def _assert_no_overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> None:
    assert span_a[1] <= span_b[0] or span_b[1] <= span_a[0], (
        f"Spans overlap: {span_a} vs {span_b}"
    )


# ---------------------------------------------------------------------------
# Pure helper tests (offsets, carving, phase1 cloning, phase2 compaction)
# ---------------------------------------------------------------------------


def test_locate_nearest_picks_occurrence_closest_to_anchor_and_never_falls_back() -> (
    None
):
    document = "Mild rash on arm. Later, mild hearing loss noted."
    chunks = {1: (0, 17), 2: (18, len(document))}
    locator = EvidenceLocator(document, chunks)

    anchor = (
        document.index("mild hearing loss"),
        document.index("mild hearing loss") + 17,
    )
    located = locator.locate_nearest("mild", [1, 2], anchor_span=anchor)
    assert located.basis == "evidence_in_chunk"
    assert (located.start, located.end) == (
        document.index("mild hearing"),
        document.index("mild hearing") + 4,
    )

    # Immediate neighbor lookup when cited chunk lacks the modifier word.
    neighbor_located = locator.locate_nearest("Mild", [2], include_neighbors=True)
    assert neighbor_located.basis == "evidence_in_chunk"
    assert document[neighbor_located.start : neighbor_located.end] == "mild"

    # Unresolved quote NEVER falls back to chunk_span.
    missing = locator.locate_nearest(
        "paraphrased severity", [2], include_neighbors=True
    )
    assert missing.start is None
    assert missing.end is None
    assert missing.basis == "unresolved"


def test_carve_modifier_span_covers_all_four_cases() -> None:
    # Case 1: no overlap
    doc = "Severe anemia and thrombocytopenia."
    span, keep = carve_modifier_span(doc, (18, 34), (0, 6))
    assert (span, keep) == ((18, 34), True)

    # Case 2: leading edge on word boundary
    doc2 = "mild left-sided hearing loss"
    span, keep = carve_modifier_span(doc2, (0, len(doc2)), (0, 4))
    assert keep is True
    assert doc2[span[0] : span[1]] == "left-sided hearing loss"

    # Case 3: trailing edge on word boundary
    doc3 = "hearing loss, mild"
    span, keep = carve_modifier_span(doc3, (0, len(doc3)), (14, 18))
    assert keep is True
    assert doc3[span[0] : span[1]] == "hearing loss"

    # Case 4a: strictly interior
    doc4 = "hearing mild loss"
    span, keep = carve_modifier_span(doc4, (0, len(doc4)), (8, 12))
    assert (span, keep) == ((0, len(doc4)), False)

    # Case 4b: compound word without word boundary
    doc5 = "subcortical dysplasia"
    span, keep = carve_modifier_span(doc5, (0, len(doc5)), (0, 3))
    assert (span, keep) == ((0, len(doc5)), False)


def test_expand_combined_phase1_extractions_preserves_modifiers() -> None:
    extracted = [
        {
            "phrase": "cerebral cerebellar atrophy",
            "category": "abnormal",
            "chunk_ids": [1],
            "evidence_text": "progressive cerebral cerebellar atrophy",
            "modifiers": [" progressive ", "Progressive", ""],
        }
    ]
    expanded = expand_combined_phase1_extractions(extracted)
    assert [item["phrase"] for item in expanded] == [
        "cerebral atrophy",
        "cerebellar atrophy",
    ]
    assert expanded[0]["modifiers"] == ["progressive"]
    assert expanded[1]["modifiers"] == ["progressive"]
    assert expanded[0]["modifiers"] is not expanded[1]["modifiers"]


def test_compact_modifier_mapping_item_limits_candidates_and_includes_finding_term() -> (
    None
):
    item = {
        "phrase": "left-sided",
        "finding_term": {"id": "HP:0000365", "label": "Hearing impairment"},
        "grounded_context": {
            "primary_chunk_text": "mild left-sided hearing loss",
            "neighbor_chunk_texts": [],
        },
        "candidates": [
            {
                "hpo_id": f"HP:00128{idx:02d}",
                "term_name": f"Mod {idx}",
                "score": 0.9 - idx * 0.05,
            }
            for idx in range(8)
        ],
    }
    compact = compact_modifier_mapping_item(item, item_id="item_1")
    assert compact["phrase"] == "left-sided"
    assert compact["finding_term"] == {
        "id": "HP:0000365",
        "label": "Hearing impairment",
    }
    assert "neighbor_chunk_texts" not in compact
    assert len(compact["candidates"]) == DEFAULT_MAX_MODIFIER_CANDIDATES
    assert compact["item_id"] == "item_1"


def test_finding_already_expresses_modifier_conservative_cases() -> None:
    assert finding_already_expresses_modifier(
        finding_label="Bilateral ptosis",
        modifier_label="Bilateral",
        modifier_phrase="bilateral",
    )
    assert finding_already_expresses_modifier(
        finding_label="Severe intellectual disability",
        modifier_label="Severe",
        modifier_phrase="marked",
    )
    assert not finding_already_expresses_modifier(
        finding_label="Hearing impairment",
        modifier_label="Unilateral",
        modifier_phrase="left-sided",
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------


def test_extract_modifiers_false_leaves_prompt_schema_and_output_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "Patient has mild hearing loss."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypes.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "mild hearing loss",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.96,
                }
            ]
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=False),
    )

    assert provider.calls[0]["response_model"] is LLMGroundedExtractedPhenotypes
    assert len(executor.queries) == 1
    assert len(result.terms) == 1
    assert result.terms[0].term_id == "HP:0000365"
    assert result.terms[0].modifiers == []
    assert result.terms[0].warnings == []
    assert "modifiers" not in result.meta.trace
    assert "modifiers_phase2a_seconds" not in result.meta.phase_timings
    assert "modifier_phrases" not in result.meta.phase_counts


def test_disjoint_edge_carved_multiple_leading_modifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "Patient has mild left-sided hearing loss."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "mild left-sided hearing loss",
                            "modifiers": ["mild", "left-sided"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            ),
            # Modifier mapping LLM resolves "left-sided" -> HP:0012833 (Unilateral)
            LLMMappingSelection(phrase="left-sided", hpo_id="HP:0012833"),
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.95,
                }
            ],
            "mild": [{"hpo_id": "HP:0012825", "term_name": "Mild", "score": 0.97}],
            "left-sided": [
                {"hpo_id": "HP:0012833", "term_name": "Unilateral", "score": 0.78},
                {"hpo_id": "HP:0012832", "term_name": "Bilateral", "score": 0.60},
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    assert (
        provider.calls[0]["response_model"]
        is LLMGroundedExtractedPhenotypesWithModifiers
    )
    assert len(result.terms) == 1
    finding = result.terms[0]
    ev = finding.evidence_records[0]

    # Finding span is trimmed to "hearing loss".
    assert ev.start_char is not None and ev.end_char is not None
    assert text[ev.start_char : ev.end_char] == "hearing loss"
    assert ev.evidence_text == "hearing loss"

    # Both modifiers attached with exact segments and zero overlap.
    mods_by_id = {m.term_id: m for m in finding.modifiers}
    assert set(mods_by_id) == {"HP:0012825", "HP:0012833"}
    assert mods_by_id["HP:0012825"].source == "llm"
    assert mods_by_id["HP:0012833"].source == "llm"

    mild_seg = mods_by_id["HP:0012825"].segments[0]
    unilat_seg = mods_by_id["HP:0012833"].segments[0]
    assert (mild_seg.start, mild_seg.end, mild_seg.text) == (
        text.index("mild"),
        text.index("mild") + len("mild"),
        "mild",
    )
    assert (unilat_seg.start, unilat_seg.end, unilat_seg.text) == (
        text.index("left-sided"),
        text.index("left-sided") + len("left-sided"),
        "left-sided",
    )
    finding_span = (ev.start_char, ev.end_char)
    _assert_no_overlap(finding_span, (mild_seg.start, mild_seg.end))
    _assert_no_overlap(finding_span, (unilat_seg.start, unilat_seg.end))
    _assert_no_overlap(
        (mild_seg.start, mild_seg.end), (unilat_seg.start, unilat_seg.end)
    )

    # Meta accounting includes modifier phase metrics and LLM call.
    assert "modifiers" in result.meta.trace
    assert result.meta.phase_counts["modifier_phrases"] == 2
    assert result.meta.phase_counts["modifier_local_matches"] == 1
    assert result.meta.phase_counts["modifier_llm_mapped"] == 1
    assert result.meta.phase_request_counts["modifiers_phase2b_llm_requests"] == 1
    assert result.meta.request_count == 2


def test_interior_and_compound_modifiers_leave_finding_span_intact_and_empty_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    # "hearing mild loss" has an interior modifier; "mildheadache" has a compound prefix.
    text = "Notes show hearing mild loss and mildheadache."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "hearing mild loss",
                            "modifiers": ["mild"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                        {
                            "phrase": "headache",
                            "evidence_text": "mildheadache",
                            "modifiers": ["mild"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.96,
                }
            ],
            "headache": [
                {"hpo_id": "HP:0002315", "term_name": "Headache", "score": 0.96}
            ],
            "mild": [{"hpo_id": "HP:0012825", "term_name": "Mild", "score": 0.98}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    by_id = {term.term_id: term for term in result.terms}
    hearing = by_id["HP:0000365"]
    headache = by_id["HP:0002315"]

    assert hearing.evidence_records[0].evidence_text == "hearing mild loss"
    assert len(hearing.modifiers) == 1
    assert hearing.modifiers[0].term_id == "HP:0012825"
    assert hearing.modifiers[0].segments == []

    assert headache.evidence_records[0].evidence_text == "mildheadache"
    assert len(headache.modifiers) == 1
    assert headache.modifiers[0].term_id == "HP:0012825"
    assert headache.modifiers[0].segments == []


def test_shared_modifier_attaches_to_both_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "Patient presents with severe anemia and thrombocytopenia."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "anemia",
                            "evidence_text": "severe anemia",
                            "modifiers": ["severe"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                        {
                            "phrase": "thrombocytopenia",
                            "evidence_text": "thrombocytopenia",
                            "modifiers": ["severe"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "anemia": [{"hpo_id": "HP:0001903", "term_name": "Anemia", "score": 0.98}],
            "thrombocytopenia": [
                {"hpo_id": "HP:0001873", "term_name": "Thrombocytopenia", "score": 0.98}
            ],
            "severe": [{"hpo_id": "HP:0012828", "term_name": "Severe", "score": 0.99}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    by_id = {term.term_id: term for term in result.terms}
    anemia = by_id["HP:0001903"]
    thrombo = by_id["HP:0001873"]

    assert anemia.evidence_records[0].evidence_text == "anemia"
    assert thrombo.evidence_records[0].evidence_text == "thrombocytopenia"

    expected_start = text.index("severe")
    for finding in (anemia, thrombo):
        assert [m.term_id for m in finding.modifiers] == ["HP:0012828"]
        seg = finding.modifiers[0].segments[0]
        assert (seg.start, seg.end, seg.text) == (
            expected_start,
            expected_start + len("severe"),
            "severe",
        )
        ev = finding.evidence_records[0]
        assert ev.start_char is not None and ev.end_char is not None
        _assert_no_overlap((ev.start_char, ev.end_char), (seg.start, seg.end))


def test_no_double_encoding_deterministic_and_explicit_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    # 1) Deterministic suppression: "bilateral" on "Bilateral ptosis" -> 0 modifier LLM calls.
    text = "Exam shows bilateral ptosis."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "bilateral ptosis",
                            "evidence_text": "bilateral ptosis",
                            "modifiers": ["bilateral"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "bilateral ptosis": [
                {
                    "hpo_id": "HP:0000518",
                    "term_name": "Bilateral ptosis",
                    "score": 0.97,
                }
            ],
            "bilateral": [
                {"hpo_id": "HP:0012832", "term_name": "Bilateral", "score": 0.98}
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )
    assert len(provider.calls) == 1  # No modifier mapping LLM call needed
    assert result.terms[0].modifiers == []
    assert result.meta.phase_counts["modifier_suppressed_double_encoding"] == 1

    # 2) LLM explicit null suppression (semantic / non-lexical double encoding):
    # "both feet" on "Bipedal edema" -> LLM receives finding_term and returns explicit null.
    text2 = "Patient has edema of both feet."
    provider2 = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "bipedal edema",
                            "evidence_text": "edema of both feet",
                            "modifiers": ["both feet"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            ),
            LLMMappingSelection.model_validate({"phrase": "both feet", "hpo_id": None}),
        ]
    )
    executor2 = _FakeToolExecutor(
        {
            "bipedal edema": [
                {"hpo_id": "HP:0000969", "term_name": "Bipedal edema", "score": 0.95}
            ],
            "both feet": [
                {"hpo_id": "HP:0012832", "term_name": "Bilateral", "score": 0.80}
            ],
        }
    )
    pipeline2 = TwoPhaseLLMPipeline(provider=provider2, tool_executor=executor2)
    result2 = pipeline2.run(
        text=text2,
        grounded_chunks=_single_chunk(text2),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )
    assert len(provider2.calls) == 2
    payload_str = provider2.calls[1]["user_prompt"]
    payload = json.loads(
        payload_str[payload_str.index("{") : payload_str.rindex("}") + 1]
    )
    assert payload["finding_term"] == {"id": "HP:0000969", "label": "Bipedal edema"}
    assert result2.terms[0].modifiers == []


def test_negated_finding_keeps_its_modifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "Patient reports no severe headache."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "headache",
                            "evidence_text": "severe headache",
                            "experiencer": "proband",
                            "assertion": "absent",
                            "modifiers": ["severe"],
                            "category": "Normal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "headache": [
                {"hpo_id": "HP:0002315", "term_name": "Headache", "score": 0.97}
            ],
            "severe": [{"hpo_id": "HP:0012828", "term_name": "Severe", "score": 0.98}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    assert len(result.terms) == 1
    finding = result.terms[0]
    assert finding.assertion == "negated"
    assert finding.evidence_records[0].evidence_text == "headache"
    assert [m.term_id for m in finding.modifiers] == ["HP:0012828"]
    assert finding.modifiers[0].segments[0].text == "severe"


def test_modifier_mapped_as_finding_converted_when_unambiguous_and_dropped_when_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "mild hearing loss"
    finding = LLMPhenotype(
        term_id="HP:0000365",
        label="Hearing impairment",
        evidence="hearing loss",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="hearing loss",
                evidence_text="mild hearing loss",
                chunk_ids=[1],
                start_char=0,
                end_char=len(text),
                offset_basis="evidence_in_chunk",
            )
        ],
    )
    mod_as_finding = LLMPhenotype(
        term_id="HP:0012825",
        label="Mild",
        evidence="mild",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="mild",
                evidence_text="mild",
                chunk_ids=[1],
                start_char=0,
                end_char=4,
                offset_basis="evidence_in_chunk",
            )
        ],
    )

    stats: dict[str, int] = {}
    guarded, _fam, _other, dropped = TwoPhaseLLMPipeline._apply_ontology_guard(
        [finding, mod_as_finding],
        [],
        [],
        profile=HPO_DEFAULT_PROFILE,
        guard_map=TEST_GUARD_MAP,
        extract_modifiers=True,
        text=text,
        stats=stats,
    )
    assert dropped == 0
    assert stats["converted_modifier_findings"] == 1
    assert len(guarded) == 1
    assert guarded[0].evidence_records[0].evidence_text == "hearing loss"
    assert [m.term_id for m in guarded[0].modifiers] == ["HP:0012825"]
    assert guarded[0].modifiers[0].segments[0].text == "mild"

    # Ambiguous case: 2 findings with matching axes in the same chunk -> modifier dropped.
    second_finding = LLMPhenotype(
        term_id="HP:0001903",
        label="Anemia",
        evidence="anemia",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="anemia",
                evidence_text="anemia",
                chunk_ids=[1],
                start_char=10,
                end_char=16,
                offset_basis="evidence_in_chunk",
            )
        ],
    )
    finding_plain = finding.model_copy(deep=True)
    finding_plain.modifiers = []
    stats_ambig: dict[str, int] = {}
    guarded_ambig, _fam2, _other2, dropped_ambig = (
        TwoPhaseLLMPipeline._apply_ontology_guard(
            [finding_plain, second_finding, mod_as_finding],
            [],
            [],
            profile=HPO_DEFAULT_PROFILE,
            guard_map=TEST_GUARD_MAP,
            extract_modifiers=True,
            text=text,
            stats=stats_ambig,
        )
    )
    assert dropped_ambig == 1
    assert stats_ambig["converted_modifier_findings"] == 0
    assert len(guarded_ambig) == 2
    assert all(not f.modifiers for f in guarded_ambig)


def test_unrestricted_finding_roots_keeps_modifier_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "History notes stillbirth."
    unrestricted_profile = OntologyProfile(
        name="unrestricted",
        roots=OntologyRootsConfig(
            finding_roots=(),
            modifier_roots=(HPO_CLINICAL_MODIFIER_ROOT,),
        ),
    )
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypes.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "stillbirth",
                            "evidence_text": "stillbirth",
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "stillbirth": [
                {"hpo_id": "HP:0003826", "term_name": "Stillbirth", "score": 0.98}
            ]
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(
            model="test-model",
            extract_modifiers=False,
            ontology_profile=unrestricted_profile,
        ),
    )

    assert len(result.terms) == 1
    assert result.terms[0].term_id == "HP:0003826"
    assert [w.code for w in result.terms[0].warnings] == ["modifier_as_annotation"]


def test_hpo_profile_rules_exclusive_severity_and_cohort_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    # Two chunks mention anemia with Mild and Severe -> dedup merges them ->
    # post-dedup validation flags exclusive_modifier_class.
    # Also "very frequent" on seizures matches HP:0040281 (Frequency) ->
    # flagged cohort_frequency and NOT attached.
    text = "Initially mild anemia. Later severe anemia. Very frequent seizures."
    chunks = [
        {
            "chunk_id": 1,
            "text": "Initially mild anemia.",
            "start_char": 0,
            "end_char": 22,
        },
        {
            "chunk_id": 2,
            "text": "Later severe anemia.",
            "start_char": 23,
            "end_char": 43,
        },
        {
            "chunk_id": 3,
            "text": "Very frequent seizures.",
            "start_char": 44,
            "end_char": 67,
        },
    ]
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "anemia",
                            "evidence_text": "mild anemia",
                            "modifiers": ["mild"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                        {
                            "phrase": "anemia",
                            "evidence_text": "severe anemia",
                            "modifiers": ["severe"],
                            "category": "Abnormal",
                            "chunk_ids": [2],
                        },
                        {
                            "phrase": "seizure",
                            "evidence_text": "Very frequent seizures",
                            "modifiers": ["Very frequent"],
                            "category": "Abnormal",
                            "chunk_ids": [3],
                        },
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "anemia": [{"hpo_id": "HP:0001903", "term_name": "Anemia", "score": 0.98}],
            "seizure": [
                {"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.97}
            ],
            "mild": [{"hpo_id": "HP:0012825", "term_name": "Mild", "score": 0.98}],
            "severe": [{"hpo_id": "HP:0012828", "term_name": "Severe", "score": 0.98}],
            "very frequent": [
                {"hpo_id": "HP:0040281", "term_name": "Very frequent", "score": 0.99}
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=chunks,
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    by_id = {term.term_id: term for term in result.terms}
    anemia = by_id["HP:0001903"]
    seizure = by_id["HP:0001250"]

    assert {m.term_id for m in anemia.modifiers} == {"HP:0012825", "HP:0012828"}
    assert any(w.code == "exclusive_modifier_class" for w in anemia.warnings)

    assert seizure.modifiers == []
    assert any(
        w.code == "cohort_frequency" and w.term_id == "HP:0040281"
        for w in seizure.warnings
    )


def test_batched_modifier_mapping_llm_and_safety_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_guard(monkeypatch)
    text = "Patient has worsening left-sided hearing loss."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "worsening left-sided hearing loss",
                            "modifiers": ["worsening", "left-sided"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            ),
            LLMBatchMappingSelections(
                mappings=[
                    LLMBatchMappingSelection(item_id="item_1", hpo_id="HP:0003676"),
                    LLMBatchMappingSelection(item_id="item_2", hpo_id="HP:0012833"),
                ]
            ),
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.96,
                }
            ],
            "worsening": [
                {"hpo_id": "HP:0003676", "term_name": "Progressive", "score": 0.77}
            ],
            "left-sided": [
                {"hpo_id": "HP:0012833", "term_name": "Unilateral", "score": 0.79}
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(
        provider=provider, tool_executor=executor, mapping_batch_size=10
    )
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    assert len(provider.calls) == 2
    assert provider.calls[1]["response_model"] is LLMBatchMappingSelections
    finding = result.terms[0]
    assert finding.evidence_records[0].evidence_text == "hearing loss"
    assert {m.term_id for m in finding.modifiers} == {"HP:0003676", "HP:0012833"}
    assert result.meta.phase_counts["modifier_llm_mapped"] == 2


def test_chunk_span_fallback_is_not_carved_into_fake_evidence_text(
    monkeypatch,
) -> None:
    """A ``chunk_span`` fallback must never be trimmed and written back as a quote (#397)."""
    _patch_guard(monkeypatch)
    text = "The patient reports mild tremor."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "tremor",
                            "evidence_text": None,
                            "modifiers": ["mild"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "tremor": [{"hpo_id": "HP:0001337", "term_name": "Tremor", "score": 0.96}],
            "mild": [{"hpo_id": "HP:0012825", "term_name": "Mild", "score": 0.96}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    finding = result.terms[0]
    ev = finding.evidence_records[0]
    assert ev.offset_basis == "chunk_span"
    assert ev.evidence_text is None
    assert (ev.start_char, ev.end_char) == (0, len(text))
    assert len(finding.modifiers) == 1
    assert finding.modifiers[0].term_id == "HP:0012825"
    assert [(s.start, s.end, s.text) for s in finding.modifiers[0].segments] == [
        (text.index("mild"), text.index("mild") + len("mild"), "mild")
    ]


def test_converted_modifier_finding_does_not_carve_chunk_span_target() -> None:
    """Converting a standalone modifier must not carve a ``chunk_span`` target."""
    text = "The patient has mild hearing loss."
    target = LLMPhenotype(
        term_id="HP:0000365",
        label="Hearing impairment",
        evidence="hearing loss",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="hearing loss",
                evidence_text=None,
                chunk_ids=[1],
                start_char=0,
                end_char=len(text),
                offset_basis="chunk_span",
            )
        ],
    )
    mod_start = text.index("mild")
    mod_term = LLMPhenotype(
        term_id="HP:0012825",
        label="Mild",
        evidence="mild",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="mild",
                evidence_text="mild",
                chunk_ids=[1],
                start_char=mod_start,
                end_char=mod_start + len("mild"),
                offset_basis="evidence_in_chunk",
            )
        ],
    )

    TwoPhaseLLMPipeline._attach_converted_modifier_finding(
        mod_term=mod_term,
        target=target,
        text=text,
    )

    ev = target.evidence_records[0]
    assert ev.offset_basis == "chunk_span"
    assert ev.evidence_text is None
    assert (ev.start_char, ev.end_char) == (0, len(text))
    assert [m.term_id for m in target.modifiers] == ["HP:0012825"]
    assert [(s.start, s.end, s.text) for s in target.modifiers[0].segments] == [
        (mod_start, mod_start + len("mild"), "mild")
    ]


def test_modifier_locator_rejects_mid_word_inflection_substring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uninflected stem matching inside an inflected token must not split a word."""
    _patch_guard(monkeypatch)
    text = "Patient presents with severely impaired hearing."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "severely impaired hearing",
                            "modifiers": ["severe"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            ),
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.96,
                }
            ],
            "severe": [{"hpo_id": "HP:0012828", "term_name": "Severe", "score": 0.98}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    finding = result.terms[0]
    assert finding.evidence_records[0].evidence_text == "severely impaired hearing"
    assert [m.term_id for m in finding.modifiers] == ["HP:0012828"]
    assert finding.modifiers[0].segments == []


def test_modifier_batch_failure_degrades_gracefully_without_losing_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception during modifier mapping must preserve already-resolved findings."""
    _patch_guard(monkeypatch)
    text = "Patient has worsening hearing loss."

    class _FailingModifierProvider(_FakeProvider):
        def run_structured_prompt(self, **kwargs: Any) -> Any:
            if kwargs.get("response_model") is LLMMappingSelection:
                raise RuntimeError("transient 503 during modifier batch")
            return super().run_structured_prompt(**kwargs)

    provider = _FailingModifierProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "hearing impairment",
                            "evidence_text": "worsening hearing loss",
                            "modifiers": ["worsening"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        }
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "hearing impairment": [
                {
                    "hpo_id": "HP:0000365",
                    "term_name": "Hearing impairment",
                    "score": 0.96,
                }
            ],
            "worsening": [
                {"hpo_id": "HP:0003676", "term_name": "Progressive", "score": 0.77}
            ],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    assert len(result.terms) == 1
    assert result.terms[0].term_id == "HP:0000365"
    assert result.terms[0].modifiers == []


def test_post_dedup_span_reconciliation_prevents_overlap_after_merging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging a carved and uncarved duplicate mention must keep spans disjoint."""
    _patch_guard(monkeypatch)
    text = "Severe seizures noted."
    provider = _FakeProvider(
        [
            LLMGroundedExtractedPhenotypesWithModifiers.model_validate(
                {
                    "phenotypes": [
                        {
                            "phrase": "seizure",
                            "evidence_text": "Severe seizures",
                            "modifiers": ["Severe"],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                        {
                            "phrase": "seizure",
                            "evidence_text": "Severe seizures",
                            "modifiers": [],
                            "category": "Abnormal",
                            "chunk_ids": [1],
                        },
                    ]
                }
            )
        ]
    )
    executor = _FakeToolExecutor(
        {
            "seizure": [
                {"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.98}
            ],
            "severe": [{"hpo_id": "HP:0012828", "term_name": "Severe", "score": 0.99}],
        }
    )
    pipeline = TwoPhaseLLMPipeline(provider=provider, tool_executor=executor)
    result = pipeline.run(
        text=text,
        grounded_chunks=_single_chunk(text),
        config=LLMPipelineConfig(model="test-model", extract_modifiers=True),
    )

    assert len(result.terms) == 1
    finding = result.terms[0]
    assert len(finding.modifiers) == 1
    mod_seg = finding.modifiers[0].segments[0]
    assert (mod_seg.start, mod_seg.end, mod_seg.text) == (0, 6, "Severe")
    for ev in finding.evidence_records:
        assert ev.start_char is not None and ev.end_char is not None
        _assert_no_overlap((ev.start_char, ev.end_char), (mod_seg.start, mod_seg.end))


def test_terms_are_colocated_ignores_adjacent_chunk_span_fallbacks() -> None:
    """Adjacent chunk_span fallbacks in different chunks are not colocated."""
    first = LLMPhenotype(
        term_id="HP:0012825",
        label="Mild",
        evidence="mild",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="mild",
                evidence_text=None,
                chunk_ids=[1],
                start_char=0,
                end_char=50,
                offset_basis="chunk_span",
            )
        ],
    )
    second = LLMPhenotype(
        term_id="HP:0001250",
        label="Seizure",
        evidence="seizure",
        evidence_records=[
            LLMPhenotypeEvidence(
                phrase="seizure",
                evidence_text=None,
                chunk_ids=[2],
                start_char=51,
                end_char=100,
                offset_basis="chunk_span",
            )
        ],
    )
    assert TwoPhaseLLMPipeline._terms_are_colocated(first, second) is False


def test_grounded_modifier_schema_describes_evidence_text_without_changing_base() -> (
    None
):
    """Only the modifier schema adds the ``evidence_text`` description."""
    from phentrieve.llm.types import (
        LLMGroundedExtractedPhenotype,
        LLMGroundedExtractedPhenotypeWithModifiers,
    )

    base_prop = LLMGroundedExtractedPhenotype.model_json_schema()["properties"][
        "evidence_text"
    ]
    mod_prop = LLMGroundedExtractedPhenotypeWithModifiers.model_json_schema()[
        "properties"
    ]["evidence_text"]

    assert "description" not in base_prop
    assert "verbatim quote" in mod_prop["description"]
