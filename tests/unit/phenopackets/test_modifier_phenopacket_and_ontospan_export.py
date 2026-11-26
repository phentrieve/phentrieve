from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket

from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
    LLMPhenotypeEvidence,
)
from phentrieve.modifier_models import Modifier, ModifierSegment
from phentrieve.phenopackets.export_models import (
    NormalizedModifier,
    NormalizedPhenotypeExportRecord,
)
from phentrieve.phenopackets.ontospan_export import (
    ONTOSPAN_ARTIFACT_TYPE,
    ONTOSPAN_SCHEMA_VERSION,
    canonicalize_text,
    compute_text_sha256,
    export_to_ontospan_document,
    serialize_ontospan_document,
)
from phentrieve.phenopackets.utils import (
    export_phenopacket_bundle,
    format_as_phenopacket_v2,
)

pytestmark = pytest.mark.unit

# Minimal synthetic HPO ancestry map for deterministic slot-routing tests.
_SYNTHETIC_ANCESTORS: dict[str, frozenset[str]] = {
    "HP:0012828": frozenset({"HP:0012824", "HP:0012823", "HP:0000001"}),  # Severe
    "HP:0012826": frozenset({"HP:0012824", "HP:0012823", "HP:0000001"}),  # Moderate
    "HP:0003577": frozenset(
        {"HP:0003674", "HP:0012823", "HP:0000001"}
    ),  # Congenital onset
    "HP:0011463": frozenset(
        {"HP:0003674", "HP:0012823", "HP:0000001"}
    ),  # Childhood onset
    "HP:0012833": frozenset({"HP:0012832", "HP:0012823", "HP:0000001"}),  # Unilateral
    "HP:0003676": frozenset({"HP:0031797", "HP:0012823", "HP:0000001"}),  # Progressive
}


class TestNormalizedModifier:
    def test_parses_pydantic_modifier_and_legacy_dicts(self) -> None:
        pydantic_mod = Modifier(
            term_id="HP:0012828",
            label="Severe",
            segments=[ModifierSegment(start=12, end=18, text="severe")],
            source="llm",
        )
        dict_mod = {
            "id": "HP:0012833",
            "name": "Unilateral",
            "segments": [{"start": 0, "end": 10, "text": "unilateral"}],
            "source": "rule",
        }
        legacy_span_dict = {
            "term_id": "HP:0003676",
            "label": "Progressive",
            "segments": [
                {"start_char": 20, "end_char": 31, "evidence_text": "progressive"}
            ],
        }

        record = NormalizedPhenotypeExportRecord.from_legacy_dict(
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "assertion": "present",
                "modifiers": [pydantic_mod, dict_mod, legacy_span_dict],
            }
        )

        assert len(record.modifiers) == 3
        assert record.modifiers[0] == NormalizedModifier(
            term_id="HP:0012828",
            label="Severe",
            segments=record.modifiers[0].segments,
            source="llm",
        )
        assert record.modifiers[0].segments[0].start_char == 12
        assert record.modifiers[0].segments[0].end_char == 18
        assert record.modifiers[0].segments[0].text == "severe"
        assert record.modifiers[1].term_id == "HP:0012833"
        assert record.modifiers[1].label == "Unilateral"
        assert record.modifiers[1].source == "rule"
        assert record.modifiers[2].segments[0].start_char == 20


class TestPhenopacketModifierExport:
    def test_routes_severity_onset_and_other_modifiers_with_protobuf_round_trip(
        self,
    ) -> None:
        aggregated = [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "confidence": 0.92,
                "rank": 1,
                "modifiers": [
                    {"term_id": "HP:0012828", "label": "Severe"},
                    {"term_id": "HP:0003577", "label": "Congenital onset"},
                    {"term_id": "HP:0003676", "label": "Progressive"},
                    {"term_id": "HP:0012833", "label": "Unilateral"},
                ],
            }
        ]

        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated,
            hpo_version="v2025-01-01",
            ancestors_map=_SYNTHETIC_ANCESTORS,
        )
        packet_dict = json.loads(phenopacket_json)
        feature = packet_dict["phenotypicFeatures"][0]

        assert feature["severity"] == {"id": "HP:0012828", "label": "Severe"}
        assert feature["onset"] == {
            "ontologyClass": {"id": "HP:0003577", "label": "Congenital onset"}
        }
        assert feature["modifiers"] == [
            {"id": "HP:0003676", "label": "Progressive"},
            {"id": "HP:0012833", "label": "Unilateral"},
        ]

        parsed = Phenopacket()
        Parse(phenopacket_json, parsed, ignore_unknown_fields=False)
        assert parsed.phenotypic_features[0].severity.id == "HP:0012828"
        assert parsed.phenotypic_features[0].onset.ontology_class.id == "HP:0003577"
        assert [m.id for m in parsed.phenotypic_features[0].modifiers] == [
            "HP:0003676",
            "HP:0012833",
        ]

    def test_second_severity_and_second_onset_fall_back_to_modifiers(self) -> None:
        aggregated = [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "confidence": 0.9,
                "rank": 1,
                "modifiers": [
                    {"term_id": "HP:0012828", "label": "Severe"},
                    {"term_id": "HP:0012826", "label": "Moderate"},
                    {"term_id": "HP:0003577", "label": "Congenital onset"},
                    {"term_id": "HP:0011463", "label": "Childhood onset"},
                ],
            }
        ]

        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated,
            hpo_version="v2025-01-01",
            ancestors_map=_SYNTHETIC_ANCESTORS,
        )
        feature = json.loads(phenopacket_json)["phenotypicFeatures"][0]

        assert feature["severity"] == {"id": "HP:0012828", "label": "Severe"}
        assert feature["onset"] == {
            "ontologyClass": {"id": "HP:0003577", "label": "Congenital onset"}
        }
        assert feature["modifiers"] == [
            {"id": "HP:0012826", "label": "Moderate"},
            {"id": "HP:0011463", "label": "Childhood onset"},
        ]

    def test_collapsed_same_polarity_records_merge_modifiers_deduped(self) -> None:
        chunk_results = [
            {
                "chunk_idx": 0,
                "chunk_text": "Severe progressive seizures reported.",
                "matches": [
                    {
                        "id": "HP:0001250",
                        "name": "Seizure",
                        "score": 0.91,
                        "assertion_status": "affirmed",
                        "modifiers": [
                            {"term_id": "HP:0012828", "label": "Severe"},
                        ],
                    }
                ],
            },
            {
                "chunk_idx": 1,
                "chunk_text": "Progressive unilateral seizures persisted.",
                "matches": [
                    {
                        "id": "HP:0001250",
                        "name": "Seizure",
                        "score": 0.87,
                        "assertion_status": "affirmed",
                        "modifiers": [
                            {"term_id": "HP:0012828", "label": "Severe"},
                            {"term_id": "HP:0003676", "label": "Progressive"},
                            {"term_id": "HP:0012833", "label": "Unilateral"},
                        ],
                    }
                ],
            },
            {
                "chunk_idx": 2,
                "chunk_text": "No congenital seizures.",
                "matches": [
                    {
                        "id": "HP:0001250",
                        "name": "Seizure",
                        "score": 0.80,
                        "assertion_status": "negated",
                        "modifiers": [
                            {"term_id": "HP:0003577", "label": "Congenital onset"},
                        ],
                    }
                ],
            },
        ]

        bundle = export_phenopacket_bundle(
            chunk_results=chunk_results,
            hpo_version="v2025-01-01",
            ancestors_map=_SYNTHETIC_ANCESTORS,
        )
        packet_dict = json.loads(bundle["phenopacket_json"])

        assert len(packet_dict["phenotypicFeatures"]) == 1
        feature = packet_dict["phenotypicFeatures"][0]
        assert feature["severity"] == {"id": "HP:0012828", "label": "Severe"}
        assert "onset" not in feature
        assert feature["modifiers"] == [
            {"id": "HP:0003676", "label": "Progressive"},
            {"id": "HP:0012833", "label": "Unilateral"},
        ]

    def test_empty_ancestors_map_falls_back_to_modifiers_slot(self) -> None:
        aggregated = [
            {
                "id": "HP:0001250",
                "name": "Seizure",
                "confidence": 0.85,
                "rank": 1,
                "modifiers": [
                    {"term_id": "HP:0012828", "label": "Severe"},
                    {"term_id": "HP:0003577", "label": "Congenital onset"},
                ],
            }
        ]

        phenopacket_json = format_as_phenopacket_v2(
            aggregated_results=aggregated,
            hpo_version="v2025-01-01",
            ancestors_map={},
        )
        feature = json.loads(phenopacket_json)["phenotypicFeatures"][0]

        assert "severity" not in feature
        assert "onset" not in feature
        assert feature["modifiers"] == [
            {"id": "HP:0012828", "label": "Severe"},
            {"id": "HP:0003577", "label": "Congenital onset"},
        ]


class TestOntospanExport:
    def test_exports_valid_1_2_0_document_with_axes_mentions_and_stripped_modifiers(
        self,
    ) -> None:
        raw_text = (
            "\ufeffSevere unilateral tremor.\r\nNo fever.\r\nMother had migraine."
        )
        canonical = canonicalize_text(raw_text)
        expected_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert compute_text_sha256(canonical) == expected_sha

        tremor_start = canonical.index("tremor")
        tremor_end = tremor_start + len("tremor")
        severe_start = canonical.index("Severe")
        severe_end = severe_start + len("Severe")
        unilateral_start = canonical.index("unilateral")
        unilateral_end = unilateral_start + len("unilateral")

        findings = [
            {
                "term_id": "HP:0001337",
                "label": "Tremor",
                "assertion": "present",
                "experiencer": "proband",
                "temporality": "current",
                "evidence_records": [
                    {
                        "start_char": tremor_start,
                        "end_char": tremor_end,
                        "evidence_text": "tremor",
                    }
                ],
                "modifiers": [
                    Modifier(
                        term_id="HP:0012828",
                        label="Severe",
                        segments=[
                            ModifierSegment(
                                start=severe_start, end=severe_end, text="Severe"
                            )
                        ],
                        source="llm",
                    ),
                    {
                        "term_id": "HP:0012833",
                        "label": "Unilateral",
                        "segments": [
                            {
                                "start": unilateral_start,
                                "end": unilateral_end,
                                "text": "unilateral",
                            }
                        ],
                        "source": "rule",
                    },
                    # Duplicate term_id should be ignored.
                    {"term_id": "HP:0012828", "label": "Severe", "segments": []},
                ],
            },
            {
                "term_id": "HP:0001945",
                "label": "Fever",
                "assertion": "negated",
                "experiencer": "proband",
                "temporality": "unknown",
                # Unlocated finding -> must become document_level=True with empty mentions.
                "evidence_records": [
                    {
                        "start_char": 0,
                        "end_char": 5,
                        "evidence_text": "mismatch",
                    }
                ],
            },
            {
                "term_id": "HP:0002076",
                "label": "Migraine",
                "assertion": "uncertain",
                "experiencer": "family_history",
                "temporality": "historical",
            },
        ]

        fixed_time = datetime(2026, 5, 1, 12, 30, 0, tzinfo=UTC)
        doc = export_to_ontospan_document(
            findings,
            raw_text,
            document_id="case-001",
            language="en",
            annotation_set_id="set-001",
            subject_id="subj-001",
            hpo_version="2026-01-01",
            phentrieve_version="0.16.0",
            model="test-model",
            created_at=fixed_time,
        )

        assert doc["schema_version"] == ONTOSPAN_SCHEMA_VERSION == "1.2.0"
        assert doc["artifact_type"] == ONTOSPAN_ARTIFACT_TYPE == "annotated_document"
        assert doc["document"] == {
            "document_id": "case-001",
            "language": "en",
            "subject_id": "subj-001",
            "text": canonical,
            "text_sha256": expected_sha,
        }

        ann_set = doc["annotation_set"]
        assert ann_set["annotation_set_id"] == "set-001"
        assert ann_set["set_kind"] == "tool_output"
        assert ann_set["created_at"] == "2026-05-01T12:30:00Z"
        assert ann_set["layers"] == [
            {
                "layer_id": "hpo",
                "profile": "hpo-phenotype",
                "ontology": {"id": "hp", "version": "2026-01-01"},
                "allowed_roots": ["HP:0000118"],
            }
        ]
        assert len(ann_set["annotations"]) == 3

        ann_tremor, ann_fever, ann_migraine = ann_set["annotations"]

        assert ann_tremor["document_level"] is False
        assert ann_tremor["mentions"] == [
            {
                "segments": [
                    {"start": tremor_start, "end": tremor_end, "text": "tremor"}
                ],
                "cue_segments": [],
            }
        ]
        assert ann_tremor["axes"] == {
            "assertion": {"value": "present", "basis": "chosen"},
            "experiencer": {"value": "patient", "basis": "chosen"},
            "temporality": {"value": "current", "basis": "chosen"},
        }
        assert ann_tremor["modifiers"] == [
            {
                "term_id": "HP:0012828",
                "segments": [
                    {"start": severe_start, "end": severe_end, "text": "Severe"}
                ],
            },
            {
                "term_id": "HP:0012833",
                "segments": [
                    {
                        "start": unilateral_start,
                        "end": unilateral_end,
                        "text": "unilateral",
                    }
                ],
            },
        ]
        for exported_mod in ann_tremor["modifiers"]:
            assert set(exported_mod.keys()) == {"term_id", "segments"}
            assert "label" not in exported_mod
            assert "source" not in exported_mod

        assert ann_fever["document_level"] is True
        assert ann_fever["mentions"] == []
        assert ann_fever["axes"]["assertion"] == {"value": "absent", "basis": "chosen"}
        assert ann_fever["axes"]["experiencer"] == {
            "value": "patient",
            "basis": "chosen",
        }
        assert ann_fever["axes"]["temporality"] == {
            "value": "unknown",
            "basis": "chosen",
        }

        assert ann_migraine["document_level"] is True
        assert ann_migraine["mentions"] == []
        assert ann_migraine["axes"] == {
            "assertion": {"value": "uncertain", "basis": "chosen"},
            "experiencer": {"value": "family_member", "basis": "chosen"},
            "temporality": {"value": "historical", "basis": "chosen"},
        }

        serialized = serialize_ontospan_document(doc)
        assert serialized.endswith("\n")
        assert json.loads(serialized) == doc

    def test_exports_from_llm_extraction_result_including_family_and_other(
        self,
    ) -> None:
        text = "Patient has ataxia. Sister had migraine. Neighbor has rash."
        ataxia_start = text.index("ataxia")
        result = LLMExtractionResult(
            terms=[
                LLMPhenotype(
                    term_id="HP:0001251",
                    label="Ataxia",
                    assertion="present",
                    experiencer="proband",
                    temporality="current",
                    evidence_records=[
                        LLMPhenotypeEvidence(
                            phrase="ataxia",
                            evidence_text="ataxia",
                            start_char=ataxia_start,
                            end_char=ataxia_start + len("ataxia"),
                        )
                    ],
                )
            ],
            family_history_findings=[
                LLMPhenotype(
                    term_id="HP:0002076",
                    label="Migraine",
                    assertion="present",
                    experiencer="family_history",
                    temporality="historical",
                )
            ],
            other_person_findings=[
                LLMPhenotype(
                    term_id="HP:0000988",
                    label="Skin rash",
                    assertion="present",
                    experiencer="other",
                    temporality="current",
                )
            ],
            meta=LLMMeta(llm_model="synthetic-model", llm_mode="two_phase"),
        )

        doc = export_to_ontospan_document(
            result,
            text,
            document_id="case-002",
            language="en",
            hpo_version="2026-01-01",
            phentrieve_version="0.16.0",
        )

        assert doc["annotation_set"]["actors"][0]["model"] == "synthetic-model"
        annotations = doc["annotation_set"]["annotations"]
        assert [a["term_id"] for a in annotations] == [
            "HP:0001251",
            "HP:0002076",
            "HP:0000988",
        ]
        assert [a["axes"]["experiencer"]["value"] for a in annotations] == [
            "patient",
            "family_member",
            "other",
        ]
        assert annotations[0]["document_level"] is False
        assert annotations[1]["document_level"] is True
        assert annotations[2]["document_level"] is True

    def test_chunk_span_fallback_exports_as_document_level_not_exact_mention(
        self,
    ) -> None:
        text = "The patient reports mild tremor."
        result = LLMExtractionResult(
            terms=[
                LLMPhenotype(
                    term_id="HP:0001337",
                    label="Tremor",
                    assertion="present",
                    experiencer="proband",
                    temporality="current",
                    evidence_records=[
                        LLMPhenotypeEvidence(
                            phrase="tremor",
                            evidence_text=None,
                            chunk_ids=[1],
                            start_char=0,
                            end_char=len(text),
                            offset_basis="chunk_span",
                        )
                    ],
                )
            ],
            meta=LLMMeta(llm_model="synthetic-model", llm_mode="two_phase"),
        )

        doc = export_to_ontospan_document(
            result,
            text,
            document_id="case-003",
            language="en",
            hpo_version="2026-01-01",
            phentrieve_version="0.33.0",
        )
        ann = doc["annotation_set"]["annotations"][0]
        assert ann["mentions"] == []
        assert ann["document_level"] is True

    def test_negated_qualifier_derived_exports_as_document_level_not_source_span(
        self,
    ) -> None:
        text = "Episodic urticaria without angioedema."
        urticaria_start = text.index("Episodic urticaria")
        urticaria_end = urticaria_start + len("Episodic urticaria")
        result = LLMExtractionResult(
            terms=[
                LLMPhenotype(
                    term_id="HP:0100665",
                    label="Angioedema",
                    assertion="negated",
                    experiencer="proband",
                    temporality="unknown",
                    match_method="negated_qualifier_derived",
                    qualifier_surface_text="angioedema",
                    evidence_records=[
                        LLMPhenotypeEvidence(
                            phrase="episodic urticaria",
                            evidence_text="Episodic urticaria",
                            chunk_ids=[1],
                            start_char=urticaria_start,
                            end_char=urticaria_end,
                            offset_basis="evidence_in_chunk",
                        )
                    ],
                )
            ],
            meta=LLMMeta(llm_model="synthetic-model", llm_mode="two_phase"),
        )

        doc = export_to_ontospan_document(
            result,
            text,
            document_id="case-004",
            language="en",
            hpo_version="2026-01-01",
            phentrieve_version="0.33.0",
        )
        ann = doc["annotation_set"]["annotations"][0]
        assert ann["mentions"] == []
        assert ann["document_level"] is True
        assert ann["axes"]["assertion"]["value"] == "absent"

    def test_hpo_version_defaults_from_db_and_normalizes_leading_v(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "phentrieve.phenopackets.ontospan_export._read_hpo_version_from_db",
            lambda: "v2026-09-01",
        )
        doc = export_to_ontospan_document(
            [],
            "Mild tremor.",
            document_id="case-005",
            language="en",
            phentrieve_version="0.33.0",
        )
        assert doc["annotation_set"]["layers"][0]["ontology"]["version"] == "2026-09-01"

        explicit = export_to_ontospan_document(
            [],
            "Mild tremor.",
            document_id="case-006",
            language="en",
            hpo_version="v2026-09-01",
            phentrieve_version="0.33.0",
        )
        assert (
            explicit["annotation_set"]["layers"][0]["ontology"]["version"]
            == "v2026-09-01"
        )

    def test_maps_assertion_synonyms_remaps_crlf_bom_offsets_and_drops_overlapping_mod_seg(
        self,
    ) -> None:
        raw_text = (
            "\ufeffLine one.\r\nSevere tremor.\r\nPossible ataxia.\r\nDenied fever."
        )
        tremor_raw_start = raw_text.index("tremor")
        tremor_raw_end = tremor_raw_start + len("tremor")
        severe_raw_start = raw_text.index("Severe")
        severe_raw_end = severe_raw_start + len("Severe")

        findings = [
            {
                "term_id": "HP:0001337",
                "label": "Tremor",
                "assertion": "present",
                "evidence_records": [
                    {
                        "start_char": tremor_raw_start,
                        "end_char": tremor_raw_end,
                        "evidence_text": "tremor",
                    }
                ],
                "modifiers": [
                    {
                        "term_id": "HP:0012828",
                        "label": "Severe",
                        "segments": [
                            {
                                "start": severe_raw_start,
                                "end": severe_raw_end,
                                "text": "Severe",
                            },
                            # Overlapping segment must be dropped by export guard.
                            {
                                "start": tremor_raw_start,
                                "end": tremor_raw_end,
                                "text": "tremor",
                            },
                        ],
                    }
                ],
            },
            {"term_id": "HP:0001251", "label": "Ataxia", "assertion": "possible"},
            {"term_id": "HP:0001945", "label": "Fever", "assertion": "denied"},
        ]

        doc = export_to_ontospan_document(
            findings,
            raw_text,
            document_id="case-007",
            language="en",
            hpo_version="2026-09-01",
            phentrieve_version="0.33.0",
        )
        canonical = doc["document"]["text"]
        ann_tremor, ann_ataxia, ann_fever = doc["annotation_set"]["annotations"]

        assert ann_tremor["document_level"] is False
        seg = ann_tremor["mentions"][0]["segments"][0]
        assert canonical[seg["start"] : seg["end"]] == "tremor"
        mod_segs = ann_tremor["modifiers"][0]["segments"]
        assert len(mod_segs) == 1
        assert canonical[mod_segs[0]["start"] : mod_segs[0]["end"]] == "Severe"

        assert ann_ataxia["axes"]["assertion"]["value"] == "uncertain"
        assert ann_fever["axes"]["assertion"]["value"] == "absent"
