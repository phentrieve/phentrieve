"""Unit tests for clinical modifier API schemas, execution, and Phenopacket export."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket
from pydantic import ValidationError

from api.main import app
from api.schemas.text_processing_schemas import (
    ModifierSegmentAPI,
    TextProcessingRequest,
)
from phentrieve.llm.types import (
    LLMExtractionResult,
    LLMMeta,
    LLMPhenotype,
)
from phentrieve.modifier_models import Modifier, ModifierSegment, OntologyWarning
from phentrieve.text_processing.full_text_service import run_llm_backend

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _stub_llm_path_retriever(monkeypatch):
    """Keep the LLM path from loading a real model/index for its cached retriever."""
    monkeypatch.setattr(
        "api.services.text_processing_execution.get_dense_retriever_dependency",
        AsyncMock(return_value=MagicMock(model=MagicMock())),
    )


def test_text_processing_request_extract_modifiers_default_and_opt_in() -> None:
    default_req = TextProcessingRequest(text="Invented case note.")
    assert default_req.extract_modifiers is False

    opt_in_req = TextProcessingRequest(
        text="Invented case note.",
        extraction_backend="llm",
        extract_modifiers=True,
    )
    assert opt_in_req.extract_modifiers is True


def test_modifier_segment_api_validates_bounds() -> None:
    valid = ModifierSegmentAPI(start=0, end=6, text="Severe")
    assert valid.start == 0
    assert valid.end == 6
    assert valid.text == "Severe"

    with pytest.raises(ValidationError):
        ModifierSegmentAPI(start=6, end=6, text="Severe")

    with pytest.raises(ValidationError):
        ModifierSegmentAPI(start=0, end=5, text="Severe")


def test_text_processing_router_passes_extract_modifiers_and_returns_modifiers(
    client, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {
                "extraction_backend": "llm",
                "llm_provider": kwargs["llm_provider"],
                "llm_model": kwargs["llm_model"],
                "llm_mode": kwargs["llm_mode"],
                "extract_modifiers": True,
            },
            "processed_chunks": [
                {
                    "chunk_id": 1,
                    "text": "Severe congenital bilateral hypotonia.",
                    "status": "affirmed",
                    "hpo_matches": [
                        {"id": "HP:0001252", "name": "Hypotonia", "score": 0.94}
                    ],
                    "start_char": 0,
                    "end_char": 38,
                }
            ],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "confidence": 0.94,
                    "status": "present",
                    "experiencer": "proband",
                    "temporality": "current",
                    "excluded": False,
                    "evidence_count": 1,
                    "source_chunk_ids": [1],
                    "top_evidence_chunk_id": 1,
                    "text_attributions": [
                        {
                            "chunk_id": 1,
                            "start_char": 28,
                            "end_char": 37,
                            "matched_text_in_chunk": "hypotonia",
                        }
                    ],
                    "modifiers": [
                        {
                            "term_id": "HP:0012828",
                            "label": "Severe",
                            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
                            "source": "llm",
                        },
                        {
                            "term_id": "HP:0012832",
                            "label": "Bilateral",
                            "segments": [],
                            "source": "rule",
                        },
                    ],
                    "warnings": [
                        {
                            "code": "non_canonical_modifier_branch",
                            "message": "Invented warning for verification.",
                            "term_id": "HP:0012832",
                        }
                    ],
                    "score": 0.94,
                }
            ],
        }

    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        fake_run_full_text_service,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Severe congenital bilateral hypotonia.",
            "extraction_backend": "llm",
            "extract_modifiers": True,
        },
    )

    assert response.status_code == 200
    assert captured["extract_modifiers"] is True
    payload = response.json()
    assert payload["meta"]["extract_modifiers"] is True
    term = payload["aggregated_hpo_terms"][0]
    assert term["modifiers"] == [
        {
            "term_id": "HP:0012828",
            "label": "Severe",
            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
            "source": "llm",
        },
        {
            "term_id": "HP:0012832",
            "label": "Bilateral",
            "segments": [],
            "source": "rule",
        },
    ]
    assert term["warnings"] == [
        {
            "code": "non_canonical_modifier_branch",
            "message": "Invented warning for verification.",
            "term_id": "HP:0012832",
        }
    ]


def test_text_processing_router_defaults_extract_modifiers_false(
    client, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {
                "extraction_backend": "llm",
                "llm_provider": kwargs["llm_provider"],
                "llm_model": kwargs["llm_model"],
                "llm_mode": kwargs["llm_mode"],
            },
            "processed_chunks": [
                {
                    "chunk_id": 1,
                    "text": "Hypotonia noted.",
                    "status": "affirmed",
                    "hpo_matches": [
                        {"id": "HP:0001252", "name": "Hypotonia", "score": 0.9}
                    ],
                    "start_char": 0,
                    "end_char": 16,
                }
            ],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "confidence": 0.9,
                    "status": "present",
                    "evidence_count": 1,
                    "source_chunk_ids": [1],
                    "top_evidence_chunk_id": 1,
                    "text_attributions": [],
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(
        "api.services.text_processing_execution.run_full_text_service",
        fake_run_full_text_service,
    )

    response = client.post(
        "/api/v1/text/process",
        json={
            "text": "Hypotonia noted.",
            "extraction_backend": "llm",
        },
    )

    assert response.status_code == 200
    assert captured["extract_modifiers"] is False
    term = response.json()["aggregated_hpo_terms"][0]
    assert term["modifiers"] == []
    assert term["warnings"] == []


def test_run_llm_backend_extract_modifiers_wires_prompt_config_and_output(
    mocker,
) -> None:
    provider = mocker.Mock()
    provider.provider_name = "gemini"
    provider.model_name = "gemini-3.1-flash-lite"
    provider.base_url = None
    provider.count_tokens.return_value = {"prompt_tokens": 10, "total_tokens": 10}

    pipeline = mocker.Mock()
    pipeline.run.return_value = LLMExtractionResult(
        terms=[
            LLMPhenotype(
                term_id="HP:0001252",
                label="Hypotonia",
                evidence="hypotonia",
                assertion="present",
                evidence_records=[
                    {
                        "phrase": "hypotonia",
                        "evidence_text": "hypotonia",
                        "chunk_ids": [1],
                        "start_char": 7,
                        "end_char": 16,
                        "match_method": "local",
                    }
                ],
                modifiers=[
                    Modifier(
                        term_id="HP:0012828",
                        label="Severe",
                        segments=[ModifierSegment(start=0, end=6, text="Severe")],
                        source="llm",
                    )
                ],
                warnings=[
                    OntologyWarning(
                        code="modifier_ungrounded",
                        message="Invented warning message.",
                        term_id="HP:0012828",
                    )
                ],
            )
        ],
        meta=LLMMeta(
            llm_provider="gemini",
            llm_model="gemini-3.1-flash-lite",
            llm_mode="two_phase",
        ),
    )

    mod_prompt = mocker.Mock()
    mod_prompt.render_system_prompt.return_value = "modifier system prompt"
    mod_prompt.render_grounded_user_prompt.return_value = "grounded user prompt"
    mod_prompt.render_user_prompt.return_value = "user prompt"
    std_prompt = mocker.Mock()

    get_mod_prompt = mocker.patch(
        "phentrieve.text_processing.full_text_service.get_modifier_extraction_prompt",
        return_value=mod_prompt,
    )
    get_std_prompt = mocker.patch(
        "phentrieve.text_processing.full_text_service.get_prompt",
        return_value=std_prompt,
    )
    preprocess = mocker.patch(
        "phentrieve.text_processing.full_text_service.preprocess_grounded_document",
        return_value=SimpleNamespace(
            grounded_chunks=[
                {
                    "chunk_id": 1,
                    "text": "Severe hypotonia.",
                    "start_char": 0,
                    "end_char": 17,
                    "status": "affirmed",
                }
            ],
            extraction_groups=[],
        ),
    )

    result = run_llm_backend(
        text="Severe hypotonia.",
        language="en",
        extract_modifiers=True,
        provider_factory=lambda **_kwargs: provider,
        pipeline_factory=lambda **_kwargs: pipeline,
    )

    get_mod_prompt.assert_called_once_with("en")
    get_std_prompt.assert_not_called()
    assert preprocess.call_args.kwargs["extraction_prompt"] is mod_prompt
    assert pipeline.run.call_args.kwargs["config"].extract_modifiers is True
    assert result["meta"]["extract_modifiers"] is True
    term = result["aggregated_hpo_terms"][0]
    assert term["modifiers"] == [
        {
            "term_id": "HP:0012828",
            "label": "Severe",
            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
            "source": "llm",
        }
    ]
    assert term["warnings"] == [
        {
            "code": "modifier_ungrounded",
            "message": "Invented warning message.",
            "term_id": "HP:0012828",
        }
    ]


def test_phenopacket_export_router_emits_severity_onset_and_modifiers(
    client, monkeypatch
) -> None:
    fake_ancestors = {
        "HP:0012828": frozenset({"HP:0012824", "HP:0012823", "HP:0000001"}),
        "HP:0003577": frozenset({"HP:0003674", "HP:0012823", "HP:0000001"}),
        "HP:0012832": frozenset({"HP:0012831", "HP:0012823", "HP:0000001"}),
    }
    monkeypatch.setattr(
        "phentrieve.phenopackets.utils._load_ancestors_map_safe",
        lambda: fake_ancestors,
    )

    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-modifiers-1",
            "input_text": "Severe congenital bilateral hypotonia.",
            "phenotypes": [
                {
                    "hpo_id": "HP:0001252",
                    "label": "Hypotonia",
                    "assertion_status": "affirmed",
                    "confidence": 0.93,
                    "modifiers": [
                        {
                            "term_id": "HP:0012828",
                            "label": "Severe",
                            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
                            "source": "llm",
                        },
                        {
                            "term_id": "HP:0003577",
                            "label": "Congenital onset",
                            "segments": [{"start": 7, "end": 17, "text": "congenital"}],
                            "source": "llm",
                        },
                        {
                            "term_id": "HP:0012832",
                            "label": "Bilateral",
                            "segments": [],
                            "source": "rule",
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    Parse(payload["phenopacket_json"], Phenopacket(), ignore_unknown_fields=False)

    phenopacket = json.loads(payload["phenopacket_json"])
    feature = phenopacket["phenotypicFeatures"][0]
    assert feature["type"] == {"id": "HP:0001252", "label": "Hypotonia"}
    assert feature["severity"] == {"id": "HP:0012828", "label": "Severe"}
    assert feature["onset"] == {
        "ontologyClass": {"id": "HP:0003577", "label": "Congenital onset"}
    }
    assert feature["modifiers"] == [{"id": "HP:0012832", "label": "Bilateral"}]


@pytest.mark.parametrize(
    ("modifier_payload", "expected_fragment"),
    [
        (
            {
                "term_id": "not-a-curie",
                "label": "Severe",
            },
            "term_id",
        ),
        (
            {
                "term_id": "HP:0012828",
                "label": "Severe",
                "segments": [{"start": 0, "end": 4, "text": "Severe"}],
            },
            "end - start",
        ),
    ],
)
def test_phenopacket_export_router_rejects_invalid_modifiers(
    client, modifier_payload, expected_fragment
) -> None:
    response = client.post(
        "/api/v1/phenopackets/export",
        json={
            "case_id": "case-invalid-modifier",
            "phenotypes": [
                {
                    "hpo_id": "HP:0001252",
                    "label": "Hypotonia",
                    "modifiers": [modifier_payload],
                }
            ],
        },
    )

    assert response.status_code == 422
    assert expected_fragment in response.text


def test_mcp_coerce_export_phenotype_forwards_valid_and_rejects_falsy_modifiers() -> (
    None
):
    from pydantic import ValidationError

    from api.mcp.service_adapters import _coerce_export_phenotype
    from api.schemas.phenopacket_schemas import ExportPhenotypeRequest

    coerced = _coerce_export_phenotype(
        ExportPhenotypeRequest,
        {
            "hpo_id": "HP:0001252",
            "label": "Hypotonia",
            "modifiers": [{"term_id": "HP:0012828", "label": "Severe"}],
        },
        0,
    )
    assert coerced is not None
    assert [m.term_id for m in coerced.modifiers] == ["HP:0012828"]

    with pytest.raises(ValidationError):
        _coerce_export_phenotype(
            ExportPhenotypeRequest,
            {
                "hpo_id": "HP:0001252",
                "label": "Hypotonia",
                "modifiers": False,
            },
            0,
        )
