from __future__ import annotations

import json
import logging

import pytest

from phentrieve.benchmark.data_loader import (
    _get_case_text,
    _validate_document_benchmark_payload,
    load_json_benchmark_data,
    load_phenobert_data,
    normalize_benchmark_assertion,
    parse_gold_terms,
)


def test_load_json_benchmark_data_uses_generic_list_error(tmp_path) -> None:
    test_file = tmp_path / "cases.json"
    test_file.write_text('{"unexpected": true}', encoding="utf-8")

    with pytest.raises(
        ValueError, match="Benchmark JSON objects must contain a 'documents' list."
    ):
        load_json_benchmark_data(test_file)


def test_validate_document_benchmark_payload_uses_generic_empty_error() -> None:
    with pytest.raises(ValueError, match="No benchmark documents found"):
        _validate_document_benchmark_payload({"documents": []}, source_name="sample")


def test_get_case_text_uses_generic_non_empty_text_error() -> None:
    with pytest.raises(
        ValueError, match="Each benchmark case must provide non-empty text."
    ):
        _get_case_text({})


def test_load_phenobert_data_accepts_raghpo_paper_datasets(tmp_path) -> None:
    annotations_dir = tmp_path / "CSC" / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "CSC_1.json").write_text(
        json.dumps(
            {
                "doc_id": "CSC_1",
                "full_text": "Patient has obesity.",
                "annotations": [
                    {
                        "hpo_id": "HP:0001513",
                        "label": "Obesity",
                        "assertion_status": "affirmed",
                        "evidence_spans": [],
                    },
                    {
                        "hpo_id": "HP:0001250",
                        "label": "Seizure",
                        "assertion_status": "family_history",
                        "evidence_spans": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = load_phenobert_data(tmp_path, dataset="CSC")

    assert payload["metadata"]["dataset_name"] == "rag_hpo_CSC"
    assert payload["metadata"]["source"] == "rag_hpo_paper"
    assert payload["metadata"]["dataset_namespace"] == "rag_hpo_paper"
    assert payload["documents"][0]["id"] == "CSC_1"
    assert payload["documents"][0]["source_dataset"] == "CSC"
    assert payload["documents"][0]["gold_hpo_terms"] == [
        {
            "id": "HP:0001513",
            "label": "Obesity",
            "assertion": "PRESENT",
            "evidence_spans": [],
        }
    ]


def _write_directory_document(base_dir, dataset: str, doc_id: str) -> None:
    annotations_dir = base_dir / dataset / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / f"{doc_id}.json").write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "full_text": "The child has a wide thumb.",
                "annotations": [
                    {
                        "hpo_id": "HP:0011304",
                        "label": "Broad thumb",
                        "assertion_status": "affirmed",
                        "evidence_spans": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _incomplete_copy_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "#520" in record.getMessage()
    ]


@pytest.mark.parametrize("raw", ["normal", " Normal ", "NORMAL"])
def test_normal_assertion_scores_as_absent(raw) -> None:
    assert normalize_benchmark_assertion(raw, reject_unknown=True) == "ABSENT"
    assert normalize_benchmark_assertion(raw, reject_unknown=False) == "ABSENT"


def test_normal_gold_term_is_parsed_as_absent() -> None:
    assert parse_gold_terms([{"id": "HP:0011304", "assertion": "normal"}]) == [
        ("HP:0011304", "ABSENT")
    ]


@pytest.mark.parametrize("dataset", ["ID_68", "all"])
def test_load_phenobert_data_warns_once_for_incomplete_id_68(
    tmp_path, caplog, dataset
) -> None:
    _write_directory_document(tmp_path, "ID_68", "ID_68_1")
    _write_directory_document(tmp_path, "ID_68", "ID_68_2")
    _write_directory_document(tmp_path, "GeneReviews", "GeneReviews_1")

    with caplog.at_level(logging.WARNING, logger="phentrieve.benchmark.data_loader"):
        payload = load_phenobert_data(tmp_path, dataset=dataset)

    assert sum(doc["source_dataset"] == "ID_68" for doc in payload["documents"]) == 2
    warnings = _incomplete_copy_warnings(caplog)
    assert len(warnings) == 1
    assert "ID-68" in warnings[0]
    assert "incomplete" in warnings[0]
    assert "reported results" in warnings[0]


def test_load_phenobert_data_does_not_warn_for_other_dataset(tmp_path, caplog) -> None:
    _write_directory_document(tmp_path, "ID_68", "ID_68_1")
    _write_directory_document(tmp_path, "GeneReviews", "GeneReviews_1")

    with caplog.at_level(logging.WARNING, logger="phentrieve.benchmark.data_loader"):
        payload = load_phenobert_data(tmp_path, dataset="GeneReviews")

    assert [doc["id"] for doc in payload["documents"]] == ["GeneReviews_1"]
    assert _incomplete_copy_warnings(caplog) == []
