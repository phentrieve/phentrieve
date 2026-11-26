"""Unit tests for CLI --extract-modifiers support on `phentrieve text process`."""

from __future__ import annotations

import json

import pytest
from google.protobuf.json_format import Parse
from phenopackets import Phenopacket
from typer.testing import CliRunner

from phentrieve.cli import app
from phentrieve.profiles import Profile

pytestmark = pytest.mark.unit


def _clear_yaml_cache() -> None:
    from phentrieve.config import _load_yaml_config
    from phentrieve.utils import load_user_config

    _load_yaml_config.cache_clear()
    load_user_config.cache_clear()


def test_cli_extract_modifiers_flag_passes_true_and_emits_json_lines(
    monkeypatch,
) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {
                "extraction_backend": "llm",
                "llm_model": "gemini-3.1-flash-lite",
                "llm_mode": "two_phase",
                "extract_modifiers": True,
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "status": "present",
                    "score": 0.92,
                    "modifiers": [
                        {
                            "term_id": "HP:0012828",
                            "label": "Severe",
                            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
                            "source": "llm",
                        }
                    ],
                    "warnings": [
                        {
                            "code": "invented_warning",
                            "message": "Synthetic warning.",
                            "term_id": "HP:0012828",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        fake_run_full_text_service,
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Severe hypotonia.",
            "--extraction-backend",
            "llm",
            "--extract-modifiers",
            "--output-format",
            "json_lines",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["extract_modifiers"] is True
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["modifiers"] == [
        {
            "term_id": "HP:0012828",
            "label": "Severe",
            "segments": [{"start": 0, "end": 6, "text": "Severe"}],
            "source": "llm",
        }
    ]
    assert lines[0]["warnings"] == [
        {
            "code": "invented_warning",
            "message": "Synthetic warning.",
            "term_id": "HP:0012828",
        }
    ]


def test_cli_extract_modifiers_rich_json_summary_includes_modifiers(
    monkeypatch,
) -> None:
    runner = CliRunner()
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {
                "extraction_backend": "llm",
                "extract_modifiers": kwargs["extract_modifiers"],
            },
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "status": "present",
                    "score": 0.91,
                    "modifiers": [
                        {
                            "term_id": "HP:0012828",
                            "label": "Severe",
                            "segments": [],
                            "source": "llm",
                        }
                    ],
                }
            ],
        },
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Severe hypotonia.",
            "--extraction-backend",
            "llm",
            "--extract-modifiers",
            "--output-format",
            "rich_json_summary",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    term = payload["document"]["hpo_terms"][0]
    assert term["modifiers"] == [
        {
            "term_id": "HP:0012828",
            "label": "Severe",
            "segments": [],
            "source": "llm",
        }
    ]
    assert "warnings" not in term


def test_cli_default_no_extract_modifiers_leaves_output_unchanged(
    monkeypatch,
) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.update(kwargs)
        return {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "status": "present",
                    "score": 0.88,
                }
            ],
        }

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        fake_run_full_text_service,
    )

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Hypotonia noted.",
            "--extraction-backend",
            "llm",
            "--output-format",
            "rich_json_summary",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["extract_modifiers"] is False
    term = json.loads(result.stdout)["document"]["hpo_terms"][0]
    assert "modifiers" not in term
    assert "warnings" not in term


def test_cli_phenopacket_output_forwards_modifiers(monkeypatch) -> None:
    runner = CliRunner()
    fake_ancestors = {
        "HP:0012828": frozenset({"HP:0012824", "HP:0012823", "HP:0000001"}),
        "HP:0003577": frozenset({"HP:0003674", "HP:0012823", "HP:0000001"}),
        "HP:0012832": frozenset({"HP:0012831", "HP:0012823", "HP:0000001"}),
    }
    monkeypatch.setattr(
        "phentrieve.phenopackets.utils._load_ancestors_map_safe",
        lambda: fake_ancestors,
    )
    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        lambda **kwargs: {
            "meta": {"extraction_backend": "llm", "extract_modifiers": True},
            "processed_chunks": [],
            "aggregated_hpo_terms": [
                {
                    "id": "HP:0001252",
                    "name": "Hypotonia",
                    "status": "present",
                    "score": 0.9,
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
                            "segments": [],
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

    result = runner.invoke(
        app,
        [
            "text",
            "process",
            "Severe congenital bilateral hypotonia.",
            "--extraction-backend",
            "llm",
            "--extract-modifiers",
            "--output-format",
            "phenopacket_v2_json",
        ],
    )

    assert result.exit_code == 0, result.output
    Parse(result.stdout, Phenopacket(), ignore_unknown_fields=False)
    phenopacket = json.loads(result.stdout)
    feature = phenopacket["phenotypicFeatures"][0]
    assert feature["severity"] == {"id": "HP:0012828", "label": "Severe"}
    assert feature["onset"] == {
        "ontologyClass": {"id": "HP:0003577", "label": "Congenital onset"}
    }
    assert feature["modifiers"] == [{"id": "HP:0012832", "label": "Bilateral"}]


def test_profile_supports_extract_modifiers_and_cli_override(
    monkeypatch, tmp_path
) -> None:
    assert Profile(extract_modifiers=True).extract_modifiers is True

    (tmp_path / "phentrieve.yaml").write_text(
        "profiles:\n"
        "  llm_modifiers:\n"
        "    command: text process\n"
        "    extraction_backend: llm\n"
        "    extract_modifiers: true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _clear_yaml_cache()

    captured: dict[str, object] = {}

    def fake_run_full_text_service(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {
            "meta": {"extraction_backend": "llm"},
            "processed_chunks": [],
            "aggregated_hpo_terms": [],
        }

    monkeypatch.setattr(
        "phentrieve.cli.text_commands.run_full_text_service",
        fake_run_full_text_service,
    )

    runner = CliRunner()
    try:
        res_profile = runner.invoke(
            app,
            [
                "text",
                "process",
                "Severe hypotonia.",
                "--profile",
                "llm_modifiers",
            ],
        )
        assert res_profile.exit_code == 0, res_profile.output
        assert captured["extract_modifiers"] is True

        res_override = runner.invoke(
            app,
            [
                "text",
                "process",
                "Severe hypotonia.",
                "--profile",
                "llm_modifiers",
                "--no-extract-modifiers",
            ],
        )
        assert res_override.exit_code == 0, res_override.output
        assert captured["extract_modifiers"] is False
    finally:
        _clear_yaml_cache()
