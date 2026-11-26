"""`phentrieve data prepare-diseases` with downloads replaced by the mini fixture."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from tests.unit.disease_matching.conftest import MINI

pytestmark = pytest.mark.unit


def _fake_sources(sources_dir, hpo_version, mondo_version, hpo_base_url, force=False):
    files = {
        name: {
            "url": f"https://example.invalid/{name}",
            "path": str(MINI / name),
            "sha256": "0" * 64,
        }
        for name in ("hp.json", "phenotype.hpoa", "genes_to_phenotype.txt", "mondo.obo")
    }
    return {"hpo_version": hpo_version, "mondo_version": mondo_version, "files": files}


def _run(tmp_path: Path, **kwargs) -> None:
    from phentrieve.cli.data_commands import prepare_disease_index

    prepare_disease_index(
        data_dir=str(tmp_path),
        hpo_version="v2026-09-01",
        mondo_version="v2026-09-01",
        **kwargs,
    )


def test_builds_and_saves_index(tmp_path: Path, mocker) -> None:
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    mocker.patch("phentrieve.utils.setup_logging_cli")
    _run(tmp_path)
    fetch.assert_called_once()
    from phentrieve.disease_matching import load_index

    ix = load_index(tmp_path / "disease_index")
    assert ix.n_diseases == 4
    assert ix.meta["hpo_version"] == "v2026-09-01"
    assert ix.meta["mondo_version"] == "v2026-09-01"
    assert ix.meta["merge_orpha"] is True
    assert ix.meta["sources"]["phenotype.hpoa"]["url"].endswith("phenotype.hpoa")


def test_skips_existing_index_without_force(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    mocker.patch("phentrieve.utils.setup_logging_cli")
    _run(tmp_path)
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    _run(tmp_path)
    fetch.assert_not_called()
    _run(tmp_path, force=True)
    fetch.assert_called_once()


def test_no_orpha_merge_flag(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    mocker.patch("phentrieve.utils.setup_logging_cli")
    _run(tmp_path, no_orpha_merge=True)
    from phentrieve.disease_matching import load_index

    assert load_index(tmp_path / "disease_index").meta["merge_orpha"] is False


def test_failure_exits_with_code_1(tmp_path: Path, mocker) -> None:
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=RuntimeError("network down"),
    )
    mocker.patch("phentrieve.utils.setup_logging_cli")
    with pytest.raises(typer.Exit) as exc:
        _run(tmp_path)
    assert exc.value.exit_code == 1


def test_rebuilds_when_requested_versions_differ(tmp_path: Path, mocker) -> None:
    mocker.patch("phentrieve.utils.setup_logging_cli")
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    _run(tmp_path)
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    from phentrieve.cli.data_commands import prepare_disease_index
    from phentrieve.disease_matching import load_index

    mocker.patch(
        "phentrieve.disease_matching.index_builder.latest_mondo_version",
        return_value="v2026-09-01",
    )
    prepare_disease_index(
        data_dir=str(tmp_path), hpo_version="v2026-10-01", mondo_version=None
    )
    fetch.assert_called_once()
    assert load_index(tmp_path / "disease_index").meta["hpo_version"] == "v2026-10-01"


def test_rebuilds_a_damaged_index_without_force(tmp_path: Path, mocker) -> None:
    from phentrieve.disease_matching.index import INDEX_ARRAYS_FILENAME

    mocker.patch("phentrieve.utils.setup_logging_cli")
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    _run(tmp_path)
    arrays = tmp_path / "disease_index" / INDEX_ARRAYS_FILENAME
    arrays.write_bytes(arrays.read_bytes() + b"x")
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    _run(tmp_path)
    fetch.assert_called_once()


def test_latest_is_not_a_version_pin(tmp_path: Path, mocker) -> None:
    mocker.patch("phentrieve.utils.setup_logging_cli")
    mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    _run(tmp_path)
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    from phentrieve.cli.data_commands import prepare_disease_index

    prepare_disease_index(
        data_dir=str(tmp_path), hpo_version="latest", mondo_version="LATEST"
    )
    fetch.assert_not_called()


def test_mondo_latest_resolves_tag_on_fresh_build(tmp_path: Path, mocker) -> None:
    mocker.patch("phentrieve.utils.setup_logging_cli")
    mocker.patch(
        "phentrieve.data_processing.hpo_parser.resolve_hpo_version",
        return_value="v2026-09-01",
    )
    mocker.patch(
        "phentrieve.disease_matching.index_builder.latest_mondo_version",
        return_value="v2026-09-01",
    )
    fetch = mocker.patch(
        "phentrieve.disease_matching.index_builder.fetch_sources",
        side_effect=_fake_sources,
    )
    from phentrieve.cli.data_commands import prepare_disease_index

    prepare_disease_index(
        data_dir=str(tmp_path), hpo_version="latest", mondo_version="latest"
    )
    assert fetch.call_args.args[2] == "v2026-09-01"
