"""Saving and loading the disease index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phentrieve.disease_matching import IndexNotAvailableError, load_index, rank
from phentrieve.disease_matching.index import (
    INDEX_ARRAYS_FILENAME,
    INDEX_META_FILENAME,
    DiseaseIndex,
    index_exists,
)
from phentrieve.disease_matching.service import ProfileTerm
from tests.unit.disease_matching.conftest import build_mini

pytestmark = pytest.mark.unit


def test_roundtrip_keeps_tables_and_scores(
    tmp_path: Path, mini_index: DiseaseIndex
) -> None:
    ix = build_mini()
    ix.save(tmp_path)
    assert index_exists(tmp_path)
    loaded = load_index(tmp_path)
    for name in (
        "ic",
        "self_ic",
        "anc_ptr",
        "anc_idx",
        "dis_ptr",
        "ann_term",
        "ann_f",
        "ann_source",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(mini_index, name))
    assert loaded.disease_ids == mini_index.disease_ids
    assert loaded.twins == mini_index.twins
    assert loaded.aliases == mini_index.aliases
    assert loaded.other_terms == mini_index.other_terms
    assert loaded.meta["hpo_version"] == "v2026-09-01"
    profile = [
        ProfileTerm("HP:0000003", "affirmed"),
        ProfileTerm("HP:0000020", "affirmed"),
    ]
    assert [(d.disease_id, d.score) for d in rank(loaded, profile).diseases] == [
        (d.disease_id, d.score) for d in rank(mini_index, profile).diseases
    ]


def test_missing_index_raises(tmp_path: Path) -> None:
    assert not index_exists(tmp_path)
    with pytest.raises(IndexNotAvailableError, match="prepare-diseases"):
        load_index(tmp_path)


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    build_mini().save(tmp_path)
    path = tmp_path / INDEX_ARRAYS_FILENAME
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(IndexNotAvailableError, match="checksum"):
        load_index(tmp_path)


def test_unsupported_format_raises(tmp_path: Path) -> None:
    build_mini().save(tmp_path)
    meta_path = tmp_path / INDEX_META_FILENAME
    payload = json.loads(meta_path.read_text())
    payload["meta"]["format_version"] = 999
    meta_path.write_text(json.dumps(payload))
    with pytest.raises(IndexNotAvailableError, match="format"):
        load_index(tmp_path)


def test_unreadable_metadata_raises(tmp_path: Path) -> None:
    build_mini().save(tmp_path)
    (tmp_path / INDEX_META_FILENAME).write_text("{not json")
    with pytest.raises(IndexNotAvailableError, match="Unreadable"):
        load_index(tmp_path)


def test_info_file_is_readable_without_loading(tmp_path: Path) -> None:
    from phentrieve.disease_matching.index import read_index_info

    assert read_index_info(tmp_path) is None
    build_mini().save(tmp_path)
    info = read_index_info(tmp_path)
    assert info is not None
    assert info["hpo_version"] == "v2026-09-01"
    assert info["n_diseases"] == 4
    assert "arrays_sha256" in info
