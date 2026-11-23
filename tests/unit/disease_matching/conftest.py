from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.disease_matching.index import DiseaseIndex
from phentrieve.disease_matching.index_builder import build_index

MINI = Path(__file__).resolve().parents[2] / "data" / "disease_matching" / "mini"


def build_mini(merge_orpha: bool = True) -> DiseaseIndex:
    return build_index(
        MINI / "hp.json",
        MINI / "phenotype.hpoa",
        MINI / "genes_to_phenotype.txt",
        MINI / "mondo.obo",
        merge_orpha=merge_orpha,
        meta={"hpo_version": "v2026-09-01", "mondo_version": "v2026-09-01"},
    )


@pytest.fixture(scope="module")
def mini_index() -> DiseaseIndex:
    return build_mini(merge_orpha=True)


@pytest.fixture(scope="module")
def mini_index_plain() -> DiseaseIndex:
    return build_mini(merge_orpha=False)
