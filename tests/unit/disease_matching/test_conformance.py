"""Conformance with the frozen VarLens v2 reference output (issue #451).

Needs the HPO v2026-09-01 release files (hp.json, phenotype.hpoa, genes_to_phenotype.txt)
in ``$PHENTRIEVE_DISEASE_CONFORMANCE_HPO_DIR``; skipped otherwise. The index is built
without the ORPHA merge, which is the setting the reference was produced with.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from phentrieve.disease_matching import scorer
from phentrieve.disease_matching.index import DiseaseIndex
from phentrieve.disease_matching.index_builder import build_index

pytestmark = pytest.mark.unit

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "data"
        / "disease_matching"
        / "varlens_reference.json"
    ).read_text()
)
HPO_DIR = Path(os.environ.get("PHENTRIEVE_DISEASE_CONFORMANCE_HPO_DIR", "/nonexistent"))

needs_hpo = pytest.mark.skipif(
    not (HPO_DIR / "phenotype.hpoa").is_file(),
    reason="HPO v2026-09-01 release files not available "
    "(set PHENTRIEVE_DISEASE_CONFORMANCE_HPO_DIR)",
)


@cache
def _index() -> DiseaseIndex:
    return build_index(
        HPO_DIR / "hp.json",
        HPO_DIR / "phenotype.hpoa",
        HPO_DIR / "genes_to_phenotype.txt",
        merge_orpha=False,
    )


def _weights(ix: DiseaseIndex, terms: list[dict]) -> dict[int, float]:
    w: dict[int, float] = {}
    for t in terms:
        if t["excluded"]:  # conflictsOnly: excluded terms never score
            continue
        i = ix.resolve_term(t["hpoId"])
        if i is not None:
            w[i] = max(w.get(i, 0.0), scorer.KEY_WEIGHT if t["key"] else 1.0)
    return w


def test_reference_has_enough_profiles() -> None:
    assert len(REFERENCE["profiles"]) >= 20


@needs_hpo
def test_universe_matches_reference() -> None:
    ix = _index()
    assert ix.n_diseases == REFERENCE["n_diseases"]
    assert len(ix.gene_diseases) == REFERENCE["n_genes"]


@needs_hpo
@pytest.mark.parametrize("ref", REFERENCE["profiles"], ids=lambda r: r["id"])
def test_scores_and_ranks_match_reference(ref: dict) -> None:
    ix = _index()
    got = scorer.score(ix, _weights(ix, ref["terms"]))
    assert got is not None
    s = got.scores
    d = ix.disease_index[ref["disease"]]
    assert s[d] == pytest.approx(ref["disease_score"], abs=1e-5)
    assert scorer.avg_rank(s, d) == ref["disease_rank"]
    for disease, expected in ref["top_diseases"]:
        assert s[ix.disease_index[disease]] == pytest.approx(expected, abs=1e-5)
    genes = scorer.gene_scores(ix, s)
    for gene, expected in ref["top_genes"]:
        assert genes[gene][0] == pytest.approx(expected, abs=1e-5)
    if ref["gene_rank"] is not None:
        names = list(genes)
        values = np.array([genes[g][0] for g in names])
        assert scorer.avg_rank(values, names.index(ref["gene"])) == ref["gene_rank"]
