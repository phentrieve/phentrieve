"""Scorer: vectorised implementation against a naive reading of the formula."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from phentrieve.disease_matching import scorer
from phentrieve.disease_matching.index import ALPHA_F, DiseaseIndex

pytestmark = pytest.mark.unit


def _naive(ix: DiseaseIndex, present: dict[int, float]) -> np.ndarray:
    lanes, weights = scorer.prepare(ix, present)

    def sim(a: int, b: int) -> float:
        common = set(ix.ancestors(a).tolist()) & set(ix.ancestors(b).tolist())
        return max((float(ix.ic[c]) for c in common), default=0.0)

    out = np.zeros(ix.n_diseases)
    for d in range(ix.n_diseases):
        terms, freqs, _ = ix.annotations(d)
        num = den = 0.0
        for p, w in zip(lanes, weights, strict=True):
            # VarLens stores the frequency factor as float32; the reference follows it.
            r = (
                max(
                    sim(p, int(t)) * float(np.float32(1 - ALPHA_F + ALPHA_F * float(f)))
                    for t, f in zip(terms, freqs, strict=True)
                )
                / ix.self_ic[p]
            )
            num += w * ix.self_ic[p] * r
            den += w * ix.self_ic[p]
        fwd = num / den
        rev = sum(
            float(f) * max(sim(p, int(t)) for p in lanes) / ix.self_ic[int(t)]
            for t, f in zip(terms, freqs, strict=True)
        ) / (scorer.F0 + float(freqs.sum()))
        out[d] = scorer.GAMMA * fwd + (1 - scorer.GAMMA) * rev
    return out


def _ids(ix: DiseaseIndex, *hpo: str) -> dict[int, float]:
    return {ix.term_index[h]: 1.0 for h in hpo}


@pytest.mark.parametrize(
    "profile",
    [
        ("HP:0000003",),
        ("HP:0000003", "HP:0000020"),
        ("HP:0000012", "HP:0000004"),
        ("HP:0000011", "HP:0000020", "HP:0000004"),
    ],
)
def test_vectorised_score_matches_naive_formula(
    mini_index: DiseaseIndex, profile
) -> None:
    w = _ids(mini_index, *profile)
    got = scorer.score(mini_index, w)
    assert got is not None
    np.testing.assert_allclose(got.scores, _naive(mini_index, w), rtol=0, atol=1e-12)


def test_key_weight_changes_forward_share(mini_index: DiseaseIndex) -> None:
    a1, c = mini_index.term_index["HP:0000003"], mini_index.term_index["HP:0000020"]
    plain = scorer.score(mini_index, {a1: 1.0, c: 1.0})
    keyed = scorer.score(mini_index, {a1: scorer.KEY_WEIGHT, c: 1.0})
    assert plain is not None and keyed is not None
    np.testing.assert_allclose(
        keyed.scores, _naive(mini_index, {a1: 2.0, c: 1.0}), atol=1e-12
    )
    assert not np.allclose(plain.scores, keyed.scores)


def test_prepare_drops_ancestors_and_keeps_max_weight(mini_index: DiseaseIndex) -> None:
    organ, a1 = mini_index.term_index["HP:0000002"], mini_index.term_index["HP:0000003"]
    lanes, weights = scorer.prepare(mini_index, {organ: 2.0, a1: 1.0})
    assert lanes == [a1]
    assert weights == [2.0]


def test_root_only_profile_has_no_score(mini_index: DiseaseIndex) -> None:
    assert scorer.score(mini_index, _ids(mini_index, "HP:0000118")) is None
    assert scorer.score(mini_index, {}) is None


def test_scores_are_in_unit_interval(mini_index: DiseaseIndex) -> None:
    terms = ["HP:0000003", "HP:0000004", "HP:0000011", "HP:0000012", "HP:0000020"]
    for combo in itertools.combinations(terms, 2):
        got = scorer.score(mini_index, _ids(mini_index, *combo))
        assert got is not None
        assert np.all(got.scores >= 0) and np.all(got.scores <= 1 + 1e-12)


def test_avg_rank_averages_ties() -> None:
    values = np.array([0.5, 0.9, 0.5, 0.1])
    assert scorer.avg_rank(values, 1) == 1.0
    assert scorer.avg_rank(values, 0) == 2.5
    assert scorer.avg_rank(values, 3) == 4.0
