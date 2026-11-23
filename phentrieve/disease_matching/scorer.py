"""Phenotype similarity scorer (VarLens phenotype scoring v2, shipped setting).

Phentrieve's copy of the issue #451 port. The setting is frozen: tau = 1, background
correction off, exclusions ``conflictsOnly`` (excluded terms never change the score).

    sim(a, b) = IC(MICA(a, b)), IC from phenotype.hpoa (diseases propagate to ancestors)
    r(p, d)   = max_t sim(p, t) * (1 - a_f + a_f * f(d, t)) / IC(p)
    Fwd(d)    = sum_p w_p IC(p) r(p, d) / sum_p w_p IC(p)
    Rev(d)    = sum_t f(d, t) max_p sim(p, t) / IC(t) / (f0 + sum_t f(d, t))
    S(d)      = gamma Fwd(d) + (1 - gamma) Rev(d);  S(gene) = max over its diseases
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phentrieve.disease_matching.index import DiseaseIndex

GAMMA = 0.55
F0 = 1.0
KEY_WEIGHT = 2.0
SCORER_VERSION = "varlens-v2-shipped/phentrieve-1"


@dataclass
class ScoredProfile:
    """Disease scores plus what the explanation needs."""

    scores: np.ndarray  # S(d) for every disease
    lanes: list[int]  # present term indices after ancestor pruning
    lane_weights: list[float]
    sim: np.ndarray  # (n_terms, lanes): IC of the MICA of term t and lane l


def prepare(
    index: DiseaseIndex, weights: dict[int, float]
) -> tuple[list[int], list[float]]:
    """Drop present terms that are ancestors of other present terms; the descendant keeps the max weight."""
    w = {t: v for t, v in weights.items() if index.self_ic[t] > 0}
    terms = list(w)
    weight = [w[t] for t in terms]
    dropped = [False] * len(terms)
    for i, q in enumerate(terms):
        anc = set(index.ancestors(q).tolist())
        for j, p in enumerate(terms):
            if j != i and p in anc:
                dropped[j] = True
                weight[i] = max(weight[i], weight[j])
    keep = [i for i in range(len(terms)) if not dropped[i]]
    return [terms[i] for i in keep], [weight[i] for i in keep]


def score(index: DiseaseIndex, weights: dict[int, float]) -> ScoredProfile | None:
    """Score every disease; ``weights`` maps present term indices to w_p. None without a usable term."""
    present, w = prepare(index, weights)
    if not present:
        return None
    k = len(present)
    anc_ic = index.ic[index.anc_idx]
    starts = index.anc_ptr[:-1]
    sim = np.empty((len(index.term_ids), k))
    mask = np.zeros(len(index.term_ids), dtype=bool)
    for lane, p in enumerate(present):
        mask[:] = False
        mask[index.ancestors(p)] = True
        sim[:, lane] = np.maximum.reduceat(anc_ic * mask[index.anc_idx], starts)
    p_idx = np.asarray(present)
    lane_w = np.asarray(w) * index.self_ic[p_idx]
    lane_w = lane_w / lane_w.sum() / index.self_ic[p_idx]
    with np.errstate(divide="ignore"):
        inv_ic = np.where(index.self_ic > 0, 1 / index.self_ic, 0.0)
    reverse = sim.max(axis=1) * inv_ic
    n = index.n_diseases
    nonempty = np.diff(index.dis_ptr) > 0
    lane_max = np.zeros((n, k))
    lane_max[nonempty] = np.maximum.reduceat(
        sim[index.ann_term] * index.ann_factor[:, None],
        index.dis_ptr[:-1][nonempty],
        axis=0,
    )
    rev = np.bincount(index.ann_dis, index.ann_f * reverse[index.ann_term], n)
    fwd = lane_max @ lane_w
    scores = GAMMA * fwd + (1 - GAMMA) * rev / (F0 + index.freq_sum)
    return ScoredProfile(scores=scores, lanes=present, lane_weights=list(w), sim=sim)


def gene_scores(
    index: DiseaseIndex, scores: np.ndarray
) -> dict[str, tuple[float, int]]:
    """Gene -> (max score over its diseases, the disease that reaches it)."""
    out: dict[str, tuple[float, int]] = {}
    for gene, ds in index.gene_diseases.items():
        best = max(ds, key=lambda d: (scores[d], -d))
        out[gene] = (float(scores[best]), best)
    return out


def avg_rank(values: np.ndarray, i: int) -> float:
    """1-based rank of values[i], ties averaged; rounded to 1e-12 so summation noise never splits a tie."""
    v = np.round(values, 12)
    return float((v > v[i]).sum() + ((v == v[i]).sum() + 1) / 2)
