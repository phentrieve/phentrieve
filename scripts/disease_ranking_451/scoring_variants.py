"""Deterministic scorer variants on the cached #451 cases (post hoc, exploratory).

    python -I scoring_variants.py <hpo_dir>

No LLM or embedding calls: gold terms come from cases.jsonl, extracted terms from the
cached results/extractions_{lang}_llm_negfix.jsonl. Variants of the frozen VarLens v2 port:

- gamma: S = gamma Fwd + (1 - gamma) Rev (shipped 0.55);
- q: Fwd is the weighted mean of r(p, d) over the best patient lanes of each disease, up to a
  share q of the lane weight s_p = w_p IC(p) / sum w IC (shipped q = 1: all lanes). With
  q < 1, a few patient terms the disease does not explain cost less.

The shipped variant (gamma 0.55, q 1.0) is asserted to reproduce scorer.score.
Selection is on dev; the non-cited test split is reported descriptively (it has been seen).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import load_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GAMMAS = (0.55, 0.7, 0.85, 1.0)
QS = (1.0, 0.8, 0.6, 0.4)
BASE = (0.55, 1.0)
HITS = (1, 10, 50)
B, SEED = 2000, 451


def parts(ix: scorer.PhenotypeIndex, weights: dict[str, float]):
    """r (diseases x lanes), lane shares s (sum 1) and the normalised reverse score."""
    present, w = scorer.prepare(ix, weights)
    if not present:
        return None
    k = len(present)
    in_anc = np.zeros((len(ix.ids), k), dtype=bool)
    for lane, p in enumerate(present):
        in_anc[ix.ancestors(p), lane] = True
    vals = ix.ic[ix.anc_idx][:, None] * in_anc[ix.anc_idx]
    sim = np.maximum.reduceat(vals, ix.anc_ptr[:-1], axis=0)
    p_ic = ix.self_ic[np.asarray(present)]
    s = np.asarray(w) * p_ic
    s = s / s.sum()
    with np.errstate(divide="ignore"):
        inv_ic = np.where(ix.self_ic > 0, 1 / ix.self_ic, 0.0)
    nonempty = np.diff(ix.dis_ptr) > 0
    lane_max = np.zeros((len(ix.disease_ids), k))
    lane_max[nonempty] = np.maximum.reduceat(
        sim[ix.ann_term] * ix.ann_factor[:, None], ix.dis_ptr[:-1][nonempty], axis=0
    )
    reverse = sim.max(axis=1) * inv_ic
    rev = np.bincount(ix.ann_dis, ix.ann_f * reverse[ix.ann_term], len(ix.disease_ids))
    return lane_max / p_ic, s, rev / (scorer.F0 + ix.freq_sum)


def forward(r: np.ndarray, s: np.ndarray, q: float) -> np.ndarray:
    if q >= 1.0:
        return r @ s
    order = np.argsort(-r, axis=1, kind="stable")
    r_sorted = np.take_along_axis(r, order, axis=1)
    s_sorted = s[order]
    before = np.cumsum(s_sorted, axis=1) - s_sorted
    take = np.clip(q - before, 0.0, s_sorted)
    return (r_sorted * take).sum(axis=1) / q


def variant_scores(p, gamma: float, q: float) -> np.ndarray:
    r, s, rev = p
    return gamma * forward(r, s, q) + (1 - gamma) * rev


def hit_table(ranks: np.ndarray) -> dict[str, float]:
    out = {f"hit{h}": float(np.mean(ranks <= h)) for h in HITS}
    out["mrr"] = float(np.mean(1 / ranks))
    return out


def clustered_diff_ci(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, h: int):
    """Paired Hit@h difference a - b with a PMID-clustered bootstrap 95% CI."""
    d = (a <= h).astype(float) - (b <= h).astype(float)
    ids, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ids))]
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[i] for i in pick])
        reps.append(d[idx].mean())
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), int((d > 0).sum()), int((d < 0).sum())


def main(hpo_dir: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if not (c["split"] == "test" and c["cited"])
    ]
    ext = {
        lang: {
            e["case"]: e
            for e in load_jsonl(RESULTS / f"extractions_{lang}_llm_negfix.jsonl")
        }
        for lang in ("de", "en")
    }
    variants = [(g, q) for g in GAMMAS for q in QS]
    rows: list[dict] = []
    checked = 0
    for c in cases:
        arms = {"gold": {t["id"]: 1.0 for t in c["observed"]}}
        for lang in ("de", "en"):
            e = ext[lang].get(c["id"])
            if e is not None:
                arms[f"llm_{lang}"] = {
                    t[0]: 1.0 for t in e["terms"] if t[1] == "present" and not t[2]
                }
        d_idx = ix.disease_ids.index(c["disease"])
        for arm, w in arms.items():
            p = parts(ix, w)
            for g, q in variants:
                if p is None:
                    rank = (len(ix.disease_ids) + 1) / 2
                else:
                    s = variant_scores(p, g, q)
                    if (g, q) == BASE and checked < 50:
                        ref = scorer.score(ix, w)
                        assert np.allclose(s, ref, rtol=0, atol=1e-12), c["id"]
                        assert scorer.avg_rank(s, d_idx) == scorer.avg_rank(ref, d_idx)
                        checked += 1
                    rank = scorer.avg_rank(s, d_idx)
                rows.append(
                    {
                        "case": c["id"],
                        "split": c["split"],
                        "cited": int(c["cited"]),
                        "pmid": c["pmids"][0],
                        "arm": arm,
                        "gamma": g,
                        "q": q,
                        "disease_rank": rank,
                    }
                )
    print(f"baseline reproduces scorer.score on {checked} profiles")
    with (RESULTS / "scoring_variants_ranks.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def ranks(split: str, arm: str, g: float, q: float) -> tuple[np.ndarray, np.ndarray]:
        sel = [
            r
            for r in rows
            if r["split"] == split and r["arm"] == arm and r["gamma"] == g and r["q"] == q
        ]
        sel.sort(key=lambda r: r["case"])
        return (
            np.array([r["disease_rank"] for r in sel]),
            np.array([r["pmid"] for r in sel]),
        )

    arm_names = ("gold", "llm_de", "llm_en")
    summary = []
    for split in ("dev", "test"):
        for arm in arm_names:
            for g, q in variants:
                rk, _ = ranks(split, arm, g, q)
                summary.append(
                    {"split": split, "arm": arm, "gamma": g, "q": q, "n": len(rk)}
                    | hit_table(rk)
                )
    with (RESULTS / "scoring_variants.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)

    # Selection on dev: mean Hit@10 over the three arms; ties keep the earlier grid point.
    def dev_obj(v):
        return np.mean(
            [
                next(
                    s["hit10"]
                    for s in summary
                    if s["split"] == "dev" and s["arm"] == a and (s["gamma"], s["q"]) == v
                )
                for a in arm_names
            ]
        )

    best = max(variants, key=lambda v: (dev_obj(v), -variants.index(v)))
    print("\nDev Hit@10 (mean of gold, llm_de, llm_en):")
    print("gamma \\ q " + "".join(f"{q:>8}" for q in QS))
    for g in GAMMAS:
        print(f"{g:<10}" + "".join(f"{dev_obj((g, q)):8.3f}" for q in QS))
    print(f"\nchosen on dev: gamma={best[0]} q={best[1]}  (baseline {BASE})")
    print("\nTest, non-cited (seen before; descriptive). chosen minus baseline:")
    for arm in arm_names:
        b, pm = ranks("test", arm, *BASE)
        a, _ = ranks("test", arm, *best)
        line = f"{arm:<7} n={len(a)}"
        for h in HITS:
            m, lo, hi, win, loss = clustered_diff_ci(a, b, pm, h)
            line += (
                f" | Hit@{h} {np.mean(b <= h):.3f}->{np.mean(a <= h):.3f}"
                f" {m * 100:+.1f}pp [{lo * 100:+.1f},{hi * 100:+.1f}] +{win}/-{loss}"
            )
        print(line)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
