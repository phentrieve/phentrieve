"""Deterministic annotation enrichment on the cached #451 cases (post hoc, exploratory).

    python -I enrichment.py <hpo_dir> <mondo.obo>

No LLM or embedding calls. The scorer is unchanged (shipped VarLens v2 setting); only the
disease annotations it scores against change:

- orpha: each OMIM disease of the index also gets the HPOA annotations of every ORPHA disease
  that MONDO marks as equivalent (both xrefs `MONDO:equivalentTo` on one MONDO term); per term
  the higher frequency wins. ORPHA entries stay in the universe as before.
- union: each gene with two or more diseases gets one extra candidate, the union of its
  diseases' annotations (higher frequency wins). A gene's score is the max over its diseases
  and that candidate. This only changes gene ranks; disease ranks ignore the extra candidates.
- orpha+union: both.

There are no free parameters, so dev and the non-cited test split are both reported. Test has
been seen before, so the numbers are descriptive, not confirmatory.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run with -I: own code only
import numpy as np  # noqa: E402
import scorer  # noqa: E402
from analysis import load_jsonl  # noqa: E402
from scoring_variants import clustered_diff_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HITS = (1, 10, 50)
ARMS = ("gold", "llm_de", "llm_en")


def mondo_omim_to_orpha(path: Path) -> dict[str, set[str]]:
    """OMIM:x -> {ORPHA:y} for pairs of equivalentTo xrefs on one non-obsolete MONDO term."""
    out: dict[str, set[str]] = defaultdict(set)
    omim: list[str] = []
    orpha: list[str] = []
    obsolete = False

    def flush() -> None:
        if not obsolete:
            for o in omim:
                out[o].update(orpha)

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            flush()
            omim, orpha, obsolete = [], [], False
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif line.startswith("xref: ") and 'MONDO:equivalentTo' in line:
            ref = line[6:].split(" ", 1)[0]
            if ref.startswith("OMIM:") and ref[5:].isdigit():
                omim.append(ref)
            elif ref.startswith("Orphanet:"):
                orpha.append("ORPHA:" + ref[9:])
    flush()
    return {k: v for k, v in out.items() if v}


def base_annotations(ix: scorer.PhenotypeIndex) -> list[dict[int, float]]:
    return [
        dict(
            zip(
                ix.ann_term[ix.dis_ptr[i] : ix.dis_ptr[i + 1]].tolist(),
                ix.ann_f[ix.dis_ptr[i] : ix.dis_ptr[i + 1]].tolist(),
                strict=True,
            )
        )
        for i in range(len(ix.disease_ids))
    ]


def rebuild(
    ix: scorer.PhenotypeIndex,
    anns: list[dict[int, float]],
    extra_ids: list[str],
    gene_diseases: dict[str, list[int]],
) -> scorer.PhenotypeIndex:
    """Same index with new per-disease annotations (extra candidates appended at the end)."""
    ids = ix.disease_ids + extra_ids
    ann_term: list[int] = []
    ann_f: list[float] = []
    ptr = [0]
    for a in anns:
        for t, f in a.items():
            ann_term.append(t)
            ann_f.append(f)
        ptr.append(len(ann_term))
    f32 = scorer._f32(ann_f)
    dis_ptr = np.asarray(ptr)
    ann_dis = np.repeat(np.arange(len(ids)), np.diff(dis_ptr))
    return dataclasses.replace(
        ix,
        disease_ids=ids,
        disease_labels=ix.disease_labels + extra_ids,
        dis_ptr=dis_ptr,
        ann_dis=ann_dis,
        ann_term=np.asarray(ann_term, dtype=np.int64),
        ann_f=f32,
        ann_factor=scorer._f32(1 - scorer.ALPHA_F + scorer.ALPHA_F * f32),
        freq_sum=np.bincount(ann_dis, f32, len(ids)),
        gene_diseases=gene_diseases,
    )


def merge(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out = dict(a)
    for t, f in b.items():
        if f > out.get(t, -1.0):
            out[t] = f
    return out


def build_variants(ix, hpo_dir: Path, mondo: Path):
    _, _, aliases = scorer.parse_hp_json(hpo_dir / "hp.json")
    hpoa, _ = scorer.parse_hpoa(hpo_dir / "phenotype.hpoa", aliases)

    def to_idx(terms: dict[str, float]) -> dict[int, float]:
        out: dict[int, float] = {}
        for raw, f in terms.items():
            t = ix.index.get(aliases.get(raw, raw))
            if t is not None and ix.self_ic[t] > 0 and f > out.get(t, -1.0):
                out[t] = f
        return out

    base = base_annotations(ix)
    o2o = mondo_omim_to_orpha(mondo)
    orpha = list(base)
    enriched = 0
    for i, d in enumerate(ix.disease_ids):
        extra = [to_idx(hpoa[o]) for o in o2o.get(d, ()) if o in hpoa]
        if extra:
            merged = base[i]
            for e in extra:
                merged = merge(merged, e)
            if len(merged) > len(base[i]):
                enriched += 1
            orpha[i] = merged

    def with_union(anns: list[dict[int, float]]):
        extra_ids: list[str] = []
        extra_anns: list[dict[int, float]] = []
        genes: dict[str, list[int]] = {}
        n = len(ix.disease_ids)
        for g, ds in ix.gene_diseases.items():
            genes[g] = list(ds)
            if len(ds) >= 2:
                u: dict[int, float] = {}
                for d in ds:
                    u = merge(u, anns[d])
                genes[g].append(n + len(extra_ids))
                extra_ids.append(f"UNION:{g}")
                extra_anns.append(u)
        return rebuild(ix, anns + extra_anns, extra_ids, genes)

    variants = {
        "base": ix,
        "orpha": rebuild(ix, orpha, [], ix.gene_diseases),
        "union": with_union(base),
        "orpha+union": with_union(orpha),
    }
    info = {
        "omim_with_orpha_equivalent": sum(
            1 for d in ix.disease_ids if any(o in hpoa for o in o2o.get(d, ()))
        ),
        "omim_gaining_terms": enriched,
        "union_candidates": len(variants["union"].disease_ids) - len(ix.disease_ids),
    }
    return variants, info, base, orpha


def ranks(ix, n_real: int, case: dict, w: dict[str, float]) -> tuple[float, float]:
    s = scorer.score(ix, w)
    if s is None:
        return (n_real + 1) / 2, (len(ix.gene_diseases) + 1) / 2
    d_rank = scorer.avg_rank(s[:n_real], ix.disease_ids.index(case["disease"]))
    g_rank = math.nan
    genes = scorer.gene_scores(ix, s)
    if case["gene"] in genes:
        names = list(genes)
        g_rank = scorer.avg_rank(
            np.array([genes[g] for g in names]), names.index(case["gene"])
        )
    return d_rank, g_rank


def main(hpo_dir: Path, mondo: Path) -> None:
    ix = scorer.build_index(hpo_dir)
    n_real = len(ix.disease_ids)
    variants, info, base_anns, orpha_anns = build_variants(ix, hpo_dir, mondo)
    print(info)
    cases = [
        c
        for c in load_jsonl(HERE / "cases.jsonl")
        if not (c["split"] == "test" and c["cited"])
    ]
    tgt = [ix.disease_ids.index(c["disease"]) for c in cases]
    print(
        "case targets with ORPHA terms added:",
        sum(len(orpha_anns[t]) > len(base_anns[t]) for t in tgt),
        "of",
        len(cases),
        "| median target annotations base/orpha:",
        float(np.median([len(base_anns[t]) for t in tgt])),
        float(np.median([len(orpha_anns[t]) for t in tgt])),
    )
    ext = {
        lang: {
            e["case"]: e
            for e in load_jsonl(RESULTS / f"extractions_{lang}_llm_negfix.jsonl")
        }
        for lang in ("de", "en")
    }
    rows: list[dict] = []
    for c in cases:
        arms = {"gold": {t["id"]: 1.0 for t in c["observed"]}}
        for lang in ("de", "en"):
            e = ext[lang][c["id"]]
            arms[f"llm_{lang}"] = {
                t[0]: 1.0 for t in e["terms"] if t[1] == "present" and not t[2]
            }
        for arm, w in arms.items():
            for name, vix in variants.items():
                d, g = ranks(vix, n_real, c, w)
                rows.append(
                    {
                        "case": c["id"],
                        "split": c["split"],
                        "pmid": c["pmids"][0],
                        "arm": arm,
                        "variant": name,
                        "disease_rank": d,
                        "gene_rank": g,
                    }
                )
    with (RESULTS / "enrichment_ranks.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def get(split, arm, var, col):
        sel = sorted(
            (r for r in rows if r["split"] == split and r["arm"] == arm and r["variant"] == var),
            key=lambda r: r["case"],
        )
        return np.array([r[col] for r in sel]), np.array([r["pmid"] for r in sel])

    summary = []
    for split in ("dev", "test"):
        print(f"\n== {split} (non-cited only for test) ==")
        for col in ("disease_rank", "gene_rank"):
            for arm in ARMS:
                b, pm = get(split, arm, "base", col)
                ok = ~np.isnan(b)
                line = f"{col[:-5]:<8}{arm:<7} n={ok.sum():<4}"
                for var in variants:
                    if col == "disease_rank" and "union" in var:
                        continue
                    a, _ = get(split, arm, var, col)
                    for h in HITS:
                        m, lo, hi, win, loss = clustered_diff_ci(a[ok], b[ok], pm[ok], h)
                        summary.append(
                            {
                                "split": split,
                                "rank": col,
                                "arm": arm,
                                "variant": var,
                                "hit": h,
                                "base": float(np.mean(b[ok] <= h)),
                                "value": float(np.mean(a[ok] <= h)),
                                "diff": m,
                                "ci_lo": lo,
                                "ci_hi": hi,
                                "wins": win,
                                "losses": loss,
                            }
                        )
                    if var != "base":
                        h10 = next(
                            s for s in summary[::-1] if s["variant"] == var and s["hit"] == 10
                        )
                        h50 = summary[-1]
                        line += (
                            f" | {var}: @10 {h10['value']:.3f} ({h10['diff'] * 100:+.1f}"
                            f" [{h10['ci_lo'] * 100:+.1f},{h10['ci_hi'] * 100:+.1f}])"
                            f" @50 {h50['value']:.3f} ({h50['diff'] * 100:+.1f})"
                        )
                    else:
                        line += (
                            f" | base: @1 {np.mean(b[ok] <= 1):.3f} @10 {np.mean(b[ok] <= 10):.3f}"
                            f" @50 {np.mean(b[ok] <= 50):.3f}"
                        )
                print(line)
    with (RESULTS / "enrichment.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
