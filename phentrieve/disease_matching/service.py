"""Rank diseases or genes for a profile of HPO terms, with an explanation per row."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from phentrieve.assertion_vocab import (
    NEGATED,
    NORMAL,
    UNCERTAIN,
    canonicalize_assertion,
)
from phentrieve.disease_matching import scorer
from phentrieve.disease_matching.index import SOURCE_ORPHA, SOURCE_OWN, DiseaseIndex

Level = Literal["disease", "gene"]
Relation = Literal["exact", "broader", "narrower", "related"]

# Display hint only (never changes a score): a matched pair whose relative match r(p, d)
# stays below this is listed as not explained by the disease.
WEAK_MATCH = 0.4
MAX_TOP_K = 100


@dataclass
class ProfileTerm:
    hpo_id: str
    assertion: str | None = None


@dataclass
class TermMatch:
    """A profile term, the disease annotation it matches best, and how."""

    profile_id: str
    profile_label: str
    disease_term_id: str
    disease_term_label: str
    relation: Relation  # disease term relative to the profile term
    shared_id: str  # most informative common ancestor
    shared_label: str
    match: float  # r(p, d) in [0, 1]
    frequency: float
    source: str  # "omim", "orpha" or "both" (annotation origin)


@dataclass
class Conflict:
    """An excluded profile term that the disease is annotated with (it or a descendant)."""

    profile_id: str
    profile_label: str
    disease_term_id: str
    disease_term_label: str
    frequency: float


@dataclass
class Explanation:
    matches: list[TermMatch]
    unexplained: list[TermMatch]
    conflicts: list[Conflict]
    annotation_sources: list[str]


@dataclass
class RankedDisease:
    rank: int
    disease_id: str
    label: str
    score: float
    genes: list[str]
    twins: list[str]
    explanation: Explanation


@dataclass
class RankedGene:
    rank: int
    gene: str
    score: float
    best_disease_id: str
    best_disease_label: str
    explanation: Explanation


@dataclass
class NotScored:
    hpo_id: str
    reason: Literal["unknown_term", "not_phenotype", "uncertain", "redundant"]


@dataclass
class RankingResult:
    level: Level
    diseases: list[RankedDisease] = field(default_factory=list)
    genes: list[RankedGene] = field(default_factory=list)
    scored: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    not_scored: list[NotScored] = field(default_factory=list)
    total_candidates: int = 0
    scorer_version: str = scorer.SCORER_VERSION
    index_info: dict[str, Any] = field(default_factory=dict)


def _source_name(flags: int, own: str) -> str:
    """Annotation origin; ``own`` is the disease's own source (its ID prefix, lower case)."""
    if flags & SOURCE_OWN and flags & SOURCE_ORPHA and own != "orpha":
        return "both"
    if flags & SOURCE_ORPHA:
        return "orpha"
    return own


def _competition_ranks(values: np.ndarray) -> np.ndarray:
    """1-based rank = 1 + number of strictly higher values (scores rounded to 1e-12)."""
    v = np.round(values, 12)
    order = np.sort(v)[::-1]
    return np.searchsorted(-order, -v, side="left") + 1


class _Explainer:
    def __init__(
        self, index: DiseaseIndex, scored: scorer.ScoredProfile, excluded: list[int]
    ):
        self.ix = index
        self.sp = scored
        self.excluded = excluded

    def _label(self, t: int) -> str:
        return self.ix.term_labels[t]

    def _shared(self, p: int, t: int) -> int:
        """Most informative common ancestor; on an IC tie the most specific one."""
        ix = self.ix
        common = np.intersect1d(ix.ancestors(p), ix.ancestors(t), assume_unique=True)
        ics = ix.ic[common]
        best: list[int] = [int(c) for c in common[ics == ics.max()]]
        for preferred in (t, p):
            if preferred in best:
                return preferred
        return max(best, key=lambda c: (len(ix.ancestors(c)), -c))

    def explain(self, d: int) -> Explanation:
        ix = self.ix
        own = ix.disease_ids[d].split(":")[0].lower()
        terms, freqs, sources = ix.annotations(d)
        matches: list[TermMatch] = []
        unexplained: list[TermMatch] = []
        for lane, p in enumerate(self.sp.lanes):
            if len(terms) == 0:
                continue
            values = (
                self.sp.sim[terms, lane]
                * ix.ann_factor[ix.dis_ptr[d] : ix.dis_ptr[d + 1]]
            )
            best = int(np.argmax(values))
            t = int(terms[best])
            shared = self._shared(p, t)
            if t == p:
                relation: Relation = "exact"
            elif shared == t:
                relation = "broader"
            elif shared == p:
                relation = "narrower"
            else:
                relation = "related"
            item = TermMatch(
                profile_id=ix.term_ids[p],
                profile_label=self._label(p),
                disease_term_id=ix.term_ids[t],
                disease_term_label=self._label(t),
                relation=relation,
                shared_id=ix.term_ids[shared],
                shared_label=self._label(shared),
                match=round(float(values[best] / ix.self_ic[p]), 4),
                frequency=round(float(freqs[best]), 4),
                source=_source_name(int(sources[best]), own),
            )
            (matches if item.match >= WEAK_MATCH else unexplained).append(item)
        matches.sort(key=lambda m: -m.match)
        conflicts: list[Conflict] = []
        if self.excluded:
            term_list = terms.tolist()
            term_ancestors = [set(ix.ancestors(t).tolist()) for t in term_list]
            for e in self.excluded:
                hits = [i for i, anc in enumerate(term_ancestors) if e in anc]
                if hits:
                    i = max(hits, key=lambda h: float(freqs[h]))
                    conflicts.append(
                        Conflict(
                            profile_id=ix.term_ids[e],
                            profile_label=self._label(e),
                            disease_term_id=ix.term_ids[int(terms[i])],
                            disease_term_label=self._label(int(terms[i])),
                            frequency=round(float(freqs[i]), 4),
                        )
                    )
        flags = int(np.bitwise_or.reduce(sources)) if len(sources) else 0
        names = []
        if flags & SOURCE_OWN:
            names.append(ix.disease_ids[d].split(":")[0].lower())
        if flags & SOURCE_ORPHA:
            names.append("orpha")
        return Explanation(
            matches=matches,
            unexplained=unexplained,
            conflicts=conflicts,
            annotation_sources=sorted(set(names)),
        )


def _resolve_profile(
    index: DiseaseIndex, terms: list[ProfileTerm], include_uncertain: bool
) -> tuple[dict[int, float], list[int], list[NotScored]]:
    # Reconcile duplicate/aliased terms with explicit precedence:
    # present > excluded > uncertain.
    present: dict[int, float] = {}
    excluded_set: set[int] = set()
    excluded_order: list[int] = []
    uncertain_not_scored: dict[int, str] = {}
    other_not_scored: list[NotScored] = []
    seen_other: set[tuple[str, str]] = set()

    for term in terms:
        assertion = canonicalize_assertion(term.assertion)
        hpo_id = term.hpo_id.strip()
        t = index.resolve_term(hpo_id)
        if t is None:
            reason: Literal["not_phenotype", "unknown_term"] = (
                "not_phenotype" if hpo_id in index.other_term_set else "unknown_term"
            )
            key = (hpo_id, reason)
            if key not in seen_other:
                seen_other.add(key)
                other_not_scored.append(NotScored(hpo_id, reason))
            continue
        if index.self_ic[t] <= 0:
            key = (hpo_id, "not_phenotype")
            if key not in seen_other:
                seen_other.add(key)
                other_not_scored.append(NotScored(hpo_id, "not_phenotype"))
            continue
        if assertion in (NEGATED, NORMAL):
            if t not in present and t not in excluded_set:
                excluded_set.add(t)
                excluded_order.append(t)
                uncertain_not_scored.pop(t, None)
            continue
        if assertion == UNCERTAIN and not include_uncertain:
            if (
                t not in present
                and t not in excluded_set
                and t not in uncertain_not_scored
            ):
                uncertain_not_scored[t] = hpo_id
            continue
        present[t] = 1.0
        if t in excluded_set:
            excluded_set.remove(t)
        uncertain_not_scored.pop(t, None)

    excluded = [t for t in excluded_order if t in excluded_set]
    not_scored = [
        *other_not_scored,
        *(NotScored(hpo_id, "uncertain") for hpo_id in uncertain_not_scored.values()),
    ]
    return present, excluded, not_scored


def rank(
    index: DiseaseIndex,
    terms: list[ProfileTerm],
    level: Level = "disease",
    top_k: int = 20,
    include_uncertain: bool = False,
) -> RankingResult:
    """Rank diseases (ORPHA twins folded into their OMIM row) or genes for ``terms``."""
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    present, excluded, not_scored = _resolve_profile(index, terms, include_uncertain)
    result = RankingResult(
        level=level,
        excluded=[index.term_ids[e] for e in excluded],
        not_scored=not_scored,
        index_info={
            k: index.meta.get(k)
            for k in (
                "hpo_version",
                "mondo_version",
                "built_at",
                "n_diseases",
                "merge_orpha",
            )
        },
    )
    scored = scorer.score(index, present)
    if scored is None:
        return result
    kept = set(scored.lanes)
    result.scored = [index.term_ids[t] for t in present if t in kept]
    result.not_scored.extend(
        NotScored(index.term_ids[t], "redundant") for t in present if t not in kept
    )
    explainer = _Explainer(index, scored, excluded)
    s = scored.scores

    if level == "disease":
        rows = np.array(
            [d for d in range(index.n_diseases) if d not in index.twin_of],
            dtype=np.int64,
        )
        row_scores = s[rows].copy()
        # The disease that gives the folded row its score: the OMIM entry or a twin
        scored_by = rows.copy()
        for i, d in enumerate(rows.tolist()):
            for o in index.twins.get(d, ()):
                if s[o] > row_scores[i]:
                    row_scores[i] = s[o]
                    scored_by[i] = o
        ranks = _competition_ranks(row_scores)
        order = np.lexsort((rows, -np.round(row_scores, 12)))[:top_k]
        result.total_candidates = len(rows)
        for i in order.tolist():
            d = int(rows[i])
            result.diseases.append(
                RankedDisease(
                    rank=int(ranks[i]),
                    disease_id=index.disease_ids[d],
                    label=index.disease_labels[d],
                    score=round(float(row_scores[i]), 6),
                    genes=index.disease_genes[d],
                    twins=[index.disease_ids[o] for o in index.twins.get(d, ())],
                    explanation=explainer.explain(int(scored_by[i])),
                )
            )
        return result

    genes = scorer.gene_scores(index, s)
    names = sorted(genes)
    values = np.array([genes[g][0] for g in names])
    ranks = _competition_ranks(values)
    order = np.lexsort((np.arange(len(names)), -np.round(values, 12)))[:top_k]
    result.total_candidates = len(names)
    for i in order.tolist():
        gene = names[i]
        best = genes[gene][1]  # the disease that gives the gene its score
        result.genes.append(
            RankedGene(
                rank=int(ranks[i]),
                gene=gene,
                score=round(float(values[i]), 6),
                best_disease_id=index.disease_ids[best],
                best_disease_label=index.disease_labels[best],
                explanation=explainer.explain(best),
            )
        )
    return result
