"""Ranking service: assertions, twin folding, gene level and explanations."""

from __future__ import annotations

import pytest

from phentrieve.disease_matching import ProfileTerm, rank, scorer
from phentrieve.disease_matching.index import DiseaseIndex

pytestmark = pytest.mark.unit


def _p(*items: tuple[str, str]) -> list[ProfileTerm]:
    return [ProfileTerm(h, a) for h, a in items]


def test_twins_are_folded_into_the_omim_row(mini_index: DiseaseIndex) -> None:
    result = rank(
        mini_index, _p(("HP:0000003", "affirmed"), ("HP:0000020", "affirmed"))
    )
    ids = [d.disease_id for d in result.diseases]
    assert "ORPHA:900001" not in ids
    assert "ORPHA:900002" in ids  # an ORPHA entry without OMIM twin stays
    assert result.total_candidates == 3
    top = result.diseases[0]
    assert top.disease_id == "OMIM:100001"
    assert top.twins == ["ORPHA:900001"]
    assert top.genes == ["GENEA"]
    assert top.rank == 1


def test_folded_row_takes_the_better_twin_score(mini_index_plain: DiseaseIndex) -> None:
    profile = _p(("HP:0000020", "affirmed"))
    raw = scorer.score(
        mini_index_plain, {mini_index_plain.term_index["HP:0000020"]: 1.0}
    )
    assert raw is not None
    omim = mini_index_plain.disease_index["OMIM:100001"]
    orpha = mini_index_plain.disease_index["ORPHA:900001"]
    assert raw.scores[orpha] > raw.scores[omim]
    row = next(
        d
        for d in rank(mini_index_plain, profile).diseases
        if d.disease_id == "OMIM:100001"
    )
    assert row.score == pytest.approx(raw.scores[orpha], abs=1e-6)


def test_assertions_unknown_and_redundant_terms(mini_index: DiseaseIndex) -> None:
    result = rank(
        mini_index,
        _p(
            ("HP:0000002", "affirmed"),
            ("HP:0000003", "present"),
            ("HP:0000011", "negated"),
            ("HP:0000020", "normal"),
            ("HP:0000004", "uncertain"),
            ("HP:0000006", "affirmed"),
            ("HP:1234567", "affirmed"),
        ),
    )
    assert result.scored == ["HP:0000003"]
    assert result.excluded == ["HP:0000011", "HP:0000020"]
    reasons = {n.hpo_id: n.reason for n in result.not_scored}
    assert reasons == {
        "HP:0000004": "uncertain",
        "HP:0000006": "not_phenotype",
        "HP:1234567": "unknown_term",
        "HP:0000002": "redundant",
    }


def test_include_uncertain_scores_them_as_present(mini_index: DiseaseIndex) -> None:
    profile = _p(("HP:0000003", "affirmed"), ("HP:0000004", "uncertain"))
    off = rank(mini_index, profile)
    on = rank(mini_index, profile, include_uncertain=True)
    assert "HP:0000004" not in off.scored
    assert "HP:0000004" in on.scored
    assert [d.score for d in off.diseases] != [d.score for d in on.diseases]


def test_aliases_resolve_to_current_terms(mini_index: DiseaseIndex) -> None:
    by_alias = rank(mini_index, _p(("HP:0000099", "affirmed")))
    current = rank(mini_index, _p(("HP:0000003", "affirmed")))
    assert by_alias.scored == ["HP:0000003"]
    assert [d.score for d in by_alias.diseases] == [d.score for d in current.diseases]


def test_excluded_terms_do_not_change_scores_but_raise_conflicts(
    mini_index: DiseaseIndex,
) -> None:
    base = rank(mini_index, _p(("HP:0000003", "affirmed")))
    with_excluded = rank(
        mini_index, _p(("HP:0000003", "affirmed"), ("HP:0000011", "negated"))
    )
    assert [(d.disease_id, d.score) for d in base.diseases] == [
        (d.disease_id, d.score) for d in with_excluded.diseases
    ]
    conflicts = {
        d.disease_id: [
            (c.profile_id, c.disease_term_id) for c in d.explanation.conflicts
        ]
        for d in with_excluded.diseases
    }
    assert conflicts["OMIM:100001"] == [("HP:0000011", "HP:0000011")]
    # Disease Two is annotated with a descendant of the excluded term.
    assert conflicts["OMIM:100002"] == [("HP:0000011", "HP:0000012")]
    assert conflicts["ORPHA:900002"] == [("HP:0000011", "HP:0000011")]


def test_match_relations_and_sources(mini_index: DiseaseIndex) -> None:
    result = rank(
        mini_index, _p(("HP:0000003", "affirmed"), ("HP:0000012", "affirmed"))
    )
    one = next(d for d in result.diseases if d.disease_id == "OMIM:100001").explanation
    by_profile = {m.profile_id: m for m in one.matches + one.unexplained}
    assert by_profile["HP:0000003"].relation == "exact"
    assert by_profile["HP:0000003"].source == "both"
    assert by_profile["HP:0000012"].disease_term_id == "HP:0000011"
    assert by_profile["HP:0000012"].relation == "broader"
    assert one.annotation_sources == ["omim", "orpha"]

    narrower = rank(mini_index, _p(("HP:0000011", "affirmed")))
    two = next(
        d for d in narrower.diseases if d.disease_id == "OMIM:100002"
    ).explanation
    match = (two.matches + two.unexplained)[0]
    assert match.disease_term_id == "HP:0000012"
    assert match.relation == "narrower"


def test_weak_matches_are_listed_as_unexplained(mini_index: DiseaseIndex) -> None:
    result = rank(
        mini_index, _p(("HP:0000003", "affirmed"), ("HP:0000020", "affirmed"))
    )
    two = next(d for d in result.diseases if d.disease_id == "OMIM:100002").explanation
    # A1 only meets Disease Two's A2 via the organ-level ancestor; C is not annotated at all.
    assert {m.profile_id for m in two.unexplained} == {"HP:0000003", "HP:0000020"}
    assert two.matches == []
    assert all(m.match < 0.4 for m in two.unexplained)
    one = next(d for d in result.diseases if d.disease_id == "OMIM:100001").explanation
    assert {m.profile_id for m in one.matches} == {"HP:0000003", "HP:0000020"}


def test_gene_level_reports_the_omim_row_for_an_orpha_best(
    mini_index_plain: DiseaseIndex,
) -> None:
    result = rank(mini_index_plain, _p(("HP:0000020", "affirmed")), level="gene")
    assert result.level == "gene"
    assert result.diseases == []
    genea = next(g for g in result.genes if g.gene == "GENEA")
    assert genea.best_disease_id == "OMIM:100001"
    assert result.total_candidates == 3
    assert [g.rank for g in result.genes] == sorted(g.rank for g in result.genes)


def test_empty_profile_returns_no_rows(mini_index: DiseaseIndex) -> None:
    result = rank(mini_index, _p(("HP:0000011", "negated")))
    assert result.diseases == [] and result.genes == []
    assert result.excluded == ["HP:0000011"]


def test_top_k_is_clamped(mini_index: DiseaseIndex) -> None:
    profile = _p(("HP:0000003", "affirmed"))
    assert len(rank(mini_index, profile, top_k=1).diseases) == 1
    assert len(rank(mini_index, profile, top_k=0).diseases) == 1
    assert len(rank(mini_index, profile, top_k=10_000).diseases) == 3


def test_ties_share_a_rank(mini_index: DiseaseIndex) -> None:
    result = rank(
        mini_index, _p(("HP:0000118", "affirmed"), ("HP:0000020", "affirmed"))
    )
    scores = [d.score for d in result.diseases]
    ranks = [d.rank for d in result.diseases]
    for i in range(1, len(scores)):
        if scores[i] == scores[i - 1]:
            assert ranks[i] == ranks[i - 1]
        else:
            assert ranks[i] == i + 1


def test_folded_row_is_explained_by_the_disease_that_scored_it(
    mini_index_plain: DiseaseIndex,
) -> None:
    row = next(
        d
        for d in rank(mini_index_plain, _p(("HP:0000020", "affirmed"))).diseases
        if d.disease_id == "OMIM:100001"
    )
    # The ORPHA twin scores higher, so its annotations explain the row.
    assert [(m.profile_id, m.disease_term_id) for m in row.explanation.matches] == [
        ("HP:0000020", "HP:0000020")
    ]
    assert row.explanation.matches[0].source == "orpha"
    assert row.explanation.annotation_sources == ["orpha"]


def test_gene_level_keeps_the_orpha_best_when_the_twin_is_another_genes(
    mini_index_plain: DiseaseIndex,
) -> None:
    import dataclasses

    orpha = mini_index_plain.disease_index["ORPHA:900001"]
    genes = dict(mini_index_plain.gene_diseases)
    genes["GENEE"] = [orpha]  # ORPHA entry linked to a gene its OMIM twin is not
    ix = dataclasses.replace(mini_index_plain, gene_diseases=genes)
    result = rank(ix, _p(("HP:0000020", "affirmed")), level="gene")
    by_gene = {g.gene: g for g in result.genes}
    assert by_gene["GENEE"].best_disease_id == "ORPHA:900001"
    assert by_gene["GENEA"].best_disease_id == "OMIM:100001"
    assert by_gene["GENEA"].explanation.matches[0].source == "orpha"


def test_orpha_disease_annotations_are_labelled_orpha(mini_index: DiseaseIndex) -> None:
    result = rank(mini_index, _p(("HP:0000011", "affirmed")))
    two = next(d for d in result.diseases if d.disease_id == "ORPHA:900002")
    assert {m.source for m in two.explanation.matches} == {"orpha"}
    one = next(d for d in result.diseases if d.disease_id == "OMIM:100001")
    assert {m.source for m in one.explanation.matches} == {"omim"}
