"""Disease index construction on an invented mini ontology."""

from __future__ import annotations

import pytest

from phentrieve.disease_matching.index import SOURCE_ORPHA, SOURCE_OWN, DiseaseIndex
from phentrieve.disease_matching.index_builder import parse_mondo_equivalences
from tests.unit.disease_matching.conftest import MINI

pytestmark = pytest.mark.unit


def _annotations(ix: DiseaseIndex, disease: str) -> dict[str, tuple[float, int]]:
    terms, freqs, sources = ix.annotations(ix.disease_index[disease])
    return {
        ix.term_ids[int(t)]: (float(f), int(s))
        for t, f, s in zip(terms, freqs, sources, strict=True)
    }


def test_universe_is_gene_linked_diseases_with_phenotypes(
    mini_index: DiseaseIndex,
) -> None:
    assert mini_index.disease_ids == [
        "OMIM:100001",
        "OMIM:100002",
        "ORPHA:900001",
        "ORPHA:900002",
    ]
    assert sorted(mini_index.gene_diseases) == ["GENEA", "GENEB", "GENEC"]
    assert mini_index.disease_genes[mini_index.disease_index["ORPHA:900001"]] == [
        "GENEA"
    ]


def test_phenotype_subtree_and_aliases(mini_index: DiseaseIndex) -> None:
    assert "HP:0000118" in mini_index.term_index
    assert "HP:0000006" not in mini_index.term_index
    assert {"HP:0000001", "HP:0000005", "HP:0000006"} <= set(mini_index.other_terms)
    a1 = mini_index.term_index["HP:0000003"]
    assert mini_index.resolve_term("HP:0000097") == a1  # alt_id
    assert (
        mini_index.resolve_term("HP:0000099") == a1
    )  # replaced_by of a deprecated term


def test_frequencies_and_not_rows(mini_index_plain: DiseaseIndex) -> None:
    one = _annotations(mini_index_plain, "OMIM:100001")
    assert one["HP:0000003"][0] == pytest.approx(0.895, abs=1e-6)  # Very frequent
    assert one["HP:0000011"][0] == pytest.approx((1 + 0.8) / (2 + 2), abs=1e-6)  # 1/2
    assert "HP:0000006" not in one  # inheritance row, aspect I
    two = _annotations(mini_index_plain, "OMIM:100002")
    assert two["HP:0000012"][0] == pytest.approx(0.5, abs=1e-6)  # 50%
    assert "HP:0000020" not in two  # NOT qualifier


def test_orpha_merge_adds_terms_and_keeps_higher_frequency(
    mini_index: DiseaseIndex, mini_index_plain: DiseaseIndex
) -> None:
    plain = _annotations(mini_index_plain, "OMIM:100001")
    merged = _annotations(mini_index, "OMIM:100001")
    assert "HP:0000020" not in plain
    assert merged["HP:0000020"] == (pytest.approx(0.545, abs=1e-6), SOURCE_ORPHA)
    # ORPHA has HP:0000003 at 0.17 (via the replaced_by alias); the higher OMIM value wins.
    assert merged["HP:0000003"][0] == pytest.approx(0.895, abs=1e-6)
    assert merged["HP:0000003"][1] == SOURCE_OWN | SOURCE_ORPHA
    assert merged["HP:0000011"][1] == SOURCE_OWN
    assert mini_index.meta["n_omim_enriched"] == 1


def test_obsolete_and_non_equivalent_mondo_terms_do_not_merge(
    mini_index: DiseaseIndex,
) -> None:
    two = _annotations(mini_index, "OMIM:100002")
    assert "HP:0000011" not in two
    assert mini_index.disease_index["OMIM:100002"] not in mini_index.twins


def test_twins_link_omim_to_orpha_in_universe(mini_index: DiseaseIndex) -> None:
    omim = mini_index.disease_index["OMIM:100001"]
    orpha = mini_index.disease_index["ORPHA:900001"]
    assert mini_index.twins == {omim: [orpha]}
    assert mini_index.twin_of == {orpha: omim}


def test_parse_mondo_equivalences() -> None:
    assert parse_mondo_equivalences(MINI / "mondo.obo") == {
        "OMIM:100001": {"ORPHA:900001"}
    }


def test_merge_needs_mondo() -> None:
    from phentrieve.disease_matching.index_builder import build_index

    with pytest.raises(ValueError, match="MONDO"):
        build_index(
            MINI / "hp.json",
            MINI / "phenotype.hpoa",
            MINI / "genes_to_phenotype.txt",
            None,
            merge_orpha=True,
        )


def test_meta_records_counts_and_licences(mini_index: DiseaseIndex) -> None:
    meta = mini_index.meta
    assert meta["n_diseases"] == 4
    assert meta["n_genes"] == 3
    assert meta["merge_orpha"] is True
    assert set(meta["licences"]) == {"hpo", "orphanet", "mondo"}


def test_fetch_sources_downloads_missing_files_and_checksums(tmp_path, mocker) -> None:
    import shutil

    from phentrieve.disease_matching import index_builder

    calls: list[str] = []

    def fake_download(url: str, path) -> None:
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MINI / path.name, path)

    mocker.patch.object(index_builder, "_download", side_effect=fake_download)
    out = index_builder.fetch_sources(
        tmp_path, "v2026-09-01", "v2026-09-01", "https://hpo.example/releases"
    )
    assert calls[0] == "https://hpo.example/releases/v2026-09-01/hp.json"
    assert calls[-1].endswith("/v2026-09-01/mondo.obo")
    assert out["files"]["mondo.obo"]["sha256"] == index_builder.sha256_file(
        MINI / "mondo.obo"
    )
    calls.clear()
    index_builder.fetch_sources(
        tmp_path, "v2026-09-01", "v2026-09-01", "https://hpo.example/releases"
    )
    assert calls == []  # present files are reused
