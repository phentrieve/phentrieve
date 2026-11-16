"""Ontology guard tests (extraction contract v2 hardening).

The guard drops resolved terms KNOWN to sit outside Phenotypic abnormality
(HP:0000118) -- clinical modifiers / course / inheritance -- which are never
valid standalone phenotype annotations, while failing open on unknown ancestry
so a real finding is never dropped.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.llm.pipeline import TwoPhaseLLMPipeline
from phentrieve.llm.types import LLMPhenotype


def _guard_with_map(monkeypatch, ancestors: dict[str, set[str]]) -> None:
    monkeypatch.setattr(
        ontology_guard,
        "_ancestors_map",
        lambda: {t: frozenset(a) for t, a in ancestors.items()},
    )


def test_flags_clinical_modifier_and_keeps_phenotype(monkeypatch):
    _guard_with_map(
        monkeypatch,
        {
            "HP:0012825": {"HP:0000001", "HP:0012823"},  # Mild -> Clinical modifier
            "HP:0001250": {"HP:0000001", "HP:0000118"},  # Seizure -> phenotype
        },
    )
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0012825") is True
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0001250") is False


def test_fails_open_on_unknown_or_root(monkeypatch):
    _guard_with_map(monkeypatch, {})
    # Unknown ancestry -> kept (never silently drop a real finding).
    assert ontology_guard.is_non_phenotypic_abnormality("HP:9999999") is False
    # The phenotype root itself is kept.
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0000118") is False
    assert ontology_guard.is_non_phenotypic_abnormality("") is False


def test_empty_load_is_not_cached_and_recovers(monkeypatch, tmp_path):
    """Regression guard (Codex High): a fail-open empty/failed load must NOT be
    memoized -- otherwise the first call before hpo_data.db is present latches the
    guard OFF for the whole process. It must recover once the DB appears."""
    og = ontology_guard
    og._ANCESTORS_CACHE.clear()
    monkeypatch.setattr(og, "resolve_data_path", lambda *a, **k: tmp_path)

    db_file = tmp_path / "hpo_data.db"
    # 1) DB absent -> fail open, and DO NOT cache.
    assert og._ancestors_map() == {}
    assert og._ANCESTORS_CACHE == {}

    # 2) DB now present -> loads and caches by path.
    db_file.write_text("")
    monkeypatch.setattr(
        og,
        "_load_ancestors_map",
        lambda path: {"HP:0012825": frozenset({"HP:0012823"})},
    )
    assert og._ancestors_map() == {"HP:0012825": frozenset({"HP:0012823"})}
    assert str(db_file) in og._ANCESTORS_CACHE


def test_pipeline_drops_non_phenotypic_terms(monkeypatch):
    _guard_with_map(
        monkeypatch,
        {
            "HP:0012825": {"HP:0000001", "HP:0012823"},  # Mild (modifier) -> drop
            "HP:0001250": {"HP:0000001", "HP:0000118"},  # Seizure -> keep
        },
    )
    terms = [
        LLMPhenotype(term_id="HP:0001250", label="Seizure", evidence="seizures"),
        LLMPhenotype(term_id="HP:0012825", label="Mild", evidence="normal"),
    ]
    kept = TwoPhaseLLMPipeline._drop_non_phenotypic(terms)

    assert [t.term_id for t in kept] == ["HP:0001250"]


def test_modifier_candidates_never_reach_mapping(monkeypatch):
    """A modifier match must not cost the finding: "recurrent seizures" matches
    Seizure cluster exactly, and the post-mapping guard then dropped it."""
    guard_map = {
        "HP:0033349": frozenset({"HP:0000001", "HP:0012823"}),  # Seizure cluster
        "HP:0001250": frozenset({"HP:0000001", "HP:0000118"}),  # Seizure
    }
    lookups: list[int] = []

    def _counting_map():
        lookups.append(1)
        return guard_map

    # One lookup per retrieval, not one per candidate.
    monkeypatch.setattr(ontology_guard, "_ancestors_map", _counting_map)

    class ToolExecutor:
        def query_batch_hpo_terms(self, *, phrases, language, n_results):
            return [
                {
                    "candidates": [
                        {"hpo_id": "HP:0033349", "term_name": "Cluster", "score": 1.0},
                        {"hpo_id": "HP:0001250", "term_name": "Seizure", "score": 0.8},
                    ]
                }
                for _ in phrases
            ]

    pipeline = TwoPhaseLLMPipeline(provider=MagicMock(), tool_executor=ToolExecutor())

    results = pipeline._retrieve_candidates(
        actionable=[
            {
                "phrase": "recurrent seizures",
                "category": "abnormal",
                "negated_qualifier": None,
                "chunk_ids": [1],
                "evidence_text": "recurrent seizures",
                "start_char": 0,
                "end_char": 18,
            }
        ],
        grounded_chunks=[],
        language="en",
    )

    assert [c["hpo_id"] for c in results[0]["candidates"]] == ["HP:0001250"]
    assert len(lookups) == 1


def test_guard_counts_drops_across_proband_family_and_other(monkeypatch):
    # Drops are counted, not only logged (#493). Tested on the guard step itself,
    # so a candidate filter before mapping does not change what is checked.
    _guard_with_map(
        monkeypatch,
        {"HP:9000001": {"HP:0000001", "HP:0012823"}, "HP:9000002": {"HP:0000118"}},
    )
    keep = LLMPhenotype(term_id="HP:9000002", label="term-b")
    drop = LLMPhenotype(term_id="HP:9000001", label="term-a")

    *guarded, dropped = TwoPhaseLLMPipeline._apply_ontology_guard(
        [keep, drop], [drop], []
    )
    proband, family, other = guarded

    assert [t.term_id for t in proband] == ["HP:9000002"]
    assert family == [] and other == []
    assert dropped == 2


def test_guard_reports_unavailable_graph(monkeypatch):
    _guard_with_map(monkeypatch, {})

    assert ontology_guard.guard_available() is False


def test_missing_db_is_logged_as_error_once(monkeypatch, tmp_path, caplog):
    """Retries keep running, but only the first failure per path is an ERROR."""
    import logging

    og = ontology_guard
    og._ANCESTORS_CACHE.clear()
    monkeypatch.setattr(og, "_REPORTED_FAILURES", set())
    monkeypatch.setattr(og, "resolve_data_path", lambda *a, **k: tmp_path)

    with caplog.at_level(logging.DEBUG, logger=og.__name__):
        for _ in range(3):
            assert og._ancestors_map() == {}

    levels = [r.levelno for r in caplog.records if "not found" in r.getMessage()]
    assert levels == [logging.ERROR, logging.DEBUG, logging.DEBUG]


def test_explicit_map_skips_the_lookup(monkeypatch):
    def _fail():
        raise AssertionError("_ancestors_map must not be called")

    monkeypatch.setattr(ontology_guard, "_ancestors_map", _fail)
    guard_map = {"HP:0012825": frozenset({"HP:0000001", "HP:0012823"})}
    assert ontology_guard.is_non_phenotypic_abnormality("HP:0012825", guard_map)
    assert not ontology_guard.is_non_phenotypic_abnormality("HP:9999999", guard_map)


# --- Guard filter ahead of hybrid candidate selection (#493) -----------------
# Placeholder ids and labels: HP:90000xx sit outside the phenotype subtree,
# HP:91000xx inside it.

_NON_PHENOTYPIC_IDS = [f"HP:90000{i:02d}" for i in range(12)]
_PHENOTYPIC_IDS = [f"HP:91000{i:02d}" for i in range(6)]
_PLACEHOLDER_GUARD_MAP = {
    **{t: {"HP:0000001", "HP:0012823"} for t in _NON_PHENOTYPIC_IDS},
    **{t: {"HP:0000001", "HP:0000118"} for t in _PHENOTYPIC_IDS},
}


def _placeholder_item(phrase: str) -> dict:
    return {
        "phrase": phrase,
        "category": "abnormal",
        "negated_qualifier": None,
        "chunk_ids": [1],
        "evidence_text": phrase,
        "start_char": 0,
        "end_char": len(phrase),
    }


class _RankedHitsExecutor:
    """Returns the same ranked ``(hpo_id, label)`` hits for every query."""

    def __init__(self, hits: list[tuple[str, str]], *, shape: str) -> None:
        self.hits = hits
        self.shape = shape

    def query_batch_hpo_terms(self, *, phrases, language, n_results):
        scores = [0.99 - 0.005 * index for index in range(len(self.hits))]
        if self.shape == "candidates":
            result = {
                "candidates": [
                    {"hpo_id": hpo_id, "term_name": label, "score": score}
                    for (hpo_id, label), score in zip(self.hits, scores, strict=True)
                ]
            }
        else:
            result = {
                "metadatas": [
                    [{"hpo_id": hpo_id, "label": label} for hpo_id, label in self.hits]
                ],
                "similarities": [scores],
            }
        return [result for _ in phrases]


def _retrieve(hits: list[tuple[str, str]], *, shape: str, phrase: str):
    pipeline = TwoPhaseLLMPipeline(
        provider=MagicMock(), tool_executor=_RankedHitsExecutor(hits, shape=shape)
    )
    results = pipeline._retrieve_candidates(
        actionable=[_placeholder_item(phrase)], grounded_chunks=[], language="en"
    )
    return pipeline, results


def test_non_phenotypic_hits_do_not_exhaust_hybrid_selection(monkeypatch):
    """More non-phenotypic hits than selection slots, ranked above the one
    phenotypic hit: it must still reach routing and resolve the finding."""
    _guard_with_map(monkeypatch, _PLACEHOLDER_GUARD_MAP)
    hits = [(term_id, "term-a") for term_id in _NON_PHENOTYPIC_IDS]
    hits.append((_PHENOTYPIC_IDS[0], "term-b"))

    pipeline, results = _retrieve(hits, shape="metadatas", phrase="term-b")
    assert len(_NON_PHENOTYPIC_IDS) > pipeline.max_unique_candidates
    assert [c["hpo_id"] for c in results[0]["candidates"]] == [_PHENOTYPIC_IDS[0]]

    resolved, unresolved, counts = pipeline._route_phase2_candidates(
        phrase_candidates=results, language="en"
    )
    assert [t.term_id for t in resolved] == [_PHENOTYPIC_IDS[0]]
    assert unresolved == []
    assert counts["phase2b_no_candidate_skip_count"] == 0


def test_phenotypic_hits_fill_slots_freed_by_non_phenotypic_hits(monkeypatch):
    """A few non-phenotypic hits on top must not use up the min-fill quota:
    the phenotypic hits below them fill it, with their own scores."""
    _guard_with_map(monkeypatch, _PLACEHOLDER_GUARD_MAP)
    hits = [(term_id, "term-a") for term_id in _NON_PHENOTYPIC_IDS[:3]]
    hits += [(term_id, "term-b") for term_id in _PHENOTYPIC_IDS]

    # The phrase shares no token with any label and the threshold is out of
    # reach, so only the min-fill quota selects candidates.
    pipeline = TwoPhaseLLMPipeline(
        provider=MagicMock(),
        tool_executor=_RankedHitsExecutor(hits, shape="metadatas"),
        similarity_threshold=2.0,
    )
    results = pipeline._retrieve_candidates(
        actionable=[_placeholder_item("unrelated wording")],
        grounded_chunks=[],
        language="en",
    )

    candidates = results[0]["candidates"]
    expected = _PHENOTYPIC_IDS[: pipeline.min_unique_candidates]
    assert [c["hpo_id"] for c in candidates] == expected
    # Scores stay aligned with their hits after the filter (ranks 4 and up).
    assert candidates[0]["score"] == 0.99 - 0.005 * 3


def test_modifier_hits_never_reach_mapping_metadatas_shape(monkeypatch):
    """Counterpart of ``test_modifier_candidates_never_reach_mapping`` for
    results that go through hybrid selection; still one map lookup."""
    lookups: list[int] = []
    guard_map = {t: frozenset(a) for t, a in _PLACEHOLDER_GUARD_MAP.items()}

    def _counting_map():
        lookups.append(1)
        return guard_map

    monkeypatch.setattr(ontology_guard, "_ancestors_map", _counting_map)
    hits = [(_NON_PHENOTYPIC_IDS[0], "term-a"), (_PHENOTYPIC_IDS[0], "term-b")]

    _, results = _retrieve(hits, shape="metadatas", phrase="term-a")

    assert [c["hpo_id"] for c in results[0]["candidates"]] == [_PHENOTYPIC_IDS[0]]
    assert len(lookups) == 1


def test_candidate_filter_fails_open_without_graph(monkeypatch):
    """Graph unavailable: nothing is filtered, on either result shape."""
    _guard_with_map(monkeypatch, {})
    hits = [(term_id, "term-a") for term_id in _NON_PHENOTYPIC_IDS[:3]]
    hits.append((_PHENOTYPIC_IDS[0], "term-b"))
    all_ids = [hpo_id for hpo_id, _ in hits]

    for shape in ("metadatas", "candidates"):
        _, results = _retrieve(hits, shape=shape, phrase="term-a")
        assert [c["hpo_id"] for c in results[0]["candidates"]] == all_ids
