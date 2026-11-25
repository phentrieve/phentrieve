"""Unit tests for modifier models, ontology profiles, and role-aware ontology guard."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import phentrieve.config as phentrieve_config
import phentrieve.llm.ontology_guard as ontology_guard
from phentrieve.modifier_models import Modifier, ModifierSegment
from phentrieve.ontology_profiles import (
    HPO_CLINICAL_MODIFIER_ROOT,
    HPO_DEFAULT_PROFILE,
    HPO_FREQUENCY_ROOT,
    HPO_ONSET_ROOT,
    HPO_PHENOTYPIC_ABNORMALITY_ROOT,
    HPO_SEVERITY_ROOT,
    OntologyProfile,
    OntologyRootsConfig,
    get_default_ontology_profile,
)

pytestmark = pytest.mark.unit


# Synthetic ancestor graph using HPO branch roots and invented/standard terms.
_SYNTHETIC_ANCESTORS: dict[str, frozenset[str]] = {
    "HP:0000118": frozenset({"HP:0000001", "HP:0000118"}),
    "HP:0012823": frozenset({"HP:0000001", "HP:0012823"}),
    "HP:0012824": frozenset({"HP:0000001", "HP:0012823", "HP:0012824"}),
    "HP:0003674": frozenset({"HP:0000001", "HP:0012823", "HP:0003674"}),
    "HP:0040279": frozenset({"HP:0000001", "HP:0040279"}),
    # Finding only (Phenotypic abnormality)
    "HP:0001250": frozenset({"HP:0000001", "HP:0000118", "HP:0001250"}),
    # Modifier only: Severity terms
    "HP:0012825": frozenset({"HP:0000001", "HP:0012823", "HP:0012824", "HP:0012825"}),
    "HP:0012828": frozenset({"HP:0000001", "HP:0012823", "HP:0012824", "HP:0012828"}),
    # Modifier only: Onset terms
    "HP:0003577": frozenset({"HP:0000001", "HP:0012823", "HP:0003674", "HP:0003577"}),
    "HP:0003621": frozenset({"HP:0000001", "HP:0012823", "HP:0003674", "HP:0003621"}),
    # Modifier only: Non-exclusive spatial/laterality modifier
    "HP:0012832": frozenset({"HP:0000001", "HP:0012823", "HP:0012832"}),
    # Modifier only: Invented example such as Stillbirth under Clinical modifier
    "HP:9000101": frozenset({"HP:0000001", "HP:0012823", "HP:9000101"}),
    # Overlap term sitting under BOTH finding_roots and modifier_roots
    "HP:9000202": frozenset({"HP:0000001", "HP:0000118", "HP:0012823", "HP:9000202"}),
    # Mode of inheritance (neither finding nor modifier)
    "HP:0000006": frozenset({"HP:0000001", "HP:0000005", "HP:0000006"}),
    # Cohort frequency (disallowed modifier root HP:0040279)
    "HP:0040283": frozenset({"HP:0000001", "HP:0040279", "HP:0040283"}),
}


# ---------------------------------------------------------------------------
# ModifierSegment and Modifier model validation
# ---------------------------------------------------------------------------


def test_modifier_segment_valid_and_invalid_bounds():
    segment = ModifierSegment(start=4, end=10, text="severe")
    assert segment.start == 4
    assert segment.end == 10
    assert segment.text == "severe"

    with pytest.raises(ValidationError, match="end must be greater than start"):
        ModifierSegment(start=5, end=5, text="a")

    with pytest.raises(ValidationError, match="end must be greater than start"):
        ModifierSegment(start=8, end=5, text="abc")

    with pytest.raises(ValidationError, match="text length must equal end - start"):
        ModifierSegment(start=0, end=6, text="mild")

    with pytest.raises(ValidationError, match="text length must equal end - start"):
        ModifierSegment(start=0, end=4, text="severe")


def test_modifier_curie_validation():
    mod = Modifier(term_id="HP:0012828", label="Severe", source="llm")
    assert mod.segments == []
    assert mod.source == "llm"

    alt_mod = Modifier(term_id="SNOMEDCT_US:24484000", label="Severe", source="rule")
    assert alt_mod.term_id == "SNOMEDCT_US:24484000"

    for bad_curie in ("", "HP0012828", "HP:", "123:456", "HP:001 2828"):
        with pytest.raises(ValidationError):
            Modifier(term_id=bad_curie, label="Severe")

    with pytest.raises(ValidationError):
        Modifier(term_id="HP:0012828", label="")


# ---------------------------------------------------------------------------
# OntologyRootsConfig, OntologyProfile, and YAML loading
# ---------------------------------------------------------------------------


def test_ontology_roots_config_normalization_and_required_roots():
    cfg = OntologyRootsConfig(
        finding_roots=[" HP:0000118 ", "HP:0000118", "HP:0012823"],
        modifier_roots=("HP:0012823", " HP:0012824 "),
        index_roots={"HP:0000118"},
    )
    assert cfg.finding_roots == ("HP:0000118", "HP:0012823")
    assert cfg.modifier_roots == ("HP:0012823", "HP:0012824")
    assert cfg.index_roots == ("HP:0000118",)

    assert cfg.effective_required_roots(include_modifiers=False) == (
        "HP:0000118",
        "HP:0012823",
    )
    assert cfg.effective_required_roots(include_modifiers=True) == (
        "HP:0000118",
        "HP:0012823",
        "HP:0012824",
    )

    unrestricted = OntologyRootsConfig(finding_roots=[], modifier_roots=[])
    assert unrestricted.finding_roots == ()
    assert unrestricted.modifier_roots == ()
    assert unrestricted.index_roots == ()
    assert unrestricted.effective_required_roots(include_modifiers=True) == ()

    with pytest.raises(ValidationError):
        OntologyRootsConfig(finding_roots="HP:0000118")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        OntologyRootsConfig(finding_roots=["not-a-curie"])

    with pytest.raises(ValidationError):
        OntologyRootsConfig(finding_roots=["  "])


def test_hpo_default_profile_and_config_exports():
    assert HPO_DEFAULT_PROFILE.name == "hpo"
    assert HPO_DEFAULT_PROFILE.ontology_id == "hp"
    assert HPO_DEFAULT_PROFILE.roots.finding_roots == (HPO_PHENOTYPIC_ABNORMALITY_ROOT,)
    assert HPO_DEFAULT_PROFILE.roots.modifier_roots == (HPO_CLINICAL_MODIFIER_ROOT,)
    assert HPO_DEFAULT_PROFILE.roots.index_roots == ()
    assert HPO_DEFAULT_PROFILE.exclusive_modifier_classes == {
        "severity": HPO_SEVERITY_ROOT,
        "onset": HPO_ONSET_ROOT,
    }
    assert HPO_DEFAULT_PROFILE.disallowed_modifier_roots == {
        "cohort_frequency": HPO_FREQUENCY_ROOT,
    }
    assert HPO_DEFAULT_PROFILE.phenopacket_slot_roots == {
        "severity": HPO_SEVERITY_ROOT,
        "onset": HPO_ONSET_ROOT,
    }
    assert phentrieve_config.PHENOTYPE_ROOT == "HP:0000118"
    assert isinstance(phentrieve_config.DEFAULT_FINDING_ROOTS, tuple)
    assert isinstance(phentrieve_config.DEFAULT_MODIFIER_ROOTS, tuple)
    assert isinstance(phentrieve_config.DEFAULT_INDEX_ROOTS, tuple)


def test_get_default_ontology_profile_yaml_explicit_empty_vs_defaults(monkeypatch):
    # Omitted ontology block -> defaults
    monkeypatch.setattr(phentrieve_config, "_load_yaml_config", lambda: {})
    profile_default = get_default_ontology_profile()
    assert profile_default == HPO_DEFAULT_PROFILE

    # Explicit empty finding_roots: [] -> unrestricted (), other omitted keys keep defaults
    monkeypatch.setattr(
        phentrieve_config,
        "_load_yaml_config",
        lambda: {"ontology": {"finding_roots": []}},
    )
    profile_unrestricted = get_default_ontology_profile()
    assert profile_unrestricted.roots.finding_roots == ()
    assert profile_unrestricted.roots.modifier_roots == (HPO_CLINICAL_MODIFIER_ROOT,)
    assert profile_unrestricted.roots.index_roots == ()

    # Explicit index_roots override
    monkeypatch.setattr(
        phentrieve_config,
        "_load_yaml_config",
        lambda: {
            "ontology": {
                "index_roots": ["HP:0000118", "HP:0012823"],
                "modifier_roots": None,
            }
        },
    )
    profile_indexed = get_default_ontology_profile()
    assert profile_indexed.roots.finding_roots == (HPO_PHENOTYPIC_ABNORMALITY_ROOT,)
    assert profile_indexed.roots.modifier_roots == (HPO_CLINICAL_MODIFIER_ROOT,)
    assert profile_indexed.roots.index_roots == ("HP:0000118", "HP:0012823")


# ---------------------------------------------------------------------------
# Role-aware root queries
# ---------------------------------------------------------------------------


def test_is_under_any_root_and_term_ids_under_roots():
    assert (
        ontology_guard.is_under_any_root(
            "HP:0001250", ("HP:0000118",), _SYNTHETIC_ANCESTORS
        )
        is True
    )
    assert ontology_guard.is_under_any_root("HP:0000118", ("HP:0000118",), {}) is True
    assert (
        ontology_guard.is_under_any_root(
            "HP:0012828", ("HP:0000118",), _SYNTHETIC_ANCESTORS
        )
        is False
    )
    assert (
        ontology_guard.is_under_any_root(
            "HP:9999999", ("HP:0000118",), _SYNTHETIC_ANCESTORS
        )
        is None
    )
    assert (
        ontology_guard.is_under_any_root("", ("HP:0000118",), _SYNTHETIC_ANCESTORS)
        is None
    )

    finding_ids = ontology_guard.term_ids_under_roots(
        ("HP:0000118",), _SYNTHETIC_ANCESTORS
    )
    assert finding_ids == {"HP:0000118", "HP:0001250", "HP:9000202"}

    all_ids = ontology_guard.term_ids_under_roots((), _SYNTHETIC_ANCESTORS)
    assert all_ids == set(_SYNTHETIC_ANCESTORS.keys())


# ---------------------------------------------------------------------------
# check_finding_term
# ---------------------------------------------------------------------------


def test_check_finding_term_default_profile():
    # Finding under finding_roots -> accepted
    d_finding = ontology_guard.check_finding_term(
        "HP:0001250", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_finding.accepted is True
    assert d_finding.is_modifier_term is False
    assert d_finding.warnings == ()

    # Modifier-only term -> rejected and flagged as modifier
    d_mod = ontology_guard.check_finding_term(
        "HP:0012828", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_mod.accepted is False
    assert d_mod.is_modifier_term is True
    assert d_mod.warnings == ()

    # Non-finding, non-modifier term (Mode of inheritance) -> rejected, not a modifier
    d_inh = ontology_guard.check_finding_term(
        "HP:0000006", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_inh.accepted is False
    assert d_inh.is_modifier_term is False
    assert d_inh.warnings == ()

    # Overlap term under BOTH finding_roots and modifier_roots -> accepted as valid finding
    d_overlap = ontology_guard.check_finding_term(
        "HP:9000202", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_overlap.accepted is True
    assert d_overlap.is_modifier_term is False
    assert d_overlap.warnings == ()


def test_check_finding_term_unrestricted_emits_modifier_as_annotation():
    unrestricted_profile = OntologyProfile(
        name="hpo-unrestricted",
        roots=OntologyRootsConfig(
            finding_roots=(),
            modifier_roots=(HPO_CLINICAL_MODIFIER_ROOT,),
        ),
        disallowed_modifier_roots={"cohort_frequency": HPO_FREQUENCY_ROOT},
    )

    # Invented modifier-branch finding (e.g. Stillbirth) kept with warning
    d_stillbirth = ontology_guard.check_finding_term(
        "HP:9000101",
        profile=unrestricted_profile,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )
    assert d_stillbirth.accepted is True
    assert d_stillbirth.is_modifier_term is True
    assert len(d_stillbirth.warnings) == 1
    assert d_stillbirth.warnings[0].code == "modifier_as_annotation"
    assert d_stillbirth.warnings[0].term_id == "HP:9000101"

    # Disallowed frequency term kept with modifier_as_annotation warning
    d_freq = ontology_guard.check_finding_term(
        "HP:0040283",
        profile=unrestricted_profile,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )
    assert d_freq.accepted is True
    assert d_freq.is_modifier_term is False
    assert len(d_freq.warnings) == 1
    assert d_freq.warnings[0].code == "modifier_as_annotation"
    assert d_freq.warnings[0].term_id == "HP:0040283"

    # Normal phenotype finding has no warning
    d_pheno = ontology_guard.check_finding_term(
        "HP:0001250",
        profile=unrestricted_profile,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )
    assert d_pheno.accepted is True
    assert d_pheno.is_modifier_term is False
    assert d_pheno.warnings == ()


def test_check_finding_term_fails_open_on_unknown_ancestry():
    d_empty = ontology_guard.check_finding_term(
        "HP:0012828", profile=HPO_DEFAULT_PROFILE, ancestors_map={}
    )
    assert d_empty.accepted is True
    assert d_empty.is_modifier_term is False
    assert d_empty.warnings == ()

    d_unseen = ontology_guard.check_finding_term(
        "HP:9999999",
        profile=HPO_DEFAULT_PROFILE,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )
    assert d_unseen.accepted is True
    assert d_unseen.is_modifier_term is False
    assert d_unseen.warnings == ()


# ---------------------------------------------------------------------------
# check_modifier_term and validate_finding_modifiers
# ---------------------------------------------------------------------------


def test_check_modifier_term_accepts_rejects_and_flags_frequency():
    d_valid = ontology_guard.check_modifier_term(
        "HP:0012828", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_valid.accepted is True
    assert d_valid.warnings == ()

    d_non_mod = ontology_guard.check_modifier_term(
        "HP:0001250", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_non_mod.accepted is False
    assert d_non_mod.warnings == ()

    # Abstract container roots (Clinical modifier, Clinical course, Severity, Onset)
    # are grouping headers rather than clinical modifier values and must be rejected.
    for abstract_root in ("HP:0012823", "HP:0031797", "HP:0012824", "HP:0003674"):
        d_abs = ontology_guard.check_modifier_term(
            abstract_root,
            profile=HPO_DEFAULT_PROFILE,
            ancestors_map=_SYNTHETIC_ANCESTORS,
        )
        assert d_abs.accepted is False
        assert d_abs.warnings == ()

    d_freq = ontology_guard.check_modifier_term(
        "HP:0040283", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_freq.accepted is False
    assert len(d_freq.warnings) == 1
    assert d_freq.warnings[0].code == "cohort_frequency"
    assert d_freq.warnings[0].term_id == "HP:0040283"

    d_unknown = ontology_guard.check_modifier_term(
        "HP:9999999", profile=HPO_DEFAULT_PROFILE, ancestors_map=_SYNTHETIC_ANCESTORS
    )
    assert d_unknown.accepted is True
    assert d_unknown.warnings == ()


def test_validate_finding_modifiers_dedup_merge_filter_and_exclusive_classes():
    seg1 = ModifierSegment(start=0, end=6, text="severe")
    seg2 = ModifierSegment(start=20, end=26, text="severe")
    original_first = Modifier(
        term_id="HP:0012828", label="Severe", segments=[seg1], source="llm"
    )
    duplicate_with_extra_seg = Modifier(
        term_id="HP:0012828", label="Severe", segments=[seg1, seg2], source="rule"
    )
    second_severity = Modifier(term_id="HP:0012825", label="Mild", source="llm")
    first_onset = Modifier(term_id="HP:0003577", label="Congenital onset", source="llm")
    second_onset = Modifier(term_id="HP:0003621", label="Juvenile onset", source="llm")
    non_modifier = Modifier(term_id="HP:0001250", label="Seizure", source="llm")
    cohort_freq = Modifier(term_id="HP:0040283", label="Occasional", source="llm")

    validated, warnings = ontology_guard.validate_finding_modifiers(
        [
            original_first,
            non_modifier,
            cohort_freq,
            duplicate_with_extra_seg,
            second_severity,
            first_onset,
            second_onset,
        ],
        profile=HPO_DEFAULT_PROFILE,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )

    # Non-modifier and frequency are dropped; duplicate is merged in first-seen order;
    # both severities and both onsets are kept.
    assert [m.term_id for m in validated] == [
        "HP:0012828",
        "HP:0012825",
        "HP:0003577",
        "HP:0003621",
    ]
    assert validated[0].segments == [seg1, seg2]
    # Caller's original model is not mutated in place
    assert original_first.segments == [seg1]

    warning_codes = [w.code for w in warnings]
    assert warning_codes == [
        "cohort_frequency",
        "exclusive_modifier_class",
        "exclusive_modifier_class",
    ]
    assert warnings[0].term_id == "HP:0040283"
    assert warnings[1].term_id is None
    assert "severity" in warnings[1].message.lower()
    assert warnings[2].term_id is None
    assert "onset" in warnings[2].message.lower()


def test_validate_finding_modifiers_single_per_exclusive_class_no_warning():
    validated, warnings = ontology_guard.validate_finding_modifiers(
        [
            Modifier(term_id="HP:0012828", label="Severe"),
            Modifier(term_id="HP:0003577", label="Congenital onset"),
            Modifier(term_id="HP:0012832", label="Bilateral"),
        ],
        profile=HPO_DEFAULT_PROFILE,
        ancestors_map=_SYNTHETIC_ANCESTORS,
    )
    assert [m.term_id for m in validated] == ["HP:0012828", "HP:0003577", "HP:0012832"]
    assert warnings == []


def test_validate_finding_modifiers_empty_input():
    assert ontology_guard.validate_finding_modifiers([], ancestors_map={}) == ([], [])


# ---------------------------------------------------------------------------
# Backward compatibility: is_non_phenotypic_abnormality
# ---------------------------------------------------------------------------


def test_is_non_phenotypic_abnormality_backward_compatible():
    assert (
        ontology_guard.is_non_phenotypic_abnormality("HP:0012828", _SYNTHETIC_ANCESTORS)
        is True
    )
    assert (
        ontology_guard.is_non_phenotypic_abnormality("HP:0001250", _SYNTHETIC_ANCESTORS)
        is False
    )
    assert (
        ontology_guard.is_non_phenotypic_abnormality("HP:0000118", _SYNTHETIC_ANCESTORS)
        is False
    )
    assert (
        ontology_guard.is_non_phenotypic_abnormality("HP:9999999", _SYNTHETIC_ANCESTORS)
        is False
    )
