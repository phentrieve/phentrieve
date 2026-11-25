"""Ontology guard: role-aware validation of finding and modifier terms.

Retrieval can drift into non-phenotype HPO branches -- Clinical modifier
(HP:0012823), Clinical course (HP:0031797), Mode of inheritance (HP:0000005),
Frequency (HP:0040279), Blood group, Past medical history. Under the default
HPO profile, standalone findings live under Phenotypic abnormality
(HP:0000118) and attached clinical modifiers live under Clinical modifier
(HP:0012823), while cohort-level Frequency terms are disallowed as individual
modifiers.

Fail open: a term whose ancestry is unknown (or when the HPO graph cannot be
loaded) is kept, so incomplete data never silently drops a real finding. A
missing graph is logged as an error and reported by ``guard_available`` so a
run without the guard is visible in its trace (#493).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phentrieve.config import DEFAULT_HPO_DB_FILENAME, PHENOTYPE_ROOT
from phentrieve.modifier_models import Modifier, ModifierSegment, OntologyWarning
from phentrieve.ontology_profiles import (
    OntologyProfile,
    get_default_ontology_profile,
)
from phentrieve.utils import get_default_data_dir, resolve_data_path

logger = logging.getLogger(__name__)

# Path-keyed cache of SUCCESSFUL, non-empty ancestry loads only. A failed or
# empty load is deliberately NOT memoized so the guard recovers on a later call
# once ``hpo_data.db`` becomes available (e.g. an API process that starts before
# the data bundle is present) instead of latching the fail-open ``{}`` for the
# whole process lifetime. Keying by resolved DB path also re-loads when the data
# root changes in-process rather than serving a stale map.
_ANCESTORS_CACHE: dict[str, dict[str, frozenset[str]]] = {}
# Paths whose failed load was already logged as an error. Retries keep running,
# but repeated failures log at DEBUG so a missing graph does not flood the log.
_REPORTED_FAILURES: set[str] = set()


def _log_failure(cache_key: str, message: str, *args: object) -> None:
    level = logging.DEBUG if cache_key in _REPORTED_FAILURES else logging.ERROR
    _REPORTED_FAILURES.add(cache_key)
    logger.log(level, message, *args)


def _load_ancestors_map(db_path) -> dict[str, frozenset[str]]:
    from phentrieve.data_processing.hpo_database import HPODatabase

    with HPODatabase(db_path) as db:
        ancestors, _ = db.load_graph_data()
    return {term: frozenset(anc) for term, anc in ancestors.items()}


def _ancestors_map() -> dict[str, frozenset[str]]:
    """Return ``{term_id: frozenset(ancestor_ids)}`` from the HPO graph.

    Returns an empty map (guard disabled, fail open) when the database is
    unavailable so the pipeline never hard-depends on graph data at runtime; the
    empty result is not cached, so a later call retries.
    """
    try:
        data_dir = resolve_data_path(None, "data_dir", get_default_data_dir)
        db_path = data_dir / DEFAULT_HPO_DB_FILENAME
        cache_key = str(db_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error(
            "Ontology guard could not resolve HPO db path (%s); disabled (fail open).",
            exc,
        )
        return {}

    cached = _ANCESTORS_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        if not db_path.exists():
            _log_failure(
                cache_key,
                "HPO database not found: %s; ontology guard disabled "
                "(fail open, will retry).",
                db_path,
            )
            return {}
        loaded = _load_ancestors_map(db_path)
    except Exception as exc:  # pragma: no cover - defensive
        _log_failure(
            cache_key,
            "Ontology guard could not load HPO graph (%s); disabled "
            "(fail open, will retry).",
            exc,
        )
        return {}

    # Only memoize a successful, non-empty load.
    if loaded:
        _ANCESTORS_CACHE[cache_key] = loaded
    return loaded


def guard_available() -> bool:
    """Whether the HPO graph is loaded, i.e. the guard is active."""
    return bool(_ancestors_map())


def load_guard_map() -> Mapping[str, frozenset[str]]:
    """The ancestry map the guard uses, for callers that check many terms.

    Pass it to ``is_non_phenotypic_abnormality`` or role-aware guard helpers so
    a loop over candidates resolves the data path and loads the graph once, not
    once per term.
    """
    return _ancestors_map()


def is_non_phenotypic_abnormality(
    term_id: str, ancestors_map: Mapping[str, frozenset[str]] | None = None
) -> bool:
    """True iff ``term_id`` is KNOWN to sit outside the Phenotypic abnormality
    (HP:0000118) subtree (a clinical modifier / course / inheritance / etc.).

    Unknown terms and the phenotype root itself return ``False`` (kept).
    ``ancestors_map`` (from ``load_guard_map``) avoids a lookup per call.
    """
    term_id = (term_id or "").strip()
    if not term_id or term_id == PHENOTYPE_ROOT:
        return False
    if ancestors_map is None:
        ancestors_map = _ancestors_map()
    ancestors = ancestors_map.get(term_id)
    if ancestors is None:
        return False  # unknown ancestry -> keep (fail open)
    return PHENOTYPE_ROOT not in ancestors


def is_under_any_root(
    term_id: str,
    roots: Sequence[str],
    ancestors_map: Mapping[str, frozenset[str]] | None = None,
) -> bool | None:
    """Return whether ``term_id`` sits under any root in ``roots``.

    Returns:
        ``True`` if ``term_id`` is one of ``roots`` or has an ancestor in
        ``roots``; ``False`` if its ancestry is known and disjoint from
        ``roots``; ``None`` if ``term_id`` is empty or absent from
        ``ancestors_map`` (fail-open / unknown ancestry).
    """
    cleaned_id = (term_id or "").strip()
    if not cleaned_id:
        return None
    root_set = {root.strip() for root in roots if root and root.strip()}
    if cleaned_id in root_set:
        return True
    if ancestors_map is None:
        ancestors_map = _ancestors_map()
    ancestors = ancestors_map.get(cleaned_id)
    if ancestors is None:
        return None
    if not root_set:
        return False
    return not root_set.isdisjoint(ancestors)


def term_ids_under_roots(
    roots: Sequence[str],
    ancestors_map: Mapping[str, frozenset[str]] | None = None,
) -> set[str]:
    """Return all term IDs in ``ancestors_map`` under any root in ``roots``.

    When ``roots`` is empty, returns all term IDs present in ``ancestors_map``.
    """
    if ancestors_map is None:
        ancestors_map = _ancestors_map()
    root_set = {root.strip() for root in roots if root and root.strip()}
    if not root_set:
        return set(ancestors_map.keys())
    return {
        term_id
        for term_id, ancestors in ancestors_map.items()
        if term_id in root_set or not root_set.isdisjoint(ancestors)
    }


@dataclass(frozen=True)
class FindingGuardDecision:
    """Decision returned by ``check_finding_term``."""

    accepted: bool
    is_modifier_term: bool
    warnings: tuple[OntologyWarning, ...] = ()


def check_finding_term(
    term_id: str,
    *,
    profile: OntologyProfile | None = None,
    ancestors_map: Mapping[str, frozenset[str]] | None = None,
) -> FindingGuardDecision:
    """Validate whether ``term_id`` is an allowed standalone finding term."""
    cleaned_id = (term_id or "").strip()
    if not cleaned_id:
        return FindingGuardDecision(accepted=True, is_modifier_term=False)

    active_profile = profile if profile is not None else get_default_ontology_profile()
    amap = _ancestors_map() if ancestors_map is None else ancestors_map

    if cleaned_id not in amap:
        return FindingGuardDecision(accepted=True, is_modifier_term=False)

    finding_roots = active_profile.roots.finding_roots
    modifier_roots = active_profile.roots.modifier_roots

    in_finding = (not finding_roots) or (
        is_under_any_root(cleaned_id, finding_roots, amap) is True
    )
    in_modifier = bool(modifier_roots) and (
        is_under_any_root(cleaned_id, modifier_roots, amap) is True
    )
    in_disallowed = any(
        is_under_any_root(cleaned_id, (root_id,), amap) is True
        for root_id in active_profile.disallowed_modifier_roots.values()
    )

    if finding_roots:
        if in_finding:
            return FindingGuardDecision(accepted=True, is_modifier_term=False)
        return FindingGuardDecision(accepted=False, is_modifier_term=in_modifier)

    if in_modifier or in_disallowed:
        warning = OntologyWarning(
            code="modifier_as_annotation",
            message=(
                f"Term {cleaned_id} belongs to a modifier or disallowed branch "
                "and was kept as a finding because finding_roots is unrestricted."
            ),
            term_id=cleaned_id,
        )
        return FindingGuardDecision(
            accepted=True,
            is_modifier_term=in_modifier,
            warnings=(warning,),
        )
    return FindingGuardDecision(accepted=True, is_modifier_term=False)


@dataclass(frozen=True)
class ModifierGuardDecision:
    """Decision returned by ``check_modifier_term``."""

    accepted: bool
    warnings: tuple[OntologyWarning, ...] = ()


def check_modifier_term(
    term_id: str,
    *,
    profile: OntologyProfile | None = None,
    ancestors_map: Mapping[str, frozenset[str]] | None = None,
) -> ModifierGuardDecision:
    """Validate whether ``term_id`` is an allowed clinical modifier term."""
    cleaned_id = (term_id or "").strip()
    if not cleaned_id:
        return ModifierGuardDecision(accepted=True)

    active_profile = profile if profile is not None else get_default_ontology_profile()
    amap = _ancestors_map() if ancestors_map is None else ancestors_map

    if cleaned_id in active_profile.abstract_modifier_terms:
        return ModifierGuardDecision(accepted=False)

    for code, root_id in active_profile.disallowed_modifier_roots.items():
        if is_under_any_root(cleaned_id, (root_id,), amap) is True:
            warning = OntologyWarning(
                code=code,
                message=(
                    f"Term {cleaned_id} is under disallowed modifier root "
                    f"{root_id} ({code}) and was not attached."
                ),
                term_id=cleaned_id,
            )
            return ModifierGuardDecision(accepted=False, warnings=(warning,))

    modifier_roots = active_profile.roots.modifier_roots
    if not modifier_roots:
        return ModifierGuardDecision(accepted=True)

    under_modifier = is_under_any_root(cleaned_id, modifier_roots, amap)
    if under_modifier is None:
        return ModifierGuardDecision(accepted=True)  # fail open on unknown ancestry
    return ModifierGuardDecision(accepted=under_modifier)


def _merge_modifier_segments(
    existing: Sequence[ModifierSegment], incoming: Sequence[ModifierSegment]
) -> list[ModifierSegment]:
    merged: list[ModifierSegment] = []
    seen: set[tuple[int, int, str]] = set()
    for segment in (*existing, *incoming):
        key = (segment.start, segment.end, segment.text)
        if key not in seen:
            seen.add(key)
            merged.append(segment)
    return merged


def validate_finding_modifiers(
    modifiers: Sequence[Modifier],
    *,
    profile: OntologyProfile | None = None,
    ancestors_map: Mapping[str, frozenset[str]] | None = None,
) -> tuple[list[Modifier], list[OntologyWarning]]:
    """Deduplicate and validate clinical modifiers attached to a finding.

    Drops modifiers outside ``profile.roots.modifier_roots`` or under
    ``profile.disallowed_modifier_roots`` (recording their warnings), merges
    non-duplicate segments for repeated ``term_id`` entries, and warns when
    more than one surviving modifier falls in an exclusive modifier class
    (such as Severity or Onset) without dropping them.
    """
    if not modifiers:
        return [], []

    active_profile = profile if profile is not None else get_default_ontology_profile()
    amap = _ancestors_map() if ancestors_map is None else ancestors_map

    deduped: list[Modifier] = []
    index_by_term: dict[str, int] = {}
    for modifier in modifiers:
        existing_idx = index_by_term.get(modifier.term_id)
        if existing_idx is None:
            index_by_term[modifier.term_id] = len(deduped)
            deduped.append(
                modifier.model_copy(
                    update={"segments": _merge_modifier_segments(modifier.segments, ())}
                )
            )
        else:
            current = deduped[existing_idx]
            merged_segments = _merge_modifier_segments(
                current.segments, modifier.segments
            )
            deduped[existing_idx] = current.model_copy(
                update={"segments": merged_segments}
            )

    validated: list[Modifier] = []
    warnings: list[OntologyWarning] = []
    for modifier in deduped:
        decision = check_modifier_term(
            modifier.term_id, profile=active_profile, ancestors_map=amap
        )
        warnings.extend(decision.warnings)
        if decision.accepted:
            validated.append(modifier)

    for class_name, root_id in active_profile.exclusive_modifier_classes.items():
        matches = [
            modifier
            for modifier in validated
            if is_under_any_root(modifier.term_id, (root_id,), amap) is True
        ]
        if len(matches) > 1:
            term_summary = ", ".join(
                f"{modifier.term_id} ({modifier.label})" for modifier in matches
            )
            warnings.append(
                OntologyWarning(
                    code="exclusive_modifier_class",
                    message=f"Multiple {class_name} modifiers: {term_summary}",
                    term_id=None,
                )
            )

    return validated, warnings
