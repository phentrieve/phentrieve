"""Ontology profiles and configurable branch roots.

Defines ontology-neutral root configuration and HPO defaults so retrieval,
indexing, ontology guarding, and phenopacket mapping do not hard-code HPO
branch IDs throughout the codebase.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from phentrieve.modifier_models import CURIE_PATTERN

__all__ = [
    "HPO_PHENOTYPIC_ABNORMALITY_ROOT",
    "HPO_CLINICAL_MODIFIER_ROOT",
    "HPO_SEVERITY_ROOT",
    "HPO_ONSET_ROOT",
    "HPO_FREQUENCY_ROOT",
    "OntologyRootsConfig",
    "OntologyProfile",
    "HPO_DEFAULT_PROFILE",
    "get_default_ontology_profile",
]

HPO_PHENOTYPIC_ABNORMALITY_ROOT = "HP:0000118"
HPO_CLINICAL_MODIFIER_ROOT = "HP:0012823"
HPO_SEVERITY_ROOT = "HP:0012824"
HPO_ONSET_ROOT = "HP:0003674"
HPO_FREQUENCY_ROOT = "HP:0040279"

_CURIE_RE = re.compile(CURIE_PATTERN)


def _normalize_roots(value: Any) -> tuple[str, ...]:
    """Validate and normalize a collection of CURIE strings into a unique tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset)
    ):
        raise ValueError(
            "Ontology roots must be a list, tuple, or set of CURIE strings."
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Ontology root must be a string, got {type(item).__name__}"
            )
        cleaned = item.strip()
        if not cleaned or not _CURIE_RE.match(cleaned):
            raise ValueError(f"Invalid ontology root CURIE: {item!r}")
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return tuple(normalized)


class OntologyRootsConfig(BaseModel, frozen=True):
    """Configured ontology branch roots for findings, modifiers, and indexing.

    An empty ``index_roots`` tuple means all active ontology terms are eligible
    for indexing, keeping existing prebuilt indexes compatible. A non-empty
    ``index_roots`` restricts embedded branches to those subtrees.
    """

    finding_roots: tuple[str, ...] = (HPO_PHENOTYPIC_ABNORMALITY_ROOT,)
    modifier_roots: tuple[str, ...] = (HPO_CLINICAL_MODIFIER_ROOT,)
    index_roots: tuple[str, ...] = ()

    @field_validator("finding_roots", "modifier_roots", "index_roots", mode="before")
    @classmethod
    def _validate_root_tuples(cls, value: Any) -> tuple[str, ...]:
        return _normalize_roots(value)

    def effective_required_roots(
        self, *, include_modifiers: bool = False
    ) -> tuple[str, ...]:
        """Return deduplicated ontology roots required for the active mode."""
        if not include_modifiers:
            return self.finding_roots

        combined: list[str] = []
        seen: set[str] = set()
        for root in (*self.finding_roots, *self.modifier_roots):
            if root not in seen:
                seen.add(root)
                combined.append(root)
        return tuple(combined)


class OntologyProfile(BaseModel, frozen=True):
    """Ontology-specific branch roles and validation rules."""

    name: str = Field(min_length=1)
    ontology_id: str = "hp"
    roots: OntologyRootsConfig = Field(default_factory=OntologyRootsConfig)
    exclusive_modifier_classes: dict[str, str] = Field(default_factory=dict)
    disallowed_modifier_roots: dict[str, str] = Field(default_factory=dict)
    phenopacket_slot_roots: dict[str, str] = Field(default_factory=dict)


HPO_DEFAULT_PROFILE = OntologyProfile(
    name="hpo",
    ontology_id="hp",
    roots=OntologyRootsConfig(
        finding_roots=(HPO_PHENOTYPIC_ABNORMALITY_ROOT,),
        modifier_roots=(HPO_CLINICAL_MODIFIER_ROOT,),
        index_roots=(),
    ),
    exclusive_modifier_classes={
        "severity": HPO_SEVERITY_ROOT,
        "onset": HPO_ONSET_ROOT,
    },
    disallowed_modifier_roots={
        "cohort_frequency": HPO_FREQUENCY_ROOT,
    },
    phenopacket_slot_roots={
        "severity": HPO_SEVERITY_ROOT,
        "onset": HPO_ONSET_ROOT,
    },
)


def get_default_ontology_profile() -> OntologyProfile:
    """Build the default ontology profile, applying optional ``ontology:`` YAML overrides.

    Explicit empty lists ``[]`` in ``phentrieve.yaml`` map to ``()``
    (unrestricted), while omitted keys fall back to ``HPO_DEFAULT_PROFILE``.
    """
    import phentrieve.config as phentrieve_config

    raw_config = phentrieve_config._load_yaml_config()
    ontology_cfg = raw_config.get("ontology") if isinstance(raw_config, dict) else None
    if not isinstance(ontology_cfg, dict):
        return HPO_DEFAULT_PROFILE

    defaults = HPO_DEFAULT_PROFILE.roots
    roots_kwargs: dict[str, Any] = {
        "finding_roots": defaults.finding_roots,
        "modifier_roots": defaults.modifier_roots,
        "index_roots": defaults.index_roots,
    }
    for key in ("finding_roots", "modifier_roots", "index_roots"):
        if key in ontology_cfg and ontology_cfg[key] is not None:
            roots_kwargs[key] = ontology_cfg[key]

    return HPO_DEFAULT_PROFILE.model_copy(
        update={"roots": OntologyRootsConfig(**roots_kwargs)}
    )
