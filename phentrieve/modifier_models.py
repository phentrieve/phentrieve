"""Ontology-neutral models for clinical modifiers and validation warnings.

Shapes align with ontospan (Modifier, Segment) while keeping Phentrieve's
finding models free of hard-coded ontology IDs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CURIE_PATTERN = r"^[A-Za-z][A-Za-z0-9_.]*:\S+$"

ModifierSource = Literal["llm", "rule"]


class ModifierSegment(BaseModel):
    """A document codepoint range [start, end) and the exact text it quotes."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> ModifierSegment:
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("segment text length must equal end - start")
        return self


class Modifier(BaseModel):
    """An ontology-neutral clinical modifier attached to a finding.

    ``segments`` is empty when the modifier words lie inside the finding's own
    span (compound word or interior overlap) so finding and modifier spans
    never overlap.
    """

    term_id: str = Field(pattern=CURIE_PATTERN)
    label: str = Field(min_length=1)
    segments: list[ModifierSegment] = Field(default_factory=list)
    source: ModifierSource | None = None


class OntologyWarning(BaseModel):
    """Non-fatal ontology/profile warning attached to a finding."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    term_id: str | None = None
