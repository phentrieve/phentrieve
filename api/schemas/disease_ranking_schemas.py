"""Schemas for the disease ranking endpoint (research use, raw uncalibrated scores)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AssertionAPI = Literal[
    "affirmed", "present", "negated", "absent", "excluded", "normal", "uncertain"
]


class ProfileTermAPI(BaseModel):
    hpo_id: str = Field(
        ...,
        pattern=r"^HP:\d{7}$",
        description="HPO term ID, e.g. HP:0001166.",
        examples=["HP:0001166"],
    )
    assertion: AssertionAPI = Field(
        default="affirmed",
        description=(
            "affirmed/present terms are scored; negated/absent/excluded/normal terms "
            "never change the score and are reported as conflicts; uncertain terms are "
            "scored only with include_uncertain."
        ),
    )


class DiseaseRankingRequestAPI(BaseModel):
    terms: list[ProfileTermAPI] = Field(..., min_length=1, max_length=200)
    level: Literal["disease", "gene"] = Field(
        default="disease", description="Rank diseases or genes."
    )
    top_k: int = Field(default=20, ge=1, le=100)
    include_uncertain: bool = Field(
        default=False, description="Score uncertain terms like present ones."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "terms": [
                        {"hpo_id": "HP:0001166", "assertion": "affirmed"},
                        {"hpo_id": "HP:0001083", "assertion": "affirmed"},
                        {"hpo_id": "HP:0001249", "assertion": "negated"},
                    ],
                    "level": "disease",
                    "top_k": 20,
                }
            ]
        }
    )


class TermMatchAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    profile_id: str
    profile_label: str
    disease_term_id: str
    disease_term_label: str
    relation: Literal["exact", "broader", "narrower", "related"] = Field(
        ..., description="The disease term relative to the profile term."
    )
    shared_id: str = Field(..., description="Most informative common ancestor.")
    shared_label: str
    match: float = Field(..., description="Relative match r(p, d) in [0, 1].")
    frequency: float = Field(..., description="Annotated frequency in [0, 1].")
    source: Literal["omim", "orpha", "both"]


class ConflictAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    profile_id: str
    profile_label: str
    disease_term_id: str
    disease_term_label: str
    frequency: float


class ExplanationAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    matches: list[TermMatchAPI]
    unexplained: list[TermMatchAPI] = Field(
        ..., description="Profile terms the disease matches only weakly (display hint)."
    )
    conflicts: list[ConflictAPI]
    annotation_sources: list[str]


class RankedDiseaseAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    disease_id: str
    label: str
    score: float
    genes: list[str]
    twins: list[str] = Field(
        ..., description="MONDO-equivalent ORPHA entries folded into this row."
    )
    explanation: ExplanationAPI


class RankedGeneAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    gene: str
    score: float
    best_disease_id: str
    best_disease_label: str
    explanation: ExplanationAPI


class NotScoredAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    hpo_id: str
    reason: Literal["unknown_term", "not_phenotype", "uncertain", "redundant"]


class DiseaseIndexInfoAPI(BaseModel):
    hpo_version: str | None = None
    mondo_version: str | None = None
    built_at: str | None = None
    n_diseases: int | None = None
    merge_orpha: bool | None = None


class DiseaseRankingResponseAPI(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    level: Literal["disease", "gene"]
    diseases: list[RankedDiseaseAPI]
    genes: list[RankedGeneAPI]
    scored: list[str]
    excluded: list[str]
    not_scored: list[NotScoredAPI]
    total_candidates: int
    scorer_version: str
    index_info: DiseaseIndexInfoAPI
    score_kind: Literal["raw_uncalibrated_phenotype_similarity"] = (
        "raw_uncalibrated_phenotype_similarity"
    )
    intended_use: str


class DiseaseRankingStatusAPI(BaseModel):
    enabled: bool = False
    index_ready: bool = False
    hpo_version: str | None = None
    mondo_version: str | None = None
    built_at: str | None = None
