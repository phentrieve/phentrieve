"""Rank rare diseases and genes by phenotype similarity to an HPO profile.

Research use only. Scores are raw, uncalibrated phenotype similarity, not probabilities.

The scorer is Phentrieve's own copy of the issue #451 proof-of-concept port of the VarLens
phenotype scoring v2 at its shipped setting. The disease index adds, to each OMIM disease, the
HPOA annotations of its MONDO-equivalent ORPHA entries (see
``.planning/specs/2026-11-23-disease-ranking-design.md``).
"""

from phentrieve.disease_matching.index import (
    DiseaseIndex,
    IndexNotAvailableError,
    load_index,
)
from phentrieve.disease_matching.service import (
    ProfileTerm,
    RankedDisease,
    RankedGene,
    RankingResult,
    TermMatch,
    rank,
)

__all__ = [
    "DiseaseIndex",
    "IndexNotAvailableError",
    "ProfileTerm",
    "RankedDisease",
    "RankedGene",
    "RankingResult",
    "TermMatch",
    "load_index",
    "rank",
]
