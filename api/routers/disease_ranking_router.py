"""Disease and gene ranking from a profile of HPO terms (research use, flagged).

Mounted only when PHENTRIEVE_DISEASE_RANKING_ENABLED is set. Scores are raw,
uncalibrated phenotype similarity, not probabilities, and not a diagnosis.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from api.disease_index_state import get_disease_index
from api.research_use import (
    RESEARCH_USE_LIMITATION,
    require_research_use_acknowledgement,
    research_ack_openapi_parameter,
)
from api.schemas.disease_ranking_schemas import (
    DiseaseIndexInfoAPI,
    DiseaseRankingRequestAPI,
    DiseaseRankingResponseAPI,
)
from phentrieve.disease_matching import IndexNotAvailableError, ProfileTerm, rank
from phentrieve.utils import sanitize_log_value as _sanitize

logger = logging.getLogger(__name__)
router = APIRouter()  # Prefix is added in main.py

DISEASE_RANKING_NOTICE = (
    f"{RESEARCH_USE_LIMITATION} Disease ranking shows how well the profile matches "
    "curated disease annotations (raw, uncalibrated phenotype similarity). It is not "
    "a probability and not a diagnosis."
)


@router.post(
    "/rank",
    response_model=DiseaseRankingResponseAPI,
    operation_id="rank_diseases",
    summary="Rank diseases or genes for a profile of HPO terms",
    description=(
        "Research use only. Ranks rare diseases (ORPHA entries folded into their "
        "MONDO-equivalent OMIM row) or genes by raw, uncalibrated phenotype "
        "similarity to the given profile, with an explanation per row. No profile "
        "data is stored or logged."
    ),
    openapi_extra={"parameters": [research_ack_openapi_parameter()]},
)
async def rank_diseases(
    http_request: Request, request: DiseaseRankingRequestAPI
) -> DiseaseRankingResponseAPI:
    require_research_use_acknowledgement(http_request)
    try:
        index = await run_in_threadpool(get_disease_index)
    except IndexNotAvailableError as error:
        logger.warning("Disease ranking unavailable: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "disease_index_unavailable",
                "message": "The disease index is not available on this server.",
            },
        ) from error

    terms = [ProfileTerm(t.hpo_id, t.assertion) for t in request.terms]
    result = await run_in_threadpool(
        rank,
        index,
        terms,
        request.level,
        request.top_k,
        request.include_uncertain,
    )
    logger.info(
        "Disease ranking: level=%s terms=%d scored=%d",
        _sanitize(request.level),
        len(terms),
        len(result.scored),
    )
    return DiseaseRankingResponseAPI.model_validate(
        {
            "level": result.level,
            "diseases": result.diseases,
            "genes": result.genes,
            "scored": result.scored,
            "excluded": result.excluded,
            "not_scored": result.not_scored,
            "total_candidates": result.total_candidates,
            "scorer_version": result.scorer_version,
            "index_info": DiseaseIndexInfoAPI(**result.index_info),
            "intended_use": DISEASE_RANKING_NOTICE,
        }
    )
