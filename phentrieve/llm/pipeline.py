from __future__ import annotations

import inspect
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from phentrieve.llm.config import (
    DEFAULT_GROUNDED_PHASE1_BASE_OUTPUT_TOKENS,
    DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_GROUNDED_PHASE1_MAX_TRUNCATION_RETRIES,
    DEFAULT_GROUNDED_PHASE1_MIN_OUTPUT_TOKENS,
    DEFAULT_GROUNDED_PHASE1_OUTPUT_TOKENS_PER_CHUNK,
    DEFAULT_GROUNDED_PHASE1_TRUNCATION_RETRY_OUTPUT_TOKENS,
    DEFAULT_LLM_LANGUAGE,
    DEFAULT_LLM_MAPPING_BATCH_SIZE,
    DEFAULT_LLM_MODIFIER_MAPPING_BATCH_SIZE,
    DEFAULT_LOCAL_MATCH_THRESHOLD,
    DEFAULT_MAX_UNIQUE_CANDIDATES,
    DEFAULT_MIN_UNIQUE_CANDIDATES,
    DEFAULT_PHASE1_FALLBACK_ENABLED,
    DEFAULT_PHASE1_GROUP_MAX_WORKERS,
    DEFAULT_PHASE1_LARGE_GROUP_MAX_PROMPT_TOKENS,
    DEFAULT_PHASE1_MAX_OUTPUT_TOKENS,
    DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS,
    DEFAULT_QUERY_RESULTS_PER_PHRASE,
    DEFAULT_SIMILARITY_THRESHOLD,
    NEGATED_ASSERTION,
    PRESENT_ASSERTION,
)
from phentrieve.llm.evidence_offsets import (
    EvidenceLocator,
    carve_modifier_span,
    rendered_regions,
)
from phentrieve.llm.ontology_guard import (
    check_finding_term,
    check_modifier_term,
    guard_available,
    is_non_phenotypic_abnormality,
    load_guard_map,
    term_ids_under_roots,
    validate_finding_modifiers,
)
from phentrieve.llm.pipeline_phase1 import (
    ACTIONABLE_CATEGORIES,
)
from phentrieve.llm.pipeline_phase1 import (
    clean_text as _clean_text,
)
from phentrieve.llm.pipeline_phase1 import (
    expand_combined_phase1_extractions as _expand_combined_phase1_extractions,
)
from phentrieve.llm.pipeline_phase1 import (
    normalize_category as _normalize_category,
)
from phentrieve.llm.pipeline_phase1 import (
    normalize_modifiers as _normalize_modifiers,
)
from phentrieve.llm.pipeline_phase1 import (
    normalized_evidence_text as _normalized_evidence_text,
)
from phentrieve.llm.pipeline_phase1 import (
    phase1_chunk_text as _phase1_chunk_text,
)
from phentrieve.llm.pipeline_phase1 import (
    phase1_extraction_dedup_key as _phase1_extraction_dedup_key,
)
from phentrieve.llm.pipeline_phase1 import (
    prefer_richer_text as _prefer_richer_text,
)
from phentrieve.llm.pipeline_phase1 import (
    render_group_chunk_index_text as _render_group_chunk_index_text,
)
from phentrieve.llm.pipeline_phase1 import (
    render_phase1_user_prompt as _render_phase1_user_prompt,
)
from phentrieve.llm.pipeline_phase1 import (
    sorted_chunk_ids as _sorted_chunk_ids,
)
from phentrieve.llm.pipeline_phase1 import (
    temporality_axis as _temporality_axis,
)
from phentrieve.llm.pipeline_phase1 import (
    tokenize as _tokenize,
)
from phentrieve.llm.pipeline_phase2 import (
    CATEGORY_TO_ASSERTION,
    DEFAULT_MAX_MODIFIER_CANDIDATES,
    build_grounded_context,
    deduplicate_terms,
    extract_first_result_list,
    finding_already_expresses_modifier,
    hybrid_select_candidates,
    local_match,
    normalize_mapping_phrase_key,
    phenotype_from_candidate,
    prepare_retrieval_queries,
    run_mapping_batch,
    run_modifier_mapping_batch,
    select_candidate_ids,
)
from phentrieve.llm.pipeline_phase2 import (
    axes_disagree as _axes_disagree,
)
from phentrieve.llm.pipeline_phase2 import (
    candidate_match_text as _candidate_match_text,
)
from phentrieve.llm.pipeline_phase2 import (
    downstream_dedupe_key as _downstream_dedupe_key,
)
from phentrieve.llm.pipeline_phase2 import (
    explicit_null_item_ids as _explicit_null_item_ids,
)
from phentrieve.llm.pipeline_phase2 import (
    mapping_batch_item_id as _mapping_batch_item_id,
)
from phentrieve.llm.pipeline_retry import (
    LLMPipelinePhaseError,
)
from phentrieve.llm.pipeline_retry import (
    classify_phase1_failure as _classify_phase1_failure,
)
from phentrieve.llm.pipeline_retry import (
    group_source_chunk_ids as _group_source_chunk_ids,
)
from phentrieve.llm.pipeline_retry import (
    next_phase1_mode as _next_phase1_mode,
)
from phentrieve.llm.pipeline_trace import (
    annotate_mapping_trace_with_provenance,
    attach_phase1_failure_context,
    build_phase1_trace,
    build_phase2a_trace,
    build_phase2b_local_trace,
    phase1_extracted_trace_items,
    phase1_failure_class_from_groups,
)
from phentrieve.llm.pipeline_trace import (
    build_phase1_attempt_trace as _build_phase1_attempt_trace,
)
from phentrieve.llm.pipeline_trace import (
    sum_usage_dicts as _sum_usage_dicts,
)
from phentrieve.llm.preprocessing import build_extraction_groups
from phentrieve.llm.prompts.loader import (
    PromptTemplate,
    get_batch_mapping_prompt,
    get_batch_modifier_mapping_prompt,
    get_mapping_prompt,
    get_modifier_extraction_prompt,
    get_modifier_mapping_prompt,
    get_prompt,
)
from phentrieve.llm.provider import ToolExecutor
from phentrieve.llm.providers.base import is_output_truncation_finish_reason
from phentrieve.llm.types import (
    AnnotationMode,
    GroundedChunk,
    LLMBatchMappingSelections,
    LLMExtractedPhenotypes,
    LLMExtractedPhenotypesWithModifiers,
    LLMExtractionResult,
    LLMGroundedExtractedPhenotypes,
    LLMGroundedExtractedPhenotypesWithModifiers,
    LLMMappingSelection,
    LLMMeta,
    LLMPhenotype,
    LLMPipelineConfig,
    Phase1FailureClass,
    Phase1Mode,
)
from phentrieve.llm.utils import token_sort_similarity
from phentrieve.modifier_models import Modifier, ModifierSegment, OntologyWarning
from phentrieve.ontology_profiles import (
    HPO_PHENOTYPIC_ABNORMALITY_ROOT,
    OntologyProfile,
    get_default_ontology_profile,
)

logger = logging.getLogger(__name__)


def _grounded_phase1_output_token_budget(chunk_count: int) -> int:
    """Output-token budget for one grounded phase-1 request with ``chunk_count`` chunks."""
    proportional = (
        DEFAULT_GROUNDED_PHASE1_BASE_OUTPUT_TOKENS
        + DEFAULT_GROUNDED_PHASE1_OUTPUT_TOKENS_PER_CHUNK * max(int(chunk_count), 0)
    )
    return min(
        DEFAULT_GROUNDED_PHASE1_MAX_OUTPUT_TOKENS,
        max(DEFAULT_GROUNDED_PHASE1_MIN_OUTPUT_TOKENS, proportional),
    )


@dataclass
class _ItemResolutionOutcome:
    """Result of running one item set through the shared phase-2 resolution
    path (retrieve -> route -> map).

    The helper is PURE: it computes every accounting delta and returns it here
    instead of mutating ``run()`` state, so ``run()`` can merge the proband
    outcome into the canonical meta keys (byte-identical to the pre-B2 inline
    block) and the family outcome into a distinct ``family_*`` / ``trace["family"]``
    namespace without either overwriting the other (B2)."""

    resolved: list[LLMPhenotype]
    phase2a_seconds: float
    phase2b_local_seconds: float
    phase2b_llm_seconds: float
    candidate_sets: int
    routing_counts: dict[str, int]
    unresolved_phrases: int
    local_matches: int
    local_fallbacks: int
    llm_mapped_phrases: int
    prompt_tokens: int
    completion_tokens: int
    token_usage: dict[str, int]
    request_count: int
    phase2b_llm_requests: int
    mapping_prompt_version: str | None
    phase2a_trace: dict[str, Any]
    phase2b_local_trace: dict[str, Any]
    phase2b_llm_trace: dict[str, Any] | None
    mapping_ran: bool
    raw_modifiers_by_phenotype: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class _ModifierResolutionOutcome:
    phase2a_seconds: float = 0.0
    phase2b_local_seconds: float = 0.0
    phase2b_llm_seconds: float = 0.0
    modifier_phrases: int = 0
    local_matches: int = 0
    llm_mapped: int = 0
    suppressed_double_encoding: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    request_count: int = 0
    phase2b_llm_requests: int = 0
    mapping_ran: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


def _supports_kwarg(fn: Any, kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if kwarg in sig.parameters:
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


def _append_unique_warnings(
    target: LLMPhenotype, warnings: Sequence[OntologyWarning]
) -> None:
    if not warnings:
        return
    seen = {
        (warning.code, warning.term_id, warning.message) for warning in target.warnings
    }
    for warning in warnings:
        key = (warning.code, warning.term_id, warning.message)
        if key not in seen:
            seen.add(key)
            target.warnings.append(warning.model_copy(deep=True))


def _is_accepted_finding_candidate(
    hpo_id: str,
    *,
    profile: OntologyProfile,
    guard_map: Mapping[str, frozenset[str]],
) -> bool:
    """Check whether ``hpo_id`` is eligible as a finding candidate.

    Delegates to ``is_non_phenotypic_abnormality`` for the default HPO
    Phenotypic abnormality root so callers/tests monkeypatching that symbol
    retain identical behavior; uses ``check_finding_term`` for custom or
    unrestricted ``finding_roots``.
    """
    if profile.roots.finding_roots == (HPO_PHENOTYPIC_ABNORMALITY_ROOT,):
        return not is_non_phenotypic_abnormality(hpo_id, guard_map)
    return check_finding_term(hpo_id, profile=profile, ancestors_map=guard_map).accepted


def _build_evidence_locator(
    text: str, grounded_chunks: list[dict[str, Any]]
) -> EvidenceLocator:
    chunk_spans: dict[int, tuple[int, int]] = {}
    search_spans: dict[int, tuple[int, int]] = {}
    for region in rendered_regions(
        text,
        [(chunk.get("start_char"), chunk.get("end_char")) for chunk in grounded_chunks],
    ):
        chunk = grounded_chunks[region.index]
        chunk_id = int(chunk["chunk_id"])
        chunk_spans[chunk_id] = (region.chunk_start, region.chunk_end)
        search_spans[chunk_id] = (
            (region.start, region.end)
            if chunk.get("prompt_text")
            else (region.chunk_start, region.chunk_end)
        )
    return EvidenceLocator(text, chunk_spans, search_spans=search_spans)


def _record_secondary_outcome(
    *,
    prefix: str,
    item_count: int,
    outcome: _ItemResolutionOutcome,
    phase_timings: dict[str, float],
    phase_counts: dict[str, int],
    phase_request_counts: dict[str, int],
) -> dict[str, Any]:
    """Record a non-proband resolution outcome under ``{prefix}_*`` meta keys.

    Shared by the family and other-person paths so both report timings, counts,
    phase-2b requests and traces the same way. Returns the path's trace block.
    """
    phase_timings[f"{prefix}_phase2a_seconds"] = outcome.phase2a_seconds
    phase_timings[f"{prefix}_phase2b_local_seconds"] = outcome.phase2b_local_seconds
    phase_counts[f"{prefix}_phrases"] = item_count
    phase_counts[f"{prefix}_candidate_sets"] = outcome.candidate_sets
    phase_counts[f"{prefix}_unresolved_phrases"] = outcome.unresolved_phrases
    phase_counts[f"{prefix}_local_matches"] = outcome.local_matches
    phase_counts[f"{prefix}_resolved_phrases"] = len(outcome.resolved)
    for key, value in outcome.routing_counts.items():
        phase_counts[f"{prefix}_{key}"] = value
    trace: dict[str, Any] = {
        "phase2a": outcome.phase2a_trace,
        "phase2b_local": outcome.phase2b_local_trace,
    }
    if outcome.mapping_ran:
        phase_timings[f"{prefix}_phase2b_llm_seconds"] = outcome.phase2b_llm_seconds
        phase_request_counts[f"{prefix}_phase2b_llm_requests"] = (
            outcome.phase2b_llm_requests
        )
        phase_counts[f"{prefix}_local_fallbacks"] = outcome.local_fallbacks
        phase_counts[f"{prefix}_llm_mapped_phrases"] = outcome.llm_mapped_phrases
        trace["phase2b_llm"] = outcome.phase2b_llm_trace
    return trace


class TwoPhaseLLMPipeline:
    def __init__(
        self,
        provider,
        *,
        tool_executor: ToolExecutor | None = None,
        n_results_per_phrase: int = DEFAULT_QUERY_RESULTS_PER_PHRASE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_unique_candidates: int = DEFAULT_MIN_UNIQUE_CANDIDATES,
        max_unique_candidates: int = DEFAULT_MAX_UNIQUE_CANDIDATES,
        local_match_threshold: float = DEFAULT_LOCAL_MATCH_THRESHOLD,
        mapping_batch_size: int = DEFAULT_LLM_MAPPING_BATCH_SIZE,
        modifier_mapping_batch_size: int = DEFAULT_LLM_MODIFIER_MAPPING_BATCH_SIZE,
    ) -> None:
        self.provider = provider
        self.tool_executor = tool_executor or ToolExecutor()
        self.n_results_per_phrase = n_results_per_phrase
        self.similarity_threshold = similarity_threshold
        self.min_unique_candidates = min_unique_candidates
        self.max_unique_candidates = max_unique_candidates
        self.local_match_threshold = local_match_threshold
        self.mapping_batch_size = max(int(mapping_batch_size), 1)
        self.modifier_mapping_batch_size = max(int(modifier_mapping_batch_size), 1)

    def run(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]] | None = None,
        extraction_groups: list[dict[str, Any]] | None = None,
        config: LLMPipelineConfig,
    ) -> LLMExtractionResult:
        grounded_chunks = list(grounded_chunks or [])
        extraction_groups = list(extraction_groups or [])
        mode = AnnotationMode(config.mode)
        if mode != AnnotationMode.TWO_PHASE:
            raise ValueError(
                f"Unsupported LLM mode: {config.mode!r}. Expected 'two_phase'."
            )

        language = config.language or DEFAULT_LLM_LANGUAGE
        extract_modifiers = bool(config.extract_modifiers)
        profile = config.ontology_profile or get_default_ontology_profile()
        guard_map = load_guard_map()
        extraction_prompt = (
            get_modifier_extraction_prompt(language)
            if extract_modifiers
            else get_prompt(mode, language)
        )
        logger.debug(
            "Phase 1: extracting phenotype phrases (mode=%s, language=%s, text_chars=%d, extract_modifiers=%s)",
            mode.value,
            language,
            len(text),
            extract_modifiers,
        )
        phase1_initial_mode = self._resolve_initial_phase1_mode(
            extraction_groups=extraction_groups
        )
        phase1_fallback_count = 0
        phase1_failure_class: Phase1FailureClass = None
        phase1_attempts_trace: list[dict[str, Any]] = []
        phase1_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        phase1_request_count = 0
        phase1_elapsed = 0.0
        phase1_carried_groups_trace: list[dict[str, Any]] = []
        phase1_carried_extracted: list[dict[str, Any]] = []
        extracted: list[dict[str, Any]] = []
        current_phase1_mode = phase1_initial_mode
        current_grounded_chunks = list(grounded_chunks)
        current_extraction_groups = list(extraction_groups)

        while True:
            try:
                (
                    extracted,
                    attempt_phase1_usage,
                    attempt_phase1_request_count,
                    attempt_phase1_elapsed,
                    attempt_phase1_groups_trace,
                ) = self._run_phase1_mode(
                    mode=current_phase1_mode,
                    text=text,
                    grounded_chunks=current_grounded_chunks,
                    extraction_groups=current_extraction_groups,
                    extraction_prompt=extraction_prompt,
                    capture_phase1_debug=config.capture_phase1_debug,
                    extract_modifiers=extract_modifiers,
                )
                phase1_usage = _sum_usage_dicts(phase1_usage, attempt_phase1_usage)
                phase1_request_count += attempt_phase1_request_count
                phase1_elapsed = round(phase1_elapsed + attempt_phase1_elapsed, 6)
                phase1_final_mode = current_phase1_mode
                (
                    retry_grounded_chunks,
                    retry_failure_class,
                    retained_groups_trace,
                    terminal_failure_class,
                ) = self._phase1_partial_retry_chunks(
                    mode=current_phase1_mode,
                    grounded_chunks=current_grounded_chunks,
                    groups_trace=attempt_phase1_groups_trace,
                )
                if terminal_failure_class is not None:
                    terminal_exc = LLMPipelinePhaseError(
                        "phase1",
                        "Structured extraction failed for one or more extraction groups",
                        usage=attempt_phase1_usage,
                        request_count=attempt_phase1_request_count,
                        elapsed_seconds=attempt_phase1_elapsed,
                        groups_trace=attempt_phase1_groups_trace,
                    )
                    terminal_exc.failure_class = terminal_failure_class
                    raise terminal_exc
                if retry_grounded_chunks:
                    if phase1_failure_class is None:
                        phase1_failure_class = retry_failure_class
                    phase1_carried_extracted.extend(extracted)
                    phase1_carried_groups_trace.extend(retained_groups_trace)
                    phase1_attempts_trace.append(
                        _build_phase1_attempt_trace(
                            mode=current_phase1_mode,
                            status="partial",
                            groups_trace=attempt_phase1_groups_trace,
                            request_count=attempt_phase1_request_count,
                            elapsed_seconds=attempt_phase1_elapsed,
                            failure_class=retry_failure_class,
                        )
                    )
                    phase1_fallback_count += 1
                    current_phase1_mode = "grouped_small"
                    current_grounded_chunks = retry_grounded_chunks
                    current_extraction_groups = []
                    continue
                extracted = self._deduplicate_phase1_extractions(
                    phase1_carried_extracted + extracted
                )
                phase1_groups_trace = (
                    list(phase1_carried_groups_trace) + attempt_phase1_groups_trace
                )
                phase1_attempts_trace.append(
                    _build_phase1_attempt_trace(
                        mode=current_phase1_mode,
                        status="completed",
                        groups_trace=attempt_phase1_groups_trace,
                        request_count=attempt_phase1_request_count,
                        elapsed_seconds=attempt_phase1_elapsed,
                    )
                )
                break
            except LLMPipelinePhaseError as exc:
                failure_class = _classify_phase1_failure(exc)
                phase1_failure_class = failure_class
                phase1_usage = _sum_usage_dicts(phase1_usage, exc.usage)
                phase1_request_count += exc.request_count
                phase1_elapsed = round(
                    phase1_elapsed + float(exc.elapsed_seconds or 0.0), 6
                )
                phase1_groups_trace = list(phase1_carried_groups_trace) + list(
                    exc.groups_trace
                )
                phase1_attempts_trace.append(
                    _build_phase1_attempt_trace(
                        mode=current_phase1_mode,
                        status="failed",
                        groups_trace=exc.groups_trace,
                        request_count=exc.request_count,
                        elapsed_seconds=float(exc.elapsed_seconds or 0.0),
                        failure_class=failure_class,
                    )
                )
                next_mode = _next_phase1_mode(current_phase1_mode)
                if self._should_fallback_phase1(
                    failure_class=failure_class,
                    grounded_chunks=current_grounded_chunks,
                    next_mode=next_mode,
                ):
                    assert next_mode is not None
                    if next_mode == "grouped_large":
                        next_mode, current_extraction_groups = (
                            self._plan_grouped_large_fallback(
                                text=text,
                                grounded_chunks=current_grounded_chunks,
                                extraction_prompt=extraction_prompt,
                            )
                        )
                    phase1_fallback_count += 1
                    current_phase1_mode = next_mode
                    continue
                self._attach_phase1_failure_context(
                    exc=exc,
                    initial_mode=phase1_initial_mode,
                    final_mode=current_phase1_mode,
                    fallback_count=phase1_fallback_count,
                    failure_class=failure_class,
                    final_groups_trace=phase1_groups_trace,
                    attempts_trace=phase1_attempts_trace,
                )
                raise
        if phase1_failure_class is None:
            phase1_failure_class = self._phase1_failure_class_from_groups(
                phase1_groups_trace
            )
        logger.debug(
            "Phase 1 complete: extracted=%d prompt_tokens=%s completion_tokens=%s",
            len(extracted),
            phase1_usage.get("prompt_tokens"),
            phase1_usage.get("completion_tokens"),
        )
        logger.info(
            "Phase 1 anchor resolution: extracted=%d anchored=%d",
            len(extracted),
            sum(1 for item in extracted if item.get("chunk_ids")),
        )
        extracted = self._deduplicate_phase1_extractions(
            _expand_combined_phase1_extractions(extracted)
        )
        # After deduplication, so the set of findings is unchanged; only their
        # spans become document offsets.
        self._attach_document_offsets(
            extracted, text=text, grounded_chunks=grounded_chunks
        )
        # B2 partition: split family-experiencer phrases out of the proband set
        # BEFORE the actionable filter. A single-pass split by predicate avoids
        # the value-equality pitfall of ``item not in family_items`` when two
        # items share identical content. A finding is family when the model
        # tagged its experiencer as family_history OR it carries the legacy
        # Family_History category (the latter clause catches a category-only
        # finding whose experiencer defaulted to proband, and the former catches
        # a family finding wearing a legacy Abnormal category that would
        # otherwise leak into the proband path).
        family_items: list[dict[str, Any]] = []
        other_person_items: list[dict[str, Any]] = []
        proband_items: list[dict[str, Any]] = []
        for item in extracted:
            # Another person who is not a relative: resolved like family
            # findings, returned in ``other_person_findings``.
            if str(item.get("experiencer") or "").strip().lower() == "other":
                other_person_items.append(item)
                continue
            is_family = (
                str(item.get("experiencer") or "").strip().lower() == "family_history"
                or _normalize_category(str(item.get("category", "")))
                == "family_history"
            )
            if is_family:
                # A family finding may have been expressed only via the legacy
                # Family_History category while its experiencer axis stayed at
                # the B1 schema default ("proband", threaded unconditionally).
                # Since the partition has classified it as family, normalize the
                # experiencer on a copy so the resolved phenotype is internally
                # consistent (experiencer=family_history) rather than a
                # self-contradictory family term tagged proband.
                family_item = dict(item)
                family_item["experiencer"] = "family_history"
                family_items.append(family_item)
            else:
                proband_items.append(item)
        actionable = [
            item
            for item in proband_items
            if _normalize_category(str(item["category"])) in ACTIONABLE_CATEGORIES
        ]

        resolved_terms: list[LLMPhenotype] = []
        prompt_tokens_total = int(phase1_usage.get("prompt_tokens", 0) or 0)
        completion_tokens_total = int(phase1_usage.get("completion_tokens", 0) or 0)
        token_usage_total = _sum_usage_dicts(phase1_usage)
        request_count_total = phase1_request_count
        prompt_version = extraction_prompt.version
        phase_timings: dict[str, float] = {
            "phase1_seconds": phase1_elapsed,
            "phase2a_seconds": 0.0,
            "phase2b_local_seconds": 0.0,
            "phase2b_llm_seconds": 0.0,
        }
        phase_counts: dict[str, int] = {
            "extracted_phrases": len(extracted),
            "actionable_phrases": len(actionable),
            "candidate_sets": 0,
            "unresolved_phrases": 0,
            "local_matches": 0,
            "llm_mapped_phrases": 0,
            "local_fallbacks": 0,
            "phase1_fallbacks": phase1_fallback_count,
        }
        # Category/assertion contradictions, resolved by one rule (#495).
        axis_disagreements = sum(1 for item in actionable if _axes_disagree(item))
        if axis_disagreements:
            phase_counts["phase1_axis_disagreements"] = axis_disagreements
        unanchored = sum(1 for item in extracted if not item.get("chunk_ids"))
        if grounded_chunks and unanchored:
            phase_counts["phase1_unanchored_phrases"] = unanchored
        phase_request_counts: dict[str, int] = {
            "phase1_requests": phase1_request_count,
            "phase2b_llm_requests": 0,
        }
        trace: dict[str, Any] = {
            "phase1": build_phase1_trace(
                extracted=extracted,
                groups_trace=phase1_groups_trace,
                attempts_trace=phase1_attempts_trace,
                initial_mode=phase1_initial_mode,
                final_mode=phase1_final_mode,
                fallback_count=phase1_fallback_count,
                failure_class=phase1_failure_class,
            )
        }
        all_phase1_groups = [
            group
            for attempt in phase1_attempts_trace
            for group in attempt.get("groups", [])
        ]
        if all_phase1_groups:
            phase_counts["phase1_completed_groups"] = sum(
                1 for group in all_phase1_groups if group.get("status") == "completed"
            )
            phase_counts["phase1_failed_groups"] = sum(
                1 for group in all_phase1_groups if group.get("status") == "failed"
            )
            phase_counts["phase1_partial_failures"] = int(
                phase_counts["phase1_failed_groups"] > 0
                and phase_counts["phase1_completed_groups"] > 0
            )
            phase_counts["phase1_truncation_retries"] = sum(
                1 for group in all_phase1_groups if group.get("truncation_retry")
            )
            # Groups whose failure no later fallback recovered: their findings
            # are missing from this (still successful) result (#395).
            phase_counts["phase1_unrecovered_groups"] = sum(
                1 for group in phase1_groups_trace if group.get("status") == "failed"
            )
        resolved_family: list[LLMPhenotype] = []
        resolved_other_person: list[LLMPhenotype] = []
        raw_modifiers_by_phenotype: dict[int, list[str]] = {}
        if actionable:
            # Proband path: merge the outcome into the CANONICAL meta keys so the
            # result stays byte-identical to the pre-B2 inline block. The gated
            # ``mapping_ran`` branch mirrors the old ``if unresolved:`` guard
            # exactly (phase2b_llm_* keys otherwise keep their initialized zeros).
            proband_outcome = self._resolve_items(
                actionable,
                grounded_chunks=grounded_chunks,
                language=language,
                profile=profile,
                guard_map=guard_map,
            )
            resolved_terms.extend(proband_outcome.resolved)
            raw_modifiers_by_phenotype.update(
                proband_outcome.raw_modifiers_by_phenotype
            )
            phase_timings["phase2a_seconds"] = proband_outcome.phase2a_seconds
            phase_timings["phase2b_local_seconds"] = (
                proband_outcome.phase2b_local_seconds
            )
            phase_counts["candidate_sets"] = proband_outcome.candidate_sets
            phase_counts["unresolved_phrases"] = proband_outcome.unresolved_phrases
            phase_counts["local_matches"] = proband_outcome.local_matches
            phase_counts.update(proband_outcome.routing_counts)
            trace["phase2a"] = proband_outcome.phase2a_trace
            trace["phase2b_local"] = proband_outcome.phase2b_local_trace
            if proband_outcome.mapping_ran:
                phase_timings["phase2b_llm_seconds"] = (
                    proband_outcome.phase2b_llm_seconds
                )
                prompt_version = (
                    f"{prompt_version}+{proband_outcome.mapping_prompt_version}"
                )
                prompt_tokens_total += proband_outcome.prompt_tokens
                completion_tokens_total += proband_outcome.completion_tokens
                token_usage_total = _sum_usage_dicts(
                    token_usage_total,
                    proband_outcome.token_usage,
                )
                request_count_total += proband_outcome.request_count
                phase_request_counts["phase2b_llm_requests"] = (
                    proband_outcome.phase2b_llm_requests
                )
                phase_counts["local_fallbacks"] = proband_outcome.local_fallbacks
                phase_counts["llm_mapped_phrases"] = proband_outcome.llm_mapped_phrases
                trace["phase2b_llm"] = proband_outcome.phase2b_llm_trace

        # Family path: resolve ALL family phrases through the SAME retrieve ->
        # route -> map path (they are phenotype phrases regardless of the legacy
        # category; they must NOT be re-dropped via ACTIONABLE_CATEGORIES).
        # Family results land in the separate ``resolved_family`` list; they are
        # NEVER merged into ``resolved_terms``. Findings of another person who
        # is not a relative take the same path into ``resolved_other_person``.
        # Their cost ALWAYS folds into the shared token/request accumulators so
        # an extra LLM pass is not underreported; per-path breakdowns use a
        # ``family_*`` / ``other_person_*`` namespace (and ``trace[prefix]``) so
        # proband metrics are never overwritten.
        for prefix, items, resolved_sink in (
            ("family", family_items, resolved_family),
            ("other_person", other_person_items, resolved_other_person),
        ):
            if not items:
                continue
            outcome = self._resolve_items(
                items,
                grounded_chunks=grounded_chunks,
                language=language,
                profile=profile,
                guard_map=guard_map,
            )
            resolved_sink.extend(outcome.resolved)
            raw_modifiers_by_phenotype.update(outcome.raw_modifiers_by_phenotype)
            prompt_tokens_total += outcome.prompt_tokens
            completion_tokens_total += outcome.completion_tokens
            token_usage_total = _sum_usage_dicts(token_usage_total, outcome.token_usage)
            request_count_total += outcome.request_count
            trace[prefix] = _record_secondary_outcome(
                prefix=prefix,
                item_count=len(items),
                outcome=outcome,
                phase_timings=phase_timings,
                phase_counts=phase_counts,
                phase_request_counts=phase_request_counts,
            )

        if actionable or family_items or other_person_items:
            self._check_index_roots_compatibility(
                language=language,
                profile=profile,
                guard_map=guard_map,
                include_modifiers=extract_modifiers,
            )

        # B3: map each resolved proband finding's negated qualifier Y (the
        # absent portion of an "X without Y" phrase) to a GENERATED excluded
        # finding. Y is resolved through the SAME route -> map path as any
        # phrase; when the chosen term scores at or above the confidence floor
        # (``self.similarity_threshold``) an excluded ``LLMPhenotype`` is emitted
        # for Y. Below the floor (or when nothing is chosen) no term is
        # generated and the source finding's ``negated_qualifier`` metadata
        # string is retained. This runs AFTER proband resolution and BEFORE the
        # final dedup so the generated terms flow through the same dedup key.
        # The generated exclusions are secondary: when resolving them fails the
        # resolved findings are still returned, each keeping its
        # ``negated_qualifier`` string, and the failure is counted.
        try:
            qualifier_exclusions, qualifier_outcome = self._build_qualifier_exclusions(
                resolved_terms=resolved_terms,
                grounded_chunks=grounded_chunks,
                language=language,
                profile=profile,
                guard_map=guard_map,
            )
        except LLMPipelinePhaseError as exc:
            logger.warning(
                "Negated-qualifier resolution failed in %s (%s); "
                "returning findings without generated exclusions",
                exc.phase,
                exc,
            )
            qualifier_exclusions, qualifier_outcome = [], None
            token_usage_total = _sum_usage_dicts(token_usage_total, exc.usage)
            prompt_tokens_total += int(exc.usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(exc.usage.get("completion_tokens", 0) or 0)
            request_count_total += exc.request_count
            phase_counts["negated_qualifier_failures"] = 1
            trace["negated_qualifier"] = {"failed_phase": exc.phase}
        if qualifier_outcome is not None:
            prompt_tokens_total += qualifier_outcome.prompt_tokens
            completion_tokens_total += qualifier_outcome.completion_tokens
            token_usage_total = _sum_usage_dicts(
                token_usage_total, qualifier_outcome.token_usage
            )
            request_count_total += qualifier_outcome.request_count
            trace["negated_qualifier"] = _record_secondary_outcome(
                prefix="negated_qualifier",
                item_count=len(
                    {(term.negated_qualifier or "").strip() for term in resolved_terms}
                    - {""}
                ),
                outcome=qualifier_outcome,
                phase_timings=phase_timings,
                phase_counts=phase_counts,
                phase_request_counts=phase_request_counts,
            )

        proband_terms = resolved_terms + qualifier_exclusions
        all_terms = (
            len(proband_terms) + len(resolved_family) + len(resolved_other_person)
        )
        guard_stats: dict[str, int] = {}
        guarded_terms, guarded_family, guarded_other_person, dropped_terms = (
            self._apply_ontology_guard(
                proband_terms,
                resolved_family,
                resolved_other_person,
                profile=profile,
                guard_map=guard_map,
                extract_modifiers=extract_modifiers,
                text=text,
                stats=guard_stats,
            )
        )
        converted_modifier_findings = guard_stats.get("converted_modifier_findings", 0)
        # Counted and traced, so lost findings and an inactive guard show (#493).
        if dropped_terms:
            phase_counts["ontology_guard_dropped_terms"] = dropped_terms
        if converted_modifier_findings:
            phase_counts["ontology_guard_converted_modifier_findings"] = (
                converted_modifier_findings
            )
        if all_terms:
            guard_trace: dict[str, Any] = {
                "available": guard_available(),
                "dropped_terms": dropped_terms,
            }
            if converted_modifier_findings:
                guard_trace["converted_modifier_findings"] = converted_modifier_findings
            trace["ontology_guard"] = guard_trace

        if extract_modifiers:
            surviving_findings = [
                *guarded_terms,
                *guarded_family,
                *guarded_other_person,
            ]
            if any(
                raw_modifiers_by_phenotype.get(id(finding))
                for finding in surviving_findings
            ):
                modifier_outcome = self._resolve_modifiers(
                    findings=surviving_findings,
                    raw_modifiers_by_phenotype=raw_modifiers_by_phenotype,
                    text=text,
                    grounded_chunks=grounded_chunks,
                    language=language,
                    profile=profile,
                    guard_map=guard_map,
                )
                phase_timings["modifiers_phase2a_seconds"] = (
                    modifier_outcome.phase2a_seconds
                )
                phase_timings["modifiers_phase2b_local_seconds"] = (
                    modifier_outcome.phase2b_local_seconds
                )
                phase_counts["modifier_phrases"] = modifier_outcome.modifier_phrases
                phase_counts["modifier_local_matches"] = modifier_outcome.local_matches
                phase_counts["modifier_llm_mapped"] = modifier_outcome.llm_mapped
                if modifier_outcome.suppressed_double_encoding:
                    phase_counts["modifier_suppressed_double_encoding"] = (
                        modifier_outcome.suppressed_double_encoding
                    )
                if modifier_outcome.mapping_ran:
                    phase_timings["modifiers_phase2b_llm_seconds"] = (
                        modifier_outcome.phase2b_llm_seconds
                    )
                    phase_request_counts["modifiers_phase2b_llm_requests"] = (
                        modifier_outcome.phase2b_llm_requests
                    )
                prompt_tokens_total += modifier_outcome.prompt_tokens
                completion_tokens_total += modifier_outcome.completion_tokens
                token_usage_total = _sum_usage_dicts(
                    token_usage_total, modifier_outcome.token_usage
                )
                request_count_total += modifier_outcome.request_count
                trace["modifiers"] = modifier_outcome.trace

        deduped_terms = self._deduplicate_terms(guarded_terms)
        deduped_family = self._deduplicate_terms(guarded_family)
        deduped_other_person = self._deduplicate_terms(guarded_other_person)

        for finding in (*deduped_terms, *deduped_family, *deduped_other_person):
            if finding.modifiers:
                self._reconcile_post_dedup_spans(finding=finding, text=text)
                validated_modifiers, mod_warnings = validate_finding_modifiers(
                    finding.modifiers,
                    profile=profile,
                    ancestors_map=guard_map,
                )
                finding.modifiers = validated_modifiers
                _append_unique_warnings(finding, mod_warnings)

        return LLMExtractionResult(
            terms=deduped_terms,
            family_history_findings=deduped_family,
            other_person_findings=deduped_other_person,
            meta=LLMMeta(
                llm_provider=config.provider,
                llm_model=config.model,
                llm_mode=config.mode,
                prompt_version=prompt_version,
                token_input=prompt_tokens_total,
                token_output=completion_tokens_total,
                token_count_source=getattr(
                    self.provider,
                    "token_count_source",
                    None,
                ),
                token_usage=token_usage_total,
                request_count=request_count_total,
                phase_timings=phase_timings,
                phase_counts=phase_counts,
                phase_request_counts=phase_request_counts,
                trace=trace,
            ),
        )

    def _build_qualifier_exclusions(
        self,
        *,
        resolved_terms: list[LLMPhenotype],
        grounded_chunks: list[dict[str, Any]],
        language: str,
        profile: OntologyProfile | None = None,
        guard_map: Mapping[str, frozenset[str]] | None = None,
    ) -> tuple[list[LLMPhenotype], _ItemResolutionOutcome | None]:
        """Generate an excluded finding for each proband finding's negated
        qualifier Y that maps at or above the confidence floor (B3).

        For every resolved proband finding carrying a truthy
        ``negated_qualifier``, resolve Y through the shared retrieve -> route ->
        map path (#490) and, when the chosen term scores
        ``>= self.similarity_threshold``, emit a generated excluded
        ``LLMPhenotype`` for Y. When nothing is chosen (or below the floor) no
        term is generated and the source finding's ``negated_qualifier``
        metadata string is left untouched. Also returns the resolution outcome
        so its cost is accounted, or ``None`` when there is no qualifier.

        F5: the generated term INHERITS the source finding's evidence records so
        it carries valid ``chunk_ids`` (the "X without Y" span lives in the same
        chunk as X). A generated term with empty/invalid chunk refs is silently
        dropped at the aggregation service boundary
        (``full_text_service.py``)."""
        sources_by_qualifier: dict[str, list[LLMPhenotype]] = {}
        synthetic_items: list[dict[str, Any]] = []
        for finding in resolved_terms:
            qualifier = (finding.negated_qualifier or "").strip()
            if not qualifier:
                continue
            if qualifier not in sources_by_qualifier:
                # Inherit the source finding's chunk ids so the retrieval item
                # stays anchored to the source span (F5).
                synthetic_items.append(
                    {
                        "phrase": qualifier,
                        "category": "Abnormal",
                        "chunk_ids": [
                            chunk_id
                            for record in finding.evidence_records
                            for chunk_id in record.chunk_ids
                        ],
                        "evidence_text": qualifier,
                        "negated_qualifier": None,
                        "start_char": None,
                        "end_char": None,
                        "experiencer": "proband",
                        "assertion": None,
                    }
                )
            sources_by_qualifier.setdefault(qualifier, []).append(finding)
        if not synthetic_items:
            return [], None

        # Y takes the same route -> map path as any phrase, so the local-match
        # check and the mapping model decide, not the top-1 candidate (#490).
        outcome = self._resolve_items(
            synthetic_items,
            grounded_chunks=grounded_chunks,
            language=language,
            profile=profile,
            guard_map=guard_map,
        )
        exclusions: list[LLMPhenotype] = []
        for mapped in outcome.resolved:
            top_score = float(mapped.score or 0.0)
            if top_score < self.similarity_threshold:
                continue
            qualifier = mapped.evidence_records[0].phrase
            for finding in sources_by_qualifier.get(qualifier, []):
                # F5: copy the source finding's evidence records so the
                # generated term carries at least one valid chunk reference.
                inherited_records = [
                    record.model_copy(deep=True) for record in finding.evidence_records
                ]
                exclusions.append(
                    LLMPhenotype(
                        term_id=mapped.term_id,
                        label=mapped.label,
                        evidence=qualifier,
                        assertion=NEGATED_ASSERTION,
                        experiencer="proband",
                        negated_qualifier=None,
                        qualifier_surface_text=qualifier,
                        match_method="negated_qualifier_derived",
                        confidence=top_score,
                        score=top_score,
                        evidence_records=inherited_records,
                    )
                )
        return exclusions, outcome

    def _resolve_items(
        self,
        items: list[dict[str, Any]],
        *,
        grounded_chunks: list[dict[str, Any]],
        language: str,
        profile: OntologyProfile | None = None,
        guard_map: Mapping[str, frozenset[str]] | None = None,
    ) -> _ItemResolutionOutcome:
        """Run one item set through the shared phase-2 path (retrieve -> route
        -> map) and return every accounting delta as a pure outcome.

        Extracted verbatim from the pre-B2 inline ``run()`` resolution block so
        both the proband set and the family set flow through IDENTICAL retrieval,
        routing, and mapping logic. The method never mutates ``run()`` state;
        ``run()`` merges the returned deltas into the appropriate meta
        namespace."""
        resolved: list[LLMPhenotype] = []
        raw_modifiers_by_phenotype: dict[int, list[str]] = {}
        logger.debug(
            "Phase 2A: retrieving candidates for actionable_phrases=%d",
            len(items),
        )
        phase2a_start = time.perf_counter()
        try:
            phrase_candidates = self._retrieve_candidates(
                actionable=items,
                grounded_chunks=grounded_chunks,
                language=language,
                profile=profile,
                guard_map=guard_map,
            )
        except LLMPipelinePhaseError:
            raise
        except Exception as exc:
            # A typed phase error lets callers record the document as failed
            # instead of aborting a whole batch run (#494).
            raise LLMPipelinePhaseError(
                "phase2a", "Candidate retrieval failed"
            ) from exc
        phase2a_seconds = round(time.perf_counter() - phase2a_start, 6)
        candidate_sets = len(phrase_candidates)
        phase2a_trace = build_phase2a_trace(phrase_candidates)
        logger.debug(
            "Phase 2A complete: candidate_sets=%d",
            len(phrase_candidates),
        )
        logger.debug(
            "Phase 2B-local: resolving local matches for candidate_sets=%d",
            len(phrase_candidates),
        )
        phase2b_local_start = time.perf_counter()
        (
            locally_resolved,
            unresolved,
            routing_counts,
        ) = self._route_phase2_candidates(
            phrase_candidates=phrase_candidates,
            language=language,
            raw_modifiers_sink=raw_modifiers_by_phenotype,
        )
        phase2b_local_seconds = round(time.perf_counter() - phase2b_local_start, 6)
        resolved.extend(locally_resolved)
        phase2b_local_trace = build_phase2b_local_trace(
            locally_resolved=locally_resolved,
            unresolved=unresolved,
        )
        logger.debug(
            "Phase 2B-local complete: matched=%d unresolved=%d",
            len(locally_resolved),
            len(unresolved),
        )

        phase2b_llm_seconds = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        token_usage: dict[str, int] = {}
        request_count = 0
        phase2b_llm_requests = 0
        local_fallbacks = 0
        llm_mapped_phrases = 0
        mapping_prompt_version: str | None = None
        phase2b_llm_trace: dict[str, Any] | None = None
        mapping_ran = False

        if unresolved:
            mapping_ran = True
            mapping_prompt = (
                get_batch_mapping_prompt(language)
                if len(unresolved) > 1 and self.mapping_batch_size > 1
                else get_mapping_prompt(language)
            )
            mapping_prompt_version = mapping_prompt.version
            logger.debug(
                "Phase 2B-llm: mapping unresolved phrases=%d",
                len(unresolved),
            )
            phase2b_llm_start = time.perf_counter()
            try:
                (
                    mapped_terms,
                    mapping_prompt_tokens,
                    mapping_completion_tokens,
                    mapping_token_usage,
                    mapping_request_count,
                    local_fallback_count,
                    mapping_trace,
                ) = self._resolve_with_mapping_prompt(
                    unresolved=unresolved,
                    mapping_prompt=mapping_prompt,
                    raw_modifiers_sink=raw_modifiers_by_phenotype,
                )
            except LLMPipelinePhaseError:
                raise
            except Exception as exc:
                raise LLMPipelinePhaseError(
                    "phase2b",
                    "Phase 2 mapping failed",
                    usage=dict(getattr(self.provider, "last_usage", {}) or {}),
                    request_count=int(
                        getattr(self.provider, "last_request_count", 0) or 0
                    ),
                ) from exc
            phase2b_llm_seconds = round(time.perf_counter() - phase2b_llm_start, 6)
            prompt_tokens = mapping_prompt_tokens
            completion_tokens = mapping_completion_tokens
            token_usage = mapping_token_usage
            request_count = mapping_request_count
            phase2b_llm_requests = mapping_request_count
            resolved.extend(mapped_terms)
            local_fallbacks = local_fallback_count
            llm_mapped_phrases = len(mapped_terms) - local_fallback_count
            phase2b_llm_trace = {"resolved": mapping_trace}
            logger.debug(
                "Phase 2B-llm complete: mapped=%d prompt_tokens=%d completion_tokens=%d",
                llm_mapped_phrases,
                mapping_prompt_tokens,
                mapping_completion_tokens,
            )

        return _ItemResolutionOutcome(
            resolved=resolved,
            phase2a_seconds=phase2a_seconds,
            phase2b_local_seconds=phase2b_local_seconds,
            phase2b_llm_seconds=phase2b_llm_seconds,
            candidate_sets=candidate_sets,
            routing_counts=routing_counts,
            unresolved_phrases=len(unresolved),
            local_matches=len(locally_resolved),
            local_fallbacks=local_fallbacks,
            llm_mapped_phrases=llm_mapped_phrases,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            token_usage=token_usage,
            request_count=request_count,
            phase2b_llm_requests=phase2b_llm_requests,
            mapping_prompt_version=mapping_prompt_version,
            phase2a_trace=phase2a_trace,
            phase2b_local_trace=phase2b_local_trace,
            phase2b_llm_trace=phase2b_llm_trace,
            mapping_ran=mapping_ran,
            raw_modifiers_by_phenotype=raw_modifiers_by_phenotype,
        )

    def warmup(self, *, language: str) -> None:
        if hasattr(self.tool_executor, "warmup"):
            self.tool_executor.warmup(language=language)

    @staticmethod
    def _render_chunk_index(grounded_chunks: list[dict[str, Any]]) -> str:
        if not grounded_chunks:
            return "[]"
        return "\n".join(
            f"- chunk_id={chunk['chunk_id']}: {_phase1_chunk_text(chunk)}"
            for chunk in grounded_chunks
        )

    def _extract_phase1_phenotypes(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_prompt,
        chunk_index_text: str | None = None,
        capture_debug: bool = False,
        max_output_tokens: int | None = None,
        extract_modifiers: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, dict[str, Any] | None]:
        user_prompt = _render_phase1_user_prompt(
            extraction_prompt=extraction_prompt,
            text=text,
            grounded_chunks=grounded_chunks,
            chunk_index_text=chunk_index_text,
        )
        if max_output_tokens is None:
            max_output_tokens = (
                _grounded_phase1_output_token_budget(len(grounded_chunks))
                if grounded_chunks or chunk_index_text is not None
                else DEFAULT_PHASE1_MAX_OUTPUT_TOKENS
            )
        logger.debug(
            "Phase 1 request shape: grounded=%s chunk_count=%d user_prompt_chars=%d max_output_tokens=%d",
            bool(grounded_chunks),
            len(grounded_chunks),
            len(user_prompt),
            max_output_tokens,
        )
        is_grounded = bool(grounded_chunks or chunk_index_text is not None)
        response_model: type[Any]
        if extract_modifiers:
            response_model = (
                LLMGroundedExtractedPhenotypesWithModifiers
                if is_grounded
                else LLMExtractedPhenotypesWithModifiers
            )
        else:
            response_model = (
                LLMGroundedExtractedPhenotypes
                if is_grounded
                else LLMExtractedPhenotypes
            )
        try:
            response = self.provider.run_structured_prompt(
                system_prompt=extraction_prompt.render_system_prompt(),
                user_prompt=user_prompt,
                response_model=response_model,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            logger.exception("Phase 1 structured extraction failed")
            phase_error = LLMPipelinePhaseError(
                "phase1",
                "Structured extraction failed",
                usage=dict(getattr(self.provider, "last_usage", {}) or {}),
                request_count=int(getattr(self.provider, "last_request_count", 0) or 0),
            )
            phase_error.output_truncated = is_output_truncation_finish_reason(
                getattr(self.provider, "last_finish_reason", None)
            )
            raise phase_error from exc

        usage = dict(getattr(self.provider, "last_usage", {}) or {})
        request_count = int(getattr(self.provider, "last_request_count", 0) or 0)
        parsed: list[dict[str, Any]] = []
        for phenotype in response.phenotypes:
            parsed_item: dict[str, Any] = {
                "phrase": phenotype.phrase.strip(),
                "category": _normalize_category(phenotype.category),
                "negated_qualifier": getattr(phenotype, "negated_qualifier", None),
                "chunk_ids": list(getattr(phenotype, "chunk_ids", [])),
                "evidence_text": getattr(phenotype, "evidence_text", None),
                "start_char": getattr(phenotype, "start_char", None),
                "end_char": getattr(phenotype, "end_char", None),
                "experiencer": getattr(phenotype, "experiencer", None),
                "temporality": (
                    phenotype.temporality
                    if "temporality" in phenotype.model_fields_set
                    else None
                ),
                # Thread the model's assertion ONLY when the model actually
                # emitted it. The wire schema declares assertion with a
                # default of "present", so a category-only finding (e.g.
                # Normal/Suspected) always has a non-None attribute; keying
                # on ``model_fields_set`` distinguishes a model-emitted
                # assertion from the schema default. When unset we thread
                # None so the downstream category fallback re-engages and a
                # ruled-out/uncertain finding does not silently flip to
                # present (B1).
                "assertion": (
                    phenotype.assertion
                    if "assertion" in phenotype.model_fields_set
                    else None
                ),
            }
            if extract_modifiers:
                parsed_item["modifiers"] = _normalize_modifiers(
                    getattr(phenotype, "modifiers", [])
                )
            parsed.append(parsed_item)
        debug_payload: dict[str, Any] | None = None
        if capture_debug:
            structured_payload = (
                dict(getattr(self.provider, "last_structured_payload", {}) or {})
                or None
            )
            debug_payload = {
                "source_text": chunk_index_text
                if chunk_index_text is not None
                else text,
                "user_prompt": user_prompt,
                "structured_response": structured_payload,
                "parsed_extracted": list(parsed),
            }
        return (
            parsed,
            usage,
            request_count,
            debug_payload,
        )

    def _extract_grouped_phase1_phenotypes(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_groups: list[dict[str, Any]],
        extraction_prompt,
        capture_debug: bool = False,
        retry_truncated_groups: bool = False,
        extract_modifiers: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, list[dict[str, Any]]]:
        extracted: list[dict[str, Any]] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        total_request_count = 0
        phase1_groups_trace: list[dict[str, Any]] = []
        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in grounded_chunks}
        phase1_started = time.perf_counter()
        indexed_groups = [
            {"group_index": index, **dict(group)}
            for index, group in enumerate(extraction_groups)
        ]
        indexed_results: list[
            tuple[
                int,
                dict[str, Any],
                list[dict[str, Any]] | None,
                dict[str, int] | None,
                int,
                float,
                dict[str, Any] | None,
                LLMPipelinePhaseError | None,
                bool,
            ]
        ] = []
        max_workers = min(len(indexed_groups), DEFAULT_PHASE1_GROUP_MAX_WORKERS) or 1
        retry_budget = (
            threading.Semaphore(DEFAULT_GROUNDED_PHASE1_MAX_TRUNCATION_RETRIES)
            if retry_truncated_groups
            else None
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_phase1_group_with_truncation_retry,
                    retry_budget=retry_budget,
                    extraction_group=group,
                    text=text,
                    extraction_prompt=extraction_prompt,
                    chunk_lookup=chunk_lookup,
                    capture_debug=capture_debug,
                    extract_modifiers=extract_modifiers,
                ): group
                for group in indexed_groups
            }
            for future in as_completed(futures):
                group = futures[future]
                group_index = int(group.get("group_index", 0))
                try:
                    (
                        group_extracted,
                        group_usage,
                        group_request_count,
                        group_elapsed,
                        phase1_debug,
                        truncation_retried,
                    ) = future.result()
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            group_extracted,
                            group_usage,
                            group_request_count,
                            group_elapsed,
                            phase1_debug,
                            None,
                            truncation_retried,
                        )
                    )
                except LLMPipelinePhaseError as exc:
                    indexed_results.append(
                        (
                            group_index,
                            group,
                            None,
                            exc.usage,
                            exc.request_count,
                            float(exc.elapsed_seconds or 0.0),
                            None,
                            exc,
                            exc.truncation_retried,
                        )
                    )

        indexed_results.sort(key=lambda item: item[0])
        first_group_error: LLMPipelinePhaseError | None = None

        for (
            _group_index,
            group,
            group_extracted_opt,
            group_usage_opt,
            group_request_count,
            group_elapsed,
            phase1_debug,
            group_error,
            truncation_retried,
        ) in indexed_results:
            group_chunk_ids = [int(chunk_id) for chunk_id in group.get("chunk_ids", [])]
            retry_trace = {"truncation_retry": True} if truncation_retried else {}
            if group_error is not None:
                if first_group_error is None:
                    first_group_error = group_error
                prompt_tokens_total += int(
                    (group_usage_opt or {}).get("prompt_tokens", 0) or 0
                )
                completion_tokens_total += int(
                    (group_usage_opt or {}).get("completion_tokens", 0) or 0
                )
                total_request_count += group_request_count
                logger.warning(
                    "Continuing after grouped phase 1 failure: group_id=%s chunk_ids=%s error=%s",
                    group.get("group_id"),
                    group_chunk_ids,
                    str(group_error),
                )
                phase1_groups_trace.append(
                    {
                        "group_id": group.get("group_id"),
                        "chunk_ids": group_chunk_ids,
                        "source_chunk_ids": group_chunk_ids,
                        "status": "failed",
                        "error_code": "phase1_group_error",
                        "error": "Grouped phase 1 extraction failed",
                        "error_type": type(
                            group_error.__cause__ or group_error
                        ).__name__,
                        "failure_class": _classify_phase1_failure(group_error),
                        "extracted_count": 0,
                        "extracted": [],
                        "elapsed_seconds": group_elapsed,
                        **retry_trace,
                        **({"debug": phase1_debug} if phase1_debug else {}),
                    }
                )
                continue

            assert group_extracted_opt is not None
            assert group_usage_opt is not None
            group_extracted = group_extracted_opt
            group_usage = group_usage_opt
            prompt_tokens_total += int(group_usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(group_usage.get("completion_tokens", 0) or 0)
            total_request_count += group_request_count
            extracted.extend(group_extracted)
            group_trace_extracted = phase1_extracted_trace_items(group_extracted)
            phase1_groups_trace.append(
                {
                    "group_id": group.get("group_id"),
                    "chunk_ids": group_chunk_ids,
                    "source_chunk_ids": group_chunk_ids,
                    "status": "completed",
                    "extracted_count": len(group_trace_extracted),
                    "prompt_tokens": int(group_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        group_usage.get("completion_tokens", 0) or 0
                    ),
                    "request_count": group_request_count,
                    "elapsed_seconds": group_elapsed,
                    "extracted": group_trace_extracted,
                    **retry_trace,
                    **({"debug": phase1_debug} if phase1_debug else {}),
                }
            )

        any_group_succeeded = any(
            group.get("status") == "completed" for group in phase1_groups_trace
        )
        if extraction_groups and not any_group_succeeded:
            raise LLMPipelinePhaseError(
                "phase1",
                "Structured extraction failed for all extraction groups",
                usage={
                    "prompt_tokens": prompt_tokens_total,
                    "completion_tokens": completion_tokens_total,
                    "total_tokens": prompt_tokens_total + completion_tokens_total,
                },
                request_count=total_request_count,
                elapsed_seconds=round(time.perf_counter() - phase1_started, 6),
                groups_trace=phase1_groups_trace,
            ) from first_group_error

        extracted = self._deduplicate_phase1_extractions(extracted)

        return (
            extracted,
            {
                "prompt_tokens": prompt_tokens_total,
                "completion_tokens": completion_tokens_total,
                "total_tokens": prompt_tokens_total + completion_tokens_total,
            },
            total_request_count,
            round(time.perf_counter() - phase1_started, 6),
            phase1_groups_trace,
        )

    def _run_phase1_group_with_truncation_retry(
        self,
        *,
        retry_budget: threading.Semaphore | None,
        **group_kwargs: Any,
    ) -> tuple[
        list[dict[str, Any]], dict[str, int], int, float, dict[str, Any] | None, bool
    ]:
        """Run one phase-1 group; retry it ONCE at a fixed ceiling if truncated.

        Only the last fallback level (``grouped_small``) retries: there is no
        smaller group left to fall back to, so without the retry a group whose
        findings legitimately exceed the scaled budget would be dropped (#395).
        The first request keeps the scaled budget, so a runaway generation is
        still cut early, and the retry is bounded by the fixed ceiling.
        ``retry_budget`` is shared by all groups of one request and caps the
        number of ceiling retries; ``None`` disables retries.
        """
        try:
            return (*self._run_phase1_group(**group_kwargs), False)
        except LLMPipelinePhaseError as exc:
            if not (
                retry_budget is not None
                and exc.output_truncated
                and retry_budget.acquire(blocking=False)
            ):
                raise
            first_error = exc
        extraction_group = group_kwargs["extraction_group"]
        logger.warning(
            "Phase 1 group truncated at its output budget; retrying once with "
            "max_output_tokens=%d: group_id=%s chunk_ids=%s",
            DEFAULT_GROUNDED_PHASE1_TRUNCATION_RETRY_OUTPUT_TOKENS,
            extraction_group.get("group_id"),
            list(extraction_group.get("chunk_ids", [])),
        )
        first_elapsed = float(first_error.elapsed_seconds or 0.0)
        try:
            extracted, usage, request_count, elapsed, debug = self._run_phase1_group(
                **group_kwargs,
                max_output_tokens=DEFAULT_GROUNDED_PHASE1_TRUNCATION_RETRY_OUTPUT_TOKENS,
            )
        except LLMPipelinePhaseError as retry_error:
            retry_error.usage = _sum_usage_dicts(first_error.usage, retry_error.usage)
            retry_error.request_count += first_error.request_count
            retry_error.elapsed_seconds = round(
                first_elapsed + float(retry_error.elapsed_seconds or 0.0), 6
            )
            retry_error.truncation_retried = True
            raise
        return (
            extracted,
            _sum_usage_dicts(first_error.usage, usage),
            request_count + first_error.request_count,
            round(first_elapsed + elapsed, 6),
            debug,
            True,
        )

    def _run_phase1_group(
        self,
        *,
        extraction_group: dict[str, Any],
        text: str,
        extraction_prompt: PromptTemplate,
        chunk_lookup: dict[int, dict[str, Any]],
        capture_debug: bool = False,
        max_output_tokens: int | None = None,
        extract_modifiers: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, dict[str, Any] | None]:
        group_chunk_ids = [
            int(chunk_id) for chunk_id in extraction_group.get("chunk_ids", [])
        ]
        group_grounded_chunks = [
            chunk_lookup[chunk_id]
            for chunk_id in group_chunk_ids
            if chunk_id in chunk_lookup
        ]
        group_started = time.perf_counter()
        try:
            group_extracted, group_usage, group_request_count, phase1_debug = (
                self._extract_phase1_phenotypes(
                    text=text,
                    grounded_chunks=group_grounded_chunks,
                    chunk_index_text=_render_group_chunk_index_text(
                        group=extraction_group,
                        grounded_chunks=group_grounded_chunks,
                    ),
                    extraction_prompt=extraction_prompt,
                    capture_debug=capture_debug,
                    max_output_tokens=max_output_tokens,
                    extract_modifiers=extract_modifiers,
                )
            )
        except LLMPipelinePhaseError as exc:
            exc.elapsed_seconds = round(time.perf_counter() - group_started, 6)
            raise
        return (
            group_extracted,
            group_usage,
            group_request_count,
            round(time.perf_counter() - group_started, 6),
            phase1_debug if capture_debug else None,
        )

    def _resolve_initial_phase1_mode(
        self,
        *,
        extraction_groups: list[dict[str, Any]],
    ) -> Phase1Mode:
        return "grouped_large" if extraction_groups else "ungrouped"

    def _run_phase1_mode(
        self,
        *,
        mode: Phase1Mode,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_groups: list[dict[str, Any]],
        extraction_prompt: PromptTemplate,
        capture_phase1_debug: bool = False,
        extract_modifiers: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int, float, list[dict[str, Any]]]:
        phase1_started = time.perf_counter()
        try:
            if mode == "ungrouped":
                extracted, phase1_usage, phase1_request_count, phase1_debug = (
                    self._extract_phase1_phenotypes(
                        text=text,
                        grounded_chunks=grounded_chunks,
                        extraction_prompt=extraction_prompt,
                        capture_debug=capture_phase1_debug,
                        extract_modifiers=extract_modifiers,
                    )
                )
                groups_trace: list[dict[str, Any]] = []
                if capture_phase1_debug and phase1_debug is not None:
                    groups_trace = [
                        {
                            "group_id": 1,
                            "chunk_ids": [
                                int(chunk["chunk_id"])
                                for chunk in grounded_chunks
                                if "chunk_id" in chunk
                            ],
                            "source_chunk_ids": [
                                int(chunk["chunk_id"])
                                for chunk in grounded_chunks
                                if "chunk_id" in chunk
                            ],
                            "status": "completed",
                            "extracted_count": len(extracted),
                            "prompt_tokens": int(
                                phase1_usage.get("prompt_tokens", 0) or 0
                            ),
                            "completion_tokens": int(
                                phase1_usage.get("completion_tokens", 0) or 0
                            ),
                            "request_count": phase1_request_count,
                            "elapsed_seconds": round(
                                time.perf_counter() - phase1_started, 6
                            ),
                            "extracted": phase1_extracted_trace_items(extracted),
                            "debug": phase1_debug,
                        }
                    ]
                return (
                    extracted,
                    phase1_usage,
                    phase1_request_count,
                    round(time.perf_counter() - phase1_started, 6),
                    groups_trace,
                )

            active_extraction_groups = (
                extraction_groups
                if mode == "grouped_large" and extraction_groups
                else self._build_large_extraction_groups(
                    text=text,
                    grounded_chunks=grounded_chunks,
                    extraction_prompt=extraction_prompt,
                )
                if mode == "grouped_large"
                else self._build_small_extraction_groups(
                    grounded_chunks=grounded_chunks
                )
            )
            return self._extract_grouped_phase1_phenotypes(
                text=text,
                grounded_chunks=grounded_chunks,
                extraction_groups=active_extraction_groups,
                extraction_prompt=extraction_prompt,
                capture_debug=capture_phase1_debug,
                retry_truncated_groups=mode == "grouped_small",
                extract_modifiers=extract_modifiers,
            )
        except LLMPipelinePhaseError as exc:
            if exc.elapsed_seconds is None:
                exc.elapsed_seconds = round(time.perf_counter() - phase1_started, 6)
            raise
        except Exception as exc:
            raise LLMPipelinePhaseError(
                "phase1",
                f"Phase 1 grouped setup failed: {exc}",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                request_count=0,
                elapsed_seconds=round(time.perf_counter() - phase1_started, 6),
                groups_trace=[],
            ) from exc

    def _phase1_partial_retry_chunks(
        self,
        *,
        mode: Phase1Mode,
        grounded_chunks: list[dict[str, Any]],
        groups_trace: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        Phase1FailureClass,
        list[dict[str, Any]],
        Phase1FailureClass,
    ]:
        if mode != "grouped_large":
            return [], None, list(groups_trace), None

        next_mode = _next_phase1_mode(mode)
        retry_chunk_ids: set[int] = set()
        retained_groups_trace: list[dict[str, Any]] = []
        retry_failure_class: Phase1FailureClass = None
        terminal_failure_class: Phase1FailureClass = None

        for group in groups_trace:
            if group.get("status") != "failed":
                retained_groups_trace.append(group)
                continue

            failure_class = group.get("failure_class")
            if self._should_fallback_phase1(
                failure_class=failure_class,
                grounded_chunks=grounded_chunks,
                next_mode=next_mode,
            ):
                if retry_failure_class is None:
                    retry_failure_class = failure_class
                retry_chunk_ids.update(_group_source_chunk_ids(group))
                continue
            retained_groups_trace.append(group)
            if terminal_failure_class is None:
                terminal_failure_class = failure_class

        if terminal_failure_class is not None:
            return [], None, list(groups_trace), terminal_failure_class
        if not retry_chunk_ids:
            return [], None, list(groups_trace), None

        retry_grounded_chunks = [
            chunk
            for chunk in grounded_chunks
            if int(chunk.get("chunk_id", 0)) in retry_chunk_ids
        ]
        return (
            retry_grounded_chunks,
            retry_failure_class,
            retained_groups_trace,
            None,
        )

    def _should_fallback_phase1(
        self,
        *,
        failure_class: Phase1FailureClass,
        grounded_chunks: list[dict[str, Any]],
        next_mode: Phase1Mode | None,
    ) -> bool:
        if not DEFAULT_PHASE1_FALLBACK_ENABLED:
            return False
        if next_mode is None or not grounded_chunks:
            return False
        return failure_class in {
            "provider_timeout",
            "structured_json_invalid",
            "structured_schema_validation_failed",
        }

    def _plan_grouped_large_fallback(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_prompt: PromptTemplate,
    ) -> tuple[Phase1Mode, list[dict[str, Any]]]:
        """Choose the fallback after a failed ungrouped request.

        A single large group holding every chunk re-sends essentially the same
        request (only the list rendering differs), so it would repeat the
        failure, whether that was a truncation or an invalid response. In that
        case go straight to small groups.
        """
        try:
            large_groups = self._build_large_extraction_groups(
                text=text,
                grounded_chunks=grounded_chunks,
                extraction_prompt=extraction_prompt,
            )
        except Exception:
            # Let the regular grouped_large path build and report the failure.
            return "grouped_large", []
        if len(large_groups) <= 1:
            logger.warning(
                "Phase 1 request failed; skipping grouped_large because it "
                "would resend all %d chunks as one group",
                len(grounded_chunks),
            )
            return "grouped_small", []
        return "grouped_large", large_groups

    def _build_large_extraction_groups(
        self,
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
        extraction_prompt: PromptTemplate,
    ) -> list[dict[str, Any]]:
        if not grounded_chunks:
            return []
        grounded_chunk_models = [
            GroundedChunk(
                chunk_id=int(chunk["chunk_id"]),
                text=str(chunk.get("text", "")),
                start_char=chunk.get("start_char"),
                end_char=chunk.get("end_char"),
                status=str(chunk.get("status", "unknown")),
                prompt_text=chunk.get("prompt_text"),
            )
            for chunk in grounded_chunks
        ]
        try:
            return [
                asdict(group)
                for group in build_extraction_groups(
                    grounded_chunks=grounded_chunk_models,
                    provider=self.provider,
                    system_prompt=extraction_prompt.render_system_prompt(),
                    max_prompt_tokens=max(
                        int(DEFAULT_PHASE1_LARGE_GROUP_MAX_PROMPT_TOKENS),
                        1,
                    ),
                )
            ]
        except (NotImplementedError, TypeError):
            pass

        return [
            {
                "group_id": 1,
                "chunk_ids": [int(chunk["chunk_id"]) for chunk in grounded_chunks],
                "text": self._render_chunk_index(grounded_chunks),
                "estimated_prompt_tokens": 0,
            }
        ]

    def _build_small_extraction_groups(
        self,
        *,
        grounded_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not grounded_chunks:
            return []

        max_chunks = max(int(DEFAULT_PHASE1_SMALL_GROUP_MAX_CHUNKS), 1)
        groups: list[dict[str, Any]] = []
        start_index = 0
        while start_index < len(grounded_chunks):
            group_chunks = grounded_chunks[start_index : start_index + max_chunks]
            groups.append(
                {
                    "group_id": len(groups) + 1,
                    "chunk_ids": [int(chunk["chunk_id"]) for chunk in group_chunks],
                    "text": self._render_chunk_index(group_chunks),
                    "estimated_prompt_tokens": 0,
                }
            )
            if start_index + max_chunks >= len(grounded_chunks):
                break
            start_index += max_chunks - 1 if max_chunks > 1 else 1
        return groups

    @staticmethod
    def _phase1_failure_class_from_groups(
        phase1_groups_trace: list[dict[str, Any]],
    ) -> Phase1FailureClass:
        return phase1_failure_class_from_groups(phase1_groups_trace)

    @staticmethod
    def _attach_phase1_failure_context(
        *,
        exc: LLMPipelinePhaseError,
        initial_mode: Phase1Mode,
        final_mode: Phase1Mode,
        fallback_count: int,
        failure_class: Phase1FailureClass,
        final_groups_trace: list[dict[str, Any]],
        attempts_trace: list[dict[str, Any]],
    ) -> None:
        attach_phase1_failure_context(
            exc=exc,
            initial_mode=initial_mode,
            final_mode=final_mode,
            fallback_count=fallback_count,
            failure_class=failure_class,
            final_groups_trace=final_groups_trace,
            attempts_trace=attempts_trace,
        )

    def _check_index_roots_compatibility(
        self,
        *,
        language: str,
        profile: OntologyProfile,
        guard_map: Mapping[str, frozenset[str]],
        include_modifiers: bool,
    ) -> None:
        if not hasattr(self.tool_executor, "check_index_roots_compatibility"):
            return
        if (
            isinstance(self.tool_executor, ToolExecutor)
            and self.tool_executor._cached_retriever is None
            and self.tool_executor._retriever is None
        ):
            return
        required_roots = profile.roots.effective_required_roots(
            include_modifiers=include_modifiers
        )
        try:
            warnings = self.tool_executor.check_index_roots_compatibility(
                language=language,
                configured_index_roots=profile.roots.index_roots,
                required_roots=required_roots,
                ancestors_map=guard_map,
                strict=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Index root compatibility check skipped: %s", exc)
            return
        for warning_msg in warnings:
            logger.warning("Index roots compatibility: %s", warning_msg)

    def _query_batch_hpo_terms(
        self,
        *,
        phrases: list[str],
        language: str,
        n_results: int,
        allowed_term_ids: set[str] | frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        if allowed_term_ids is not None and _supports_kwarg(
            self.tool_executor.query_batch_hpo_terms, "allowed_term_ids"
        ):
            return list(
                self.tool_executor.query_batch_hpo_terms(
                    phrases=phrases,
                    language=language,
                    n_results=n_results,
                    allowed_term_ids=allowed_term_ids,
                )
            )
        return list(
            self.tool_executor.query_batch_hpo_terms(
                phrases=phrases,
                language=language,
                n_results=n_results,
            )
        )

    def _retrieve_candidates(
        self,
        *,
        actionable: list[dict[str, Any]],
        grounded_chunks: list[dict[str, Any]],
        language: str,
        profile: OntologyProfile | None = None,
        guard_map: Mapping[str, frozenset[str]] | None = None,
    ) -> list[dict[str, Any]]:
        unique_actionable: list[dict[str, Any]] = []
        actionable_groups: dict[
            tuple[str, str, tuple[str, ...]], list[dict[str, Any]]
        ] = {}
        for item in actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            actionable_groups.setdefault(dedupe_key, []).append(item)
            if len(actionable_groups[dedupe_key]) == 1:
                unique_actionable.append(item)

        expanded_queries: list[str] = []
        expanded_query_keys: list[tuple[str, str, tuple[str, ...]]] = []
        for item in unique_actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            query_variants = prepare_retrieval_queries(
                str(item["phrase"]),
                strip_cue=(
                    _normalize_category(str(item.get("category", ""))) == "normal"
                    or str(item.get("assertion") or "").strip().lower()
                    in {"absent", NEGATED_ASSERTION}
                ),
                language=language,
            )
            if not query_variants:
                query_variants = [str(item["phrase"])]
            for query in query_variants:
                expanded_queries.append(query)
                expanded_query_keys.append(dedupe_key)

        batched_variant_results = self.tool_executor.query_batch_hpo_terms(
            phrases=expanded_queries,
            language=language,
            n_results=self.n_results_per_phrase,
        )

        shared_results: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
        grouped_variant_results: dict[
            tuple[str, str, tuple[str, ...]], list[tuple[str, dict[str, Any]]]
        ] = {}
        for index, dedupe_key in enumerate(expanded_query_keys):
            query = expanded_queries[index]
            batch_result = (
                batched_variant_results[index]
                if index < len(batched_variant_results)
                else {}
            )
            grouped_variant_results.setdefault(dedupe_key, []).append(
                (query, batch_result)
            )

        # Loaded once: the guard checks every candidate of every phrase below.
        active_profile = (
            profile if profile is not None else get_default_ontology_profile()
        )
        active_guard_map = load_guard_map() if guard_map is None else guard_map
        for item in unique_actionable:
            phrase = str(item["phrase"])
            category = str(item["category"])
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            merged_candidates: dict[str, dict[str, Any]] = {}
            for query, batch_result in grouped_variant_results.get(dedupe_key, []):
                if "candidates" in batch_result:
                    candidates = [
                        {
                            **dict(candidate),
                            "retrieval_query": query,
                        }
                        for candidate in batch_result.get("candidates", [])
                        if isinstance(candidate, dict)
                    ]
                else:
                    metadatas = self._extract_first_result_list(
                        batch_result, "metadatas"
                    )
                    similarities = self._extract_first_result_list(
                        batch_result, "similarities"
                    )
                    # Filter before selection: a non-finding hit would use up
                    # a selection slot or the min-fill quota and then be removed
                    # below without replacement (#493).
                    phenotypic_indices = [
                        index
                        for index, metadata in enumerate(metadatas)
                        if _is_accepted_finding_candidate(
                            str(metadata.get("hpo_id", "")),
                            profile=active_profile,
                            guard_map=active_guard_map,
                        )
                    ]
                    if len(phenotypic_indices) < len(metadatas):
                        similarities = [
                            similarities[index]
                            for index in phenotypic_indices
                            if index < len(similarities)
                        ]
                        metadatas = [metadatas[index] for index in phenotypic_indices]
                    candidates = [
                        {
                            **candidate,
                            "retrieval_query": query,
                        }
                        for candidate in self._hybrid_select_candidates(
                            phrase=query,
                            metadatas=metadatas,
                            similarities=similarities,
                        )
                    ]

                for candidate in candidates:
                    hpo_id = str(candidate.get("hpo_id", "")).strip()
                    # Filter before mapping: post-mapping guard runs after a term is
                    # chosen, so a finding mapped to a disallowed branch would be lost.
                    if not hpo_id or not _is_accepted_finding_candidate(
                        hpo_id,
                        profile=active_profile,
                        guard_map=active_guard_map,
                    ):
                        continue
                    existing = merged_candidates.get(hpo_id)
                    if existing is None or float(
                        candidate.get("score", 0.0) or 0.0
                    ) > float(existing.get("score", 0.0) or 0.0):
                        merged_candidates[hpo_id] = candidate

            merged = sorted(
                merged_candidates.values(),
                key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
                reverse=True,
            )[: self.n_results_per_phrase]
            shared_results[dedupe_key] = {
                "phrase": phrase,
                "category": category,
                "candidates": merged,
            }
            logger.debug(
                "Phase 2A candidate retrieval: phrase=%r candidates=%d",
                phrase,
                len(shared_results[dedupe_key]["candidates"]),
            )

        results: list[dict[str, Any]] = []
        for item in actionable:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=False)
            shared_result = dict(shared_results.get(dedupe_key, {}))
            shared_result.setdefault("phrase", str(item["phrase"]))
            shared_result.setdefault("category", str(item["category"]))
            shared_result["chunk_ids"] = list(item.get("chunk_ids", []))
            shared_result["evidence_text"] = item.get("evidence_text")
            shared_result["negated_qualifier"] = item.get("negated_qualifier")
            shared_result["start_char"] = item.get("start_char")
            shared_result["end_char"] = item.get("end_char")
            shared_result["offset_basis"] = item.get("offset_basis")
            shared_result["experiencer"] = item.get("experiencer")
            shared_result["assertion"] = item.get("assertion")
            shared_result["temporality"] = item.get("temporality")
            if "modifiers" in item:
                shared_result["modifiers"] = list(item.get("modifiers") or [])
            shared_result["grounded_context"] = self._build_grounded_context(
                item=item,
                grounded_chunks=grounded_chunks,
            )
            shared_result["candidates"] = list(shared_result.get("candidates", []))
            results.append(shared_result)
        return results

    @staticmethod
    def _extract_first_result_list(
        batch_result: dict[str, Any],
        key: str,
    ) -> list[Any]:
        return extract_first_result_list(batch_result, key)

    @staticmethod
    def _build_grounded_context(
        *,
        item: dict[str, Any],
        grounded_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_grounded_context(item=item, grounded_chunks=grounded_chunks)

    def _hybrid_select_candidates(
        self,
        *,
        phrase: str,
        metadatas: list[dict[str, Any]],
        similarities: list[float],
    ) -> list[dict[str, Any]]:
        return hybrid_select_candidates(
            phrase=phrase,
            metadatas=metadatas,
            similarities=similarities,
            max_unique_candidates=self.max_unique_candidates,
            min_unique_candidates=self.min_unique_candidates,
            similarity_threshold=self.similarity_threshold,
        )

    @staticmethod
    def _deduplicate_phase1_extractions(
        extracted: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deduplicated: list[dict[str, Any]] = []
        seen_items: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in extracted:
            key = _phase1_extraction_dedup_key(item)
            if key in seen_items:
                if "modifiers" in item:
                    TwoPhaseLLMPipeline._merge_phase1_modifiers(seen_items[key], item)
                continue
            merged = False
            for existing in deduplicated:
                if not TwoPhaseLLMPipeline._should_merge_phase1_extractions(
                    existing, item
                ):
                    continue
                TwoPhaseLLMPipeline._merge_phase1_extraction(existing, item)
                seen_items[key] = existing
                merged = True
                break
            if merged:
                continue
            copied = dict(item)
            if "modifiers" in item:
                copied["modifiers"] = _normalize_modifiers(item.get("modifiers"))
            deduplicated.append(copied)
            seen_items[key] = copied
        return deduplicated

    @staticmethod
    def _should_merge_phase1_extractions(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> bool:
        # ``experiencer`` and ``assertion`` are orthogonal axes (B1). The fuzzy
        # merge exists only to consolidate fragments/duplicates of the SAME
        # finding; a differing experiencer (proband vs family_history) or
        # assertion (present vs absent) marks a genuinely DIFFERENT finding, so
        # refuse the merge before any span/evidence heuristics run. This runs
        # BEFORE the B2 family partition, so collapsing here would silently drop
        # a co-located family or opposite-polarity mention.
        for axis in ("experiencer", "assertion"):
            if (
                str(existing.get(axis) or "").strip().lower()
                != str(incoming.get(axis) or "").strip().lower()
            ):
                return False
        if _temporality_axis(existing) != _temporality_axis(incoming):
            return False
        if (
            str(existing.get("phrase", "")).strip().lower()
            != str(incoming.get("phrase", "")).strip().lower()
        ):
            return False
        if _normalize_category(
            str(existing.get("category", ""))
        ) != _normalize_category(str(incoming.get("category", ""))):
            return False

        existing_chunk_ids = set(_sorted_chunk_ids(existing.get("chunk_ids", [])))
        incoming_chunk_ids = set(_sorted_chunk_ids(incoming.get("chunk_ids", [])))
        if not existing_chunk_ids or not incoming_chunk_ids:
            return False
        if not (existing_chunk_ids & incoming_chunk_ids):
            return False

        existing_evidence = _normalized_evidence_text(existing)
        incoming_evidence = _normalized_evidence_text(incoming)
        if not existing_evidence or not incoming_evidence:
            return False
        return (
            existing_evidence == incoming_evidence
            or existing_evidence in incoming_evidence
            or incoming_evidence in existing_evidence
        )

    @staticmethod
    def _attach_document_offsets(
        extracted: list[dict[str, Any]],
        *,
        text: str,
        grounded_chunks: list[dict[str, Any]],
    ) -> None:
        """Replace model-reported offsets with document offsets of the evidence.

        A chunk whose phase-1 rendering (``prompt_text``) carries the text that
        cleanup dropped around it is searched over exactly the source text that
        rendering shows, because the model may quote that text.
        """
        locator = _build_evidence_locator(text, grounded_chunks)
        for item in extracted:
            located = locator.locate(
                item.get("evidence_text"), item.get("chunk_ids") or []
            )
            item["start_char"] = located.start
            item["end_char"] = located.end
            item["offset_basis"] = located.basis

    @staticmethod
    def _merge_phase1_modifiers(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        if "modifiers" in existing or "modifiers" in incoming:
            existing["modifiers"] = _normalize_modifiers(
                [
                    *(existing.get("modifiers") or []),
                    *(incoming.get("modifiers") or []),
                ]
            )

    @staticmethod
    def _merge_phase1_extraction(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> None:
        existing["chunk_ids"] = _sorted_chunk_ids(
            [*existing.get("chunk_ids", []), *incoming.get("chunk_ids", [])]
        )
        existing["evidence_text"] = _prefer_richer_text(
            existing.get("evidence_text"),
            incoming.get("evidence_text"),
        )
        TwoPhaseLLMPipeline._merge_phase1_modifiers(existing, incoming)

    def _route_phase2_candidates(
        self,
        *,
        phrase_candidates: list[dict[str, Any]],
        language: str,
        raw_modifiers_sink: dict[int, list[str]] | None = None,
    ) -> tuple[list[LLMPhenotype], list[dict[str, Any]], dict[str, int]]:
        resolved: list[LLMPhenotype] = []
        unresolved: list[dict[str, Any]] = []
        counts = {
            "phase2b_local_accept_count": 0,
            "phase2b_deferred_count": 0,
            "phase2b_no_candidate_skip_count": 0,
        }

        for item in phrase_candidates:
            candidates = item.get("candidates", [])
            if not candidates:
                logger.warning(
                    "Phase 2B-local: phrase=%r has no candidates; skipping mapping",
                    item["phrase"],
                )
                counts["phase2b_no_candidate_skip_count"] += 1
                continue

            decision = self._decide_phase2_routing(item=item, language=language)
            if decision != "local":
                unresolved.append(item)
                counts["phase2b_deferred_count"] += 1
                logger.debug(
                    "Phase 2B-local: phrase=%r unresolved_candidates=%d",
                    item["phrase"],
                    len(candidates),
                )
                continue

            local_match = self._try_local_match(item["phrase"], candidates)
            if local_match is None:
                unresolved.append(item)
                counts["phase2b_deferred_count"] += 1
                logger.debug(
                    "Phase 2B-local: phrase=%r deferred_after_missing_local_match",
                    item["phrase"],
                )
                continue

            phenotype = self._phenotype_from_candidate(item=item, candidate=local_match)
            if raw_modifiers_sink is not None and item.get("modifiers"):
                raw_modifiers_sink[id(phenotype)] = list(item["modifiers"])
            resolved.append(phenotype)
            counts["phase2b_local_accept_count"] += 1
            logger.debug(
                "Phase 2B-local: phrase=%r matched=%s",
                item["phrase"],
                local_match["hpo_id"],
            )
        return resolved, unresolved, counts

    def _decide_phase2_routing(self, *, item: dict[str, Any], language: str) -> str:
        candidates = list(item.get("candidates", []))
        local_match = self._try_local_match(str(item["phrase"]), candidates)
        if local_match is None:
            return "defer"

        normalized_language = (language or "").strip().lower()
        phrase_clean = _clean_text(str(item["phrase"]))
        term_clean = _clean_text(_candidate_match_text(local_match))
        phrase_tokens = _tokenize(str(item["phrase"]))
        term_tokens = _tokenize(_candidate_match_text(local_match))
        match_score = float(local_match.get("score", 0.0) or 0.0)

        if normalized_language == "en":
            if phrase_clean == term_clean or (
                phrase_tokens and term_tokens and phrase_tokens == term_tokens
            ):
                return "local" if match_score >= 0.85 else "defer"
            if term_clean and re.search(rf"\b{re.escape(term_clean)}\b", phrase_clean):
                return "local" if match_score >= 0.85 else "defer"
            if token_sort_similarity(phrase_clean, term_clean) >= max(
                self.local_match_threshold, 90.0
            ):
                return "local" if match_score >= 0.85 else "defer"
            return "defer"

        if normalized_language == "de":
            if phrase_clean == term_clean or (
                phrase_tokens and term_tokens and phrase_tokens == term_tokens
            ):
                return "local"
            return "defer"

        return "defer"

    def _try_local_match(
        self,
        phrase: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        return local_match(
            phrase=phrase,
            candidates=candidates,
            local_match_threshold=self.local_match_threshold,
        )

    def _resolve_with_mapping_prompt(
        self,
        *,
        unresolved: list[dict[str, Any]],
        mapping_prompt,
        raw_modifiers_sink: dict[int, list[str]] | None = None,
    ) -> tuple[
        list[LLMPhenotype],
        int,
        int,
        dict[str, int],
        int,
        int,
        list[dict[str, Any]],
    ]:
        resolved: list[LLMPhenotype] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        token_usage_total: dict[str, int] = {}
        request_count_total = 0
        local_fallback_count = 0
        mapping_trace: list[dict[str, Any]] = []
        unresolved_groups: dict[
            tuple[str, str, tuple[str, ...]], list[dict[str, Any]]
        ] = {}
        unique_unresolved: list[dict[str, Any]] = []
        for item in unresolved:
            dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=True)
            unresolved_groups.setdefault(dedupe_key, []).append(item)
            if len(unresolved_groups[dedupe_key]) == 1:
                unique_unresolved.append(item)

        for start in range(0, len(unique_unresolved), self.mapping_batch_size):
            batch = unique_unresolved[start : start + self.mapping_batch_size]
            # The model chooses among the candidates it is shown; local matching
            # and the fallback below keep the full retrieved list (#523).
            prompt_batch = [self._mapping_prompt_item(item) for item in batch]
            mapping_response, batch_usage = self._run_mapping_batch(
                batch=prompt_batch,
                mapping_prompt=mapping_prompt,
            )
            request_count_total += int(
                getattr(self.provider, "last_request_count", 0) or 0
            )
            token_usage_total = _sum_usage_dicts(token_usage_total, batch_usage)
            prompt_tokens_total += int(batch_usage.get("prompt_tokens", 0) or 0)
            completion_tokens_total += int(batch_usage.get("completion_tokens", 0) or 0)
            selected_ids = self._select_candidate_ids(
                mapping_response=mapping_response,
                batch=prompt_batch,
            )
            explicit_nulls = _explicit_null_item_ids(
                mapping_response=mapping_response, batch=batch
            )
            for index, item in enumerate(batch):
                dedupe_key = _downstream_dedupe_key(item, include_candidate_ids=True)
                grouped_items = unresolved_groups[dedupe_key]
                selected_id = selected_ids.get(_mapping_batch_item_id(index))
                explicit_null = (
                    selected_id is None
                    and _mapping_batch_item_id(index) in explicit_nulls
                )
                (
                    item_resolved,
                    item_local_fallback_count,
                    item_trace,
                ) = self._resolve_mapping_selection(
                    item=item, selected_id=selected_id, explicit_null=explicit_null
                )
                if raw_modifiers_sink is not None and item.get("modifiers"):
                    for phenotype in item_resolved:
                        raw_modifiers_sink[id(phenotype)] = list(item["modifiers"])
                resolved.extend(item_resolved)
                local_fallback_count += item_local_fallback_count
                mapping_trace.append(item_trace)
                for grouped_item in grouped_items[1:]:
                    (
                        grouped_resolved,
                        grouped_local_fallback_count,
                        grouped_item_trace,
                    ) = self._resolve_mapping_selection(
                        item=grouped_item,
                        selected_id=selected_id,
                        explicit_null=explicit_null,
                    )
                    if raw_modifiers_sink is not None and grouped_item.get("modifiers"):
                        for phenotype in grouped_resolved:
                            raw_modifiers_sink[id(phenotype)] = list(
                                grouped_item["modifiers"]
                            )
                    resolved.extend(grouped_resolved)
                    local_fallback_count += grouped_local_fallback_count
                    mapping_trace.append(
                        self._annotate_mapping_trace_with_provenance(
                            grouped_item_trace,
                            grouped_item,
                        )
                    )
        return (
            resolved,
            prompt_tokens_total,
            completion_tokens_total,
            token_usage_total,
            request_count_total,
            local_fallback_count,
            mapping_trace,
        )

    def _mapping_prompt_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Copy of ``item`` with the candidates the mapping prompt may carry.

        Retrieval returns ``n_results_per_phrase`` candidates per query variant
        so that merging across variants sees the wider lists; the prompt carries
        the ``max_unique_candidates`` best of the merged list, by score.
        """
        ranked = sorted(
            item.get("candidates", []),
            key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
            reverse=True,
        )
        return {**item, "candidates": ranked[: max(self.max_unique_candidates, 0)]}

    @staticmethod
    def _annotate_mapping_trace_with_provenance(
        trace_entry: dict[str, Any],
        item: dict[str, Any],
    ) -> dict[str, Any]:
        return annotate_mapping_trace_with_provenance(trace_entry, item)

    def _run_mapping_batch(
        self,
        *,
        batch: list[dict[str, Any]],
        mapping_prompt,
    ) -> tuple[LLMMappingSelection | LLMBatchMappingSelections, dict[str, int]]:
        return run_mapping_batch(
            provider=self.provider,
            batch=batch,
            mapping_prompt=mapping_prompt,
        )

    def _resolve_mapping_selection(
        self,
        *,
        item: dict[str, Any],
        selected_id: str | None,
        explicit_null: bool = False,
    ) -> tuple[list[LLMPhenotype], int, dict[str, Any]]:
        if not selected_id:
            # An explicit null rejects every candidate, including the local
            # match routing already declined (#490).
            if explicit_null:
                fallback = None
            else:
                logger.warning(
                    "Phase 2B-llm: phrase=%r produced no valid mapping; trying local fallback",
                    item["phrase"],
                )
                fallback = self._try_local_match(item["phrase"], item["candidates"])
            if fallback is None:
                return (
                    [],
                    0,
                    {
                        "phrase": str(item["phrase"]),
                        "selected_id": None,
                        "term_id": None,
                        "label": None,
                        "assertion": CATEGORY_TO_ASSERTION.get(
                            _normalize_category(str(item.get("category", ""))),
                            PRESENT_ASSERTION,
                        ),
                        "category": _normalize_category(str(item.get("category", ""))),
                        "match_method": "llm",
                        "local_fallback": False,
                    },
                )
            logger.debug(
                "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                item["phrase"],
                fallback["hpo_id"],
            )
            phenotype = self._phenotype_from_candidate(
                item=item,
                candidate=fallback,
                match_method="local",
            )
            return (
                [phenotype],
                1,
                {
                    "phrase": str(item["phrase"]),
                    "selected_id": None,
                    "term_id": phenotype.term_id,
                    "label": phenotype.label,
                    "assertion": phenotype.assertion,
                    "category": phenotype.category,
                    "match_method": "llm",
                    "local_fallback": True,
                },
            )

        candidate = next(
            (
                candidate
                for candidate in item["candidates"]
                if candidate["hpo_id"] == selected_id
            ),
            None,
        )
        if candidate is None:
            logger.warning(
                "Phase 2B-llm: phrase=%r selected_id=%s not in candidates; trying local fallback",
                item["phrase"],
                selected_id,
            )
            fallback = self._try_local_match(item["phrase"], item["candidates"])
            if fallback is None:
                return (
                    [],
                    0,
                    {
                        "phrase": str(item["phrase"]),
                        "selected_id": selected_id,
                        "term_id": None,
                        "label": None,
                        "assertion": CATEGORY_TO_ASSERTION.get(
                            _normalize_category(str(item.get("category", ""))),
                            PRESENT_ASSERTION,
                        ),
                        "category": _normalize_category(str(item.get("category", ""))),
                        "match_method": "llm",
                        "local_fallback": False,
                    },
                )
            logger.debug(
                "Phase 2B-llm: phrase=%r local fallback resolved=%s",
                item["phrase"],
                fallback["hpo_id"],
            )
            phenotype = self._phenotype_from_candidate(
                item=item,
                candidate=fallback,
                match_method="local",
            )
            return (
                [phenotype],
                1,
                {
                    "phrase": str(item["phrase"]),
                    "selected_id": selected_id,
                    "term_id": phenotype.term_id,
                    "label": phenotype.label,
                    "assertion": phenotype.assertion,
                    "category": phenotype.category,
                    "match_method": "llm",
                    "local_fallback": True,
                },
            )

        logger.debug(
            "Phase 2B-llm: phrase=%r mapped=%s",
            item["phrase"],
            selected_id,
        )
        phenotype = self._phenotype_from_candidate(
            item=item,
            candidate=candidate,
            match_method="llm",
        )
        return (
            [phenotype],
            0,
            {
                "phrase": str(item["phrase"]),
                "selected_id": selected_id,
                "term_id": phenotype.term_id,
                "label": phenotype.label,
                "assertion": phenotype.assertion,
                "category": phenotype.category,
                "match_method": "llm",
                "local_fallback": False,
            },
        )

    @staticmethod
    def _phenotype_from_candidate(
        *,
        item: dict[str, Any],
        candidate: dict[str, Any],
        match_method: str = "local",
    ) -> LLMPhenotype:
        return phenotype_from_candidate(
            item=item,
            candidate=candidate,
            match_method=match_method,
        )

    @classmethod
    def _select_candidate_ids(
        cls,
        *,
        mapping_response: LLMMappingSelection | LLMBatchMappingSelections,
        batch: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        return select_candidate_ids(mapping_response=mapping_response, batch=batch)

    @staticmethod
    def _deduplicate_terms(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
        return deduplicate_terms(terms)

    def _extract_modifier_candidates(
        self,
        *,
        phrase: str,
        batch_result: dict[str, Any],
        profile: OntologyProfile,
        guard_map: Mapping[str, frozenset[str]],
    ) -> tuple[list[dict[str, Any]], list[OntologyWarning], bool]:
        """Parse and validate retrieved modifier candidates for one phrase.

        Returns ``(accepted_candidates, warnings, matched_disallowed)``.
        """
        raw_candidates: list[dict[str, Any]] = []
        if "candidates" in batch_result:
            raw_candidates = [
                {**dict(candidate), "retrieval_query": phrase}
                for candidate in batch_result.get("candidates", [])
                if isinstance(candidate, dict)
            ]
        else:
            metadatas = self._extract_first_result_list(batch_result, "metadatas")
            similarities = self._extract_first_result_list(batch_result, "similarities")
            for idx, metadata in enumerate(metadatas):
                if not isinstance(metadata, dict):
                    continue
                hpo_id = str(metadata.get("hpo_id", "")).strip()
                label = str(metadata.get("label", "")).strip()
                score = float(similarities[idx]) if idx < len(similarities) else 0.0
                if hpo_id:
                    raw_candidates.append(
                        {
                            "hpo_id": hpo_id,
                            "term_name": label,
                            "score": score,
                            "retrieval_query": phrase,
                        }
                    )

        deduped: dict[str, dict[str, Any]] = {}
        for candidate in raw_candidates:
            hpo_id = str(candidate.get("hpo_id", "")).strip()
            if not hpo_id:
                continue
            existing = deduped.get(hpo_id)
            if existing is None or float(candidate.get("score", 0.0) or 0.0) > float(
                existing.get("score", 0.0) or 0.0
            ):
                deduped[hpo_id] = candidate

        ranked = sorted(
            deduped.values(),
            key=lambda candidate: float(candidate.get("score", 0.0) or 0.0),
            reverse=True,
        )

        accepted: list[dict[str, Any]] = []
        disallowed: list[dict[str, Any]] = []
        disallowed_warnings: dict[str, tuple[OntologyWarning, ...]] = {}
        for candidate in ranked:
            hpo_id = str(candidate.get("hpo_id", "")).strip()
            decision = check_modifier_term(
                hpo_id, profile=profile, ancestors_map=guard_map
            )
            if decision.accepted:
                accepted.append(candidate)
            elif decision.warnings:
                disallowed.append(candidate)
                disallowed_warnings[hpo_id] = decision.warnings

        warnings: list[OntologyWarning] = []
        matched_disallowed = False
        if disallowed:
            local_disallowed = self._try_local_match(phrase, disallowed)
            if local_disallowed is not None:
                warnings.extend(
                    disallowed_warnings.get(str(local_disallowed["hpo_id"]), ())
                )
                matched_disallowed = True
            else:
                top_raw = ranked[0] if ranked else None
                if (
                    top_raw is not None
                    and str(top_raw.get("hpo_id", "")) in disallowed_warnings
                    and float(top_raw.get("score", 0.0) or 0.0)
                    >= self.similarity_threshold
                ):
                    warnings.extend(disallowed_warnings[str(top_raw["hpo_id"])])
                    matched_disallowed = True

        return (
            accepted[: min(self.n_results_per_phrase, 25)],
            warnings,
            matched_disallowed,
        )

    def _resolve_modifiers(
        self,
        *,
        findings: list[LLMPhenotype],
        raw_modifiers_by_phenotype: dict[int, list[str]],
        text: str,
        grounded_chunks: list[dict[str, Any]],
        language: str,
        profile: OntologyProfile,
        guard_map: Mapping[str, frozenset[str]],
    ) -> _ModifierResolutionOutcome:
        outcome = _ModifierResolutionOutcome()
        # Collect unique modifier phrases preserving order.
        unique_phrases: list[str] = []
        phrase_by_key: dict[str, str] = {}
        for finding in findings:
            for raw_phrase in raw_modifiers_by_phenotype.get(id(finding), []):
                key = normalize_mapping_phrase_key(raw_phrase)
                if not key:
                    continue
                if key not in phrase_by_key:
                    phrase_by_key[key] = raw_phrase
                    unique_phrases.append(raw_phrase)

        if not unique_phrases:
            return outcome

        outcome.modifier_phrases = len(unique_phrases)
        modifier_term_ids: set[str] | None = None
        if profile.roots.modifier_roots and guard_map:
            retrieval_roots = (
                *profile.roots.modifier_roots,
                *profile.disallowed_modifier_roots.values(),
            )
            modifier_term_ids = term_ids_under_roots(retrieval_roots, guard_map)

        phase2a_start = time.perf_counter()
        try:
            batched_results = self._query_batch_hpo_terms(
                phrases=unique_phrases,
                language=language,
                n_results=min(self.n_results_per_phrase, 25),
                allowed_term_ids=modifier_term_ids,
            )
        except Exception as exc:
            logger.warning(
                "Modifier retrieval failed; continuing without modifiers: %s",
                exc,
            )
            outcome.phase2a_seconds = round(time.perf_counter() - phase2a_start, 6)
            return outcome
        candidates_by_phrase_key: dict[str, list[dict[str, Any]]] = {}
        warnings_by_phrase_key: dict[str, list[OntologyWarning]] = {}
        disallowed_phrase_keys: set[str] = set()
        for idx, phrase in enumerate(unique_phrases):
            phrase_key = normalize_mapping_phrase_key(phrase)
            batch_result = batched_results[idx] if idx < len(batched_results) else {}
            accepted_cands, phrase_warnings, matched_disallowed = (
                self._extract_modifier_candidates(
                    phrase=phrase,
                    batch_result=batch_result,
                    profile=profile,
                    guard_map=guard_map,
                )
            )
            candidates_by_phrase_key[phrase_key] = accepted_cands
            if phrase_warnings:
                warnings_by_phrase_key[phrase_key] = phrase_warnings
            if matched_disallowed:
                disallowed_phrase_keys.add(phrase_key)
        outcome.phase2a_seconds = round(time.perf_counter() - phase2a_start, 6)

        # Deduplicate tasks by (phrase_key, finding.term_id, finding.label, chunk_ids)
        # so occurrences in distinct chunks retain their own grounded context.
        TaskKey = tuple[str, str, str, tuple[int, ...]]
        tasks: dict[TaskKey, dict[str, Any]] = {}
        finding_tasks: list[tuple[LLMPhenotype, str, TaskKey]] = []
        for finding in findings:
            chunk_ids = [
                cid for record in finding.evidence_records for cid in record.chunk_ids
            ]
            chunk_key = tuple(sorted(set(chunk_ids)))
            for raw_phrase in raw_modifiers_by_phenotype.get(id(finding), []):
                phrase_key = normalize_mapping_phrase_key(raw_phrase)
                if not phrase_key:
                    continue
                task_key: TaskKey = (
                    phrase_key,
                    finding.term_id,
                    finding.label,
                    chunk_key,
                )
                finding_tasks.append((finding, raw_phrase, task_key))
                if task_key not in tasks:
                    primary_ev_text = (
                        finding.evidence_records[0].evidence_text
                        if finding.evidence_records
                        else finding.evidence
                    )
                    tasks[task_key] = {
                        "phrase": raw_phrase,
                        "phrase_key": phrase_key,
                        "finding_term": {
                            "id": finding.term_id,
                            "label": finding.label,
                        },
                        "grounded_context": build_grounded_context(
                            item={
                                "chunk_ids": chunk_ids,
                                "evidence_text": primary_ev_text,
                                "phrase": raw_phrase,
                            },
                            grounded_chunks=grounded_chunks,
                        ),
                        "candidates": candidates_by_phrase_key.get(phrase_key, []),
                        "task_key": task_key,
                    }

        # Attach any disallowed warnings to the findings referencing those phrases.
        for finding, _raw_phrase, task_key in finding_tasks:
            disallowed_phrase_warnings = warnings_by_phrase_key.get(task_key[0])
            if disallowed_phrase_warnings:
                _append_unique_warnings(finding, disallowed_phrase_warnings)

        # Phase 2B-local for modifiers.
        phase2b_local_start = time.perf_counter()
        resolved_by_task: dict[TaskKey, tuple[str, str]] = {}
        deferred_tasks: list[dict[str, Any]] = []
        local_trace: list[dict[str, Any]] = []
        suppressed_trace: list[dict[str, Any]] = []

        for task_key, task in tasks.items():
            phrase_key = task["phrase_key"]
            phrase = task["phrase"]
            finding_label = task["finding_term"]["label"]
            accepted_cands = task["candidates"]

            if phrase_key in disallowed_phrase_keys:
                continue
            if not accepted_cands:
                continue

            if finding_already_expresses_modifier(
                finding_label=finding_label,
                modifier_label="",
                modifier_phrase=phrase,
            ):
                outcome.suppressed_double_encoding += 1
                suppressed_trace.append(
                    {
                        "phrase": phrase,
                        "finding_term_id": task["finding_term"]["id"],
                        "reason": "double_encoding_phrase",
                    }
                )
                continue

            route = self._decide_phase2_routing(
                item={"phrase": phrase, "candidates": accepted_cands},
                language=language,
            )
            matched = (
                self._try_local_match(phrase, accepted_cands)
                if route == "local"
                else None
            )
            if matched is not None:
                if finding_already_expresses_modifier(
                    finding_label=finding_label,
                    modifier_label=str(matched["term_name"]),
                    modifier_phrase=phrase,
                ):
                    outcome.suppressed_double_encoding += 1
                    suppressed_trace.append(
                        {
                            "phrase": phrase,
                            "finding_term_id": task["finding_term"]["id"],
                            "modifier_term_id": matched["hpo_id"],
                            "reason": "double_encoding_local",
                        }
                    )
                    continue
                resolved_by_task[task_key] = (
                    str(matched["hpo_id"]),
                    str(matched["term_name"]),
                )
                outcome.local_matches += 1
                local_trace.append(
                    {
                        "phrase": phrase,
                        "finding_term_id": task["finding_term"]["id"],
                        "term_id": matched["hpo_id"],
                        "label": matched["term_name"],
                        "match_method": "local",
                    }
                )
                continue

            deferred_tasks.append(
                {
                    **task,
                    "candidates": accepted_cands[:DEFAULT_MAX_MODIFIER_CANDIDATES],
                }
            )
        outcome.phase2b_local_seconds = round(
            time.perf_counter() - phase2b_local_start, 6
        )

        # Phase 2B-llm for unresolved modifiers.
        llm_trace: list[dict[str, Any]] = []
        if deferred_tasks:
            outcome.mapping_ran = True
            modifier_mapping_prompt = (
                get_batch_modifier_mapping_prompt(language)
                if len(deferred_tasks) > 1 and self.modifier_mapping_batch_size > 1
                else get_modifier_mapping_prompt(language)
            )
            phase2b_llm_start = time.perf_counter()
            for start in range(
                0, len(deferred_tasks), self.modifier_mapping_batch_size
            ):
                batch = deferred_tasks[start : start + self.modifier_mapping_batch_size]
                try:
                    mapping_response, batch_usage = run_modifier_mapping_batch(
                        provider=self.provider,
                        batch=batch,
                        mapping_prompt=modifier_mapping_prompt,
                    )
                except Exception as exc:
                    batch_reqs = int(
                        getattr(self.provider, "last_request_count", 0) or 0
                    )
                    outcome.request_count += batch_reqs
                    outcome.phase2b_llm_requests += batch_reqs
                    logger.warning(
                        "Modifier mapping batch failed; continuing without LLM "
                        "mappings for %d modifier(s): %s",
                        len(batch),
                        exc,
                    )
                    for mod_item in batch:
                        llm_trace.append(
                            {
                                "phrase": mod_item["phrase"],
                                "finding_term_id": mod_item["finding_term"]["id"],
                                "term_id": None,
                                "error": type(exc).__name__,
                            }
                        )
                    continue
                batch_reqs = int(getattr(self.provider, "last_request_count", 0) or 0)
                outcome.request_count += batch_reqs
                outcome.phase2b_llm_requests += batch_reqs
                outcome.token_usage = _sum_usage_dicts(outcome.token_usage, batch_usage)
                outcome.prompt_tokens += int(batch_usage.get("prompt_tokens", 0) or 0)
                outcome.completion_tokens += int(
                    batch_usage.get("completion_tokens", 0) or 0
                )

                selected_ids = self._select_candidate_ids(
                    mapping_response=mapping_response,
                    batch=batch,
                )
                explicit_nulls = _explicit_null_item_ids(
                    mapping_response=mapping_response,
                    batch=batch,
                )
                for index, mod_item in enumerate(batch):
                    item_id = _mapping_batch_item_id(index)
                    selected_id = selected_ids.get(item_id)
                    explicit_null = selected_id is None and item_id in explicit_nulls
                    task_key = mod_item["task_key"]
                    finding_label = mod_item["finding_term"]["label"]
                    phrase = mod_item["phrase"]

                    chosen: dict[str, Any] | None = None
                    if selected_id is not None:
                        chosen = next(
                            (
                                c
                                for c in mod_item["candidates"]
                                if c["hpo_id"] == selected_id
                            ),
                            None,
                        )
                    elif not explicit_null:
                        chosen = self._try_local_match(
                            phrase,
                            candidates_by_phrase_key.get(mod_item["phrase_key"], []),
                        )

                    if chosen is None:
                        llm_trace.append(
                            {
                                "phrase": phrase,
                                "finding_term_id": mod_item["finding_term"]["id"],
                                "term_id": None,
                                "explicit_null": explicit_null,
                            }
                        )
                        continue

                    if finding_already_expresses_modifier(
                        finding_label=finding_label,
                        modifier_label=str(chosen["term_name"]),
                        modifier_phrase=phrase,
                    ):
                        outcome.suppressed_double_encoding += 1
                        suppressed_trace.append(
                            {
                                "phrase": phrase,
                                "finding_term_id": mod_item["finding_term"]["id"],
                                "modifier_term_id": chosen["hpo_id"],
                                "reason": "double_encoding_llm_guard",
                            }
                        )
                        continue

                    resolved_by_task[task_key] = (
                        str(chosen["hpo_id"]),
                        str(chosen["term_name"]),
                    )
                    outcome.llm_mapped += 1
                    llm_trace.append(
                        {
                            "phrase": phrase,
                            "finding_term_id": mod_item["finding_term"]["id"],
                            "term_id": chosen["hpo_id"],
                            "label": chosen["term_name"],
                            "match_method": "llm",
                        }
                    )
            outcome.phase2b_llm_seconds = round(
                time.perf_counter() - phase2b_llm_start, 6
            )

        # Locate modifier spans and enforce disjoint span carving per finding.
        locator = _build_evidence_locator(text, grounded_chunks)
        resolved_per_finding: dict[int, list[tuple[str, str, str]]] = {}
        finding_by_id: dict[int, LLMPhenotype] = {}
        for finding, raw_phrase, task_key in finding_tasks:
            resolved_mod = resolved_by_task.get(task_key)
            if resolved_mod is None:
                continue
            finding_by_id[id(finding)] = finding
            resolved_per_finding.setdefault(id(finding), []).append(
                (resolved_mod[0], resolved_mod[1], raw_phrase)
            )

        for fid, mod_entries in resolved_per_finding.items():
            self._attach_and_carve_modifiers(
                finding=finding_by_id[fid],
                resolved_mods=mod_entries,
                text=text,
                locator=locator,
            )

        outcome.trace = {
            "local": local_trace,
            "llm": llm_trace,
            "suppressed": suppressed_trace,
        }
        return outcome

    @staticmethod
    def _attach_and_carve_modifiers(
        *,
        finding: LLMPhenotype,
        resolved_mods: list[tuple[str, str, str]],
        text: str,
        locator: EvidenceLocator,
    ) -> None:
        # Only carve a genuinely located evidence span, never a ``chunk_span``
        # fallback (#397): trimming a whole-chunk fallback and writing
        # ``text[start:end]`` back into ``evidence_text`` would fabricate a
        # fake exact quote out of surrounding chunk wording.
        primary_ev = next(
            (
                ev
                for ev in finding.evidence_records
                if ev.start_char is not None
                and ev.end_char is not None
                and ev.offset_basis != "chunk_span"
            ),
            None,
        )
        finding_chunk_ids = [
            cid for ev in finding.evidence_records for cid in ev.chunk_ids
        ]
        anchor_span: tuple[int, int] | None = (
            (primary_ev.start_char, primary_ev.end_char)
            if primary_ev is not None
            and primary_ev.start_char is not None
            and primary_ev.end_char is not None
            else None
        )

        located_spans: dict[int, tuple[int, int] | None] = {}
        for idx, (_term_id, _label, raw_phrase) in enumerate(resolved_mods):
            if not finding_chunk_ids or not text:
                located_spans[idx] = None
                continue
            located = locator.locate_nearest(
                raw_phrase,
                finding_chunk_ids,
                anchor_span=anchor_span,
                include_neighbors=True,
            )
            if (
                located.start is not None
                and located.end is not None
                and 0 <= located.start < located.end <= len(text)
            ):
                located_spans[idx] = (located.start, located.end)
            else:
                located_spans[idx] = None

        segments_by_idx: dict[int, list[ModifierSegment]] = {
            idx: [] for idx in range(len(resolved_mods))
        }

        if primary_ev is not None and anchor_span is not None:
            current_span = anchor_span
            pending: list[tuple[int, int, int]] = []
            for idx, m_span in located_spans.items():
                if m_span is None:
                    continue
                m_start, m_end = m_span
                if m_end <= current_span[0] or m_start >= current_span[1]:
                    segments_by_idx[idx].append(
                        ModifierSegment(
                            start=m_start,
                            end=m_end,
                            text=text[m_start:m_end],
                        )
                    )
                else:
                    pending.append((idx, m_start, m_end))

            while pending:
                progressed = False
                next_pending: list[tuple[int, int, int]] = []
                for idx, m_start, m_end in sorted(
                    pending, key=lambda entry: (entry[1], entry[2])
                ):
                    new_span, keep = carve_modifier_span(
                        text, current_span, (m_start, m_end)
                    )
                    if keep:
                        current_span = new_span
                        segments_by_idx[idx].append(
                            ModifierSegment(
                                start=m_start,
                                end=m_end,
                                text=text[m_start:m_end],
                            )
                        )
                        progressed = True
                    else:
                        next_pending.append((idx, m_start, m_end))
                pending = next_pending
                if not pending:
                    break

                next_pending = []
                for idx, m_start, m_end in sorted(
                    pending, key=lambda entry: (-entry[2], -entry[1])
                ):
                    new_span, keep = carve_modifier_span(
                        text, current_span, (m_start, m_end)
                    )
                    if keep:
                        current_span = new_span
                        segments_by_idx[idx].append(
                            ModifierSegment(
                                start=m_start,
                                end=m_end,
                                text=text[m_start:m_end],
                            )
                        )
                        progressed = True
                    else:
                        next_pending.append((idx, m_start, m_end))
                pending = next_pending
                if not progressed:
                    break

            if current_span != anchor_span:
                primary_ev.start_char, primary_ev.end_char = current_span
                primary_ev.evidence_text = text[current_span[0] : current_span[1]]
        else:
            for idx, m_span in located_spans.items():
                if m_span is not None:
                    m_start, m_end = m_span
                    segments_by_idx[idx].append(
                        ModifierSegment(
                            start=m_start,
                            end=m_end,
                            text=text[m_start:m_end],
                        )
                    )

        existing_by_id = {
            modifier.term_id: idx for idx, modifier in enumerate(finding.modifiers)
        }
        for idx, (term_id, label, _raw_phrase) in enumerate(resolved_mods):
            new_segments = segments_by_idx[idx]
            existing_idx = existing_by_id.get(term_id)
            if existing_idx is None:
                existing_by_id[term_id] = len(finding.modifiers)
                finding.modifiers.append(
                    Modifier(
                        term_id=term_id,
                        label=label,
                        segments=new_segments,
                        source="llm",
                    )
                )
            else:
                current = finding.modifiers[existing_idx]
                seen_segs = {(s.start, s.end, s.text) for s in current.segments}
                merged_segs = list(current.segments)
                for seg in new_segments:
                    key = (seg.start, seg.end, seg.text)
                    if key not in seen_segs:
                        seen_segs.add(key)
                        merged_segs.append(seg)
                finding.modifiers[existing_idx] = current.model_copy(
                    update={"segments": merged_segs, "source": current.source or "llm"}
                )

    @staticmethod
    def _reconcile_post_dedup_spans(
        *,
        finding: LLMPhenotype,
        text: str,
    ) -> None:
        """Ensure finding evidence spans and modifier segments stay disjoint after dedup.

        When ``_deduplicate_terms`` merges two mentions of the same finding (one
        carved and one uncarved), an inherited evidence record could overlap a
        surviving modifier segment. Re-carve any overlapping located evidence
        record, or drop modifier segments that still overlap an uncarvable span.
        """
        located_records = [
            ev
            for ev in finding.evidence_records
            if ev.start_char is not None
            and ev.end_char is not None
            and ev.offset_basis != "chunk_span"
        ]
        if not located_records or not finding.modifiers:
            return

        updated_modifiers: list[Modifier] = []
        for modifier in finding.modifiers:
            kept_segments: list[ModifierSegment] = []
            for seg in modifier.segments:
                seg_valid = True
                for ev in located_records:
                    if ev.start_char is None or ev.end_char is None:
                        continue
                    if seg.end <= ev.start_char or seg.start >= ev.end_char:
                        continue
                    if text and 0 <= ev.start_char < ev.end_char <= len(text):
                        new_span, keep = carve_modifier_span(
                            text, (ev.start_char, ev.end_char), (seg.start, seg.end)
                        )
                        if keep:
                            if new_span != (ev.start_char, ev.end_char):
                                ev.start_char, ev.end_char = new_span
                                ev.evidence_text = text[new_span[0] : new_span[1]]
                            continue
                    seg_valid = False
                    break
                if seg_valid:
                    kept_segments.append(seg)
            if len(kept_segments) != len(modifier.segments):
                updated_modifiers.append(
                    modifier.model_copy(update={"segments": kept_segments})
                )
            else:
                updated_modifiers.append(modifier)
        finding.modifiers = updated_modifiers

        deduped_records = []
        seen_spans: set[tuple[int | None, int | None, str | None]] = set()
        for ev in finding.evidence_records:
            key = (ev.start_char, ev.end_char, ev.offset_basis)
            if ev.start_char is not None and ev.end_char is not None:
                if key in seen_spans:
                    continue
                seen_spans.add(key)
            deduped_records.append(ev)
        finding.evidence_records = deduped_records

    @staticmethod
    def _terms_are_colocated(first: LLMPhenotype, second: LLMPhenotype) -> bool:
        first_chunks = {
            cid for record in first.evidence_records for cid in record.chunk_ids
        }
        second_chunks = {
            cid for record in second.evidence_records for cid in record.chunk_ids
        }
        if first_chunks and second_chunks and (first_chunks & second_chunks):
            return True
        for rec_a in first.evidence_records:
            if (
                rec_a.start_char is None
                or rec_a.end_char is None
                or rec_a.offset_basis == "chunk_span"
            ):
                continue
            for rec_b in second.evidence_records:
                if (
                    rec_b.start_char is None
                    or rec_b.end_char is None
                    or rec_b.offset_basis == "chunk_span"
                ):
                    continue
                if (
                    rec_a.start_char <= rec_b.end_char + 2
                    and rec_b.start_char <= rec_a.end_char + 2
                ):
                    return True
        return False

    @staticmethod
    def _attach_converted_modifier_finding(
        *,
        mod_term: LLMPhenotype,
        target: LLMPhenotype,
        text: str,
    ) -> None:
        segments: list[ModifierSegment] = []
        mod_ev = next(
            (
                ev
                for ev in mod_term.evidence_records
                if ev.start_char is not None
                and ev.end_char is not None
                and ev.offset_basis != "chunk_span"
            ),
            None,
        )
        target_ev = next(
            (
                ev
                for ev in target.evidence_records
                if ev.start_char is not None
                and ev.end_char is not None
                and ev.offset_basis != "chunk_span"
            ),
            None,
        )
        if (
            mod_ev is not None
            and mod_ev.start_char is not None
            and mod_ev.end_char is not None
        ):
            m_start, m_end = mod_ev.start_char, mod_ev.end_char
            if text and 0 <= m_start < m_end <= len(text):
                if (
                    target_ev is not None
                    and target_ev.start_char is not None
                    and target_ev.end_char is not None
                ):
                    f_span = (target_ev.start_char, target_ev.end_char)
                    new_span, keep = carve_modifier_span(text, f_span, (m_start, m_end))
                    if keep:
                        if new_span != f_span:
                            target_ev.start_char, target_ev.end_char = new_span
                            target_ev.evidence_text = text[new_span[0] : new_span[1]]
                        segments.append(
                            ModifierSegment(
                                start=m_start,
                                end=m_end,
                                text=text[m_start:m_end],
                            )
                        )
                else:
                    segments.append(
                        ModifierSegment(
                            start=m_start,
                            end=m_end,
                            text=text[m_start:m_end],
                        )
                    )
            elif (
                mod_ev.evidence_text
                and len(mod_ev.evidence_text) == m_end - m_start
                and m_end > m_start
                and (
                    target_ev is None
                    or target_ev.start_char is None
                    or target_ev.end_char is None
                    or m_end <= target_ev.start_char
                    or m_start >= target_ev.end_char
                )
            ):
                segments.append(
                    ModifierSegment(
                        start=m_start,
                        end=m_end,
                        text=mod_ev.evidence_text,
                    )
                )

        target.modifiers.append(
            Modifier(
                term_id=mod_term.term_id,
                label=mod_term.label,
                segments=segments,
                source="llm",
            )
        )

    @classmethod
    def _apply_ontology_guard(
        cls,
        *term_lists: list[LLMPhenotype],
        profile: OntologyProfile | None = None,
        guard_map: Mapping[str, frozenset[str]] | None = None,
        extract_modifiers: bool = False,
        text: str = "",
        stats: dict[str, int] | None = None,
    ) -> tuple[Any, ...]:
        """Guard every list and return them plus the number of dropped terms."""
        active_profile = (
            profile if profile is not None else get_default_ontology_profile()
        )
        active_guard_map = load_guard_map() if guard_map is None else guard_map
        guarded_lists: list[list[LLMPhenotype]] = []
        dropped = 0
        converted = 0

        use_legacy_default_guard = (
            not extract_modifiers
            and active_profile.roots.finding_roots == (HPO_PHENOTYPIC_ABNORMALITY_ROOT,)
        )

        for terms in term_lists:
            accepted: list[LLMPhenotype] = []
            modifier_candidates: list[LLMPhenotype] = []
            for term in terms:
                if use_legacy_default_guard:
                    if is_non_phenotypic_abnormality(term.term_id, active_guard_map):
                        logger.info(
                            "ontology guard dropped non-phenotype term %s (%s)",
                            term.term_id,
                            term.label,
                        )
                        dropped += 1
                    else:
                        accepted.append(term)
                    continue
                decision = check_finding_term(
                    term.term_id,
                    profile=active_profile,
                    ancestors_map=active_guard_map,
                )
                if decision.accepted:
                    _append_unique_warnings(term, decision.warnings)
                    accepted.append(term)
                elif decision.is_modifier_term and extract_modifiers:
                    modifier_candidates.append(term)
                else:
                    logger.info(
                        "ontology guard dropped non-finding term %s (%s)",
                        term.term_id,
                        term.label,
                    )
                    dropped += 1

            for mod_term in modifier_candidates:
                targets = [
                    candidate
                    for candidate in accepted
                    if (
                        candidate.experiencer,
                        candidate.assertion,
                        candidate.temporality,
                    )
                    == (
                        mod_term.experiencer,
                        mod_term.assertion,
                        mod_term.temporality,
                    )
                    and cls._terms_are_colocated(mod_term, candidate)
                ]
                if len(targets) != 1:
                    logger.info(
                        "ontology guard dropped modifier-as-finding %s (%s): targets=%d",
                        mod_term.term_id,
                        mod_term.label,
                        len(targets),
                    )
                    dropped += 1
                    continue
                target = targets[0]
                mod_decision = check_modifier_term(
                    mod_term.term_id,
                    profile=active_profile,
                    ancestors_map=active_guard_map,
                )
                if not mod_decision.accepted:
                    _append_unique_warnings(target, mod_decision.warnings)
                    dropped += 1
                    continue
                if finding_already_expresses_modifier(
                    finding_label=target.label,
                    modifier_label=mod_term.label,
                    modifier_phrase=mod_term.evidence or mod_term.label,
                ):
                    dropped += 1
                    continue
                cls._attach_converted_modifier_finding(
                    mod_term=mod_term,
                    target=target,
                    text=text,
                )
                converted += 1

            guarded_lists.append(accepted)

        if stats is not None:
            stats["converted_modifier_findings"] = converted
        return (*guarded_lists, dropped)

    @staticmethod
    def _drop_non_phenotypic(terms: list[LLMPhenotype]) -> list[LLMPhenotype]:
        """Drop resolved terms that are KNOWN to sit outside the Phenotypic
        abnormality (HP:0000118) subtree -- clinical modifiers / course /
        inheritance nodes (e.g. "Mild" HP:0012825, "Spatial pattern" HP:0012836,
        "Slowly progressive" HP:0003677). These are qualifiers, never standalone
        phenotypes, so they are not valid annotations. Fail open: terms with
        unknown ancestry (or when the HPO graph is unavailable) are kept, so a
        real finding is never silently dropped."""
        kept: list[LLMPhenotype] = []
        for term in terms:
            if is_non_phenotypic_abnormality(term.term_id):
                logger.info(
                    "ontology guard dropped non-phenotype term %s (%s)",
                    term.term_id,
                    term.label,
                )
                continue
            kept.append(term)
        return kept
