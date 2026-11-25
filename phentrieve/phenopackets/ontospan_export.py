"""Export Phentrieve findings into an ontospan 1.2.0 AnnotatedDocument."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from phentrieve.config import PHENOTYPE_ROOT
from phentrieve.llm.types import LLMExtractionResult

ONTOSPAN_SCHEMA_VERSION = "1.2.0"
ONTOSPAN_ARTIFACT_TYPE = "annotated_document"
_UTF8_BOM = "\ufeff"

_ASSERTION_MAP: dict[str, str] = {
    "present": "present",
    "affirmed": "present",
    "negated": "absent",
    "absent": "absent",
    "normal": "absent",
    "uncertain": "uncertain",
    "suspected": "uncertain",
}

_EXPERIENCER_MAP: dict[str, str] = {
    "proband": "patient",
    "patient": "patient",
    "family_history": "family_member",
    "family_member": "family_member",
    "family history": "family_member",
    "other": "other",
}

_TEMPORALITY_VALUES = frozenset({"current", "historical", "unknown"})


def canonicalize_text(raw: str) -> str:
    """Return NFC text with LF line endings and no leading BOM."""
    normalized = raw.lstrip(_UTF8_BOM).replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized)


def compute_text_sha256(text: str) -> str:
    """Compute SHA-256 hex digest of UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_field(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item and item[name] is not None:
                return item[name]
        return default
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return default


def _map_assertion(finding: Any) -> str:
    if _get_field(finding, "excluded") is True:
        return "absent"
    raw = _get_field(finding, "assertion", "assertion_status", "status")
    normalized = str(raw).strip().lower() if raw is not None else "present"
    return _ASSERTION_MAP.get(normalized, "present")


def _map_experiencer(finding: Any, default_experiencer: str = "proband") -> str:
    raw = _get_field(finding, "experiencer")
    normalized = (
        str(raw).strip().lower() if raw is not None else default_experiencer.lower()
    )
    return _EXPERIENCER_MAP.get(normalized, "unknown")


def _map_temporality(finding: Any) -> str:
    raw = _get_field(finding, "temporality")
    normalized = str(raw).strip().lower() if raw is not None else "unknown"
    return normalized if normalized in _TEMPORALITY_VALUES else "unknown"


def _is_valid_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _extract_mentions(finding: Any, canonical_doc: str) -> list[dict[str, Any]]:
    raw_records = _get_field(finding, "evidence_records", "spans", default=()) or ()
    mentions: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()
    doc_len = len(canonical_doc)

    for record in raw_records:
        # A ``chunk_span`` fallback covers the whole cited chunk rather than a
        # located mention (#397); do not export it as an exact text mention.
        if _get_field(record, "offset_basis") == "chunk_span":
            continue
        start = _get_field(record, "start_char", "start")
        end = _get_field(record, "end_char", "end")
        if not (_is_valid_int(start) and _is_valid_int(end)):
            continue
        if start < 0 or end <= start or end > doc_len:
            continue
        exact_slice = canonical_doc[start:end]
        if not exact_slice:
            continue
        declared_text = _get_field(record, "evidence_text", "text")
        if declared_text is not None and str(declared_text) != exact_slice:
            continue
        span_key = (start, end)
        if span_key in seen_ranges:
            continue
        seen_ranges.add(span_key)
        mentions.append(
            {
                "segments": [{"start": start, "end": end, "text": exact_slice}],
                "cue_segments": [],
            }
        )
    return mentions


def _extract_modifiers(finding: Any, canonical_doc: str) -> list[dict[str, Any]]:
    raw_modifiers = _get_field(finding, "modifiers", default=()) or ()
    exported: list[dict[str, Any]] = []
    seen_term_ids: set[str] = set()
    doc_len = len(canonical_doc)

    for modifier in raw_modifiers:
        term_id = _get_field(modifier, "term_id", "id")
        if not term_id:
            continue
        term_id_str = str(term_id).strip()
        if not term_id_str or term_id_str in seen_term_ids:
            continue
        seen_term_ids.add(term_id_str)

        raw_segments = _get_field(modifier, "segments", default=()) or ()
        candidates: list[tuple[int, int, str]] = []
        for seg in raw_segments:
            start = _get_field(seg, "start", "start_char")
            end = _get_field(seg, "end", "end_char")
            text = _get_field(seg, "text", "evidence_text")
            if not (_is_valid_int(start) and _is_valid_int(end)):
                continue
            if start < 0 or end <= start or end > doc_len:
                continue
            exact_slice = canonical_doc[start:end]
            if text is not None and str(text) != exact_slice:
                continue
            if not exact_slice:
                continue
            candidates.append((start, end, exact_slice))

        candidates.sort(key=lambda item: (item[0], item[1]))
        valid_segments: list[dict[str, Any]] = []
        prev_end = -1
        for start, end, exact_slice in candidates:
            if start < prev_end:
                continue
            valid_segments.append({"start": start, "end": end, "text": exact_slice})
            prev_end = end

        exported.append(
            {
                "term_id": term_id_str,
                "segments": valid_segments,
            }
        )
    return exported


def _collect_findings(
    findings: LLMExtractionResult | Sequence[Any],
) -> tuple[list[tuple[Any, str]], str | None]:
    if isinstance(findings, LLMExtractionResult):
        collected: list[tuple[Any, str]] = []
        collected.extend((item, "proband") for item in findings.terms)
        collected.extend(
            (item, "family_history") for item in findings.family_history_findings
        )
        collected.extend((item, "other") for item in findings.other_person_findings)
        return collected, findings.meta.llm_model
    return [(item, "proband") for item in findings], None


def _format_created_at(created_at: datetime | None) -> str:
    if created_at is None:
        resolved = datetime.now(UTC)
    elif created_at.tzinfo is None:
        resolved = created_at.replace(tzinfo=UTC)
    else:
        resolved = created_at.astimezone(UTC)
    return resolved.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_phentrieve_version(phentrieve_version: str | None) -> str:
    if phentrieve_version is not None:
        return phentrieve_version
    try:
        from phentrieve import __version__

        return __version__
    except ImportError:
        return "unknown"


def export_to_ontospan_document(
    findings: LLMExtractionResult | Sequence[Any],
    document_text: str,
    *,
    document_id: str,
    language: str = "en",
    annotation_set_id: str | None = None,
    subject_id: str | None = None,
    hpo_version: str = "unknown",
    finding_roots: Sequence[str] = (PHENOTYPE_ROOT,),
    actor_id: str = "phentrieve",
    actor_name: str = "phentrieve",
    phentrieve_version: str | None = None,
    model: str | None = None,
    run_id: str | None = None,
    created_at: datetime | None = None,
    layer_id: str = "hpo",
    profile: str = "hpo-phenotype",
    ontology_id: str = "hp",
) -> dict[str, Any]:
    """Export Phentrieve findings into an ontospan 1.2.0 AnnotatedDocument dict."""
    canonical_doc = canonicalize_text(document_text)
    doc_sha256 = compute_text_sha256(canonical_doc)
    collected_findings, inferred_model = _collect_findings(findings)
    resolved_model = model if model is not None else inferred_model
    resolved_version = _resolve_phentrieve_version(phentrieve_version)

    actor: dict[str, Any] = {
        "actor_id": actor_id,
        "kind": "tool",
        "name": actor_name,
        "version": resolved_version,
    }
    if resolved_model is not None:
        actor["model"] = resolved_model
    if run_id is not None:
        actor["run_id"] = run_id

    annotations: list[dict[str, Any]] = []
    for index, (finding, default_experiencer) in enumerate(collected_findings, start=1):
        term_id = _get_field(finding, "term_id", "hpo_id", "id")
        if not term_id:
            continue
        mentions = _extract_mentions(finding, canonical_doc)
        document_level = len(mentions) == 0
        modifiers = _extract_modifiers(finding, canonical_doc)
        annotations.append(
            {
                "annotation_id": f"ann-{index:04d}",
                "layer_id": layer_id,
                "term_id": str(term_id).strip(),
                "axes": {
                    "assertion": {
                        "value": _map_assertion(finding),
                        "basis": "chosen",
                    },
                    "experiencer": {
                        "value": _map_experiencer(finding, default_experiencer),
                        "basis": "chosen",
                    },
                    "temporality": {
                        "value": _map_temporality(finding),
                        "basis": "chosen",
                    },
                },
                "mentions": mentions,
                "document_level": document_level,
                "modifiers": modifiers,
                "provenance": {
                    "actor": actor,
                    "method": "tool",
                    "source_refs": [],
                    "notes": [],
                },
            }
        )

    document_block: dict[str, Any] = {
        "document_id": document_id,
        "language": language,
        "text": canonical_doc,
        "text_sha256": doc_sha256,
    }
    if subject_id is not None:
        document_block["subject_id"] = subject_id

    return {
        "schema_version": ONTOSPAN_SCHEMA_VERSION,
        "artifact_type": ONTOSPAN_ARTIFACT_TYPE,
        "document": document_block,
        "annotation_set": {
            "annotation_set_id": annotation_set_id or f"phentrieve-{uuid.uuid4()}",
            "set_kind": "tool_output",
            "actors": [actor],
            "created_at": _format_created_at(created_at),
            "layers": [
                {
                    "layer_id": layer_id,
                    "profile": profile,
                    "ontology": {
                        "id": ontology_id,
                        "version": hpo_version,
                    },
                    "allowed_roots": list(finding_roots),
                }
            ],
            "annotations": annotations,
        },
    }


def serialize_ontospan_document(doc: dict[str, Any]) -> str:
    """Serialize an ontospan AnnotatedDocument dict to stable UTF-8 JSON."""
    return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "ONTOSPAN_ARTIFACT_TYPE",
    "ONTOSPAN_SCHEMA_VERSION",
    "canonicalize_text",
    "compute_text_sha256",
    "export_to_ontospan_document",
    "serialize_ontospan_document",
]
