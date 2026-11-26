"""Export Phentrieve findings into an ontospan 1.2.0 AnnotatedDocument."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from phentrieve.assertion_vocab import UNCERTAIN, canonicalize_assertion, is_excluded
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


def _build_raw_to_canonical_offsets(raw: str) -> list[int]:
    """Map each raw codepoint index (0..len(raw)) to its index in canonicalize_text(raw)."""
    raw_len = len(raw)
    offsets = [0] * (raw_len + 1)
    idx = 0
    while idx < raw_len and raw[idx] == _UTF8_BOM:
        offsets[idx] = 0
        idx += 1

    canon_pos = 0
    while idx < raw_len:
        if raw[idx] == "\r":
            offsets[idx] = canon_pos
            if idx + 1 < raw_len and raw[idx + 1] == "\n":
                offsets[idx + 1] = canon_pos
                idx += 2
            else:
                idx += 1
            canon_pos += 1
            continue
        end = idx + 1
        while end < raw_len and unicodedata.combining(raw[end]):
            end += 1
        cluster = unicodedata.normalize("NFC", raw[idx:end])
        for pos in range(idx, end):
            offsets[pos] = canon_pos
        canon_pos += len(cluster)
        idx = end
    offsets[raw_len] = canon_pos
    return offsets


def _resolve_span_in_canonical(
    start: int,
    end: int,
    declared_text: str | None,
    canonical_doc: str,
    raw_to_canonical: list[int] | None,
) -> tuple[int, int, str] | None:
    """Resolve ``(start, end)`` against ``canonical_doc``, remapping raw offsets if needed."""
    doc_len = len(canonical_doc)
    if 0 <= start < end <= doc_len:
        direct_slice = canonical_doc[start:end]
        if direct_slice and (
            declared_text is None
            or declared_text == direct_slice
            or canonicalize_text(declared_text) == direct_slice
        ):
            return start, end, direct_slice

    if raw_to_canonical is not None and 0 <= start < end < len(raw_to_canonical):
        c_start = raw_to_canonical[start]
        c_end = raw_to_canonical[end]
        if 0 <= c_start < c_end <= doc_len:
            mapped_slice = canonical_doc[c_start:c_end]
            if mapped_slice and (
                declared_text is None
                or declared_text == mapped_slice
                or canonicalize_text(declared_text) == mapped_slice
            ):
                return c_start, c_end, mapped_slice

    return None


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
    if normalized in _ASSERTION_MAP:
        return _ASSERTION_MAP[normalized]
    if is_excluded(normalized):
        return "absent"
    if canonicalize_assertion(normalized) == UNCERTAIN:
        return "uncertain"
    return "present"


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


def _extract_mentions(
    finding: Any,
    canonical_doc: str,
    raw_to_canonical: list[int] | None = None,
) -> list[dict[str, Any]]:
    # A ``negated_qualifier_derived`` finding inherits its source finding's
    # evidence records purely for chunk provenance (B3/F5); exporting them
    # would point at the source finding rather than the derived exclusion.
    if _get_field(finding, "match_method") == "negated_qualifier_derived":
        return []
    raw_records = _get_field(finding, "evidence_records", "spans", default=()) or ()
    mentions: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for record in raw_records:
        # A ``chunk_span`` fallback covers the whole cited chunk rather than a
        # located mention (#397); do not export it as an exact text mention.
        if _get_field(record, "offset_basis") == "chunk_span":
            continue
        start = _get_field(record, "start_char", "start")
        end = _get_field(record, "end_char", "end")
        if not (_is_valid_int(start) and _is_valid_int(end)):
            continue
        declared_raw = _get_field(record, "evidence_text", "text")
        declared_text = str(declared_raw) if declared_raw is not None else None
        resolved = _resolve_span_in_canonical(
            start, end, declared_text, canonical_doc, raw_to_canonical
        )
        if resolved is None:
            continue
        c_start, c_end, exact_slice = resolved
        span_key = (c_start, c_end)
        if span_key in seen_ranges:
            continue
        seen_ranges.add(span_key)
        mentions.append(
            {
                "segments": [{"start": c_start, "end": c_end, "text": exact_slice}],
                "cue_segments": [],
            }
        )
    return mentions


def _extract_modifiers(
    finding: Any,
    canonical_doc: str,
    *,
    mentions: Sequence[Mapping[str, Any]] = (),
    raw_to_canonical: list[int] | None = None,
) -> list[dict[str, Any]]:
    raw_modifiers = _get_field(finding, "modifiers", default=()) or ()
    exported: list[dict[str, Any]] = []
    seen_term_ids: set[str] = set()
    mention_ranges: list[tuple[int, int]] = [
        (int(seg["start"]), int(seg["end"]))
        for mention in mentions
        for seg in mention.get("segments", ())
    ]

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
            declared_text = str(text) if text is not None else None
            resolved = _resolve_span_in_canonical(
                start, end, declared_text, canonical_doc, raw_to_canonical
            )
            if resolved is None:
                continue
            c_start, c_end, exact_slice = resolved
            if any(
                c_start < m_end and m_start < c_end for m_start, m_end in mention_ranges
            ):
                continue
            candidates.append((c_start, c_end, exact_slice))

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


def _read_hpo_version_from_db() -> str:
    """Read HPO version from local SQLite DB without importing protobuf deps."""
    try:
        from phentrieve.config import DEFAULT_HPO_DB_FILENAME
        from phentrieve.data_processing.hpo_database import HPODatabase
        from phentrieve.utils import get_default_data_dir

        package_root = Path(__file__).resolve().parents[2]
        candidates = (
            get_default_data_dir() / DEFAULT_HPO_DB_FILENAME,
            Path.cwd() / "data" / DEFAULT_HPO_DB_FILENAME,
            package_root / "data" / DEFAULT_HPO_DB_FILENAME,
        )
        db_file = next((path for path in candidates if path.exists()), None)
        if db_file is None:
            return "unknown"
        with HPODatabase(db_file) as db:
            return db.get_metadata("hpo_version") or "unknown"
    except Exception:
        return "unknown"


def _resolve_hpo_version(hpo_version: str | None) -> str:
    if hpo_version is not None:
        return hpo_version
    resolved = _read_hpo_version_from_db()
    if not resolved or resolved == "unknown":
        return "unknown"
    # Match hp.json release identifiers (e.g. "2026-09-01") when DB metadata
    # carries a leading "v" prefix ("v2026-09-01").
    if len(resolved) > 1 and resolved[0] in ("v", "V") and resolved[1].isdigit():
        return resolved[1:]
    return resolved


def export_to_ontospan_document(
    findings: LLMExtractionResult | Sequence[Any],
    document_text: str,
    *,
    document_id: str,
    language: str = "en",
    annotation_set_id: str | None = None,
    subject_id: str | None = None,
    hpo_version: str | None = None,
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
    raw_to_canonical = (
        _build_raw_to_canonical_offsets(document_text)
        if document_text != canonical_doc
        else None
    )
    doc_sha256 = compute_text_sha256(canonical_doc)
    collected_findings, inferred_model = _collect_findings(findings)
    resolved_model = model if model is not None else inferred_model
    resolved_version = _resolve_phentrieve_version(phentrieve_version)
    resolved_hpo_version = _resolve_hpo_version(hpo_version)

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
        mentions = _extract_mentions(finding, canonical_doc, raw_to_canonical)
        document_level = len(mentions) == 0
        modifiers = _extract_modifiers(
            finding,
            canonical_doc,
            mentions=mentions,
            raw_to_canonical=raw_to_canonical,
        )
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
                        "version": resolved_hpo_version,
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
