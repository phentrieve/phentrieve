"""Document offsets for phase-1 evidence, computed in code.

The model quotes ``evidence_text`` and cites ``chunk_ids``; character offsets
it reports itself are unreliable. The span is the evidence text located inside
the cited chunks, whose document positions are known from preprocessing.
Matching tolerates typography a model may alter (quotes, dashes, repeated or
non-breaking whitespace, letter case, Unicode composition, soft hyphens and
hyphenation at line breaks); offsets always refer to the original document.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from phentrieve.utils import LINE_BREAK_HYPHEN as _LINE_BREAK_HYPHEN
from phentrieve.utils import SOFT_HYPHEN as _SOFT_HYPHEN

OffsetBasis = Literal[
    "evidence_in_chunk", "evidence_in_chunk_hull", "chunk_span", "unresolved"
]

_TYPOGRAPHY = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


_QUOTED_LINE_BREAK_HYPHEN = re.compile(r"(?<=\w)- (?=\w)")

# Separators the phase-1 chunk index shows after the chunk they follow.
_SEPARATORS = " \t\r\n.,;:!?)]-+"
# Separators stripped when trimming a carved modifier from a finding span edge.
_CARVE_SEPARATORS = " \t\r\n,;:-"
# Dropped text up to here belongs to the chunk before it ("<finding>: no").
_LINE_OR_SENTENCE_END = re.compile(r"[\r\n]|[.!?](?=\s|$)")


@dataclass(frozen=True)
class EvidenceSpan:
    start: int | None
    end: int | None
    basis: OffsetBasis


def _normalized(text: str) -> tuple[str, list[int], list[int]]:
    """Normalized text; character i covers original [starts[i], ends[i])."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        if index > 0 and (
            text[index - 1].isalnum() or unicodedata.combining(text[index - 1])
        ):
            hyphenation = _LINE_BREAK_HYPHEN.match(text, index)
            if hyphenation:
                index = hyphenation.end()
                continue
        if text[index] == _SOFT_HYPHEN:
            index += 1
            continue
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            characters.append(" ")
            starts.append(index)
            ends.append(end)
            index = end
            continue
        # A base character and its combining marks are composed and case-folded
        # together; every resulting character maps to the whole cluster.
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = unicodedata.normalize("NFC", text[index:end]).casefold()
        for character in cluster.translate(_TYPOGRAPHY):
            characters.append(character)
            starts.append(index)
            ends.append(end)
        index = end
    return "".join(characters), starts, ends


def is_located(start: object, end: object) -> bool:
    """Whether a chunk has a document position.

    Chunks the text pipeline could not locate carry ``-1`` (or no) positions.
    """
    return isinstance(start, int) and isinstance(end, int) and 0 <= start <= end


@dataclass(frozen=True)
class RenderedRegion:
    """Document region the phase-1 chunk index shows for one located chunk.

    ``index`` is the chunk's position in the input sequence. The region runs
    from the source text that cleanup dropped before the chunk (``start``) to
    the separators that follow it (``end``); ``chunk_start``/``chunk_end`` are
    the chunk's own positions.
    """

    index: int
    start: int
    end: int
    chunk_start: int
    chunk_end: int


def rendered_regions(
    text: str, spans: Sequence[tuple[object, object]]
) -> list[RenderedRegion]:
    """Walk the gaps between located chunks, given as ``(start, end)`` pairs.

    Dropped text after a chunk up to the first line break or sentence end is
    shown after that chunk, as is the text after the last chunk (#487); the
    rest of a gap is shown before the next chunk, without the separators that
    are shown after the previous chunk instead. Next to an unlocated chunk only
    separators are shown, because that gap still holds the unlocated chunk's
    own text.
    """
    located: list[tuple[int, int, int]] = [
        (index, start, end)
        for index, (start, end) in enumerate(spans)
        if isinstance(start, int) and isinstance(end, int) and is_located(start, end)
    ]
    located_indices = {index for index, _, _ in located}
    regions: list[RenderedRegion] = []
    previous_end = 0
    for position, (index, start, end) in enumerate(located):
        if index > 0 and index - 1 not in located_indices:
            region_start = start
        else:
            gap_start = min(previous_end, start)
            gap = text[gap_start:start]
            region_start = start - len(gap.lstrip(_SEPARATORS))
        next_start = (
            located[position + 1][1] if position + 1 < len(located) else len(text)
        )
        following = text[end : max(end, next_start)]
        region_end = end + len(following) - len(following.lstrip(_SEPARATORS))
        if index + 1 == len(spans) or (
            position + 1 < len(located) and located[position + 1][0] == index + 1
        ):
            boundary = _LINE_OR_SENTENCE_END.search(following)
            if boundary is None:
                cut = len(following) if index + 1 == len(spans) else 0
            else:
                cut = boundary.end() if boundary.group() in ".!?" else boundary.start()
            region_end = max(region_end, end + len(following[:cut].rstrip()))
        regions.append(RenderedRegion(index, region_start, region_end, start, end))
        previous_end = max(previous_end, region_end)
    return regions


def _adjacent_runs(chunk_ids: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for chunk_id in chunk_ids:
        if runs and chunk_id == runs[-1][-1] + 1:
            runs[-1].append(chunk_id)
        else:
            runs.append([chunk_id])
    return runs


class EvidenceLocator:
    """Locates evidence in one document.

    Each search region is normalized once and reused for every finding that
    cites it, so long notes with many findings do not re-normalize the same
    chunks over and over.
    """

    def __init__(
        self,
        document: str,
        chunk_spans: Mapping[int, tuple[int, int]],
        *,
        search_spans: Mapping[int, tuple[int, int]] | None = None,
    ) -> None:
        self._document = document
        self._chunk_spans = chunk_spans
        self._search_spans = search_spans or chunk_spans
        self._normalized_regions: dict[
            tuple[int, int], tuple[str, list[int], list[int]]
        ] = {}

    def _find(self, evidence: str, start: int, end: int) -> tuple[int, int] | None:
        """First match of ``evidence`` lying within document[start:end]."""
        needle = _normalized(evidence.strip())[0]
        if not needle:
            return None
        region = self._normalized_regions.get((start, end))
        if region is None:
            region = _normalized(self._document[start:end])
            self._normalized_regions[(start, end)] = region
        haystack, starts, ends = region
        position = haystack.find(needle)
        if position == -1:
            # The chunk index renders "hemo-\nglobin" as "hemo- globin", so a quote
            # can keep a hyphen that the normalized document no longer has.
            needle = _QUOTED_LINE_BREAK_HYPHEN.sub("", needle)
            position = haystack.find(needle)
        if position == -1:
            return None
        return start + starts[position], start + ends[position + len(needle) - 1]

    def _find_all(self, evidence: str, start: int, end: int) -> list[tuple[int, int]]:
        """All matches of ``evidence`` lying within document[start:end]."""
        needle = _normalized(evidence.strip())[0]
        if not needle:
            return []
        region = self._normalized_regions.get((start, end))
        if region is None:
            region = _normalized(self._document[start:end])
            self._normalized_regions[(start, end)] = region
        haystack, starts, ends = region

        def _collect(pattern: str) -> list[tuple[int, int]]:
            if not pattern:
                return []
            spans: list[tuple[int, int]] = []
            pos = 0
            while pos <= len(haystack) - len(pattern):
                found = haystack.find(pattern, pos)
                if found == -1:
                    break
                span = (
                    start + starts[found],
                    start + ends[found + len(pattern) - 1],
                )
                if not spans or spans[-1] != span:
                    spans.append(span)
                pos = found + 1
            return spans

        matches = _collect(needle)
        if not matches:
            collapsed = _QUOTED_LINE_BREAK_HYPHEN.sub("", needle)
            if collapsed != needle:
                matches = _collect(collapsed)
        return matches

    def locate(
        self, evidence_text: str | None, chunk_ids: Sequence[int]
    ) -> EvidenceSpan:
        """Document span of one piece of evidence (see ``locate_evidence``)."""
        chunk_spans = self._chunk_spans
        ids = [
            c
            for c in sorted(set(chunk_ids))
            if c in chunk_spans and is_located(*chunk_spans[c])
        ]
        if not ids:
            return EvidenceSpan(None, None, "unresolved")
        runs = _adjacent_runs(ids)
        if evidence_text:
            regions = {c: self._search_spans.get(c, chunk_spans[c]) for c in ids}
            for chunk_id in ids:
                found = self._find(evidence_text, *regions[chunk_id])
                if found:
                    return EvidenceSpan(*found, "evidence_in_chunk")
            for run in runs:
                if len(run) < 2:
                    continue
                hull = (
                    min(regions[c][0] for c in run),
                    max(regions[c][1] for c in run),
                )
                found = self._find(evidence_text, *hull)
                if found:
                    return EvidenceSpan(*found, "evidence_in_chunk_hull")
        run = runs[0]
        return EvidenceSpan(
            chunk_spans[run[0]][0], chunk_spans[run[-1]][1], "chunk_span"
        )

    def locate_nearest(
        self,
        evidence_text: str | None,
        chunk_ids: Sequence[int],
        *,
        anchor_span: tuple[int, int] | None = None,
        include_neighbors: bool = True,
        require_word_boundaries: bool = True,
    ) -> EvidenceSpan:
        """Locate ``evidence_text`` verbatim, preferring the match nearest ``anchor_span``.

        Searches all occurrences in the cited chunks (then hulls of adjacent
        cited chunks, then immediate neighbor chunks when ``include_neighbors``
        is True). Unlike ``locate()``, never falls back to ``chunk_span`` so a
        modifier quote that cannot be found verbatim stays ``unresolved``.
        When ``require_word_boundaries`` is True (default), mid-word substring
        matches (for example an uninflected stem inside an inflected token) are
        rejected so modifier spans never split words.
        """
        if not evidence_text or not evidence_text.strip():
            return EvidenceSpan(None, None, "unresolved")

        chunk_spans = self._chunk_spans
        ids = [
            c
            for c in sorted(set(chunk_ids))
            if c in chunk_spans and is_located(*chunk_spans[c])
        ]
        if not ids:
            return EvidenceSpan(None, None, "unresolved")

        regions = {c: self._search_spans.get(c, chunk_spans[c]) for c in ids}
        matches: list[tuple[int, int, OffsetBasis]] = []
        seen_spans: set[tuple[int, int]] = set()

        def _is_bounded(entry: tuple[int, int, OffsetBasis]) -> bool:
            return _is_word_boundary(self._document, entry[0]) and _is_word_boundary(
                self._document, entry[1]
            )

        # If the modifier text appears inside the finding's own anchor span only
        # as a compound/sub-word (e.g. "mild" inside "mildheadache"), do not
        # jump to an unrelated bounded occurrence elsewhere in the chunk; leave
        # it unresolved so modifier.segments stays empty.
        if require_word_boundaries and anchor_span is not None:
            a_start_check, a_end_check = anchor_span
            in_anchor_hits = self._find_all(evidence_text, a_start_check, a_end_check)
            if in_anchor_hits and not any(
                _is_word_boundary(self._document, s)
                and _is_word_boundary(self._document, e)
                for s, e in in_anchor_hits
            ):
                return EvidenceSpan(None, None, "unresolved")

        def _add_matches(
            found_spans: Sequence[tuple[int, int]], basis: OffsetBasis
        ) -> None:
            for span_start, span_end in found_spans:
                if require_word_boundaries and not (
                    _is_word_boundary(self._document, span_start)
                    and _is_word_boundary(self._document, span_end)
                ):
                    continue
                if (span_start, span_end) not in seen_spans:
                    seen_spans.add((span_start, span_end))
                    matches.append((span_start, span_end, basis))

        for chunk_id in ids:
            _add_matches(
                self._find_all(evidence_text, *regions[chunk_id]),
                "evidence_in_chunk",
            )

        if not matches:
            for run in _adjacent_runs(ids):
                if len(run) < 2:
                    continue
                hull = (
                    min(regions[c][0] for c in run),
                    max(regions[c][1] for c in run),
                )
                _add_matches(
                    self._find_all(evidence_text, *hull),
                    "evidence_in_chunk_hull",
                )

        if not matches and include_neighbors:
            cited_set = set(ids)
            neighbor_ids = [
                cid
                for cid in (min(ids) - 1, max(ids) + 1)
                if cid in chunk_spans
                and cid not in cited_set
                and is_located(*chunk_spans[cid])
            ]
            for cid in neighbor_ids:
                region = self._search_spans.get(cid, chunk_spans[cid])
                _add_matches(
                    self._find_all(evidence_text, *region),
                    "evidence_in_chunk",
                )

        if not matches:
            return EvidenceSpan(None, None, "unresolved")

        if anchor_span is None:
            best = matches[0]
            return EvidenceSpan(best[0], best[1], best[2])

        a_start, a_end = anchor_span

        def _distance_key(
            entry: tuple[int, int, OffsetBasis],
        ) -> tuple[int, int, int, int]:
            m_start, m_end, _ = entry
            if m_end <= a_start:
                gap = a_start - m_end
            elif m_start >= a_end:
                gap = m_start - a_end
            else:
                gap = 0
            center_distance = abs((m_start + m_end) - (a_start + a_end))
            return (0 if _is_bounded(entry) else 1, gap, center_distance, m_start)

        best = min(matches, key=_distance_key)
        return EvidenceSpan(best[0], best[1], best[2])


_INTRA_WORD_HYPHENS = "-\u2010\u2011"


def _is_word_boundary(document: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(document):
        return True
    if unicodedata.combining(document[pos]):
        return False
    prev_idx = pos - 1
    while prev_idx > 0 and unicodedata.combining(document[prev_idx]):
        prev_idx -= 1
    prev_ch = document[prev_idx]
    curr_ch = document[pos]
    if prev_ch.isalnum() and curr_ch.isalnum():
        return False
    # Treat intra-word hyphens (e.g. "left" inside "left-sided") as non-boundary
    # so modifier carving never splits hyphenated compounds into fragments.
    if (
        curr_ch in _INTRA_WORD_HYPHENS
        and prev_ch.isalnum()
        and pos + 1 < len(document)
        and document[pos + 1].isalnum()
    ):
        return False
    if (
        prev_ch in _INTRA_WORD_HYPHENS
        and curr_ch.isalnum()
        and prev_idx > 0
        and document[prev_idx - 1].isalnum()
    ):
        return False
    return True


def _is_carve_separator(ch: str) -> bool:
    return ch.isspace() or ch in _CARVE_SEPARATORS


def carve_modifier_span(
    document: str,
    finding_span: tuple[int, int],
    modifier_span: tuple[int, int],
) -> tuple[tuple[int, int], bool]:
    """Enforce disjoint finding and modifier spans (A80/A101).

    Returns ``(updated_finding_span, keep_modifier_segment)``:
    - No overlap: keeps both spans untouched and returns ``True``.
    - Leading-edge overlap on a word boundary: trims the finding's leading edge
      past the modifier and adjacent separators, returning ``True``.
    - Trailing-edge overlap on a word boundary: trims the finding's trailing
      edge before the modifier and adjacent separators, returning ``True``.
    - Interior overlap, full coverage, or mid-word split: leaves the finding
      span intact and returns ``False`` so the caller leaves
      ``modifier.segments = []``.
    """
    f_start, f_end = finding_span
    m_start, m_end = modifier_span

    if f_start >= f_end or m_start >= m_end:
        return finding_span, False

    if m_end <= f_start or m_start >= f_end:
        return finding_span, True

    if m_start <= f_start and m_end < f_end and _is_word_boundary(document, m_end):
        new_start = m_end
        while new_start < f_end and _is_carve_separator(document[new_start]):
            new_start += 1
        if new_start < f_end:
            return (new_start, f_end), True

    if f_start < m_start and f_end <= m_end and _is_word_boundary(document, m_start):
        new_end = m_start
        while new_end > f_start and _is_carve_separator(document[new_end - 1]):
            new_end -= 1
        if f_start < new_end:
            return (f_start, new_end), True

    return finding_span, False


def locate_evidence(
    document: str,
    evidence_text: str | None,
    chunk_ids: Sequence[int],
    chunk_spans: Mapping[int, tuple[int, int]],
    *,
    search_spans: Mapping[int, tuple[int, int]] | None = None,
) -> EvidenceSpan:
    """Document span of one piece of evidence.

    Tried in order: the evidence inside one cited chunk (first in chunk-id
    order), inside the hull of a run of adjacent cited chunks (evidence that
    crosses chunk borders; a hull never spans an uncited chunk), and finally
    the first run of adjacent cited chunks themselves. ``search_spans`` may
    widen a chunk's search region (for example by text that cleanup dropped
    before it); the fallback always uses ``chunk_spans``. Chunks without a
    located position are ignored. To locate many findings in one document,
    reuse one ``EvidenceLocator``.
    """
    return EvidenceLocator(document, chunk_spans, search_spans=search_spans).locate(
        evidence_text, chunk_ids
    )
