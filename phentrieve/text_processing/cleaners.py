"""
Text cleaning and normalization utilities for Phentrieve.

This module provides functions for pre-processing text before chunking
and assertion detection, ensuring consistent text formatting.
"""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from phentrieve.utils import LINE_BREAK_HYPHEN, SOFT_HYPHEN

_HYPHENATION_OR_SOFT_HYPHEN = re.compile(f"{LINE_BREAK_HYPHEN.pattern}|{SOFT_HYPHEN}")


@dataclass(frozen=True)
class RejoinedText:
    """Text with characters removed, and the map back to the original text.

    ``cut_positions[i]`` is the position in ``text`` where original characters
    were removed; ``removed_before[i]`` is the number of original characters
    removed up to and including that cut.
    """

    text: str
    cut_positions: list[int] = field(default_factory=list)
    removed_before: list[int] = field(default_factory=list)

    def original_start(self, position: int) -> int:
        """Original position of the character at ``position`` (inclusive start)."""
        cuts = bisect_right(self.cut_positions, position)
        return position + (self.removed_before[cuts - 1] if cuts else 0)

    def original_end(self, position: int) -> int:
        """Original exclusive end for a span that ends at ``position``.

        Characters removed right after the span are not part of it.
        """
        cuts = bisect_left(self.cut_positions, position)
        return position + (self.removed_before[cuts - 1] if cuts else 0)


def rejoin_line_break_hyphens(text: str) -> RejoinedText:
    """Rejoin words split by a hyphen at a line break; drop soft hyphens.

    "Entwick-\nlung" becomes "Entwicklung". The hyphen must follow a letter and
    the next line must start with a lowercase letter, so a compound that happens
    to break at its hyphen ("Down-\nSyndrom") is kept. Soft hyphens (U+00AD) are
    invisible and are always dropped; one at a line break joins the word under
    the same rule. Nothing else is changed, so every other character keeps its
    relative order and the result maps back to ``text`` exactly.
    """
    parts: list[str] = []
    cut_positions: list[int] = []
    removed_before: list[int] = []
    copied_until = 0
    length = 0
    removed = 0
    for match in _HYPHENATION_OR_SOFT_HYPHEN.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start else ""
        joins = (
            end - start > 1
            and (before.isalpha() or (before and unicodedata.combining(before)))
            and end < len(text)
            and text[end].islower()
        )
        if not joins:
            if text[start] != SOFT_HYPHEN:
                continue
            end = start + 1
        parts.append(text[copied_until:start])
        length += start - copied_until
        removed += end - start
        cut_positions.append(length)
        removed_before.append(removed)
        copied_until = end
    parts.append(text[copied_until:])
    return RejoinedText("".join(parts), cut_positions, removed_before)


def normalize_line_endings(text: str) -> str:
    """
    Normalize different line ending styles to Unix-style newlines.

    Args:
        text: Input text with potentially mixed line endings

    Returns:
        Text with normalized line endings (all '\n')
    """
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_internal_newlines_and_extra_spaces(text_chunk: str) -> str:
    """
    Replace internal newlines with spaces and normalize whitespace.

    This function:
    1. Replaces any newlines surrounded by optional whitespace with a single space
    2. Replaces sequences of 2+ spaces with a single space
    3. Strips leading/trailing whitespace

    Args:
        text_chunk: Text chunk to clean

    Returns:
        Cleaned text with normalized spacing
    """
    if not text_chunk:
        return ""
    cleaned = re.sub(r"\s*\n\s*", " ", text_chunk)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()
