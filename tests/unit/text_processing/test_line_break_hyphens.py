"""#519: words split by a hyphen at a line break in the standard pipeline.

Text pasted from PDFs carries "Entwick-\\nlung"; the fragments were chunked as
they were. All sentences are invented.
"""

import pytest

from phentrieve.text_processing.cleaners import (
    RejoinedText,
    rejoin_line_break_hyphens,
)
from phentrieve.text_processing.pipeline import TextProcessingPipeline

pytestmark = pytest.mark.unit

# Model-free chunking.
_CFG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "fine_grained_punctuation"},
    {"type": "conjunction"},
    {"type": "final_chunk_cleaner"},
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Entwick-\nlungsverzoegerung", "Entwicklungsverzoegerung"),
        ("Mikro-\r\nzephalie", "Mikrozephalie"),
        ("Mikro-\rzephalie", "Mikrozephalie"),
        ("Mikro- \t\nzephalie", "Mikrozephalie"),
        ("Mikro-\n   zephalie", "Mikrozephalie"),
        ("Mikro‐\nzephalie", "Mikrozephalie"),
        ("Mikro­\nzephalie", "Mikrozephalie"),
        ("Mikro­zephalie und Ata­xie", "Mikrozephalie und Ataxie"),
        ("hypo-\ntonia and macro-\ncephaly", "hypotonia and macrocephaly"),
    ],
)
def test_split_word_is_rejoined(text: str, expected: str) -> None:
    assert rejoin_line_break_hyphens(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "Down-\nSyndrom",  # compound: next line starts uppercase
        "Typ-\n2",  # next line starts with a digit
        "Vitamin B12-\nmangel",  # hyphen follows a digit
        "Befund -\nunauffaellig",  # dash, not a word hyphen
        "Mikro-zephalie",  # hyphen not at a line end
        "Hypotonie - Ataxie",
        "Hypotonie\nAtaxie",  # line break without hyphen
        "Mikro-\n",  # nothing follows
        "",
    ],
)
def test_other_text_is_untouched(text: str) -> None:
    rejoined = rejoin_line_break_hyphens(text)

    assert rejoined.text == text
    assert rejoined.cut_positions == []


def test_soft_hyphen_before_uppercase_line_keeps_the_line_break() -> None:
    assert rejoin_line_break_hyphens("Down­\nSyndrom").text == "Down\nSyndrom"


@pytest.mark.parametrize(
    "text",
    [
        "Pat. mit Entwick-\nlungsverzoegerung und Mikro-  \r\n  zephalie.",
        "Ata­xie, Mikro­\nzephalie, Down-\nSyndrom",
        "a­­b-\nc",
    ],
)
def test_every_position_maps_back_to_its_original_character(text: str) -> None:
    rejoined = rejoin_line_break_hyphens(text)

    assert rejoined.cut_positions  # something was removed
    for position, character in enumerate(rejoined.text):
        start = rejoined.original_start(position)
        assert text[start] == character
        assert rejoined.original_end(position + 1) == start + 1
    assert rejoined.original_end(0) == 0


def test_span_over_a_rejoined_word_covers_the_original_fragments() -> None:
    text = "mit Mikro-\nzephalie."
    rejoined = rejoin_line_break_hyphens(text)
    start = rejoined.text.index("Mikrozephalie")
    end = start + len("Mikrozephalie")

    original = text[rejoined.original_start(start) : rejoined.original_end(end)]

    assert original == "Mikro-\nzephalie"


def test_span_ending_at_a_cut_excludes_the_removed_characters() -> None:
    text = "Mikro-\nzephalie"
    rejoined = rejoin_line_break_hyphens(text)

    assert text[rejoined.original_start(0) : rejoined.original_end(5)] == "Mikro"
    assert text[rejoined.original_start(5) : rejoined.original_end(13)] == "zephalie"


def test_text_without_cuts_maps_to_itself() -> None:
    unchanged = RejoinedText("abc")

    assert (unchanged.original_start(1), unchanged.original_end(3)) == (1, 3)


def _pipeline(**kwargs: bool) -> TextProcessingPipeline:
    return TextProcessingPipeline(
        language="de",
        chunking_pipeline_config=_CFG,
        assertion_config={"disable": True},
        **kwargs,
    )


_NOTE = "Pat. mit Entwick-\r\nlungsverzoegerung, Mikro- \nzephalie und Down-\nSyndrom."


def test_pipeline_chunks_hold_the_rejoined_words() -> None:
    texts = " | ".join(chunk["text"] for chunk in _pipeline().process(_NOTE))

    assert "Entwicklungsverzoegerung" in texts
    assert "Mikrozephalie" in texts
    assert "Down-" in texts  # compound hyphen kept
    assert "Entwick-" not in texts


def test_pipeline_positions_index_the_original_text() -> None:
    chunks = _pipeline().process(_NOTE, include_positions=True)
    by_word = {
        word: chunk
        for word in ("Entwicklungsverzoegerung", "Mikrozephalie")
        for chunk in chunks
        if word in chunk["text"]
    }

    assert set(by_word) == {"Entwicklungsverzoegerung", "Mikrozephalie"}
    development = by_word["Entwicklungsverzoegerung"]
    source = _NOTE[development["start_char"] : development["end_char"]]
    assert "Entwick-\r\nlungsverzoegerung" in source
    microcephaly = by_word["Mikrozephalie"]
    source = _NOTE[microcephaly["start_char"] : microcephaly["end_char"]]
    assert "Mikro- \nzephalie" in source

    for chunk in chunks:
        assert 0 <= chunk["start_char"] < chunk["end_char"] <= len(_NOTE)
        # Removing the hyphenation from the original slice gives the chunk text.
        source = _NOTE[chunk["start_char"] : chunk["end_char"]]
        assert " ".join(rejoin_line_break_hyphens(source).text.split()) == chunk["text"]


def test_pipeline_positions_are_unchanged_without_hyphenation() -> None:
    note = "Hypotonie, Ataxie und Mikrozephalie."

    assert _pipeline().process(note, include_positions=True) == _pipeline(
        rejoin_line_break_hyphens=False
    ).process(note, include_positions=True)


def test_pipeline_can_keep_the_text_as_it_is() -> None:
    chunks = _pipeline(rejoin_line_break_hyphens=False).process(
        _NOTE, include_positions=True
    )
    texts = " | ".join(chunk["text"] for chunk in chunks)

    assert "Entwick-" in texts
    assert "Entwicklungsverzoegerung" not in texts
