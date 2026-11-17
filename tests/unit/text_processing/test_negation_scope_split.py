"""#521: a negation cue negates only its scope inside a chunk.

A chunk has one status, so a cue used to negate everything chunking left beside
it. The text outside the cue's scope now becomes its own chunk. The first four
sentences are the ones of the issue; all others are invented.
"""

import warnings

import pytest

from phentrieve.config import DEFAULT_ASSERTION_CONFIG
from phentrieve.text_processing.assertion_detection import get_spacy_model
from phentrieve.text_processing.pipeline import (
    TextProcessingPipeline,
    _content_span,
    _function_words,
)

pytestmark = pytest.mark.skipif(
    get_spacy_model("en") is None or get_spacy_model("de") is None,
    reason="needs the en and de spaCy models",
)

_CFG = [
    {"type": "paragraph"},
    {"type": "sentence"},
    {"type": "fine_grained_punctuation"},
    {"type": "conjunction"},
    {"type": "final_chunk_cleaner"},
]


def _pipeline(language: str) -> TextProcessingPipeline:
    warnings.filterwarnings("ignore")
    return TextProcessingPipeline(
        language=language,
        chunking_pipeline_config=_CFG,
        assertion_config=dict(DEFAULT_ASSERTION_CONFIG),
    )


@pytest.fixture(scope="module")
def en() -> TextProcessingPipeline:
    return _pipeline("en")


@pytest.fixture(scope="module")
def de() -> TextProcessingPipeline:
    return _pipeline("de")


def _chunks(pipeline: TextProcessingPipeline, text: str) -> list[tuple[str, str]]:
    """(chunk text, status) per chunk; also checks the chunk positions."""
    chunks = pipeline.process(text, include_positions=True)
    for chunk in chunks:
        assert text[chunk["start_char"] : chunk["end_char"]] == chunk["text"]
    return [(chunk["text"], chunk["status"].value) for chunk in chunks]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Seizures since age 2, but free of fever.",
            [("Seizures since age 2", "affirmed"), ("free of fever", "negated")],
        ),
        (
            "Seizures since age 2 and lack of fever.",
            [("Seizures since age 2", "affirmed"), ("lack of fever", "negated")],
        ),
        (
            "Seizures since age 2 with absence of fever.",
            [("Seizures since age 2", "affirmed"), ("absence of fever", "negated")],
        ),
        (
            "Seizures since age 2 with no fever.",
            [("Seizures since age 2", "affirmed"), ("no fever", "negated")],
        ),
    ],
)
def test_issue_sentences(en, text, expected) -> None:
    assert _chunks(en, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # forward cue: the text before it is out of scope
        (
            "Short stature with no evidence of scoliosis.",
            [("Short stature", "affirmed"), ("no evidence of scoliosis", "negated")],
        ),
        (
            "Patient has tremor with no rigidity.",
            [("Patient has tremor", "affirmed"), ("no rigidity", "negated")],
        ),
        # backward cue: the text after it is out of scope
        (
            "Fever is absent with persistent cough.",
            [("Fever is absent", "negated"), ("persistent cough", "affirmed")],
        ),
        # a scope terminator ends the scope
        (
            "No abnormalities except mild ptosis.",
            [("abnormalities", "negated"), ("mild ptosis", "affirmed")],
        ),
        (
            "Ataxia although scoliosis was excluded.",
            [("Ataxia", "affirmed"), ("scoliosis was excluded", "negated")],
        ),
        # the part before the cue keeps its own status
        (
            "Scoliosis cannot be ruled out with no fever.",
            [("Scoliosis cannot be ruled out", "affirmed"), ("no fever", "negated")],
        ),
    ],
)
def test_text_outside_the_scope_gets_its_own_chunk(en, text, expected) -> None:
    assert _chunks(en, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Kleinwuchs seit der Geburt ohne Skoliose.",
            [("Kleinwuchs seit der Geburt", "affirmed"), ("ohne Skoliose", "negated")],
        ),
        (
            "Eine Skoliose wurde ausgeschlossen bei bestehender Ataxie.",
            [
                ("Skoliose wurde ausgeschlossen", "negated"),
                ("bestehender Ataxie", "affirmed"),
            ],
        ),
    ],
)
def test_german_scope(de, text, expected) -> None:
    assert _chunks(de, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # the whole chunk is in scope
        "No fever.",
        "Fever is absent.",
        "Scoliosis with ataxia was ruled out.",
        "No fever with no rash.",
        # only function words outside the scope
        "And with no fever.",
        "There is no nystagmus.",
        # cue and the text on its other side are one clause
        "Seizures were not observed.",
        "The scan showed no lesion.",
        "Patient has no fever.",
        "She denies headaches.",
    ],
)
def test_chunk_in_scope_stays_one_negated_chunk(en, text) -> None:
    chunks = _chunks(en, text)

    assert [status for _, status in chunks] == ["negated"], chunks


@pytest.mark.parametrize(
    "text",
    [
        "Kein Fieber.",
        "Krampfanfaelle wurden nicht beobachtet.",
        "Ausgeschlossen wurde eine Epilepsie.",
        "Der Patient hat kein Fieber.",
        "Fieber besteht nicht.",
    ],
)
def test_german_chunk_in_scope_stays_one_negated_chunk(de, text) -> None:
    chunks = _chunks(de, text)

    assert [status for _, status in chunks] == ["negated"], chunks


@pytest.mark.parametrize(
    "text",
    [
        "Echocardiography without pathological findings.",
        "Abdominal ultrasound without abnormalities.",
        "Brain MRI with no abnormality.",
        "Cranial MRI with no signs of pathology.",
        "Abdominal ultrasound without evidence of abnormalities.",
        # the examination is the agent of the excluding verb
        "Seizures were excluded by EEG.",
        "Epilepsy was ruled out by EEG.",
    ],
)
def test_examination_with_normal_result_stays_one_negated_chunk(en, text) -> None:
    assert _chunks(en, text) == [(text.rstrip("."), "negated")]


@pytest.mark.parametrize(
    "text",
    [
        "Sonographie der Nieren ohne Auff\u00e4lligkeiten.",
        "Echokardiographie ohne pathologischen Befund.",
        "EEG ohne relevante Auff\u00e4lligkeiten.",
        "Sch\u00e4del-MRT ohne Besonderheiten.",
        "Sonographie des Abdomens ohne Hinweis auf Pathologie.",
    ],
)
def test_german_examination_with_normal_result_stays_one_negated_chunk(
    de, text
) -> None:
    assert _chunks(de, text) == [(text.rstrip("."), "negated")]


def test_generic_noun_beside_a_finding_still_splits(en, de) -> None:
    assert _chunks(en, "Short stature with no signs of scoliosis.") == [
        ("Short stature", "affirmed"),
        ("no signs of scoliosis", "negated"),
    ]
    assert _chunks(de, "Kleinwuchs ohne Hinweis auf Skoliose.") == [
        ("Kleinwuchs", "affirmed"),
        ("ohne Hinweis auf Skoliose", "negated"),
    ]


def test_finding_after_a_terminator_still_splits_from_a_normal_result(en) -> None:
    text = "Echocardiography without abnormalities except mild mitral regurgitation."

    assert _chunks(en, text) == [
        ("Echocardiography without abnormalities", "negated"),
        ("mild mitral regurgitation", "affirmed"),
    ]


def test_split_positions_index_the_text_with_line_break_hyphens(de) -> None:
    text = "Entwick-\nlungsverzoegerung besteht. Kleinwuchs seit der Ge-\nburt ohne Sko-\nliose."

    chunks = de.process(text, include_positions=True)

    assert [(chunk["text"], chunk["status"].value) for chunk in chunks][1:] == [
        ("Kleinwuchs seit der Geburt", "affirmed"),
        ("ohne Skoliose", "negated"),
    ]
    for chunk in chunks:
        original = text[chunk["start_char"] : chunk["end_char"]]
        assert original.replace("-\n", "") == chunk["text"]
    assert text[chunks[1]["start_char"] :].startswith("Kleinwuchs seit der Ge-\nburt")
    assert text[: chunks[2]["end_char"]].endswith("ohne Sko-\nliose")


def test_negated_lists_are_unchanged(en, de) -> None:
    assert _chunks(en, "No hepatomegaly or splenomegaly.") == [
        ("hepatomegaly", "negated"),
        ("splenomegaly", "negated"),
    ]
    assert _chunks(en, "No fever, rash or vomiting.") == [
        ("fever", "negated"),
        ("rash", "negated"),
        ("vomiting", "negated"),
    ]
    assert _chunks(de, "Keine Krampfanfaelle, Ataxie oder Hoerminderung.") == [
        ("Keine Krampfanfaelle", "negated"),
        ("Ataxie", "negated"),
        ("Hoerminderung", "negated"),
    ]


def test_list_after_a_split_cue_is_still_negated(en) -> None:
    assert _chunks(en, "Muscle weakness without ataxia, tremor or rigidity.") == [
        ("Muscle weakness", "affirmed"),
        ("without ataxia", "negated"),
        ("tremor", "negated"),
        ("rigidity", "negated"),
    ]


def test_split_keeps_details_and_optional_positions(en) -> None:
    whole, part = en.process("No fever. Tremor without rigidity.")[1:]

    assert "start_char" not in whole and "start_char" not in part
    assert part["assertion_details"]["negated_scope_texts"] == ["rigidity"]
    assert whole["assertion_details"]["final_status"] == "affirmed"


def test_content_span_drops_function_words_at_the_edges() -> None:
    leading, trailing = _function_words("en")

    assert _content_span("and the tremor with", 0, 19, leading, trailing) == (8, 14)
    assert _content_span("and with the", 0, 12, leading, trailing) is None
    assert _content_span("since 2", 0, 7, frozenset({"since"}), frozenset()) is None
