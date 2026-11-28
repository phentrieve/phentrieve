from __future__ import annotations

from pathlib import Path

import pytest

from phentrieve.llm.prompts import loader
from phentrieve.llm.prompts.identity import (
    PROMPT_BEHAVIOR_VERSION,
    PROMPT_MODIFIERS_BEHAVIOR_VERSION,
    build_prompt_bundle_identity,
)
from phentrieve.llm.prompts.safety import (
    UNTRUSTED_DOCUMENT_INSTRUCTION,
    UNTRUSTED_PAYLOAD_INSTRUCTION,
)
from phentrieve.llm.types import AnnotationMode

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_prompt_loader_cache():
    loader.load_prompt_template.cache_clear()
    yield
    loader.load_prompt_template.cache_clear()


def test_modifier_extraction_prompt_loads_and_stays_compact() -> None:
    base = loader.get_prompt(AnnotationMode.TWO_PHASE, "en")
    mod = loader.get_modifier_extraction_prompt("en")

    assert base.version == "v3.5.0"
    assert mod.version == "v3.6.0"
    assert Path(mod.source_path).as_posix().endswith("two_phase/en_modifiers.yaml")
    assert mod.system_prompt.isascii()
    assert "few_shot_examples" not in Path(mod.source_path).read_text(encoding="utf-8")

    # Compactness guard: modifier prompt must stay <= 15% larger than en.yaml.
    growth_ratio = len(mod.system_prompt) / len(base.system_prompt)
    assert growth_ratio <= 1.15, f"Unexpected prompt growth: {growth_ratio:.3f}"

    assert '"modifiers"' in mod.system_prompt
    assert "severe anemia and thrombocytopenia" in mod.system_prompt
    assert "bilateral ptosis" in mod.system_prompt
    assert "mode of inheritance" in mod.system_prompt
    assert "{chunk_index}" in mod.user_prompt_template
    assert "{text}" in mod.user_prompt_template
    assert "{chunk_index}" in mod.grounded_user_prompt_template

    rendered_system = mod.render_system_prompt()
    assert UNTRUSTED_DOCUMENT_INSTRUCTION in rendered_system


def test_modifier_extraction_prompt_falls_back_to_english_for_other_languages() -> None:
    template = loader.get_modifier_extraction_prompt("de")

    assert template.version == "v3.6.0"
    assert Path(template.source_path).as_posix().endswith("two_phase/en_modifiers.yaml")


@pytest.mark.parametrize(
    ("getter", "expected_filename"),
    [
        (loader.get_modifier_mapping_prompt, "two_phase/en_modifier_mapping.yaml"),
        (
            loader.get_batch_modifier_mapping_prompt,
            "two_phase/en_modifier_mapping_batch.yaml",
        ),
    ],
)
def test_modifier_mapping_prompts_load_with_payload_safety_and_rules(
    getter, expected_filename: str
) -> None:
    template = getter("de")

    assert template.version == "v4.3.0"
    assert template.language == "de"
    assert Path(template.source_path).as_posix().endswith(expected_filename)
    assert template.system_prompt.isascii()
    assert "few_shot_examples" not in Path(template.source_path).read_text(
        encoding="utf-8"
    )

    rendered_system = template.render_system_prompt(language=template.language)
    assert UNTRUSTED_PAYLOAD_INSTRUCTION in rendered_system
    assert "The input may be in de" in rendered_system
    assert "finding_term" in rendered_system
    assert "No double encoding" in rendered_system
    assert "cohort frequency" in rendered_system
    assert "Never invent an HPO id outside the candidates list." in rendered_system

    rendered_user = template.render_user_prompt('{"phrase":"severe"}')
    assert "UNTRUSTED_MAPPING_PAYLOAD_BEGIN" in rendered_user
    assert "UNTRUSTED_MAPPING_PAYLOAD_END" in rendered_user


def test_modifier_mapping_prompt_output_contracts() -> None:
    single = loader.get_modifier_mapping_prompt("en").system_prompt
    batch = loader.get_batch_modifier_mapping_prompt("en").system_prompt

    assert '{"phrase": "example", "hpo_id": "HP:XXXXXXX"}' in single
    assert '{"phrase": "example", "hpo_id": null}' in single
    assert '{"mappings": [{"item_id": "item_1", "hpo_id": "HP:XXXXXXX"}]}' in batch
    assert '{"mappings": [{"item_id": "item_1", "hpo_id": null}]}' in batch


def test_prompt_bundle_identity_default_unchanged_and_modifiers_distinct() -> None:
    default_identity = build_prompt_bundle_identity("two_phase", "en")
    explicit_false = build_prompt_bundle_identity(
        "two_phase", "en", extract_modifiers=False
    )
    modifiers_identity = build_prompt_bundle_identity(
        "two_phase", "en", extract_modifiers=True
    )

    assert default_identity == explicit_false
    assert default_identity.prompt_behavior_version == PROMPT_BEHAVIOR_VERSION
    assert default_identity.prompt_behavior_version == "11"
    assert tuple(c.name for c in default_identity.components) == (
        "extraction",
        "mapping",
        "batch_mapping",
    )

    assert (
        modifiers_identity.prompt_behavior_version == PROMPT_MODIFIERS_BEHAVIOR_VERSION
    )
    assert modifiers_identity.prompt_behavior_version == "12"
    assert tuple(c.name for c in modifiers_identity.components) == (
        "extraction",
        "mapping",
        "batch_mapping",
        "modifier_mapping",
        "batch_modifier_mapping",
    )
    assert modifiers_identity.sha256 != default_identity.sha256
    assert modifiers_identity.components[0].version == "v3.6.0"
    assert modifiers_identity.components[3].version == "v4.3.0"
    assert modifiers_identity.components[4].version == "v4.3.0"
