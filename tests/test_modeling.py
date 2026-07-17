from types import SimpleNamespace

import pytest

from snuaichal.modeling import (
    ModelFamily,
    apply_model_chat_template,
    create_4bit_config,
    default_lora_rank,
    detect_model_family,
    resolve_model_class,
    select_lora_target_modules,
)


@pytest.mark.parametrize(
    ("model_type", "architecture", "expected"),
    [
        ("qwen2_vl", "Qwen2VLForConditionalGeneration", ModelFamily.QWEN2_VL),
        ("qwen3_vl", "Qwen3VLForConditionalGeneration", ModelFamily.QWEN3_VL),
        (
            "qwen3_vl_moe",
            "Qwen3VLMoeForConditionalGeneration",
            ModelFamily.QWEN3_VL_MOE,
        ),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration", ModelFamily.QWEN3_5),
        (
            "qwen3_5_moe",
            "Qwen3_5MoeForConditionalGeneration",
            ModelFamily.QWEN3_5_MOE,
        ),
    ],
)
def test_model_factory_detects_family_from_config(
    model_type: str, architecture: str, expected: ModelFamily
) -> None:
    config = SimpleNamespace(model_type=model_type, architectures=[architecture])

    assert detect_model_family(config) is expected


def test_model_factory_rejects_unsupported_architecture() -> None:
    config = SimpleNamespace(model_type="llava", architectures=["LlavaForConditionalGeneration"])

    with pytest.raises(ValueError, match="Unsupported vision-language model"):
        detect_model_family(config)


def test_model_factory_resolves_configured_class_without_heavy_import() -> None:
    expected_class = object()
    fake_transformers = SimpleNamespace(Qwen3VLForConditionalGeneration=expected_class)
    config = SimpleNamespace(
        model_type="qwen3_vl", architectures=["Qwen3VLForConditionalGeneration"]
    )

    assert resolve_model_class(config, fake_transformers) is expected_class


def test_qwen35_chat_template_disables_thinking() -> None:
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return "prompt"

    result = apply_model_chat_template(
        FakeProcessor(),
        [{"role": "user", "content": "question"}],
        family=ModelFamily.QWEN3_5,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert result == "prompt"
    assert calls == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
    ]


def test_4bit_config_uses_nf4_double_quant_and_bfloat16() -> None:
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_torch = SimpleNamespace(bfloat16="bf16")
    fake_transformers = SimpleNamespace(BitsAndBytesConfig=FakeBitsAndBytesConfig)

    create_4bit_config(fake_torch, fake_transformers)

    assert captured == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
    }


def test_lora_targets_include_language_projections_but_exclude_vision() -> None:
    class FakeModel:
        def named_modules(self):
            names = [
                "model.visual.blocks.0.attn.q_proj",
                "model.visual.blocks.0.attn.v_proj",
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.self_attn.k_proj",
                "model.language_model.layers.0.self_attn.v_proj",
                "model.language_model.layers.0.self_attn.o_proj",
                "model.language_model.layers.0.mlp.gate_proj",
                "model.language_model.layers.0.mlp.up_proj",
                "model.language_model.layers.0.mlp.down_proj",
                "lm_head",
            ]
            return [(name, object()) for name in names]

    targets = select_lora_target_modules(FakeModel())

    assert "model.visual.blocks.0.attn.q_proj" not in targets
    assert "model.language_model.layers.0.self_attn.q_proj" in targets
    assert "model.language_model.layers.0.mlp.down_proj" in targets
    assert len(targets) == 7


def test_qwen35_starts_with_lower_lora_rank() -> None:
    assert default_lora_rank(ModelFamily.QWEN3_VL) == 16
    assert default_lora_rank(ModelFamily.QWEN3_5) == 8


def test_model_factory_rejects_contradictory_config() -> None:
    config = SimpleNamespace(
        model_type="qwen3_vl",
        architectures=["Qwen3_5ForConditionalGeneration"],
    )

    with pytest.raises(ValueError, match="contradictory"):
        detect_model_family(config)


def test_qwen35_moe_chat_template_also_disables_thinking() -> None:
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return "prompt"

    apply_model_chat_template(
        FakeProcessor(),
        [],
        family=ModelFamily.QWEN3_5_MOE,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert calls[0]["enable_thinking"] is False
