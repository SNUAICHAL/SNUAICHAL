"""Config-driven model dispatch shared by training and inference."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ModelFamily(str, Enum):
    """Supported Qwen vision-language architecture families."""

    QWEN2_VL = "qwen2_vl"
    QWEN3_VL = "qwen3_vl"
    QWEN3_VL_MOE = "qwen3_vl_moe"
    QWEN3_5 = "qwen3_5"
    QWEN3_5_MOE = "qwen3_5_moe"


_MODEL_TYPES = {family.value: family for family in ModelFamily}
_ARCHITECTURES = {
    "Qwen2VLForConditionalGeneration": ModelFamily.QWEN2_VL,
    "Qwen3VLForConditionalGeneration": ModelFamily.QWEN3_VL,
    "Qwen3VLMoeForConditionalGeneration": ModelFamily.QWEN3_VL_MOE,
    "Qwen3_5ForConditionalGeneration": ModelFamily.QWEN3_5,
    "Qwen3_5MoeForConditionalGeneration": ModelFamily.QWEN3_5_MOE,
}


def detect_model_family(config: Any) -> ModelFamily:
    """Determine the supported family from a loaded Transformers config."""
    model_type = str(getattr(config, "model_type", ""))
    if model_type in _MODEL_TYPES:
        family = _MODEL_TYPES[model_type]
        architecture_families = {
            _ARCHITECTURES[architecture]
            for architecture in (getattr(config, "architectures", None) or [])
            if architecture in _ARCHITECTURES
        }
        if architecture_families and architecture_families != {family}:
            raise ValueError(
                "Model config is contradictory: "
                f"model_type={model_type!r}, "
                f"architectures={getattr(config, 'architectures', None)!r}"
            )
        return family
    for architecture in getattr(config, "architectures", None) or []:
        if architecture in _ARCHITECTURES:
            return _ARCHITECTURES[architecture]
    raise ValueError(
        "Unsupported vision-language model: "
        f"model_type={model_type!r}, architectures={getattr(config, 'architectures', None)!r}"
    )


def resolve_model_class(config: Any, transformers_module: Any) -> Any:
    """Resolve an explicit configured class, falling back to the multimodal AutoModel."""
    family = detect_model_family(config)
    configured = list(getattr(config, "architectures", None) or [])
    defaults = {
        ModelFamily.QWEN2_VL: "Qwen2VLForConditionalGeneration",
        ModelFamily.QWEN3_VL: "Qwen3VLForConditionalGeneration",
        ModelFamily.QWEN3_VL_MOE: "Qwen3VLMoeForConditionalGeneration",
        ModelFamily.QWEN3_5: "Qwen3_5ForConditionalGeneration",
        ModelFamily.QWEN3_5_MOE: "Qwen3_5MoeForConditionalGeneration",
    }
    for class_name in [*configured, defaults[family]]:
        model_class = getattr(transformers_module, class_name, None)
        if model_class is not None:
            return model_class
    auto_class = getattr(transformers_module, "AutoModelForImageTextToText", None)
    if auto_class is None:
        raise RuntimeError(
            f"Installed Transformers does not support {family.value}; upgrade requirements"
        )
    return auto_class


def apply_model_chat_template(
    processor: Any,
    messages: Any,
    *,
    family: ModelFamily,
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    """Apply the model chat template with family-specific deterministic options."""
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if family in {ModelFamily.QWEN3_5, ModelFamily.QWEN3_5_MOE}:
        kwargs["enable_thinking"] = False
    return processor.apply_chat_template(messages, **kwargs)


def create_4bit_config(torch_module: Any, transformers_module: Any) -> Any:
    """Create the reproducible NF4 QLoRA quantization configuration."""
    return transformers_module.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_module.bfloat16,
    )


def default_lora_rank(family: ModelFamily) -> int:
    """Return the initial rank validated or requested for each model scale."""
    if family in {ModelFamily.QWEN3_5, ModelFamily.QWEN3_5_MOE}:
        return 8
    return 16


def select_lora_target_modules(
    model: Any, family: ModelFamily | None = None
) -> list[str]:
    """Select exact language projection paths while leaving vision frozen."""
    projection_names = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    }
    if family in {ModelFamily.QWEN3_VL_MOE, ModelFamily.QWEN3_5_MOE}:
        projection_names = {"q_proj", "k_proj", "v_proj", "o_proj"}
    vision_markers = {"visual", "vision_model", "vision_tower", "image_encoder"}
    targets = []
    for module_name, _ in model.named_modules():
        parts = set(module_name.split("."))
        if module_name.rsplit(".", 1)[-1] not in projection_names:
            continue
        if parts.intersection(vision_markers):
            continue
        targets.append(module_name)
    if not targets:
        raise ValueError("No supported language projection modules found for LoRA")
    return sorted(set(targets))


def freeze_vision_parameters(model: Any) -> int:
    """Freeze parameters belonging to the vision tower and return their count."""
    vision_markers = {"visual", "vision_model", "vision_tower", "image_encoder"}
    frozen = 0
    for parameter_name, parameter in model.named_parameters():
        if set(parameter_name.split(".")).intersection(vision_markers):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def count_parameters(model: Any) -> tuple[int, int]:
    """Return trainable and total parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total
