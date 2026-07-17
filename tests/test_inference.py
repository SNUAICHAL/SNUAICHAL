from pathlib import Path
from types import SimpleNamespace

import pytest

from snuaichal import inference
from snuaichal.inference import build_parser


def test_inference_accepts_a_local_lora_adapter() -> None:
    args = build_parser().parse_args(
        [
            "--adapter-path",
            "outputs/run/final",
            "--validation-fraction",
            "0.05",
            "--metrics-output",
            "outputs/metrics.json",
        ]
    )

    assert args.adapter_path == Path("outputs/run/final")
    assert args.validation_fraction == 0.05
    assert args.metrics_output == Path("outputs/metrics.json")
    assert args.clean_validation is True
    assert args.max_new_tokens == 16
    assert args.min_pixels == 56 * 28 * 28
    assert args.image_size == 512
    assert args.max_pixels is None
    assert args.tta == 1


def test_inference_accepts_explicit_precision_and_manifest() -> None:
    args = build_parser().parse_args(
        [
            "--precision",
            "bf16",
            "--validation-manifest",
            "outputs/run/split_manifest.json",
        ]
    )

    assert args.precision == "bf16"
    assert args.validation_manifest == Path("outputs/run/split_manifest.json")


def test_manifest_validation_rows_follow_manifest_order(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        '{"validation_ids": ["valid-2", "valid-1"]}\n', encoding="utf-8"
    )
    rows = [
        {"Id": "train-1"},
        {"Id": "valid-1", "Answer": "[1, 2, 3, 4]"},
        {"Id": "valid-2", "Answer": "[2, 1, 3, 4]"},
    ]

    selected = inference.load_manifest_validation_rows(rows, manifest)

    assert [row["Id"] for row in selected] == ["valid-2", "valid-1"]


def test_manifest_validation_rows_reject_missing_and_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        '{"validation_ids": ["valid-1", "missing"]}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing"):
        inference.load_manifest_validation_rows([{"Id": "valid-1"}], manifest)

    manifest.write_text(
        '{"validation_ids": ["valid-1", "valid-1"]}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unique"):
        inference.load_manifest_validation_rows([{"Id": "valid-1"}], manifest)


def test_nf4_and_bf16_model_loading_arguments_are_mutually_explicit() -> None:
    quantization_config = object()
    nf4 = inference.build_model_load_kwargs(
        precision="nf4",
        dtype="bf16",
        device_map="auto",
        local_files_only=True,
        quantization_config=quantization_config,
    )
    bf16 = inference.build_model_load_kwargs(
        precision="bf16",
        dtype="bf16",
        device_map="auto",
        local_files_only=True,
    )

    assert nf4["device_map"] == {"": 0}
    assert nf4["quantization_config"] is quantization_config
    assert bf16["device_map"] == "auto"
    assert "quantization_config" not in bf16
    with pytest.raises(ValueError, match="requires"):
        inference.build_model_load_kwargs(
            precision="nf4",
            dtype="bf16",
            device_map="auto",
            local_files_only=True,
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("True", True), ("false", False), (1, True)],
)
def test_no_ordering_parser_is_analysis_only(raw: object, expected: bool) -> None:
    assert inference.parse_no_ordering(raw) is expected


def test_runtime_state_records_precision_adapter_eval_and_cache() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    model.is_loaded_in_4bit = False
    model.peft_config = {"default": object()}
    model.config = SimpleNamespace(use_cache=True)
    model.eval()

    state = inference.collect_model_runtime_state(model, torch, precision="bf16")

    assert state["precision"] == "bf16"
    assert state["quantization_applied"] is False
    assert state["adapter_loaded"] is True
    assert state["model_eval"] is True
    assert state["use_cache"] is True
    assert state["parameter_dtypes"] == {"torch.bfloat16": 2}


def test_inference_canonicalizes_a_view_with_keyword_only_tta_api() -> None:
    assert inference.canonicalize_view_result(
        [2, 4, 1, 3], (2, 3, 4, 1)
    ) == [3, 2, 4, 1]
    assert inference.canonicalize_view_result(None, (2, 3, 4, 1)) is None
