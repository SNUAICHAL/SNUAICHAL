from pathlib import Path

import pytest

from snuaichal.physical_memory import (
    _nvml_handle_for_visible_device,
    _valid_used_bytes,
    command_identity,
    cuda_workload_identity,
    process_identity_matches,
    select_physical_measurement,
)
from snuaichal.training import (
    Qwen2VLCollator,
    answer_positions_to_chronological,
    balance_training_rows,
    build_parser,
    build_training_argument_kwargs,
    build_training_messages,
    build_training_summary,
    build_vram_measurements,
    image_dhash,
    permute_training_row,
    seed_training,
    select_training_rows,
    split_rows,
    split_rows_without_image_overlap,
    write_or_validate_manifest,
)


def test_answer_positions_are_converted_to_chronological_labels() -> None:
    assert answer_positions_to_chronological("[3, 2, 4, 1]") == [4, 2, 1, 3]


@pytest.mark.parametrize("answer", ["not a list", "[1, 2, 3]", "[1, 1, 2, 3]"])
def test_invalid_training_answer_is_rejected(answer: str) -> None:
    with pytest.raises(ValueError, match="Invalid training answer"):
        answer_positions_to_chronological(answer)


def test_training_messages_end_with_chronological_assistant_answer(
    tmp_path: Path,
) -> None:
    row = {
        "Id": "sample",
        "Sentence": "First this, then that.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in ("one.jpg", "two.jpg", "three.jpg", "four.jpg"):
        (sample_dir / filename).touch()

    messages = build_training_messages(row, tmp_path)

    assert messages[-1] == {"role": "assistant", "content": "[4, 2, 1, 3]"}
    assert [item["type"] for item in messages[0]["content"]].count("image") == 4


def test_collator_masks_prompt_and_padding_tokens(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def __getattr__(self, name: str):
            return self[name]

    class FakeProcessor:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "full" if messages[-1]["role"] == "assistant" else "prompt"

        def __call__(self, *, text, **kwargs):
            if text == ["full"]:
                return FakeBatch(
                    input_ids=torch.tensor([[10, 11, 12, 13, 0]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
                )
            return FakeBatch(
                input_ids=torch.tensor([[10, 11, 12]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )

    row = {
        "Id": "sample",
        "Sentence": "First this, then that.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in ("one.jpg", "two.jpg", "three.jpg", "four.jpg"):
        (sample_dir / filename).touch()

    collator = Qwen2VLCollator(
        FakeProcessor(),
        tmp_path,
        process_vision_info_fn=lambda messages: (["images"], None),
    )

    batch = collator([row])

    assert batch["labels"].tolist() == [[-100, -100, -100, 13, -100]]


def test_collator_masks_left_padded_batch_prompts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class FakeBatch(dict):
        def __getattr__(self, name: str):
            return self[name]

    class LeftPaddingProcessor:
        tokenizer = type(
            "Tokenizer", (), {"pad_token_id": 0, "padding_side": "left"}
        )()

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "full" if messages[-1]["role"] == "assistant" else "prompt"

        def __call__(self, *, text, **kwargs):
            if text == ["full", "full"]:
                return FakeBatch(
                    input_ids=torch.tensor(
                        [[0, 10, 11, 12, 13], [20, 21, 22, 23, 24]]
                    ),
                    attention_mask=torch.tensor(
                        [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
                    ),
                )
            return FakeBatch(
                input_ids=torch.tensor(
                    [[0, 0, 10, 11, 12], [0, 20, 21, 22, 23]]
                ),
                attention_mask=torch.tensor(
                    [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]
                ),
            )

    rows = []
    for index in range(2):
        sample_id = f"sample-{index}"
        sample_dir = tmp_path / sample_id
        sample_dir.mkdir()
        row = {
            "Id": sample_id,
            "Sentence": "First this, then that.",
            "Answer": "[3, 2, 4, 1]",
        }
        for slot in range(1, 5):
            filename = f"image-{slot}.jpg"
            (sample_dir / filename).touch()
            row[f"Input_{slot}"] = filename
        rows.append(row)

    collator = Qwen2VLCollator(
        LeftPaddingProcessor(),
        tmp_path,
        process_vision_info_fn=lambda messages: (["images"], None),
    )

    batch = collator(rows)

    assert batch["labels"].tolist() == [
        [-100, -100, -100, -100, 13],
        [-100, -100, -100, -100, 24],
    ]


def test_split_rows_is_deterministic_and_disjoint() -> None:
    rows = [{"Id": str(index)} for index in range(20)]

    train_rows, validation_rows = split_rows(rows, validation_fraction=0.2, seed=7)

    assert len(train_rows) == 16
    assert len(validation_rows) == 4
    assert {row["Id"] for row in train_rows}.isdisjoint(
        row["Id"] for row in validation_rows
    )
    assert split_rows(rows, validation_fraction=0.2, seed=7) == (
        train_rows,
        validation_rows,
    )


def test_training_row_limit_applies_after_the_full_dataset_split() -> None:
    rows = [
        {"Id": f"sample-{index}", "Answer": "[1, 2, 3, 4]"}
        for index in range(20)
    ]
    full_train, full_validation = split_rows(
        rows, validation_fraction=0.2, seed=42
    )

    train_rows, validation_rows = select_training_rows(
        rows,
        image_dir=Path("unused"),
        validation_fraction=0.2,
        seed=42,
        limit=10,
        clean_validation=False,
    )

    assert train_rows == full_train[:8]
    assert validation_rows == full_validation[:2]


def test_zero_validation_fraction_uses_all_rows_for_final_training() -> None:
    rows = [{"Id": f"sample-{index}"} for index in range(5)]

    train_rows, validation_rows = select_training_rows(
        rows,
        image_dir=Path("unused"),
        validation_fraction=0.0,
        seed=42,
        limit=None,
        clean_validation=True,
    )

    assert train_rows == rows
    assert validation_rows == []


def test_zero_validation_fraction_honors_smoke_limit() -> None:
    rows = [{"Id": f"sample-{index}"} for index in range(5)]

    train_rows, validation_rows = select_training_rows(
        rows,
        image_dir=Path("unused"),
        validation_fraction=0.0,
        seed=42,
        limit=2,
        clean_validation=True,
    )

    assert train_rows == rows[:2]
    assert validation_rows == []


def test_training_defaults_fit_a_single_24gb_gpu() -> None:
    args = build_parser().parse_args([])

    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert args.image_size == 512
    assert args.max_pixels is None
    assert args.lora_rank is None
    assert args.balance_inputs is True
    assert args.clean_validation is True
    assert args.validation_fraction == 0.1
    assert args.save_total_limit is None

    training_kwargs = build_training_argument_kwargs(args, bf16=True)
    assert training_kwargs["eval_strategy"] == "no"
    assert training_kwargs["save_total_limit"] is None
    assert training_kwargs["lr_scheduler_type"] == "cosine"
    assert training_kwargs["max_steps"] == -1


def test_training_rejects_nonpositive_save_total_limit() -> None:
    args = build_parser().parse_args(["--save-total-limit", "0"])

    with pytest.raises(ValueError, match="positive integer"):
        build_training_argument_kwargs(args, bf16=True)


def test_training_summary_records_optimizer_step_timing() -> None:
    summary = build_training_summary(
        global_step=2,
        epoch=0.01,
        learning_rate=1e-4,
        training_loss=0.5,
        peak_vram_bytes=20_000,
        train_metrics={"train_runtime": 100.0, "train_steps_per_second": 0.02},
    )

    assert summary["train_runtime_seconds"] == 100.0
    assert summary["train_steps_per_second"] == 0.02
    assert summary["seconds_per_optimizer_step"] == 50.0


def test_wddm_unavailable_process_memory_is_not_converted_to_int() -> None:
    assert _valid_used_bytes(None) is None


def test_wddm_gui_process_is_not_classified_as_cuda_workload(monkeypatch) -> None:
    import psutil

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def name(self) -> str:
            return "explorer.exe" if self.pid == 10 else "python.exe"

        def cmdline(self) -> list[str]:
            if self.pid == 10:
                return ["C:/Windows/explorer.exe"]
            return ["python.exe", "-m", "snuaichal.training"]

        def create_time(self) -> float:
            return 1.0

    monkeypatch.setattr(psutil, "Process", FakeProcess)

    assert cuda_workload_identity(10) is None
    identity = cuda_workload_identity(11)
    assert identity is not None
    assert identity["pid"] == 11
    assert identity["create_time"] == 1.0
    assert identity["process_name"] == "python.exe"
    assert len(identity["command_identity"]) == 64


def test_process_identity_requires_exact_creation_time() -> None:
    command = ["python.exe", "-m", "snuaichal.training"]

    class FakeProcess:
        pid = 11

        @staticmethod
        def create_time() -> float:
            return 1.000000001

        @staticmethod
        def cmdline() -> list[str]:
            return command

    assert not process_identity_matches(
        FakeProcess(),
        expected_pid=11,
        expected_create_time=1.0,
        expected_command_identity=command_identity(command),
    )


def test_partial_process_samples_use_conservative_device_fallback() -> None:
    report = select_physical_measurement(
        physical_total_vram_bytes=24 * 1024**3,
        process_peak_bytes=5 * 1024**3,
        process_sample_count=1,
        device_peak_bytes=12 * 1024**3,
        device_sample_count=3,
        process_memory_unavailable=True,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )

    assert report["physical_measurement_status"] == "valid"
    assert report["physical_measurement_source"] == "nvml_device_memory_info_used"
    assert report["physical_peak_observed_bytes"] == 12 * 1024**3
    assert report["sample_count"] == 3


def _physical_report(
    *,
    source: str = "nvml_per_process_used_bytes",
    observed: int,
    total: int,
    sample_count: int = 4,
) -> dict:
    return {
        "schema_version": 1,
        "physical_measurement_status": "valid",
        "physical_peak_observed_bytes": observed,
        "physical_total_vram_bytes": total,
        "physical_measurement_source": source,
        "sample_count": sample_count,
        "sample_interval_seconds": 0.5,
        "sampling_started_at": "2026-07-18T00:00:00+00:00",
        "sampling_ended_at": "2026-07-18T00:01:00+00:00",
        "sampling_started_before_model_load": True,
        "sampling_finished_after_work": True,
        "process_identity_match": True,
        "trusted_cuda_child_seen": True,
        "unexpected_cuda_processes": [],
        "device_fallback_exclusive": True,
    }


def test_vram_measurements_never_sum_allocated_and_reserved() -> None:
    gib = 1024**3
    measurements = build_vram_measurements(
        logical_peak_allocated_bytes=20 * gib,
        logical_peak_reserved_bytes=23 * gib,
        allocator_backend="native",
        physical_measurement=_physical_report(
            observed=21 * gib, total=24 * gib
        ),
    )

    assert 20 * gib + 23 * gib > 24 * gib
    assert measurements["logical_peak_allocated_bytes"] == 20 * gib
    assert measurements["logical_peak_reserved_bytes"] == 23 * gib
    assert measurements["continuation_gate_source"] == (
        "nvml_per_process_used_bytes"
    )
    assert measurements["continuation_gate_bytes"] == 21 * gib


def test_logical_allocated_may_exceed_physical_under_wddm() -> None:
    gib = 1024**3
    measurements = build_vram_measurements(
        logical_peak_allocated_bytes=28 * gib,
        logical_peak_reserved_bytes=29 * gib,
        allocator_backend="native",
        physical_measurement=_physical_report(
            source="nvml_device_memory_info_used",
            observed=23 * gib,
            total=24 * gib,
        ),
    )

    assert measurements["logical_peak_allocated_bytes"] > measurements[
        "physical_total_vram_bytes"
    ]
    assert measurements["continuation_gate_source"] == (
        "nvml_device_memory_info_used"
    )


def test_nvml_process_measurement_has_exact_precedence() -> None:
    gib = 1024**3
    selected = select_physical_measurement(
        physical_total_vram_bytes=48 * gib,
        process_peak_bytes=31 * gib,
        process_sample_count=4,
        device_peak_bytes=33 * gib,
        device_sample_count=4,
        process_memory_unavailable=False,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )
    assert selected["physical_measurement_source"] == (
        "nvml_per_process_used_bytes"
    )
    assert selected["physical_peak_observed_bytes"] == 31 * gib


def test_wddm_device_fallback_requires_exclusive_trusted_workload() -> None:
    gib = 1024**3
    valid = select_physical_measurement(
        physical_total_vram_bytes=24 * gib,
        process_peak_bytes=None,
        process_sample_count=0,
        device_peak_bytes=23 * gib,
        device_sample_count=10,
        process_memory_unavailable=True,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )
    invalid = select_physical_measurement(
        physical_total_vram_bytes=24 * gib,
        process_peak_bytes=None,
        process_sample_count=0,
        device_peak_bytes=23 * gib,
        device_sample_count=10,
        process_memory_unavailable=True,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[{"pid": 999}],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )
    assert valid["physical_measurement_source"] == (
        "nvml_device_memory_info_used"
    )
    assert invalid["physical_measurement_status"] == "indeterminate"


def test_zero_sample_or_peak_above_physical_blocks_measurement() -> None:
    gib = 1024**3
    zero_sample = select_physical_measurement(
        physical_total_vram_bytes=24 * gib,
        process_peak_bytes=None,
        process_sample_count=0,
        device_peak_bytes=None,
        device_sample_count=0,
        process_memory_unavailable=True,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )
    too_high = select_physical_measurement(
        physical_total_vram_bytes=24 * gib,
        process_peak_bytes=None,
        process_sample_count=0,
        device_peak_bytes=25 * gib,
        device_sample_count=4,
        process_memory_unavailable=True,
        trusted_cuda_child_seen=True,
        process_identity_match=True,
        unexpected_cuda_processes=[],
        sampling_started_before_model_load=True,
        sampling_finished_after_work=True,
    )
    assert zero_sample["physical_measurement_status"] == "indeterminate"
    assert too_high["physical_measurement_status"] == "indeterminate"


def test_training_seed_reproduces_torch_and_python_randomness() -> None:
    import random

    torch = pytest.importorskip("torch")
    seed_training(17)
    first = (random.random(), torch.rand(3).tolist())
    seed_training(17)
    second = (random.random(), torch.rand(3).tolist())

    assert second == first


def test_permute_training_row_reorders_inputs_and_target() -> None:
    row = {
        "Id": "sample",
        "Sentence": "A sequence.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }

    permuted = permute_training_row(row, [2, 4, 1, 3])

    assert [permuted[f"Input_{index}"] for index in range(1, 5)] == [
        "two.jpg",
        "four.jpg",
        "one.jpg",
        "three.jpg",
    ]
    assert permuted["Answer"] == "[2, 1, 3, 4]"
    assert permuted["Id"] == row["Id"]
    assert permuted["Sentence"] == row["Sentence"]


def test_balance_training_rows_flattens_all_24_labels() -> None:
    rows = [
        {
            "Id": str(index),
            "Sentence": "A sequence.",
            "Input_1": "one.jpg",
            "Input_2": "two.jpg",
            "Input_3": "three.jpg",
            "Input_4": "four.jpg",
            "Answer": "[1, 2, 3, 4]",
        }
        for index in range(49)
    ]

    balanced = balance_training_rows(rows, seed=7)
    counts = {}
    for row in balanced:
        counts[row["Answer"]] = counts.get(row["Answer"], 0) + 1

    assert len(balanced) == len(rows)
    assert len(counts) == 24
    assert max(counts.values()) - min(counts.values()) <= 1
    assert balance_training_rows(rows, seed=7) == balanced


def test_perceptual_hash_matches_recompressed_image(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (96, 64))
    draw = ImageDraw.Draw(image)
    for x in range(96):
        draw.line((x, 0, x, 63), fill=(x * 2, 255 - x * 2, x))
    draw.rectangle((20, 15, 55, 45), fill=(230, 30, 80))
    high_quality = tmp_path / "high.jpg"
    low_quality = tmp_path / "low.jpg"
    image.save(high_quality, quality=95)
    image.save(low_quality, quality=45)

    assert high_quality.read_bytes() != low_quality.read_bytes()
    assert image_dhash(high_quality) == image_dhash(low_quality)


def test_clean_split_excludes_rows_with_images_reused_elsewhere(tmp_path: Path) -> None:
    import random

    from PIL import Image

    rows = []
    labels = ["[1, 2, 3, 4]"] * 3 + ["[4, 3, 2, 1]"] * 3
    for index, label in enumerate(labels):
        sample_id = f"sample-{index}"
        sample_dir = tmp_path / sample_id
        sample_dir.mkdir()
        row = {"Id": sample_id, "Answer": label}
        for slot in range(1, 5):
            filename = f"image-{slot}.jpg"
            image_seed = index * 4 + slot
            if slot == 1 and index in {0, 3}:
                image_seed = 10_000
            image_bytes = random.Random(image_seed).randbytes(32 * 32 * 3)
            Image.frombytes("RGB", (32, 32), image_bytes).save(sample_dir / filename)
            row[f"Input_{slot}"] = filename
        rows.append(row)

    train_rows, validation_rows = split_rows_without_image_overlap(
        rows,
        image_dir=tmp_path,
        validation_fraction=1 / 3,
        seed=7,
    )

    assert len(validation_rows) == 2
    assert {row["Id"] for row in validation_rows}.isdisjoint({"sample-0", "sample-3"})
    assert {row["Answer"] for row in validation_rows} == {
        "[1, 2, 3, 4]",
        "[4, 3, 2, 1]",
    }
    assert {row["Id"] for row in train_rows}.union(
        row["Id"] for row in validation_rows
    ) == {row["Id"] for row in rows}


def test_split_rows_is_label_stratified() -> None:
    rows = [
        {"Id": f"a-{index}", "Answer": "[1, 2, 3, 4]"} for index in range(10)
    ] + [{"Id": f"b-{index}", "Answer": "[4, 3, 2, 1]"} for index in range(10)]

    train_rows, validation_rows = split_rows(rows, validation_fraction=0.2, seed=3)

    assert len(validation_rows) == 4
    assert sum(row["Answer"] == "[1, 2, 3, 4]" for row in validation_rows) == 2
    assert sum(row["Answer"] == "[4, 3, 2, 1]" for row in validation_rows) == 2
    assert len(train_rows) == 16


def test_run_manifest_rejects_incompatible_resume_settings(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    write_or_validate_manifest(path, {"scheduler_horizon_steps": 6438})
    write_or_validate_manifest(path, {"scheduler_horizon_steps": 6438})

    with pytest.raises(ValueError, match="does not match"):
        write_or_validate_manifest(path, {"scheduler_horizon_steps": 4292})
class _FakeNvml:
    def nvmlDeviceGetHandleByIndex(self, index: int) -> tuple[str, int]:
        return ("index", index)

    def nvmlDeviceGetHandleByUUID(self, value: str) -> tuple[str, str]:
        return ("uuid", value)

    def nvmlDeviceGetHandleByPciBusId(self, value: str) -> tuple[str, str]:
        return ("pci", value)


def test_physical_monitor_respects_scheduler_visible_device(monkeypatch) -> None:
    nvml = _FakeNvml()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert _nvml_handle_for_visible_device(nvml) == ("index", 3)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
    assert _nvml_handle_for_visible_device(nvml) == ("uuid", "GPU-assigned")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0000:65:00.0")
    assert _nvml_handle_for_visible_device(nvml) == ("pci", "0000:65:00.0")
