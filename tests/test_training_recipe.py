import json

import pytest

from snuaichal.training import (
    build_parser,
    build_training_argument_kwargs,
    resolve_max_pixels,
    validate_effective_batch,
    write_split_manifest,
)


def test_qwen3_recipe_defaults_keep_scheduler_horizon_separate() -> None:
    args = build_parser().parse_args([])

    assert args.validation_fraction == 0.10
    assert args.image_size == 512
    assert resolve_max_pixels(args) == 512 * 512
    assert args.epochs == 6
    assert args.stop_after_steps == 4292
    assert args.save_steps == 1073
    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 8

    kwargs = build_training_argument_kwargs(args, bf16=True)
    assert kwargs["num_train_epochs"] == 6
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["max_steps"] == -1


def test_effective_batch_must_equal_eight() -> None:
    args = build_parser().parse_args(
        ["--batch-size", "2", "--gradient-accumulation-steps", "4"]
    )
    validate_effective_batch(args)

    invalid = build_parser().parse_args(
        ["--batch-size", "2", "--gradient-accumulation-steps", "2"]
    )
    with pytest.raises(ValueError, match="effective batch size must be exactly 8"):
        validate_effective_batch(invalid)


def test_split_manifest_preserves_reproducible_id_lists(tmp_path) -> None:
    path = write_split_manifest(
        tmp_path,
        train_rows=[{"Id": "train-2"}, {"Id": "train-1"}],
        validation_rows=[{"Id": "valid-1"}],
        validation_fraction=0.10,
        seed=42,
        clean_validation=True,
    )

    assert path == tmp_path / "split_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "clean_validation": True,
        "seed": 42,
        "train_ids": ["train-2", "train-1"],
        "validation_fraction": 0.1,
        "validation_ids": ["valid-1"],
    }
