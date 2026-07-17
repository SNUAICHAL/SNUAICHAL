"""Training utilities for supervised Qwen2-VL frame-order fine-tuning."""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from snuaichal.augmentation import DatasetEpochCallback, EpochShuffleDataset
from snuaichal.inference import build_messages
from snuaichal.modeling import ModelFamily, apply_model_chat_template
from snuaichal.scheduling import (
    HorizonTrainer,
    StopAtStepCallback,
    build_schedule_plan,
    validate_resume_checkpoint,
)
from snuaichal.submission import answer_to_string, chronological_to_positions, is_permutation


def seed_training(seed: int) -> None:
    """Seed Python, NumPy, and Torch before model and LoRA initialization."""
    from transformers import set_seed

    set_seed(seed)


def image_dhash(image_path: Path) -> int:
    """Return a 64-bit difference hash stable across ordinary JPEG recompression."""
    from PIL import Image

    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        if hasattr(grayscale, "get_flattened_data"):
            pixels = list(grayscale.get_flattened_data())
        else:  # Pillow < 12 compatibility.
            pixels = list(grayscale.getdata())
    fingerprint = 0
    for y in range(8):
        for x in range(8):
            fingerprint = (fingerprint << 1) | (
                pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            )
    return fingerprint


def split_rows(
    rows: list[Any], validation_fraction: float, seed: int
) -> tuple[list[Any], list[Any]]:
    """Split rows reproducibly, stratifying by Answer when it is available."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("At least two rows are required for a train/validation split")
    random_generator = random.Random(seed)
    if not all("Answer" in row for row in rows):
        shuffled = list(rows)
        random_generator.shuffle(shuffled)
        validation_size = max(1, round(len(shuffled) * validation_fraction))
        return shuffled[validation_size:], shuffled[:validation_size]

    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(str(row["Answer"]), []).append(row)

    train_rows: list[Any] = []
    validation_rows: list[Any] = []
    for label in sorted(groups):
        group = list(groups[label])
        random_generator.shuffle(group)
        validation_size = round(len(group) * validation_fraction)
        if len(group) > 1:
            validation_size = max(1, min(validation_size, len(group) - 1))
        else:
            validation_size = 0
        validation_rows.extend(group[:validation_size])
        train_rows.extend(group[validation_size:])

    random_generator.shuffle(train_rows)
    random_generator.shuffle(validation_rows)
    if not validation_rows:
        validation_rows.append(train_rows.pop())
    return train_rows, validation_rows


def split_rows_without_image_overlap(
    rows: list[Any], image_dir: Path, validation_fraction: float, seed: int
) -> tuple[list[Any], list[Any]]:
    """Build a stratified holdout whose image bytes occur in no other row."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("At least two rows are required for a train/validation split")

    row_hashes: list[list[int]] = []
    hash_counts: Counter[int] = Counter()
    for row in rows:
        sample_hashes = []
        for slot in range(1, 5):
            image_path = image_dir / str(row["Id"]) / str(row[f"Input_{slot}"])
            digest = image_dhash(image_path)
            sample_hashes.append(digest)
            hash_counts[digest] += 1
        row_hashes.append(sample_hashes)

    candidates = [
        row
        for row, sample_hashes in zip(rows, row_hashes)
        if all(hash_counts[digest] == 1 for digest in sample_hashes)
    ]
    validation_size = max(1, round(len(rows) * validation_fraction))
    if len(candidates) < validation_size:
        raise ValueError(
            f"Only {len(candidates)} image-disjoint rows are available for "
            f"a {validation_size}-row validation split"
        )

    groups: dict[str, list[Any]] = {}
    for row in candidates:
        groups.setdefault(str(row["Answer"]), []).append(row)
    random_generator = random.Random(seed)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label, group in groups.items():
        ideal = validation_size * len(group) / len(candidates)
        quotas[label] = int(ideal)
        remainders.append((ideal - quotas[label], label))
        random_generator.shuffle(group)
    for _, label in sorted(remainders, reverse=True)[
        : validation_size - sum(quotas.values())
    ]:
        quotas[label] += 1

    validation_rows = [
        row for label, group in groups.items() for row in group[: quotas[label]]
    ]
    validation_ids = {str(row["Id"]) for row in validation_rows}
    train_rows = [row for row in rows if str(row["Id"]) not in validation_ids]
    random_generator.shuffle(train_rows)
    random_generator.shuffle(validation_rows)
    return train_rows, validation_rows


def permute_training_row(row: Any, input_order: list[int]) -> dict[str, Any]:
    """Reorder input slots and move each frame's target position with it."""
    if not is_permutation(input_order):
        raise ValueError(f"Invalid input permutation: {input_order!r}")

    try:
        positions = ast.literal_eval(str(row["Answer"]))
    except (KeyError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid training answer: {row.get('Answer')!r}") from exc
    if not is_permutation(positions):
        raise ValueError(f"Invalid training answer: {row.get('Answer')!r}")

    permuted = dict(row)
    permuted_positions: list[int] = []
    for new_index, old_index in enumerate(input_order, start=1):
        permuted[f"Input_{new_index}"] = row[f"Input_{old_index}"]
        permuted_positions.append(positions[old_index - 1])
    permuted["Answer"] = answer_to_string(permuted_positions)
    return permuted


def balance_training_rows(rows: list[Any], seed: int) -> list[dict[str, Any]]:
    """Assign input permutations so the 24 target labels are nearly uniform."""
    targets = list(itertools.permutations(range(1, 5)))
    random_generator = random.Random(seed)
    schedule: list[tuple[int, ...]] = []
    while len(schedule) < len(rows):
        block = list(targets)
        random_generator.shuffle(block)
        schedule.extend(block)

    balanced: list[dict[str, Any]] = []
    for row, target in zip(rows, schedule):
        try:
            positions = ast.literal_eval(str(row["Answer"]))
        except (KeyError, SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid training answer: {row.get('Answer')!r}") from exc
        if not is_permutation(positions):
            raise ValueError(f"Invalid training answer: {row.get('Answer')!r}")
        input_order = [positions.index(position) + 1 for position in target]
        balanced.append(permute_training_row(row, input_order))
    return balanced


def answer_positions_to_chronological(answer_text: str) -> list[int]:
    """Convert a CSV position answer into chronological image labels."""
    try:
        positions = ast.literal_eval(answer_text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid training answer: {answer_text!r}") from exc
    if not is_permutation(positions):
        raise ValueError(f"Invalid training answer: {answer_text!r}")
    return chronological_to_positions(positions)


def build_training_messages(row: Any, image_dir: Path) -> list[dict[str, Any]]:
    """Build one user request and its chronological assistant target."""
    messages = build_messages(row, image_dir)
    chronological = answer_positions_to_chronological(str(row["Answer"]))
    messages.append({"role": "assistant", "content": answer_to_string(chronological)})
    return messages


class QwenVLCollator:
    """Prepare multimodal batches while masking user-prompt and padding labels."""

    def __init__(
        self,
        processor: Any,
        image_dir: Path,
        process_vision_info_fn: Callable[[Any], tuple[Any, Any]] | None = None,
        family: ModelFamily = ModelFamily.QWEN2_VL,
    ) -> None:
        if process_vision_info_fn is None:
            from qwen_vl_utils import process_vision_info

            process_vision_info_fn = process_vision_info
        self.processor = processor
        self.image_dir = image_dir
        self.process_vision_info = process_vision_info_fn
        self.family = family
        self._logged_visual_stats = False

    def __call__(self, rows: list[Any]) -> dict[str, Any]:
        import torch

        prompt_messages = [build_messages(row, self.image_dir) for row in rows]
        full_messages = [build_training_messages(row, self.image_dir) for row in rows]
        prompt_texts = [
            apply_model_chat_template(
                self.processor,
                messages,
                family=self.family,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in prompt_messages
        ]
        full_texts = [
            apply_model_chat_template(
                self.processor,
                messages,
                family=self.family,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in full_messages
        ]

        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for messages in full_messages:
            vision_kwargs = (
                {} if self.family is ModelFamily.QWEN2_VL else {"image_patch_size": 16}
            )
            sample_images, sample_videos = self.process_vision_info(
                messages, **vision_kwargs
            )
            image_inputs.extend(sample_images or [])
            video_inputs.extend(sample_videos or [])

        processor_kwargs = {
            "images": image_inputs or None,
            "videos": video_inputs or None,
            "padding": True,
            "return_tensors": "pt",
        }
        batch = self.processor(text=full_texts, **processor_kwargs)
        prompt_batch = self.processor(text=prompt_texts, **processor_kwargs)

        if not self._logged_visual_stats and "image_grid_thw" in batch:
            grids = batch["image_grid_thw"].tolist()
            merge_size = int(
                getattr(getattr(self.processor, "image_processor", None), "merge_size", 2)
            )
            visual_tokens = [
                grid[0] * grid[1] * grid[2] // (merge_size * merge_size)
                for grid in grids
            ]
            print(
                json.dumps(
                    {
                        "image_grid_thw": grids,
                        "visual_tokens": visual_tokens,
                        "merge_size": merge_size,
                    },
                    sort_keys=True,
                )
            )
            self._logged_visual_stats = True

        labels = batch.input_ids.clone()
        prompt_lengths = prompt_batch.attention_mask.sum(dim=1)
        full_lengths = batch.attention_mask.sum(dim=1)
        left_padding = getattr(self.processor.tokenizer, "padding_side", "right") == "left"
        for row_index, (prompt_length, full_length) in enumerate(
            zip(prompt_lengths.tolist(), full_lengths.tolist())
        ):
            non_padding_start = labels.shape[1] - full_length if left_padding else 0
            answer_start = non_padding_start + prompt_length
            labels[row_index, :answer_start] = -100
        labels = torch.where(batch.attention_mask == 0, -100, labels)
        batch["labels"] = labels
        return dict(batch)


Qwen2VLCollator = QwenVLCollator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen2-VL-2B-Instruct")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen2-vl-lora")
    )
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--min-pixels", type=int, default=56 * 28 * 28)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--stop-after-steps", type=int, default=4292)
    parser.add_argument("--save-steps", type=int, default=1073)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--train-vision-encoder",
        action="store_true",
        help="Opt in to vision-tower training; frozen by default",
    )
    parser.add_argument(
        "--balance-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Permute training image slots to flatten the 24 target classes",
    )
    parser.add_argument(
        "--clean-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hold out only rows whose image bytes never occur in another row",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def resolve_max_pixels(args: argparse.Namespace) -> int:
    """Resolve an area budget without forcing image aspect ratios."""
    if args.image_size <= 0:
        raise ValueError("image_size must be positive")
    return args.max_pixels if args.max_pixels is not None else args.image_size**2


def validate_effective_batch(args: argparse.Namespace, world_size: int = 1) -> None:
    """Enforce the recipe's exact optimizer-update batch size."""
    effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
    if effective_batch != 8:
        raise ValueError(
            "effective batch size must be exactly 8; "
            f"got {args.batch_size} * {args.gradient_accumulation_steps} * "
            f"{world_size} = {effective_batch}"
        )


def write_split_manifest(
    output_dir: Path,
    *,
    train_rows: list[Any],
    validation_rows: list[Any],
    validation_fraction: float,
    seed: int,
    clean_validation: bool,
) -> Path:
    """Persist exact split IDs under outputs for resume and evaluation reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split_manifest.json"
    payload = {
        "clean_validation": clean_validation,
        "seed": seed,
        "train_ids": [str(row["Id"]) for row in train_rows],
        "validation_fraction": validation_fraction,
        "validation_ids": [str(row["Id"]) for row in validation_rows],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise ValueError(f"Existing split manifest does not match this run: {path}")
    else:
        path.write_text(serialized, encoding="utf-8")
    return path


def write_or_validate_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write immutable run metadata, rejecting incompatible resume settings."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"Existing run manifest does not match: {path}")
        return
    path.write_text(serialized, encoding="utf-8")


def build_training_argument_kwargs(
    args: argparse.Namespace, bf16: bool
) -> dict[str, Any]:
    """Build Trainer settings while reserving validation for exact-match inference."""
    return {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "eval_strategy": "no",
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": None,
        "bf16": bf16,
        "fp16": not bf16,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "report_to": "none",
        "optim": "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        "max_steps": -1,
        "seed": args.seed,
        "data_seed": args.seed,
    }


def run(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoProcessor, TrainingArguments

    from snuaichal.modeling import (
        count_parameters,
        create_4bit_config,
        default_lora_rank,
        detect_model_family,
        freeze_vision_parameters,
        resolve_model_class,
        select_lora_target_modules,
    )

    seed_training(args.seed)

    train_csv = args.data_dir / "train.csv"
    image_dir = args.data_dir / "train"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Train image directory not found: {image_dir}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Local model not found: {args.model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen-VL fine-tuning")
    validate_effective_batch(args)
    if args.resume_from_checkpoint is not None:
        validate_resume_checkpoint(args.resume_from_checkpoint, args.stop_after_steps)

    with train_csv.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required_columns = {
        "Id",
        "Sentence",
        "Input_1",
        "Input_2",
        "Input_3",
        "Input_4",
        "Answer",
    }
    if not rows:
        raise ValueError("train.csv contains no samples")
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(f"Missing train.csv columns: {sorted(missing_columns)}")
    if args.clean_validation:
        train_rows, validation_rows = split_rows_without_image_overlap(
            rows,
            image_dir=image_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        train_rows, validation_rows = split_rows(
            rows, validation_fraction=args.validation_fraction, seed=args.seed
        )
    if args.limit is not None:
        if args.limit < 2:
            raise ValueError("limit must be at least 2")
        limited_validation_size = max(1, round(args.limit * args.validation_fraction))
        validation_rows = validation_rows[:limited_validation_size]
        train_rows = train_rows[: args.limit - limited_validation_size]
    write_split_manifest(
        args.output_dir,
        train_rows=train_rows,
        validation_rows=validation_rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        clean_validation=args.clean_validation,
    )
    train_dataset = EpochShuffleDataset(
        train_rows, seed=args.seed, augment=args.balance_inputs
    )
    schedule = build_schedule_plan(
        train_samples=len(train_dataset),
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=1,
        scheduler_epochs=args.epochs,
        stop_after_steps=args.stop_after_steps,
        save_steps=args.save_steps,
    )
    write_or_validate_manifest(
        args.output_dir / "schedule.json",
        {
            "updates_per_epoch": schedule.updates_per_epoch,
            "scheduler_horizon_steps": schedule.scheduler_horizon_steps,
            "stop_after_steps": schedule.stop_after_steps,
            "checkpoint_steps": list(schedule.checkpoint_steps),
        },
    )

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    family = detect_model_family(config)
    model_class = resolve_model_class(config, transformers)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=resolve_max_pixels(args),
        local_files_only=True,
    )
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "quantization_config": create_4bit_config(torch, transformers),
                "device_map": {"": 0},
            }
        )
    model = model_class.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    frozen_vision_parameters = (
        0 if args.train_vision_encoder else freeze_vision_parameters(model)
    )
    target_modules = select_lora_target_modules(model, family=family)
    lora_rank = args.lora_rank or default_lora_rank(family)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        ),
    )
    trainable_parameters, total_parameters = count_parameters(model)
    model_manifest = {
        "family": family.value,
        "architecture": model_class.__name__,
        "model_type": config.model_type,
        "model_path": str(args.model_path),
        "load_in_4bit": args.load_in_4bit,
        "lora_rank": lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "target_module_count": len(target_modules),
        "frozen_vision_parameters": frozen_vision_parameters,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "max_pixels": resolve_max_pixels(args),
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": version("peft"),
            "accelerate": version("accelerate"),
            "bitsandbytes": version("bitsandbytes"),
            "qwen-vl-utils": version("qwen-vl-utils"),
        },
    }
    write_or_validate_manifest(
        args.output_dir / "model_manifest.json", model_manifest
    )
    print(json.dumps(model_manifest, sort_keys=True))

    training_args = TrainingArguments(**build_training_argument_kwargs(args, bf16))
    trainer = HorizonTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QwenVLCollator(processor, image_dir, family=family),
        callbacks=[
            DatasetEpochCallback(train_dataset),
            StopAtStepCallback(args.stop_after_steps),
        ],
        scheduler_horizon_steps=schedule.scheduler_horizon_steps,
    )
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    summary = {
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "learning_rate": trainer._get_learning_rate(),
        "training_loss": train_result.training_loss,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    final_dir = args.output_dir / "final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
