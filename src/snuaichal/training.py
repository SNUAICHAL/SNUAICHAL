"""Training utilities for supervised Qwen2-VL frame-order fine-tuning."""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from snuaichal.inference import build_messages
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


class Qwen2VLCollator:
    """Prepare multimodal batches while masking user-prompt and padding labels."""

    def __init__(
        self,
        processor: Any,
        image_dir: Path,
        process_vision_info_fn: Callable[[Any], tuple[Any, Any]] | None = None,
    ) -> None:
        if process_vision_info_fn is None:
            from qwen_vl_utils import process_vision_info

            process_vision_info_fn = process_vision_info
        self.processor = processor
        self.image_dir = image_dir
        self.process_vision_info = process_vision_info_fn

    def __call__(self, rows: list[Any]) -> dict[str, Any]:
        import torch

        prompt_messages = [build_messages(row, self.image_dir) for row in rows]
        full_messages = [build_training_messages(row, self.image_dir) for row in rows]
        prompt_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in prompt_messages
        ]
        full_texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in full_messages
        ]

        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for messages in full_messages:
            sample_images, sample_videos = self.process_vision_info(messages)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen2-VL-2B-Instruct")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/qwen2-vl-lora")
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=56 * 28 * 28)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
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
        "logging_steps": args.logging_steps,
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
        "max_steps": args.max_steps,
        "seed": args.seed,
        "data_seed": args.seed,
    }


def run(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
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
        raise RuntimeError("CUDA GPU is required for Qwen2-VL fine-tuning")

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
    if args.balance_inputs:
        train_rows = balance_training_rows(train_rows, seed=args.seed)

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(**build_training_argument_kwargs(args, bf16))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        data_collator=Qwen2VLCollator(processor, image_dir),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    final_dir = args.output_dir / "final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
