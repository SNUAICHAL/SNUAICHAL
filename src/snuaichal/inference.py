"""Offline inference entry point for the Qwen2-VL zero-shot baseline."""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any

from snuaichal.submission import IDENTITY_ORDER, answer_to_string, parse_model_output

REQUIRED_COLUMNS = {"Id", "Sentence", "Input_1", "Input_2", "Input_3", "Input_4"}


def build_messages(row: Any, image_dir: Path) -> list[dict[str, Any]]:
    """Build one multimodal chat request from a test row."""
    sample_id = str(row["Id"])
    content: list[dict[str, str]] = []

    for image_number in range(1, 5):
        image_path = image_dir / sample_id / str(row[f"Input_{image_number}"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        content.extend(
            [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": f"\nImage {image_number}\n"},
            ]
        )

    content.append(
        {
            "type": "text",
            "text": (
                f'Thinking about the sentence: "{row["Sentence"]}"\n'
                "Look at the 4 images above labeled Image 1 to Image 4. "
                "Determine the correct chronological order of these images to "
                "match the sentence. Provide the answer ONLY as a Python list "
                "of integers. Example: [1, 2, 3, 4]"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def _resolve_dtype(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, requested)


def run(args: argparse.Namespace) -> None:
    """Run deterministic inference and write both submission and audit logs."""
    try:
        import pandas as pd
        import torch
        from qwen_vl_utils import process_vision_info
        from tqdm.auto import tqdm
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_csv = args.test_csv or args.data_dir / "test.csv"
    image_dir = args.image_dir or args.data_dir / "test"
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {image_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Local model not found: {args.model_path}. See models/README.md."
        )
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        raise FileNotFoundError(f"Local LoRA adapter not found: {args.adapter_path}")

    test_df = pd.read_csv(test_csv, dtype={"Id": str})
    missing_columns = REQUIRED_COLUMNS.difference(test_df.columns)
    if missing_columns:
        raise ValueError(f"Missing test.csv columns: {sorted(missing_columns)}")
    if args.validation_fraction is not None:
        if "Answer" not in test_df.columns:
            raise ValueError("Validation evaluation requires an Answer column")
        from snuaichal.training import split_rows, split_rows_without_image_overlap

        split_function = (
            split_rows_without_image_overlap if args.clean_validation else split_rows
        )
        split_kwargs = {
            "rows": test_df.to_dict("records"),
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        }
        if args.clean_validation:
            split_kwargs["image_dir"] = image_dir
        _, validation_rows = split_function(**split_kwargs)
        test_df = pd.DataFrame(validation_rows)
    if args.limit is not None:
        test_df = test_df.head(args.limit)

    dtype = _resolve_dtype(torch, args.dtype)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        local_files_only=not args.allow_network,
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            local_files_only=not args.allow_network,
        )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=not args.allow_network,
    )
    model.eval()

    predictions: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    metric_predictions: list[list[int] | None] = []
    metric_references: list[list[int]] = []
    parse_failures = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        messages = build_messages(row, image_dir)
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed_answer = parse_model_output(output_text)
        if args.validation_fraction is not None:
            reference = ast.literal_eval(str(row["Answer"]))
            metric_predictions.append(parsed_answer)
            metric_references.append(reference)
        parse_ok = parsed_answer is not None
        if not parse_ok:
            parse_failures += 1
            parsed_answer = IDENTITY_ORDER.copy()

        sample_id = str(row["Id"])
        predictions.append(
            {"Id": sample_id, "Answer": answer_to_string(parsed_answer)}
        )
        audit_rows.append(
            {
                "Id": sample_id,
                "raw_output": output_text,
                "answer": parsed_answer,
                "parse_ok": parse_ok,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions, columns=["Id", "Answer"]).to_csv(
        args.output, index=False, encoding="utf-8"
    )
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_log.open("w", encoding="utf-8", newline="\n") as log_file:
        for row in audit_rows:
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(predictions)} predictions to {args.output}")
    print(f"Parse failures: {parse_failures}/{len(predictions)}")
    if metric_references:
        from snuaichal.evaluation import compute_exact_match_metrics

        metrics = compute_exact_match_metrics(metric_predictions, metric_references)
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metrics, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen2-VL-2B-Instruct")
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Evaluate the deterministic held-out fraction of a labeled CSV",
    )
    parser.add_argument(
        "--clean-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the image-disjoint training holdout",
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("outputs/validation_metrics.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument(
        "--audit-log", type=Path, default=Path("outputs/raw_predictions.jsonl")
    )
    parser.add_argument("--min-pixels", type=int, default=56 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Smoke-test only the first N rows")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Transformers to access the network (disabled by default)",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

