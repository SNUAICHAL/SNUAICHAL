"""Offline inference entry point for the Qwen2-VL zero-shot baseline."""

from __future__ import annotations

import argparse
import ast
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from snuaichal.confidence import align_answer_confidence
from snuaichal.modeling import apply_model_chat_template
from snuaichal.submission import (
    answer_to_string,
    is_permutation,
    parse_model_output,
    validate_submission_records,
)
from snuaichal.tta import (
    CYCLIC_TTA_ORDERS,
    aggregate_tta_modes,
    aggregate_tta_predictions,
    canonicalize_view_prediction,
    permute_input_row,
    tta_agreement_pattern,
)

REQUIRED_COLUMNS = {"Id", "Sentence", "Input_1", "Input_2", "Input_3", "Input_4"}


def parse_tta_orders_json(value: str) -> tuple[tuple[int, ...], ...]:
    """Parse and validate an explicit ordered list of TTA permutations."""
    try:
        raw_orders = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("TTA orders must be valid JSON") from exc
    if not isinstance(raw_orders, list) or not raw_orders:
        raise argparse.ArgumentTypeError("TTA orders must be a nonempty JSON list")
    orders: list[tuple[int, ...]] = []
    for raw_order in raw_orders:
        if not is_permutation(raw_order):
            raise argparse.ArgumentTypeError(
                f"invalid TTA input order: {raw_order!r}"
            )
        orders.append(tuple(raw_order))
    if len(set(orders)) != len(orders):
        raise argparse.ArgumentTypeError("TTA orders must be unique")
    return tuple(orders)


@dataclass(frozen=True)
class ViewGeneration:
    """One generated view plus score, timing, and visual-token evidence."""

    output_text: str
    prediction: list[int] | None
    generated_token_ids: list[int]
    generated_token_logprobs: list[float]
    parsed_answer_substring: str | None
    answer_token_start: int | None
    answer_token_end: int | None
    answer_token_ids: list[int]
    answer_token_logprobs: list[float]
    answer_logprob_sum: float | None
    answer_logprob_mean: float | None
    answer_confidence_valid: bool
    alignment_method: str | None
    alignment_failure_reason: str | None
    image_grid_thw: list[list[int]]
    visual_tokens: list[int]
    elapsed_seconds: float


def load_manifest_validation_rows(
    rows: list[dict[str, Any]], manifest_path: Path
) -> list[dict[str, Any]]:
    """Select validation rows in the exact order persisted by training."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_ids = [str(sample_id) for sample_id in payload.get("validation_ids", [])]
    if not validation_ids:
        raise ValueError("Validation manifest contains no validation_ids")
    if len(set(validation_ids)) != len(validation_ids):
        raise ValueError("Validation manifest IDs must be unique")

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["Id"])
        if sample_id in rows_by_id:
            raise ValueError(f"Input CSV IDs must be unique; duplicate: {sample_id}")
        rows_by_id[sample_id] = row
    missing_ids = [sample_id for sample_id in validation_ids if sample_id not in rows_by_id]
    if missing_ids:
        raise ValueError(
            f"Validation manifest IDs are missing from the input CSV: {missing_ids[:5]}"
        )
    selected = [rows_by_id[sample_id] for sample_id in validation_ids]
    if any("Answer" not in row for row in selected):
        raise ValueError("Manifest validation evaluation requires an Answer column")
    return selected


def resolve_precision(args: argparse.Namespace) -> str:
    """Resolve explicit precision while preserving the legacy 4-bit flag."""
    precision = args.precision or ("nf4" if args.load_in_4bit else "bf16")
    if args.load_in_4bit and precision != "nf4":
        raise ValueError("--load-in-4bit conflicts with --precision bf16")
    return precision


def parse_no_ordering(value: Any) -> bool:
    """Parse the train-only analysis flag without exposing it to the model."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid No_ordering value: {value!r}")


def build_model_load_kwargs(
    *,
    precision: str,
    dtype: Any,
    device_map: Any,
    local_files_only: bool,
    quantization_config: Any = None,
) -> dict[str, Any]:
    """Build auditable NF4/BF16 loader arguments without an implicit fallback."""
    if precision not in {"nf4", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": {"": 0} if precision == "nf4" else device_map,
        "local_files_only": local_files_only,
        "attn_implementation": "sdpa",
    }
    if precision == "nf4":
        if quantization_config is None:
            raise ValueError("NF4 precision requires a quantization_config")
        kwargs["quantization_config"] = quantization_config
    return kwargs


def collect_model_runtime_state(
    model: Any, torch: Any, *, precision: str
) -> dict[str, Any]:
    """Collect JSON-safe evidence for the model that was actually loaded."""
    dtype_counts = Counter(str(parameter.dtype) for parameter in model.parameters())
    base_dtype_counts = Counter(
        str(parameter.dtype)
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    quantization_config = getattr(model.config, "quantization_config", None)
    return {
        "precision": precision,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "quantization_applied": bool(
            getattr(model, "is_loaded_in_4bit", False)
            or quantization_config is not None
        ),
        "quantization_config_type": (
            type(quantization_config).__name__ if quantization_config is not None else None
        ),
        "adapter_loaded": bool(getattr(model, "peft_config", None)),
        "model_eval": not model.training,
        "use_cache": bool(getattr(model.config, "use_cache", False)),
        "parameter_dtypes": dict(sorted(dtype_counts.items())),
        "base_parameter_dtypes": dict(sorted(base_dtype_counts.items())),
    }


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
                f'Given the sentence: "{row["Sentence"]}"\n'
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
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, requested)


def _generate_view(
    *,
    row: Any,
    image_dir: Path,
    model: Any,
    processor: Any,
    family: Any,
    process_vision_info: Any,
    torch: Any,
    max_new_tokens: int,
) -> ViewGeneration:
    messages = build_messages(row, image_dir)
    prompt = apply_model_chat_template(
        processor,
        messages,
        family=family,
        tokenize=False,
        add_generation_prompt=True,
    )
    vision_kwargs = {} if family.value == "qwen2_vl" else {"image_patch_size": 16}
    image_inputs, video_inputs = process_vision_info(messages, **vision_kwargs)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    grids = (
        inputs["image_grid_thw"].detach().cpu().tolist()
        if "image_grid_thw" in inputs
        else []
    )
    merge_size = int(
        getattr(getattr(processor, "image_processor", None), "merge_size", 2)
    )
    visual_tokens = [
        int(grid[0] * grid[1] * grid[2] // (merge_size * merge_size))
        for grid in grids
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    prompt_length = inputs.input_ids.shape[1]
    trimmed_ids = generation.sequences[:, prompt_length:]
    generated_token_ids = trimmed_ids[0].detach().cpu().tolist()
    generated_token_logprobs = []
    for score, token_id in zip(generation.scores, generated_token_ids, strict=False):
        logprob = torch.log_softmax(score[0].float(), dim=-1)[token_id]
        generated_token_logprobs.append(float(logprob.item()))
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    confidence = align_answer_confidence(
        raw_output=output_text,
        generated_token_ids=generated_token_ids,
        generated_token_logprobs=generated_token_logprobs,
        tokenizer=processor.tokenizer,
    )
    return ViewGeneration(
        output_text=output_text,
        prediction=parse_model_output(output_text),
        generated_token_ids=generated_token_ids,
        generated_token_logprobs=generated_token_logprobs,
        parsed_answer_substring=confidence.parsed_answer_substring,
        answer_token_start=confidence.answer_token_start,
        answer_token_end=confidence.answer_token_end,
        answer_token_ids=confidence.answer_token_ids,
        answer_token_logprobs=confidence.answer_token_logprobs,
        answer_logprob_sum=confidence.answer_logprob_sum,
        answer_logprob_mean=confidence.answer_logprob_mean,
        answer_confidence_valid=confidence.answer_confidence_valid,
        alignment_method=confidence.alignment_method,
        alignment_failure_reason=confidence.alignment_failure_reason,
        image_grid_thw=grids,
        visual_tokens=visual_tokens,
        elapsed_seconds=elapsed_seconds,
    )


def canonicalize_view_result(
    view_prediction: list[int] | None, view_order: tuple[int, int, int, int]
) -> list[int] | None:
    if view_prediction is None:
        return None
    return canonicalize_view_prediction(view_prediction, view_order=view_order)


def run(args: argparse.Namespace) -> None:
    """Run deterministic inference and write both submission and audit logs."""
    try:
        import pandas as pd
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info
        from tqdm.auto import tqdm
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    from snuaichal.model_manifest import verify_model_manifest
    from snuaichal.modeling import (
        create_4bit_config,
        detect_model_family,
        resolve_model_class,
    )

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
    verified_model_manifest = verify_model_manifest(
        args.model_manifest,
        model_root=args.model_path,
        expected_repository=args.model_repository,
        expected_revision=args.model_revision,
        expected_family=args.model_family,
    )

    test_df = pd.read_csv(test_csv, dtype={"Id": str})
    missing_columns = REQUIRED_COLUMNS.difference(test_df.columns)
    if missing_columns:
        raise ValueError(f"Missing test.csv columns: {sorted(missing_columns)}")
    if args.validation_manifest is not None:
        validation_rows = load_manifest_validation_rows(
            test_df.to_dict("records"), args.validation_manifest
        )
        test_df = pd.DataFrame(validation_rows)
    elif args.validation_fraction is not None:
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

    precision = resolve_precision(args)
    dtype = torch.bfloat16 if precision == "bf16" else _resolve_dtype(torch, args.dtype)
    config = AutoConfig.from_pretrained(
        args.model_path, local_files_only=not args.allow_network
    )
    family = detect_model_family(config)
    if family.value != args.model_family:
        raise RuntimeError(
            f"Detected model family {family.value!r} differs from declared "
            f"{args.model_family!r}"
        )
    model_class = resolve_model_class(config, transformers)
    model_kwargs = build_model_load_kwargs(
        precision=precision,
        dtype=dtype,
        device_map=args.device_map,
        local_files_only=not args.allow_network,
        quantization_config=(
            create_4bit_config(torch, transformers) if precision == "nf4" else None
        ),
    )
    model = model_class.from_pretrained(args.model_path, **model_kwargs)
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
        max_pixels=args.max_pixels or args.image_size**2,
        local_files_only=not args.allow_network,
    )
    model.eval()
    model.config.use_cache = True
    runtime_state = collect_model_runtime_state(model, torch, precision=precision)

    predictions: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    metric_predictions: list[list[int] | None] = []
    metric_predictions_by_mode: dict[str, list[list[int] | None]] = {
        mode: [] for mode in ("hard", "confidence_tiebreak", "confidence_weighted")
    }
    metric_view0_predictions: list[list[int] | None] = []
    metric_references: list[list[int]] = []
    metric_no_ordering: list[bool] = []
    metric_tta_patterns: list[str] = []
    tta_consistency: list[bool] = []
    all_image_grids: list[list[int]] = []
    all_visual_tokens: list[int] = []
    parse_failures = 0
    tta_orders = (
        args.tta_orders
        if args.tta_orders is not None
        else (CYCLIC_TTA_ORDERS[:1] if args.tta == 1 else CYCLIC_TTA_ORDERS)
    )
    if len(tta_orders) != args.tta:
        raise ValueError(
            f"Explicit TTA order count {len(tta_orders)} does not match --tta {args.tta}"
        )
    if args.fallback_policy != "identity":
        raise ValueError(f"Unsupported fallback policy: {args.fallback_policy}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        view_audit = []
        canonical_predictions: list[list[int] | None] = []
        view_confidences: list[float | None] = []
        for order in tta_orders:
            view_row = permute_input_row(row, order)
            generation = _generate_view(
                row=view_row,
                image_dir=image_dir,
                model=model,
                processor=processor,
                family=family,
                process_vision_info=process_vision_info,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
            )
            canonical_prediction = canonicalize_view_result(generation.prediction, order)
            canonical_predictions.append(canonical_prediction)
            view_confidences.append(
                generation.answer_logprob_mean
                if generation.answer_confidence_valid
                else None
            )
            all_image_grids.extend(generation.image_grid_thw)
            all_visual_tokens.extend(generation.visual_tokens)
            view_audit.append(
                {
                    "input_order": list(order),
                    "raw_output": generation.output_text,
                    "view_prediction": generation.prediction,
                    "canonical_prediction": canonical_prediction,
                    "parse_ok": generation.prediction is not None,
                    "generated_token_ids": generation.generated_token_ids,
                    "generated_token_logprobs": generation.generated_token_logprobs,
                    "parsed_answer_substring": generation.parsed_answer_substring,
                    "answer_token_start": generation.answer_token_start,
                    "answer_token_end": generation.answer_token_end,
                    "answer_token_ids": generation.answer_token_ids,
                    "answer_token_logprobs": generation.answer_token_logprobs,
                    "answer_logprob_sum": generation.answer_logprob_sum,
                    "answer_logprob_mean": generation.answer_logprob_mean,
                    "answer_confidence_valid": generation.answer_confidence_valid,
                    "alignment_method": generation.alignment_method,
                    "alignment_failure_reason": generation.alignment_failure_reason,
                    "image_grid_thw": generation.image_grid_thw,
                    "visual_tokens": generation.visual_tokens,
                    "elapsed_seconds": generation.elapsed_seconds,
                }
            )
        aggregate = aggregate_tta_predictions(canonical_predictions)
        mode_aggregates = aggregate_tta_modes(
            canonical_predictions,
            view_confidences,
            temperature=args.confidence_temperature,
        )
        selected_aggregate = mode_aggregates[args.aggregation_mode]
        agreement_pattern = tta_agreement_pattern(canonical_predictions)
        parsed_answer = list(selected_aggregate.answer)
        reference = None
        no_ordering = None
        if args.validation_manifest is not None or args.validation_fraction is not None:
            reference = ast.literal_eval(str(row["Answer"]))
            metric_predictions.append(
                parsed_answer if aggregate.valid_views > 0 else None
            )
            metric_view0_predictions.append(canonical_predictions[0])
            for mode, mode_aggregate in mode_aggregates.items():
                metric_predictions_by_mode[mode].append(
                    list(mode_aggregate.answer) if aggregate.valid_views > 0 else None
                )
            metric_references.append(reference)
            tta_consistency.append(aggregate.consistent)
            metric_tta_patterns.append(agreement_pattern)
            if "No_ordering" in row:
                no_ordering = parse_no_ordering(row["No_ordering"])
                metric_no_ordering.append(no_ordering)
        parse_ok = aggregate.valid_views > 0
        if not parse_ok:
            parse_failures += 1

        sample_id = str(row["Id"])
        predictions.append(
            {"Id": sample_id, "Answer": answer_to_string(parsed_answer)}
        )
        audit_rows.append(
            {
                "Id": sample_id,
                "answer": parsed_answer,
                "parse_ok": parse_ok,
                "valid_tta_views": aggregate.valid_views,
                "aggregation_mode": args.aggregation_mode,
                "tie_break": selected_aggregate.tie_break,
                "tta_consistent": aggregate.consistent,
                "tta_agreement_pattern": agreement_pattern,
                "aggregations": {
                    mode: asdict(mode_aggregate)
                    for mode, mode_aggregate in mode_aggregates.items()
                },
                "reference": reference,
                "no_ordering": no_ordering,
                "views": view_audit,
            }
        )

    elapsed_seconds = time.perf_counter() - inference_started
    expected_ids = [str(sample_id) for sample_id in test_df["Id"].tolist()]
    validate_submission_records(
        predictions,
        expected_ids=expected_ids,
        expected_count=len(expected_ids),
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
    peak_vram_bytes = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )
    seconds_per_sample = elapsed_seconds / len(predictions)
    metrics: dict[str, Any] = {
        "samples": len(predictions),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(predictions),
        "inference_seconds_per_sample": seconds_per_sample,
        "estimated_test_seconds": seconds_per_sample * 819,
        "peak_vram_mib": peak_vram_bytes / (1024 * 1024),
        "model_precision": precision,
    }
    if metric_references:
        from snuaichal.evaluation import (
            compare_prediction_modes,
            compute_exact_match_metrics,
        )

        metrics.update(
            compute_exact_match_metrics(
                metric_predictions,
                metric_references,
                tta_consistent=tta_consistency,
                no_ordering=metric_no_ordering or None,
                tta_agreement_patterns=metric_tta_patterns,
                elapsed_seconds=elapsed_seconds,
                peak_vram_bytes=peak_vram_bytes,
                model_precision=precision,
                image_grid_thw=all_image_grids,
                visual_tokens=all_visual_tokens,
            )
        )
        metrics.update(
            {
                "non_identity_exact_matches": sum(
                    prediction is not None and prediction == reference
                    for prediction, reference, no_ordering in zip(
                        metric_predictions,
                        metric_references,
                        metric_no_ordering,
                        strict=True,
                    )
                    if not no_ordering
                ),
                "aggregation_comparison": {
                    mode: {
                        "vs_hard": compare_prediction_modes(
                            mode_predictions,
                            metric_predictions_by_mode["hard"],
                            metric_references,
                        ),
                        "vs_view0": compare_prediction_modes(
                            mode_predictions,
                            metric_view0_predictions,
                            metric_references,
                        ),
                    }
                    for mode, mode_predictions in metric_predictions_by_mode.items()
                },
            }
        )
    metrics.update(
        {
            "aggregation_mode": args.aggregation_mode,
            "adapter_path": str(args.adapter_path) if args.adapter_path else None,
            "base_model_path": str(args.model_path),
            "model_repository": args.model_repository,
            "model_family": family.value,
            "detected_model_family": family.value,
            "declared_model_family": args.model_family,
            "model_revision": args.model_revision,
            "model_manifest_path": str(args.model_manifest),
            "model_manifest_sha256": verified_model_manifest["manifest_sha256"],
            "verified_model_tree_sha256": verified_model_manifest["tree_sha256"],
            "confidence_temperature": args.confidence_temperature,
            "fallback_policy": args.fallback_policy,
            "max_pixels": args.max_pixels or args.image_size**2,
            "min_pixels": args.min_pixels,
            "runtime_state": runtime_state,
            "tta_orders": [list(order) for order in tta_orders],
            "tta_views": len(tta_orders),
        }
    )
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument(
        "--model-family",
        choices=("qwen2_vl", "qwen3_vl", "qwen3_vl_moe", "qwen3_5"),
        required=True,
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    adapter_group = parser.add_mutually_exclusive_group(required=True)
    adapter_group.add_argument("--adapter-path", type=Path)
    adapter_group.add_argument(
        "--no-adapter",
        action="store_true",
        help="Explicitly run the base model without a PEFT adapter",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        help="Evaluate the deterministic held-out fraction of a labeled CSV",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="Use validation_ids from an immutable training split manifest",
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
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--precision",
        choices=("nf4", "bf16"),
        required=True,
        help="Explicit base-model inference precision",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Deprecated compatibility alias for --precision nf4",
    )
    parser.add_argument("--tta", type=int, choices=(1, 4), required=True)
    parser.add_argument(
        "--tta-orders-json",
        dest="tta_orders",
        type=parse_tta_orders_json,
        help="Explicit ordered JSON list of TTA input permutations",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=("hard", "confidence_tiebreak", "confidence_weighted"),
        required=True,
    )
    parser.add_argument(
        "--fallback-policy",
        choices=("identity",),
        default="identity",
        help="Explicit policy used when every TTA view fails to parse",
    )
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
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

