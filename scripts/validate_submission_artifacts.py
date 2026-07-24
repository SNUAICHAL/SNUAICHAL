from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from snuaichal.submission import parse_model_output
from snuaichal.tta import (
    CYCLIC_TTA_ORDERS,
    aggregate_tta_modes,
    aggregate_tta_predictions,
    canonicalize_view_prediction,
    tta_agreement_pattern,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a snuaichallenge submission and inference audit")
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-tta", type=int, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument(
        "--aggregation-mode",
        choices=("hard", "confidence_tiebreak", "confidence_weighted"),
        default="hard",
    )
    return parser.parse_args()


def parse_answer(raw: str) -> list[int]:
    try:
        answer = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid Answer literal: {raw!r}") from exc
    if not isinstance(answer, list) or len(answer) != 4 or sorted(answer) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid Answer permutation: {raw!r}")
    return answer


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_tta <= 0:
        raise ValueError("expected_tta must be positive")

    with args.test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        expected_ids = [row["Id"] for row in csv.DictReader(handle)]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Test IDs must be non-empty and unique")

    with args.submission.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Id", "Answer"]:
            raise ValueError(f"Invalid submission columns: {reader.fieldnames}")
        submission_rows = list(reader)

    submission_ids = [row["Id"] for row in submission_rows]
    if submission_ids != expected_ids:
        raise ValueError("Submission IDs or row order do not match test.csv")
    answers = [parse_answer(row["Answer"]) for row in submission_rows]

    audit_rows = [
        json.loads(line)
        for line in args.audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(audit_rows) != len(expected_ids):
        raise ValueError(
            f"Audit row count mismatch: expected {len(expected_ids)}, got {len(audit_rows)}"
        )
    audit_ids = [row.get("Id") for row in audit_rows]
    if audit_ids != expected_ids:
        raise ValueError("Audit IDs or row order do not match test.csv")

    expected_orders = (
        [list(order) for order in CYCLIC_TTA_ORDERS]
        if args.expected_tta == len(CYCLIC_TTA_ORDERS)
        else None
    )
    parse_failures = 0
    view_parse_failures = 0
    inconsistent_tta = 0
    for index, (audit, answer) in enumerate(zip(audit_rows, answers, strict=True)):
        views = audit.get("views")
        if not isinstance(views, list) or len(views) != args.expected_tta:
            raise ValueError(
                f"Expected {args.expected_tta} audit views at row {index}, got {views!r}"
            )
        orders = [view.get("input_order") for view in views]
        if expected_orders is not None and orders != expected_orders:
            raise ValueError(f"Final Latin4 order drift at row {index}: {orders!r}")
        if len({tuple(order) for order in orders if isinstance(order, list)}) != len(
            orders
        ):
            raise ValueError(f"Duplicate TTA order at row {index}")

        canonical_predictions: list[list[int] | None] = []
        confidence_means: list[float | None] = []
        for view_index, view in enumerate(views):
            raw_output = view.get("raw_output")
            if not isinstance(raw_output, str):
                raise ValueError(
                    f"Missing raw_output at row {index}, view {view_index}"
                )
            parsed = parse_model_output(raw_output)
            stored_view_prediction = view.get("view_prediction")
            if parsed != stored_view_prediction:
                raise ValueError(
                    f"Raw-output/view prediction mismatch at row {index}, "
                    f"view {view_index}"
                )
            stored_parse_ok = view.get("parse_ok")
            if stored_parse_ok is not (parsed is not None):
                raise ValueError(
                    f"View parse_ok mismatch at row {index}, view {view_index}"
                )
            if parsed is None:
                canonical = None
                view_parse_failures += 1
            else:
                canonical = canonicalize_view_prediction(
                    parsed, view_order=orders[view_index]
                )
            if canonical != view.get("canonical_prediction"):
                raise ValueError(
                    f"Canonical prediction mismatch at row {index}, "
                    f"view {view_index}"
                )
            canonical_predictions.append(canonical)
            confidence = view.get("answer_logprob_mean")
            confidence_means.append(
                float(confidence)
                if view.get("answer_confidence_valid") is True
                and isinstance(confidence, (int, float))
                else None
            )

        hard = aggregate_tta_predictions(canonical_predictions)
        modes = aggregate_tta_modes(canonical_predictions, confidence_means)
        aggregation_mode = getattr(args, "aggregation_mode", None) or audit.get(
            "aggregation_mode"
        )
        if aggregation_mode not in modes:
            raise ValueError(f"Invalid aggregation mode at row {index}")
        selected = modes[aggregation_mode]
        if audit.get("aggregation_mode") != aggregation_mode:
            raise ValueError(f"Audit aggregation mode drift at row {index}")
        if audit.get("answer") != selected.answer or answer != selected.answer:
            raise ValueError(f"Semantic aggregation mismatch at row {index}")
        if audit.get("valid_tta_views") != hard.valid_views:
            raise ValueError(f"valid_tta_views mismatch at row {index}")
        if audit.get("parse_ok") is not (hard.valid_views > 0):
            raise ValueError(f"parse_ok mismatch at row {index}")
        if audit.get("tta_consistent") is not hard.consistent:
            raise ValueError(f"tta_consistent mismatch at row {index}")
        if audit.get("tta_agreement_pattern") != tta_agreement_pattern(
            canonical_predictions
        ):
            raise ValueError(f"agreement pattern mismatch at row {index}")
        if audit.get("tie_break") != selected.tie_break:
            raise ValueError(f"tie_break mismatch at row {index}")

        recorded_modes = audit.get("aggregations")
        if not isinstance(recorded_modes, dict):
            raise ValueError(f"Missing aggregation audit at row {index}")
        for mode_name, recomputed in modes.items():
            recorded = recorded_modes.get(mode_name)
            if not isinstance(recorded, dict):
                raise ValueError(
                    f"Missing {mode_name} aggregation at row {index}"
                )
            for key in (
                "answer",
                "valid_views",
                "used_fallback",
                "tie_break",
                "vote_counts",
                "valid_confidence_views",
            ):
                if recorded.get(key) != getattr(recomputed, key):
                    raise ValueError(
                        f"{mode_name}.{key} mismatch at row {index}"
                    )

        if hard.valid_views == 0:
            parse_failures += 1
        if not hard.consistent:
            inconsistent_tta += 1

    if parse_failures or view_parse_failures:
        raise ValueError(
            "Final audit contains parse failures: "
            f"rows={parse_failures}, views={view_parse_failures}"
        )

    summary = {
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "audit_rows": len(audit_rows),
        "expected_tta": args.expected_tta,
        "inconsistent_tta_rows": inconsistent_tta,
        "parse_failures": parse_failures,
        "rows": len(submission_rows),
        "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        "unique_ids": len(set(submission_ids)),
        "view_parse_failures": view_parse_failures,
    }
    metrics_path = getattr(args, "metrics", None)
    if metrics_path is not None:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected_metrics = {
            "aggregation_mode": getattr(args, "aggregation_mode", "hard"),
            "model_repository": "Qwen/Qwen3.6-27B",
            "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            "verified_model_tree_sha256": (
                "e4107e6508793261ca372faf4b560dcb55a5b6ba79a5ab921bfe1b25a207ec07"
            ),
            "tta_views": args.expected_tta,
        }
        for key, expected in expected_metrics.items():
            if metrics.get(key) != expected:
                raise ValueError(f"metrics {key} mismatch")
        if expected_orders is not None and metrics.get("tta_orders") != expected_orders:
            raise ValueError("metrics Latin4 order drift")
        if metrics.get("parse_failures") != 0:
            raise ValueError("metrics report parse failures")
        if metrics.get("physical_measurement_status") != "valid":
            raise ValueError("physical VRAM measurement is not valid")
        if metrics.get("unexpected_cuda_processes") != []:
            raise ValueError("unexpected CUDA process was observed")
        for key in (
            "trusted_cuda_child_seen",
            "process_identity_match",
            "sampling_started_before_model_load",
            "sampling_finished_after_work",
        ):
            if metrics.get(key) is not True:
                raise ValueError(f"physical VRAM evidence {key} is not true")
        samples = metrics.get("sample_count")
        peak = metrics.get("physical_peak_observed_bytes")
        total = metrics.get("physical_total_vram_bytes")
        if (
            isinstance(samples, bool)
            or not isinstance(samples, int)
            or samples <= 0
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak < 0
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
        ):
            raise ValueError("physical VRAM evidence is incomplete")
        if peak > total or peak > 24 * 1024**3:
            raise ValueError("physical peak exceeds the 24 GiB deployment contract")
        summary["metrics_sha256"] = hashlib.sha256(
            metrics_path.read_bytes()
        ).hexdigest()
        summary["physical_peak_observed_bytes"] = peak
    return summary


def main() -> None:
    print(json.dumps(validate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
