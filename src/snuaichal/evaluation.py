"""Validation metrics for frame-order predictions."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from typing import Any

from snuaichal.submission import IDENTITY_ORDER, answer_to_string


def compare_prediction_modes(
    candidate: Sequence[list[int] | None],
    baseline: Sequence[list[int] | None],
    references: Sequence[list[int]],
) -> dict[str, int | float]:
    """Return paired exact-match changes from a baseline to a candidate mode."""
    if not (len(candidate) == len(baseline) == len(references)):
        raise ValueError("Candidate, baseline, and references must have equal length")
    if not references:
        raise ValueError("At least one reference is required")
    baseline_correct = [value == target for value, target in zip(baseline, references)]
    candidate_correct = [value == target for value, target in zip(candidate, references)]
    corrected = sum(
        not old and new for old, new in zip(baseline_correct, candidate_correct)
    )
    worsened = sum(
        old and not new for old, new in zip(baseline_correct, candidate_correct)
    )
    exact_matches = sum(candidate_correct)
    return {
        "accuracy": exact_matches / len(references),
        "corrected": corrected,
        "exact_matches": exact_matches,
        "net_gain": corrected - worsened,
        "prediction_changes": sum(
            new != old for new, old in zip(candidate, baseline)
        ),
        "worsened": worsened,
    }


def compute_exact_match_metrics(
    predictions: Sequence[list[int] | None],
    references: Sequence[list[int]],
    *,
    tta_consistent: Sequence[bool] | None = None,
    no_ordering: Sequence[bool] | None = None,
    tta_agreement_patterns: Sequence[str] | None = None,
    elapsed_seconds: float | None = None,
    peak_vram_bytes: int | None = None,
    model_precision: str | None = None,
    image_grid_thw: Sequence[Sequence[int]] | None = None,
    visual_tokens: Sequence[int] | None = None,
    expected_test_samples: int = 819,
) -> dict[str, Any]:
    """Compute competition accuracy and diagnostics without masking parse failures."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    if not references:
        raise ValueError("At least one reference is required")

    exact = [prediction == reference for prediction, reference in zip(predictions, references)]
    identity_indices = [
        index for index, reference in enumerate(references) if reference == IDENTITY_ORDER
    ]
    non_identity_indices = [
        index for index, reference in enumerate(references) if reference != IDENTITY_ORDER
    ]
    parse_failures = sum(prediction is None for prediction in predictions)
    labels = [answer_to_string(order) for order in itertools.permutations(range(1, 5))]

    def accuracy(indices: list[int]) -> float:
        return sum(exact[index] for index in indices) / len(indices) if indices else 0.0

    metrics: dict[str, Any] = {
        "samples": len(references),
        "exact_matches": sum(exact),
        "exact_match": sum(exact) / len(references),
        "identity_samples": len(identity_indices),
        "identity_exact_match": accuracy(identity_indices),
        "non_identity_samples": len(non_identity_indices),
        "non_identity_exact_match": accuracy(non_identity_indices),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(references),
    }
    metrics["class_accuracy"] = {
        label: {
            "accuracy": accuracy(indices),
            "exact_matches": sum(exact[index] for index in indices),
            "samples": len(indices),
        }
        for label in labels
        for indices in [
            [
                index
                for index, reference in enumerate(references)
                if answer_to_string(reference) == label
            ]
        ]
    }
    if tta_consistent is not None:
        if len(tta_consistent) != len(references):
            raise ValueError("TTA consistency flags must match references")
        metrics["tta_consistency"] = sum(tta_consistent) / len(references)
    if elapsed_seconds is not None:
        seconds_per_sample = elapsed_seconds / len(references)
        metrics["inference_seconds_per_sample"] = seconds_per_sample
        metrics["estimated_test_seconds"] = seconds_per_sample * expected_test_samples
    if peak_vram_bytes is not None:
        metrics["peak_vram_mib"] = peak_vram_bytes / (1024 * 1024)
    if model_precision is not None:
        metrics["model_precision"] = model_precision
    if no_ordering is not None:
        if len(no_ordering) != len(references):
            raise ValueError("No_ordering flags must match references")
        metrics["no_ordering_accuracy"] = {
            key: {
                "accuracy": accuracy(indices),
                "exact_matches": sum(exact[index] for index in indices),
                "samples": len(indices),
            }
            for flag, key in ((True, "true"), (False, "false"))
            for indices in [
                [index for index, value in enumerate(no_ordering) if value is flag]
            ]
        }
    if tta_agreement_patterns is not None:
        if len(tta_agreement_patterns) != len(references):
            raise ValueError("TTA agreement patterns must match references")
        metrics["tta_agreement_patterns"] = dict(
            sorted(Counter(tta_agreement_patterns).items())
        )
    if image_grid_thw is not None:
        metrics["image_grid_thw"] = dict(
            sorted(Counter(str(list(grid)) for grid in image_grid_thw).items())
        )
    if visual_tokens is not None:
        values = list(visual_tokens)
        if values:
            metrics["visual_tokens"] = {
                "count": len(values),
                "maximum": max(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
            }
    if any(
        value is not None
        for value in (tta_consistent, elapsed_seconds, peak_vram_bytes)
    ):
        labels = [answer_to_string(order) for order in itertools.permutations(range(1, 5))]
        confusion = {
            reference: {**{prediction: 0 for prediction in labels}, "parse_failure": 0}
            for reference in labels
        }
        for prediction, reference in zip(predictions, references):
            reference_key = answer_to_string(reference)
            prediction_key = (
                answer_to_string(prediction) if prediction is not None else "parse_failure"
            )
            confusion[reference_key][prediction_key] += 1
        metrics["confusion"] = confusion
    return metrics
