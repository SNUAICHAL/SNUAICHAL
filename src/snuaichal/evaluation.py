"""Validation metrics for frame-order predictions."""

from __future__ import annotations

from collections.abc import Sequence

from snuaichal.submission import IDENTITY_ORDER


def compute_exact_match_metrics(
    predictions: Sequence[list[int] | None], references: Sequence[list[int]]
) -> dict[str, int | float]:
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

    def accuracy(indices: list[int]) -> float:
        return sum(exact[index] for index in indices) / len(indices) if indices else 0.0

    return {
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
