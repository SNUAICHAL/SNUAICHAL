"""Cyclic test-time augmentation in canonical submission coordinates."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from snuaichal.submission import IDENTITY_ORDER, is_permutation

CYCLIC_TTA_ORDERS = (
    (1, 2, 3, 4),
    (2, 3, 4, 1),
    (3, 4, 1, 2),
    (4, 1, 2, 3),
)


@dataclass(frozen=True)
class TTAAggregate:
    answer: list[int]
    valid_views: int
    consistent: bool
    used_fallback: bool
    tie_break: str | None


@dataclass(frozen=True)
class TTAModeAggregate:
    answer: list[int]
    valid_views: int
    used_fallback: bool
    tie_break: str | None
    vote_counts: dict[str, int]
    confidence_sums: dict[str, float]
    confidence_weight_sums: dict[str, float]
    valid_confidence_views: int


def permute_input_row(row: Any, order: Sequence[int]) -> dict[str, Any]:
    """Return a row whose input slots follow 1-indexed original slot labels."""
    if not is_permutation(order):
        raise ValueError(f"Invalid TTA input order: {order!r}")
    permuted = dict(row)
    for new_slot, original_slot in enumerate(order, start=1):
        permuted[f"Input_{new_slot}"] = row[f"Input_{original_slot}"]
    return permuted


def canonicalize_view_prediction(
    view_prediction: Sequence[int], *, view_order: Sequence[int]
) -> list[int]:
    """Map submission positions from a permuted view back to original input slots."""
    if not is_permutation(view_prediction):
        raise ValueError(f"Invalid TTA prediction: {view_prediction!r}")
    if not is_permutation(view_order):
        raise ValueError(f"Invalid TTA input order: {view_order!r}")
    canonical = [0, 0, 0, 0]
    for view_slot, original_slot in enumerate(view_order):
        canonical[original_slot - 1] = view_prediction[view_slot]
    return canonical


def tta_agreement_pattern(
    predictions: Sequence[Sequence[int] | None],
) -> str:
    """Summarize canonical vote counts without discarding invalid views."""
    valid = [tuple(value) for value in predictions if value is not None]
    invalid = len(predictions) - len(valid)
    if not valid:
        return "invalid"
    if any(not is_permutation(value) for value in valid):
        raise ValueError(f"Invalid canonical TTA predictions: {predictions!r}")
    pattern = "-".join(
        str(count) for count in sorted(Counter(valid).values(), reverse=True)
    )
    return f"{pattern}+{invalid}i" if invalid else pattern


def aggregate_tta_predictions(
    predictions: Sequence[Sequence[int] | None],
) -> TTAAggregate:
    """Vote on canonical answers; ties resolve to the lexicographically smallest."""
    valid = [tuple(value) for value in predictions if value is not None]
    if not valid:
        return TTAAggregate(
            answer=IDENTITY_ORDER.copy(),
            valid_views=0,
            consistent=False,
            used_fallback=True,
            tie_break=None,
        )
    if any(not is_permutation(value) for value in valid):
        raise ValueError(f"Invalid canonical TTA predictions: {predictions!r}")
    counts = Counter(valid)
    most_votes = max(counts.values())
    winners = sorted(answer for answer, count in counts.items() if count == most_votes)
    return TTAAggregate(
        answer=list(winners[0]),
        valid_views=len(valid),
        consistent=len(counts) == 1 and len(valid) > 1,
        used_fallback=False,
        tie_break="lexicographic" if len(winners) > 1 else None,
    )


def aggregate_tta_modes(
    predictions: Sequence[Sequence[int] | None],
    confidence_means: Sequence[float | None],
    *,
    temperature: float = 1.0,
) -> dict[str, TTAModeAggregate]:
    """Compute hard and confidence aggregations from one set of generated views."""
    if len(predictions) != len(confidence_means):
        raise ValueError("TTA predictions and confidences must have equal length")
    if temperature <= 0:
        raise ValueError("Confidence temperature must be positive")
    hard = aggregate_tta_predictions(predictions)
    valid_predictions = [tuple(value) for value in predictions if value is not None]
    if not valid_predictions:
        fallback = TTAModeAggregate(
            answer=hard.answer,
            valid_views=0,
            used_fallback=True,
            tie_break=None,
            vote_counts={},
            confidence_sums={},
            confidence_weight_sums={},
            valid_confidence_views=0,
        )
        return {
            mode: fallback
            for mode in ("hard", "confidence_tiebreak", "confidence_weighted")
        }

    counts = Counter(valid_predictions)
    labels = {answer: str(list(answer)) for answer in counts}
    confidence_pairs = [
        (tuple(prediction), float(confidence))
        for prediction, confidence in zip(predictions, confidence_means, strict=True)
        if prediction is not None and confidence is not None and math.isfinite(confidence)
    ]
    confidence_sums = {answer: 0.0 for answer in counts}
    weight_sums = {answer: 0.0 for answer in counts}
    if confidence_pairs:
        maximum_confidence = max(confidence for _, confidence in confidence_pairs)
        for answer, confidence in confidence_pairs:
            confidence_sums[answer] += confidence
            weight_sums[answer] += math.exp(
                (confidence - maximum_confidence) / temperature
            )

    common = {
        "valid_views": len(valid_predictions),
        "used_fallback": False,
        "vote_counts": {labels[key]: counts[key] for key in sorted(counts)},
        "confidence_sums": {
            labels[key]: confidence_sums[key] for key in sorted(counts)
        },
        "confidence_weight_sums": {
            labels[key]: weight_sums[key] for key in sorted(counts)
        },
        "valid_confidence_views": len(confidence_pairs),
    }
    hard_result = TTAModeAggregate(
        answer=hard.answer,
        tie_break=hard.tie_break,
        **common,
    )

    most_votes = max(counts.values())
    vote_winners = sorted(answer for answer, count in counts.items() if count == most_votes)
    if len(vote_winners) == 1:
        tiebreak_answer = vote_winners[0]
        tiebreak_reason = None
    else:
        best_tie_score = max(weight_sums[answer] for answer in vote_winners)
        confidence_winners = [
            answer
            for answer in vote_winners
            if math.isclose(weight_sums[answer], best_tie_score)
        ]
        has_tie_confidence = any(weight_sums[answer] > 0 for answer in vote_winners)
        tiebreak_answer = sorted(confidence_winners)[0]
        tiebreak_reason = (
            "confidence"
            if has_tie_confidence and len(confidence_winners) == 1
            else "lexicographic"
        )
    tiebreak_result = TTAModeAggregate(
        answer=list(tiebreak_answer),
        tie_break=tiebreak_reason,
        **common,
    )

    if not confidence_pairs:
        weighted_answer = tuple(hard.answer)
        weighted_reason = "hard_fallback"
    else:
        best_weight = max(weight_sums.values())
        weighted_winners = sorted(
            answer
            for answer, score in weight_sums.items()
            if math.isclose(score, best_weight)
        )
        weighted_answer = weighted_winners[0]
        weighted_reason = (
            "confidence_weighted"
            if len(weighted_winners) == 1
            else "lexicographic"
        )
    weighted_result = TTAModeAggregate(
        answer=list(weighted_answer),
        tie_break=weighted_reason,
        **common,
    )
    return {
        "hard": hard_result,
        "confidence_tiebreak": tiebreak_result,
        "confidence_weighted": weighted_result,
    }
