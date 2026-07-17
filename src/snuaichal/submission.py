"""Utilities for parsing model output and validating submission files."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

FRAME_COUNT = 4
IDENTITY_ORDER = [1, 2, 3, 4]
_LIST_PATTERN = re.compile(r"\[[^\[\]]*\]")


def is_permutation(value: object, frame_count: int = FRAME_COUNT) -> bool:
    """Return whether *value* is a 1-indexed permutation of all frames."""
    if not isinstance(value, (list, tuple)) or len(value) != frame_count:
        return False
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return False
    return sorted(value) == list(range(1, frame_count + 1))


def chronological_to_positions(order: Sequence[int]) -> list[int]:
    """Convert chronological image labels to each image's chronological position.

    Example: chronological order [4, 2, 1, 3] becomes [3, 2, 4, 1].
    """
    if not is_permutation(order, len(order)):
        raise ValueError(f"Invalid frame permutation: {order!r}")

    positions = [0] * len(order)
    for position, image_number in enumerate(order, start=1):
        positions[image_number - 1] = position
    return positions


def find_last_valid_permutation_substring(
    output_text: str,
) -> tuple[str, int, int] | None:
    """Return the text and character span of the last valid bracketed answer."""
    for match in reversed(list(_LIST_PATTERN.finditer(output_text))):
        try:
            candidate = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if is_permutation(candidate):
            return match.group(0), match.start(), match.end()
    return None


def parse_model_output(output_text: str) -> list[int] | None:
    """Parse the final bracketed list and convert it to submission format.

    Returning ``None`` lets the caller count parse failures explicitly instead of
    silently treating them as confident identity-order predictions.
    """
    located = find_last_valid_permutation_substring(output_text)
    if located is None:
        return None
    candidate = ast.literal_eval(located[0])
    return chronological_to_positions(candidate)


def answer_to_string(answer: Sequence[int]) -> str:
    """Serialize a validated answer using the competition's required format."""
    if not is_permutation(answer):
        raise ValueError(f"Invalid submission answer: {answer!r}")
    return str(list(answer))


def validate_submission_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_ids: Sequence[str],
    expected_count: int | None = None,
) -> None:
    """Validate final row count, ID order, uniqueness, and permutation answers."""
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, got {len(records)}")
    ids = [str(record.get("Id")) for record in records]
    if ids != [str(sample_id) for sample_id in expected_ids]:
        raise ValueError("Submission IDs or row order do not match the input CSV")
    if len(set(ids)) != len(ids):
        raise ValueError("Submission IDs must be unique")
    for record in records:
        raw_answer = record.get("Answer")
        try:
            answer = ast.literal_eval(str(raw_answer))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid Answer for Id={record.get('Id')}: {raw_answer!r}"
            ) from exc
        if not is_permutation(answer):
            raise ValueError(
                f"Invalid Answer for Id={record.get('Id')}: {raw_answer!r}"
            )
