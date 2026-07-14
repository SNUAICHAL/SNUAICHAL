"""Utilities for parsing model output and validating submission files."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

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


def parse_model_output(output_text: str) -> list[int] | None:
    """Parse the last valid permutation and convert it to submission format.

    Returning ``None`` lets the caller count parse failures explicitly instead of
    silently treating them as confident identity-order predictions.
    """
    for match in reversed(list(_LIST_PATTERN.finditer(output_text))):
        try:
            candidate = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if is_permutation(candidate):
            return chronological_to_positions(candidate)
    return None


def answer_to_string(answer: Sequence[int]) -> str:
    """Serialize a validated answer using the competition's required format."""
    if not is_permutation(answer):
        raise ValueError(f"Invalid submission answer: {answer!r}")
    return str(list(answer))
