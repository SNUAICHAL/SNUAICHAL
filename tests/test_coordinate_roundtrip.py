"""Exhaustive coordinate-system contracts for training, TTA, and submission."""

from __future__ import annotations

import ast
import itertools

from snuaichal.submission import chronological_to_positions, parse_model_output
from snuaichal.training import (
    answer_positions_to_chronological,
    permute_training_row,
)
from snuaichal.tta import canonicalize_view_prediction


def _row(answer: tuple[int, ...]) -> dict[str, str]:
    return {
        "Id": "coordinate-contract",
        "Sentence": "A sequence.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": str(list(answer)),
    }


def test_all_24_answers_round_trip_between_csv_and_chronological_coordinates() -> None:
    for csv_answer in itertools.permutations(range(1, 5)):
        chronological = answer_positions_to_chronological(str(list(csv_answer)))
        assert chronological_to_positions(chronological) == list(csv_answer)
        assert parse_model_output(str(chronological)) == list(csv_answer)


def test_all_24_answers_times_24_input_permutations_round_trip() -> None:
    for csv_answer in itertools.permutations(range(1, 5)):
        for input_order in itertools.permutations(range(1, 5)):
            augmented = permute_training_row(_row(csv_answer), list(input_order))
            view_positions = ast.literal_eval(augmented["Answer"])
            chronological_target = answer_positions_to_chronological(augmented["Answer"])

            # A perfect model emits the chronological image-label order used in training.
            parsed_view_positions = parse_model_output(str(chronological_target))
            assert parsed_view_positions == view_positions
            assert canonicalize_view_prediction(
                parsed_view_positions, view_order=input_order
            ) == list(csv_answer)
