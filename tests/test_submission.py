import pytest

from snuaichal.submission import (
    answer_to_string,
    chronological_to_positions,
    is_permutation,
    parse_model_output,
)


def test_chronological_order_is_converted_to_submission_positions() -> None:
    assert chronological_to_positions([4, 2, 1, 3]) == [3, 2, 4, 1]


def test_parser_ignores_invalid_list_before_valid_answer() -> None:
    text = "Candidates: [1, 1, 2, 3]. Final answer: [2, 4, 1, 3]"
    assert parse_model_output(text) == [3, 1, 4, 2]


def test_parser_prefers_final_valid_answer() -> None:
    text = "Example: [1, 2, 3, 4]. Final answer: [4, 2, 1, 3]"
    assert parse_model_output(text) == [3, 2, 4, 1]


@pytest.mark.parametrize(
    "text", ["no list", "[1, 2, 3]", "[1, 2, 3, 5]", "[True, 2, 3, 4]"]
)
def test_parser_rejects_invalid_output(text: str) -> None:
    assert parse_model_output(text) is None


def test_submission_serialization_and_validation() -> None:
    assert is_permutation([1, 4, 2, 3])
    assert answer_to_string([1, 4, 2, 3]) == "[1, 4, 2, 3]"
    with pytest.raises(ValueError):
        answer_to_string([1, 1, 2, 3])
