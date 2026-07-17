from snuaichal.tta import (
    CYCLIC_TTA_ORDERS,
    aggregate_tta_modes,
    aggregate_tta_predictions,
    canonicalize_view_prediction,
    permute_input_row,
    tta_agreement_pattern,
)


def sample_row() -> dict[str, str]:
    return {
        "Id": "sample",
        "Sentence": "A sequence.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
    }


def test_cyclic_tta_places_every_frame_in_every_slot_once() -> None:
    assert CYCLIC_TTA_ORDERS == (
        (1, 2, 3, 4),
        (2, 3, 4, 1),
        (3, 4, 1, 2),
        (4, 1, 2, 3),
    )
    for slot in range(4):
        assert sorted(order[slot] for order in CYCLIC_TTA_ORDERS) == [1, 2, 3, 4]


def test_tta_row_permutation_uses_requested_input_order() -> None:
    row = permute_input_row(sample_row(), (2, 3, 4, 1))

    assert [row[f"Input_{slot}"] for slot in range(1, 5)] == [
        "two.jpg",
        "three.jpg",
        "four.jpg",
        "one.jpg",
    ]


def test_tta_prediction_is_mapped_back_to_original_input_slots() -> None:
    view_prediction = [2, 4, 1, 3]

    assert canonicalize_view_prediction(
        view_prediction, view_order=(2, 3, 4, 1)
    ) == [3, 2, 4, 1]


def test_tta_uses_majority_vote_in_canonical_answer_space() -> None:
    result = aggregate_tta_predictions(
        [[3, 2, 4, 1], [1, 2, 3, 4], [3, 2, 4, 1], None]
    )

    assert result.answer == [3, 2, 4, 1]
    assert result.valid_views == 3
    assert result.consistent is False


def test_tta_tie_break_is_lexicographically_smallest_answer() -> None:
    result = aggregate_tta_predictions(
        [[3, 2, 4, 1], [1, 2, 3, 4], [3, 2, 4, 1], [1, 2, 3, 4]]
    )

    assert result.answer == [1, 2, 3, 4]
    assert result.tie_break == "lexicographic"


def test_tta_falls_back_to_identity_only_when_all_views_are_invalid() -> None:
    result = aggregate_tta_predictions([None, None, None, None])

    assert result.answer == [1, 2, 3, 4]
    assert result.valid_views == 0
    assert result.used_fallback is True


def test_tta_agreement_pattern_records_split_votes_and_invalid_views() -> None:
    assert tta_agreement_pattern(
        [[1, 2, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4]]
    ) == "2-1-1"
    assert tta_agreement_pattern([[1, 2, 3, 4], None, None, None]) == "1+3i"
    assert tta_agreement_pattern([None, None, None, None]) == "invalid"


def test_confidence_tiebreak_only_changes_equal_vote_candidates() -> None:
    predictions = [
        [2, 1, 3, 4],
        [1, 2, 3, 4],
        [2, 1, 3, 4],
        [1, 2, 3, 4],
    ]

    results = aggregate_tta_modes(predictions, [-2.0, -0.1, -2.0, -0.2])

    assert results["hard"].answer == [1, 2, 3, 4]
    assert results["hard"].tie_break == "lexicographic"
    assert results["confidence_tiebreak"].answer == [1, 2, 3, 4]
    assert results["confidence_tiebreak"].tie_break == "confidence"
    assert results["confidence_weighted"].answer == [1, 2, 3, 4]


def test_confidence_weighted_can_override_hard_majority_but_tiebreak_cannot() -> None:
    majority = [2, 1, 3, 4]
    minority = [1, 2, 3, 4]

    results = aggregate_tta_modes(
        [majority, majority, minority, None],
        [-5.0, -5.0, -0.01, None],
    )

    assert results["hard"].answer == majority
    assert results["confidence_tiebreak"].answer == majority
    assert results["confidence_weighted"].answer == minority
    assert results["confidence_weighted"].vote_counts[str(majority)] == 2
    assert results["confidence_weighted"].vote_counts[str(minority)] == 1
    assert results["confidence_weighted"].valid_confidence_views == 3


def test_invalid_confidence_keeps_vote_and_uses_lexicographic_final_fallback() -> None:
    predictions = [[2, 1, 3, 4], [1, 2, 3, 4], None, None]

    results = aggregate_tta_modes(predictions, [None, None, -0.1, None])

    assert results["confidence_tiebreak"].answer == [1, 2, 3, 4]
    assert results["confidence_tiebreak"].tie_break == "lexicographic"
    assert results["confidence_weighted"].answer == [1, 2, 3, 4]
    assert results["confidence_weighted"].tie_break == "hard_fallback"
