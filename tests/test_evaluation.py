from snuaichal.evaluation import compute_exact_match_metrics


def test_exact_match_metrics_separate_identity_and_parse_failures() -> None:
    predictions = [[1, 2, 3, 4], None, [2, 1, 3, 4], [4, 3, 2, 1]]
    references = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 3, 4],
        [1, 2, 3, 4],
    ]

    metrics = compute_exact_match_metrics(predictions, references)

    assert metrics == {
        "samples": 4,
        "exact_matches": 2,
        "exact_match": 0.5,
        "identity_samples": 2,
        "identity_exact_match": 0.5,
        "non_identity_samples": 2,
        "non_identity_exact_match": 0.5,
        "parse_failures": 1,
        "parse_failure_rate": 0.25,
    }
