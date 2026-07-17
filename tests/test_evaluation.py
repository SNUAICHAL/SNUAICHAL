from snuaichal.evaluation import compare_prediction_modes, compute_exact_match_metrics


def test_exact_match_metrics_separate_identity_and_parse_failures() -> None:
    predictions = [[1, 2, 3, 4], None, [2, 1, 3, 4], [4, 3, 2, 1]]
    references = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 1, 3, 4],
        [1, 2, 3, 4],
    ]

    metrics = compute_exact_match_metrics(predictions, references)

    expected = {
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
    assert {key: metrics[key] for key in expected} == expected
    assert len(metrics["class_accuracy"]) == 24


def test_extended_metrics_include_tta_runtime_vram_and_full_confusion() -> None:
    metrics = compute_exact_match_metrics(
        predictions=[[1, 2, 3, 4], [2, 1, 3, 4], None],
        references=[[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]],
        tta_consistent=[True, False, False],
        elapsed_seconds=6.0,
        peak_vram_bytes=1024 * 1024 * 2048,
    )

    assert metrics["tta_consistency"] == 1 / 3
    assert metrics["inference_seconds_per_sample"] == 2.0
    assert metrics["peak_vram_mib"] == 2048.0
    confusion = metrics["confusion"]
    assert len(confusion) == 24
    assert all(len(counts) == 25 for counts in confusion.values())
    assert confusion["[1, 2, 3, 4]"]["[1, 2, 3, 4]"] == 1
    assert confusion["[1, 2, 3, 4]"]["[2, 1, 3, 4]"] == 1
    assert confusion["[4, 3, 2, 1]"]["parse_failure"] == 1


def test_phase1_metrics_include_classes_groups_agreement_and_runtime_budget() -> None:
    metrics = compute_exact_match_metrics(
        predictions=[[1, 2, 3, 4], [1, 2, 3, 4], None],
        references=[[1, 2, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]],
        no_ordering=[True, False, False],
        tta_agreement_patterns=["4", "2-1-1", "invalid"],
        elapsed_seconds=9.0,
        peak_vram_bytes=3 * 1024**3,
        model_precision="nf4",
        image_grid_thw=[[1, 32, 24], [1, 24, 32], [1, 32, 24]],
        visual_tokens=[192, 192, 192],
        expected_test_samples=819,
    )

    assert metrics["class_accuracy"]["[1, 2, 3, 4]"] == {
        "accuracy": 1.0,
        "exact_matches": 1,
        "samples": 1,
    }
    assert metrics["class_accuracy"]["[2, 1, 3, 4]"] == {
        "accuracy": 0.0,
        "exact_matches": 0,
        "samples": 2,
    }
    assert len(metrics["class_accuracy"]) == 24
    assert metrics["no_ordering_accuracy"]["true"]["accuracy"] == 1.0
    assert metrics["no_ordering_accuracy"]["false"]["accuracy"] == 0.0
    assert metrics["tta_agreement_patterns"] == {"2-1-1": 1, "4": 1, "invalid": 1}
    assert metrics["estimated_test_seconds"] == 2457.0
    assert metrics["model_precision"] == "nf4"
    assert metrics["image_grid_thw"] == {
        "[1, 24, 32]": 1,
        "[1, 32, 24]": 2,
    }
    assert metrics["visual_tokens"] == {
        "count": 3,
        "maximum": 192,
        "mean": 192.0,
        "minimum": 192,
    }


def test_prediction_mode_comparison_records_changes_corrections_and_regressions() -> None:
    references = [[1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4], [4, 1, 2, 3]]
    baseline = [[1, 2, 3, 4], [1, 2, 3, 4], [3, 1, 2, 4], None]
    candidate = [[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 3, 4], [4, 1, 2, 3]]

    comparison = compare_prediction_modes(candidate, baseline, references)

    assert comparison == {
        "accuracy": 0.75,
        "corrected": 2,
        "exact_matches": 3,
        "net_gain": 1,
        "prediction_changes": 3,
        "worsened": 1,
    }
