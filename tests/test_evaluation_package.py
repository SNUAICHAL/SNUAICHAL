from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_submission_artifacts import validate
from scripts.verify_evaluation_package import (
    verify_static_contract,
    verify_unlabeled_dataset,
)
from snuaichal.tta import (
    CYCLIC_TTA_ORDERS,
    aggregate_tta_modes,
    aggregate_tta_predictions,
    tta_agreement_pattern,
)


def _dataset(root: Path, *, include_answer: bool = False) -> Path:
    data = root / "data"
    images = data / "test" / "sample-1"
    images.mkdir(parents=True)
    for index in range(1, 5):
        (images / f"{index}.png").write_bytes(f"image-{index}".encode())
    columns = [
        "Id",
        "Input_1",
        "Input_2",
        "Input_3",
        "Input_4",
        "Sentence",
    ]
    if include_answer:
        columns.append("Answer")
    row = {
        "Id": "sample-1",
        "Input_1": "1.png",
        "Input_2": "2.png",
        "Input_3": "3.png",
        "Input_4": "4.png",
        "Sentence": "A deterministic temporal sequence.",
    }
    if include_answer:
        row["Answer"] = "[1, 2, 3, 4]"
    with (data / "test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    return data


def test_committed_final_contract_is_internally_consistent() -> None:
    result = verify_static_contract(Path("."))

    assert result["status"] == "PASS"
    assert result["checkpoint_step"] == 2726
    assert result["model_files"] == 29
    assert result["model_shards"] == 15
    assert result["public_score"] == pytest.approx(0.93542)


def test_unlabeled_dataset_preflight_accepts_exact_schema(tmp_path: Path) -> None:
    result = verify_unlabeled_dataset(_dataset(tmp_path))

    assert result["status"] == "PASS"
    assert result["answer_column_present"] is False
    assert result["rows"] == 1
    assert len(result["referenced_image_tree_sha256"]) == 64
    assert result["referenced_images"] == 4


def test_unlabeled_dataset_preflight_rejects_answer_column(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="columns must be exactly"):
        verify_unlabeled_dataset(_dataset(tmp_path, include_answer=True))


def test_unlabeled_dataset_preflight_rejects_path_escape(tmp_path: Path) -> None:
    data = _dataset(tmp_path)
    csv_path = data / "test.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(text.replace("1.png", "../1.png"), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsafe Input_1"):
        verify_unlabeled_dataset(data)


def _semantic_audit_fixture(tmp_path: Path) -> SimpleNamespace:
    data = _dataset(tmp_path)
    submission = tmp_path / "submission.csv"
    submission.write_text(
        'Id,Answer\nsample-1,"[1, 2, 3, 4]"\n', encoding="utf-8"
    )
    canonical_predictions = [[1, 2, 3, 4] for _ in CYCLIC_TTA_ORDERS]
    confidence_means = [0.0] * len(CYCLIC_TTA_ORDERS)
    modes = aggregate_tta_modes(canonical_predictions, confidence_means)
    hard = aggregate_tta_predictions(canonical_predictions)
    views = []
    for order in CYCLIC_TTA_ORDERS:
        chronological = [order.index(position) + 1 for position in range(1, 5)]
        views.append(
            {
                "input_order": list(order),
                "raw_output": str(chronological),
                "view_prediction": list(order),
                "canonical_prediction": [1, 2, 3, 4],
                "parse_ok": True,
                "answer_logprob_mean": 0.0,
                "answer_confidence_valid": True,
            }
        )
    audit_row = {
        "Id": "sample-1",
        "answer": [1, 2, 3, 4],
        "parse_ok": True,
        "valid_tta_views": 4,
        "aggregation_mode": "hard",
        "tie_break": modes["hard"].tie_break,
        "tta_consistent": hard.consistent,
        "tta_agreement_pattern": tta_agreement_pattern(canonical_predictions),
        "aggregations": {name: asdict(value) for name, value in modes.items()},
        "views": views,
    }
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps(audit_row) + "\n", encoding="utf-8")
    return SimpleNamespace(
        test_csv=data / "test.csv",
        submission=submission,
        audit=audit,
        expected_tta=4,
        aggregation_mode="hard",
        metrics=None,
    )


def test_semantic_auditor_recomputes_raw_views_and_hard_vote(
    tmp_path: Path,
) -> None:
    result = validate(_semantic_audit_fixture(tmp_path))

    assert result["rows"] == 1
    assert result["parse_failures"] == 0
    assert result["view_parse_failures"] == 0


def test_semantic_auditor_rejects_forged_raw_output(tmp_path: Path) -> None:
    args = _semantic_audit_fixture(tmp_path)
    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    payload["views"][0]["raw_output"] = "nonsense"
    args.audit.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Raw-output/view prediction mismatch"):
        validate(args)


def _valid_physical_metrics() -> dict[str, object]:
    return {
        "aggregation_mode": "hard",
        "model_repository": "Qwen/Qwen3.6-27B",
        "model_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "verified_model_tree_sha256": (
            "e4107e6508793261ca372faf4b560dcb55a5b6ba79a5ab921bfe1b25a207ec07"
        ),
        "tta_views": 4,
        "tta_orders": [list(order) for order in CYCLIC_TTA_ORDERS],
        "parse_failures": 0,
        "physical_measurement_status": "valid",
        "unexpected_cuda_processes": [],
        "trusted_cuda_child_seen": True,
        "process_identity_match": True,
        "sampling_started_before_model_load": True,
        "sampling_finished_after_work": True,
        "sample_count": 5,
        "physical_peak_observed_bytes": 21 * 1024**3,
        "physical_total_vram_bytes": 48 * 1024**3,
    }


def test_semantic_auditor_accepts_24gib_peak_on_larger_gpu(tmp_path: Path) -> None:
    args = _semantic_audit_fixture(tmp_path)
    args.metrics = tmp_path / "metrics.json"
    args.metrics.write_text(json.dumps(_valid_physical_metrics()), encoding="utf-8")

    result = validate(args)

    assert result["physical_peak_observed_bytes"] == 21 * 1024**3


def test_semantic_auditor_rejects_boolean_physical_evidence(tmp_path: Path) -> None:
    args = _semantic_audit_fixture(tmp_path)
    args.metrics = tmp_path / "metrics.json"
    metrics = _valid_physical_metrics()
    metrics["physical_peak_observed_bytes"] = True
    args.metrics.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="physical VRAM evidence is incomplete"):
        validate(args)
