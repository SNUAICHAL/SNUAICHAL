import csv
import json
import os
from pathlib import Path

from scripts import run_phase1_followup_queue as queue


def test_process_alive_recognizes_current_process() -> None:
    assert queue.process_alive(os.getpid()) is True
    assert queue.process_alive(None) is False


def test_sweep_complete_requires_all_expected_successes() -> None:
    status = {
        "experiments": {
            f"qwen3vl8b-ckpt{step}-nf4-greedy-tta1-val954": {
                "status": "succeeded"
            }
            for step in queue.EXPECTED_SWEEP_STEPS
        }
    }

    assert queue.sweep_complete(status) is True
    status["active_experiment"] = "still-running"
    assert queue.sweep_complete(status) is False


def test_read_sweep_results_selects_exact_match_then_runtime(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_sweep.csv"
    fieldnames = [
        "checkpoint",
        "global_step",
        "exact_match_count",
        "exact_match_accuracy",
        "parse_failures",
        "seconds_per_sample",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(queue.EXPECTED_SWEEP_STEPS):
            writer.writerow(
                {
                    "checkpoint": f"checkpoint-{step}",
                    "global_step": step,
                    "exact_match_count": 500 if step in (4250, 4292) else 400 + index,
                    "exact_match_accuracy": 500 / 954,
                    "parse_failures": 0,
                    "seconds_per_sample": 2.0 if step == 4292 else 2.5,
                    "status": "succeeded",
                }
            )

    results = queue.read_sweep_results(path)

    assert results[0].step == 4292
    assert results[1].step == 4250


def test_successful_metrics_reuses_only_valid_attempt(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-001"
    attempt.mkdir()
    (attempt / "exit-code.txt").write_text("0\n", encoding="utf-8")
    (attempt / "predictions.csv").write_text(
        "Id,Answer\na,\"[1, 2, 3, 4]\"\n", encoding="utf-8"
    )
    (attempt / "audit.jsonl").write_text(
        json.dumps({"Id": "a"}) + "\n", encoding="utf-8"
    )
    (attempt / "metrics.json").write_text(
        json.dumps({"samples": 1, "exact_match": 1.0}), encoding="utf-8"
    )

    assert queue.successful_metrics(tmp_path, expected_rows=1)["exact_match"] == 1.0


def test_official_inference_command_is_fixed_to_safe_second_submission(
    tmp_path: Path,
) -> None:
    command = queue.official_inference_command(tmp_path)

    assert command[command.index("--adapter-path") + 1].endswith("checkpoint-4292")
    assert command[command.index("--precision") + 1] == "nf4"
    assert command[command.index("--tta") + 1] == "4"
    assert (
        command[command.index("--aggregation-mode") + 1]
        == "confidence_tiebreak"
    )
    assert command[command.index("--output") + 1].endswith(
        queue.OFFICIAL_SUBMISSION_NAME
    )


def test_find_remote_submission_requires_file_and_description() -> None:
    rows = [
        {
            "fileName": queue.OFFICIAL_SUBMISSION_NAME,
            "description": "some other run",
            "status": "complete",
        },
        {
            "fileName": queue.OFFICIAL_SUBMISSION_NAME,
            "description": queue.OFFICIAL_DESCRIPTION,
            "status": "pending",
        },
    ]

    match = queue.find_remote_submission(
        rows,
        file_name=queue.OFFICIAL_SUBMISSION_NAME,
        description=queue.OFFICIAL_DESCRIPTION,
    )

    assert match == rows[1]
    assert queue.remote_status_complete("SubmissionStatus.COMPLETE") is True
    assert queue.remote_status_failed("SubmissionStatus.ERROR") is True
