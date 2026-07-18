import csv
import json
import os
import sys
from pathlib import Path

import psutil
import pytest

from scripts import run_phase1_sweep as runner


def _artifact_writer_command(expected_rows: int, *, exit_code: int = 0) -> list[str]:
    code = f"""
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
with (root / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Id', 'Answer'])
    writer.writeheader()
    for index in range({expected_rows}):
        writer.writerow({{'Id': str(index), 'Answer': '[1, 2, 3, 4]'}})
with (root / 'audit.jsonl').open('w', encoding='utf-8') as f:
    for index in range({expected_rows}):
        f.write(json.dumps({{'Id': str(index)}}) + '\\n')
(root / 'metrics.json').write_text(json.dumps({{'samples': {expected_rows}}}), encoding='utf-8')
raise SystemExit({exit_code})
"""
    return [sys.executable, "-c", code, "{attempt_dir}"]


def test_default_child_environment_rejects_api_keys_and_authenticated_proxy(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    monkeypatch.setenv("HTTPS_PROXY", "https://user:secret@proxy.example:443")
    before = dict(os.environ)

    child = runner.default_child_environment()

    assert dict(os.environ) == before
    assert "NVIDIA_API_KEY" not in child
    assert "HTTPS_PROXY" not in child
    assert child["NVIDIA_VISIBLE_DEVICES"] == "all"


def test_runner_records_success_and_skips_valid_completed_experiment(tmp_path: Path) -> None:
    experiment = runner.Experiment(
        experiment_id="model-ckpt1-nf4-greedy-tta1-val2",
        command=_artifact_writer_command(2),
        expected_rows=2,
    )

    assert runner.run_plan([experiment], output_root=tmp_path, poll_seconds=0.01) is True
    assert runner.run_plan([experiment], output_root=tmp_path, poll_seconds=0.01) is True

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    record = status["experiments"][experiment.experiment_id]
    assert record["status"] == "succeeded"
    assert record["attempt"] == 1
    attempt = tmp_path / experiment.experiment_id / "attempt-001"
    assert (attempt / "command.txt").is_file()
    assert (attempt / "stdout.log").is_file()
    assert (attempt / "stderr.log").is_file()
    assert (attempt / "exit-code.txt").read_text(encoding="utf-8").strip() == "0"
    assert len((tmp_path / "experiment_registry.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_runner_interrupt_records_terminal_state(monkeypatch, tmp_path: Path) -> None:
    experiment = runner.Experiment(
        experiment_id="interrupted",
        command=["tool"],
        expected_rows=1,
    )

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_command", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_plan([experiment], output_root=tmp_path, poll_seconds=0.01)

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    record = status["experiments"]["interrupted"]
    assert record["status"] == "interrupted"
    assert record["ended_at"]
    assert "active_experiment" not in status
    assert "KeyboardInterrupt" in record["artifact_errors"][0]


def test_runner_preserves_failed_attempt_and_stops_before_next_experiment(tmp_path: Path) -> None:
    failed = runner.Experiment(
        experiment_id="failed-nf4-greedy-tta1-val1",
        command=_artifact_writer_command(1, exit_code=7),
        expected_rows=1,
    )
    blocked = runner.Experiment(
        experiment_id="blocked-nf4-greedy-tta1-val1",
        command=_artifact_writer_command(1),
        expected_rows=1,
    )

    assert runner.run_plan([failed, blocked], output_root=tmp_path, poll_seconds=0.01) is False

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["experiments"][failed.experiment_id]["status"] == "failed"
    assert blocked.experiment_id not in status["experiments"]
    assert (tmp_path / failed.experiment_id / "attempt-001" / "exit-code.txt").read_text(
        encoding="utf-8"
    ).strip() == "7"


def test_artifact_validation_rejects_wrong_row_count(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt-001"
    command = _artifact_writer_command(1)
    concrete = [str(attempt) if value == "{attempt_dir}" else value for value in command]
    assert runner.run_command(concrete, attempt, poll_seconds=0.01) == 0

    errors = runner.validate_artifacts(attempt, expected_rows=2)

    assert any("predictions" in error for error in errors)
    assert any("audit" in error for error in errors)


def test_run_command_interrupt_terminates_child_and_records_exit(
    monkeypatch, tmp_path: Path
) -> None:
    attempt = tmp_path / "interrupted-attempt"

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            attempt,
            poll_seconds=0.01,
        )

    pid = int((attempt / "pid.txt").read_text(encoding="utf-8"))
    assert not psutil.pid_exists(pid)
    assert (attempt / "exit-code.txt").is_file()
    assert (attempt / "ended-at.txt").is_file()


def test_materialize_command_replaces_attempt_placeholder_inside_paths(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt-001"

    concrete = runner.materialize_command(
        ["python", "--output", "{attempt_dir}/predictions.csv"], attempt
    )

    assert concrete == ["python", "--output", str(attempt / "predictions.csv")]


def test_phase1_inference_command_has_complete_required_identity() -> None:
    command = runner.inference_command(
        adapter=Path("outputs/qwen3-vl-8b-aug/checkpoint-4292"),
        precision="nf4",
        tta=4,
        expected_rows=954,
    )

    for flag in (
        "--model-path",
        "--model-repository",
        "--model-family",
        "--model-revision",
        "--model-manifest",
        "--adapter-path",
        "--precision",
        "--tta",
        "--aggregation-mode",
        "--fallback-policy",
        "--metrics-output",
    ):
        assert flag in command


def test_refresh_checkpoint_sweep_writes_required_summary_columns(tmp_path: Path) -> None:
    attempt = tmp_path / "qwen3vl8b-ckpt3000-nf4-greedy-tta1-val954" / "attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "metrics.json").write_text(
        json.dumps(
            {
                "exact_matches": 700,
                "samples": 954,
                "exact_match": 700 / 954,
                "parse_failures": 0,
                "inference_seconds_per_sample": 3.5,
                "estimated_test_seconds": 2866.5,
                "peak_vram_mib": 7000.0,
                "visual_tokens": {"mean": 200.0},
                "model_precision": "nf4",
                "tta_views": 1,
            }
        ),
        encoding="utf-8",
    )
    status = {
        "experiments": {
            "qwen3vl8b-ckpt3000-nf4-greedy-tta1-val954": {
                "attempt_dir": str(attempt),
                "status": "succeeded",
            }
        }
    }
    runner.atomic_write_json(tmp_path / "status.json", status)

    rows = runner.refresh_checkpoint_sweep(
        tmp_path,
        csv_path=tmp_path / "checkpoint_sweep.csv",
        json_path=tmp_path / "checkpoint_sweep.json",
    )

    assert rows[0]["checkpoint"] == "checkpoint-3000"
    assert rows[0]["global_step"] == 3000
    assert rows[0]["exact_match_count"] == 700
    assert rows[0]["peak_vram_bytes"] == 7000 * 1024 * 1024
    with (tmp_path / "checkpoint_sweep.csv").open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[0]["decode_mode"] == "greedy"
    assert json.loads((tmp_path / "checkpoint_sweep.json").read_text(encoding="utf-8")) == rows
