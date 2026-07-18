"""Queue safe Phase 1 follow-ups and a Qwen3.5-27B two-step training smoke.

This process is designed to be started while ``run_phase1_sweep.py`` is still
running. It never competes for the GPU: it waits for all six sweep experiments,
then chooses the best checkpoint from validated artifacts. Suspiciously low
validation accuracy triggers a diagnostic TTA4 run instead of long training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from scripts import run_phase1_sweep as runner


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "outputs" / "phase1"
FOLLOWUP_ROOT = ROOT / "outputs" / "phase1_followup"
SWEEP_CSV = ROOT / "outputs" / "checkpoint_sweep.csv"
MODEL_8B = ROOT / "models" / "Qwen3-VL-8B-Instruct"
MODEL_8B_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_8B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_8B_MANIFEST = (
    ROOT
    / "outputs"
    / "blocked_low_validation_resume"
    / "model-manifests"
    / f"qwen3-vl-8b-{MODEL_8B_REVISION}.json"
)
MODEL_27B = ROOT / "models" / "Qwen3.5-27B"
MODEL_27B_REPOSITORY = "Qwen/Qwen3.5-27B"
MODEL_27B_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_27B_MANIFEST = (
    ROOT
    / "outputs"
    / "blocked_low_validation_resume"
    / "model-manifests"
    / f"qwen35-27b-{MODEL_27B_REVISION}.json"
)
EXPECTED_SWEEP_STEPS = (3000, 3500, 3750, 4000, 4250, 4292)
COHERENCE_THRESHOLD = 0.70
COMPETITION = "snuaichallenge"
OFFICIAL_SUBMISSION_NAME = (
    "submission_v6_ckpt4292_nf4_tta4_confidence_tiebreak.csv"
)
OFFICIAL_AUDIT_NAME = (
    "submission_v6_ckpt4292_nf4_tta4_confidence_tiebreak.jsonl"
)
OFFICIAL_DESCRIPTION = (
    "Qwen3-VL-8B QLoRA checkpoint-4292 NF4 TTA4 confidence-tiebreak"
)
OFFICIAL_STAGE_ID = (
    "qwen3vl8b-ckpt4292-nf4-greedy-tta4-confidence-tiebreak-test819"
)


@dataclass(frozen=True)
class SweepResult:
    checkpoint: str
    step: int
    exact_matches: int
    accuracy: float
    parse_failures: int
    seconds_per_sample: float


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except (AttributeError, OSError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_queue_status(**values: Any) -> None:
    path = FOLLOWUP_ROOT / "queue-status.json"
    current = load_json(path) if path.is_file() else {}
    current.update(values)
    current["updated_at"] = runner.utc_now()
    atomic_json(path, current)


def sweep_complete(status: dict[str, Any]) -> bool:
    records = status.get("experiments", {})
    for step in EXPECTED_SWEEP_STEPS:
        experiment_id = f"qwen3vl8b-ckpt{step}-nf4-greedy-tta1-val954"
        if records.get(experiment_id, {}).get("status") != "succeeded":
            return False
    return "active_experiment" not in status


def sweep_failed(status: dict[str, Any]) -> str | None:
    for experiment_id, record in status.get("experiments", {}).items():
        if record.get("status") == "failed":
            return experiment_id
    return None


def wait_for_sweep(*, runner_pid: int | None, poll_seconds: float) -> None:
    while True:
        if SWEEP_ROOT.joinpath("status.json").is_file():
            status = load_json(SWEEP_ROOT / "status.json")
            failed = sweep_failed(status)
            if failed:
                raise RuntimeError(f"checkpoint sweep failed at {failed}")
            if sweep_complete(status):
                update_queue_status(state="sweep_complete", active_experiment=None)
                return
            update_queue_status(
                state="waiting_for_sweep",
                active_experiment=status.get("active_experiment"),
                sweep_runner_alive=process_alive(runner_pid),
            )
            if runner_pid and not process_alive(runner_pid):
                raise RuntimeError("sweep runner exited before all experiments succeeded")
        time.sleep(poll_seconds)


def read_sweep_results(path: Path = SWEEP_CSV) -> list[SweepResult]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results = [
        SweepResult(
            checkpoint=str(row["checkpoint"]),
            step=int(row["global_step"]),
            exact_matches=int(row["exact_match_count"]),
            accuracy=float(row["exact_match_accuracy"]),
            parse_failures=int(row["parse_failures"]),
            seconds_per_sample=float(row["seconds_per_sample"]),
        )
        for row in rows
        if row.get("status") == "succeeded"
    ]
    steps = {result.step for result in results}
    if steps != set(EXPECTED_SWEEP_STEPS):
        raise RuntimeError(f"checkpoint sweep is incomplete: {sorted(steps)}")
    return sorted(
        results,
        key=lambda item: (
            -item.exact_matches,
            item.parse_failures,
            item.seconds_per_sample,
            item.step,
        ),
    )


def inference_command(*, step: int, precision: str, tta: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "snuaichal.inference",
        "--test-csv",
        "data/train.csv",
        "--image-dir",
        "data/train",
        "--model-path",
        str(MODEL_8B.relative_to(ROOT)),
        "--model-repository",
        MODEL_8B_REPOSITORY,
        "--model-family",
        "qwen3_vl",
        "--model-revision",
        MODEL_8B_REVISION,
        "--model-manifest",
        str(MODEL_8B_MANIFEST.relative_to(ROOT)),
        "--adapter-path",
        f"outputs/qwen3-vl-8b-aug/checkpoint-{step}",
        "--validation-manifest",
        "outputs/qwen3-vl-8b-aug/split_manifest.json",
        "--precision",
        precision,
        "--tta",
        str(tta),
        "--aggregation-mode",
        "hard",
        "--fallback-policy",
        "identity",
        "--output",
        "{attempt_dir}/predictions.csv",
        "--audit-log",
        "{attempt_dir}/audit.jsonl",
        "--metrics-output",
        "{attempt_dir}/metrics.json",
    ]


def official_inference_command(attempt: Path) -> list[str]:
    """Build the fixed, auditable command for today's second submission."""
    return [
        sys.executable,
        "-m",
        "snuaichal.inference",
        "--test-csv",
        "data/test.csv",
        "--image-dir",
        "data/test",
        "--model-path",
        str(MODEL_8B.relative_to(ROOT)),
        "--model-repository",
        MODEL_8B_REPOSITORY,
        "--model-family",
        "qwen3_vl",
        "--model-revision",
        MODEL_8B_REVISION,
        "--model-manifest",
        str(MODEL_8B_MANIFEST.relative_to(ROOT)),
        "--adapter-path",
        "outputs/qwen3-vl-8b-aug/checkpoint-4292",
        "--precision",
        "nf4",
        "--image-size",
        "512",
        "--tta",
        "4",
        "--aggregation-mode",
        "confidence_tiebreak",
        "--fallback-policy",
        "identity",
        "--output",
        str(attempt / OFFICIAL_SUBMISSION_NAME),
        "--audit-log",
        str(attempt / OFFICIAL_AUDIT_NAME),
        "--metrics-output",
        str(attempt / "metrics.json"),
    ]


def next_attempt(stage_root: Path) -> Path:
    attempts = []
    if stage_root.is_dir():
        for candidate in stage_root.glob("attempt-*"):
            try:
                attempts.append(int(candidate.name.split("-")[-1]))
            except ValueError:
                continue
    return stage_root / f"attempt-{max(attempts, default=0) + 1:03d}"


def successful_metrics(stage_root: Path, *, expected_rows: int = 954) -> dict[str, Any] | None:
    for attempt in sorted(stage_root.glob("attempt-*"), reverse=True):
        exit_path = attempt / "exit-code.txt"
        if not exit_path.is_file() or exit_path.read_text(encoding="utf-8").strip() != "0":
            continue
        if runner.validate_artifacts(attempt, expected_rows=expected_rows):
            continue
        return load_json(attempt / "metrics.json")
    return None


def run_inference_stage(
    stage_id: str,
    *,
    step: int,
    precision: str,
    tta: int,
    poll_seconds: float,
) -> dict[str, Any]:
    stage_root = FOLLOWUP_ROOT / stage_id
    existing = successful_metrics(stage_root)
    if existing is not None:
        return existing
    attempt = next_attempt(stage_root)
    command = runner.materialize_command(
        inference_command(step=step, precision=precision, tta=tta), attempt
    )

    def heartbeat(pid: int) -> None:
        update_queue_status(state="running", active_stage=stage_id, subprocess_pid=pid)

    update_queue_status(state="starting", active_stage=stage_id)
    exit_code = runner.run_command(
        command, attempt, poll_seconds=poll_seconds, on_progress=heartbeat
    )
    errors = runner.validate_artifacts(attempt, expected_rows=954)
    if exit_code != 0 or errors:
        raise RuntimeError(
            f"{stage_id} failed with exit={exit_code}, artifact_errors={errors}"
        )
    metrics = load_json(attempt / "metrics.json")
    update_queue_status(
        state="stage_complete",
        active_stage=None,
        last_stage=stage_id,
        last_exact_match=metrics.get("exact_match"),
    )
    return metrics


def run_generic_stage(
    stage_id: str,
    command: Sequence[str],
    *,
    validator: Callable[[Path], list[str]],
    poll_seconds: float,
) -> Path:
    stage_root = FOLLOWUP_ROOT / stage_id
    for attempt in sorted(stage_root.glob("attempt-*"), reverse=True):
        exit_path = attempt / "exit-code.txt"
        if (
            exit_path.is_file()
            and exit_path.read_text(encoding="utf-8").strip() == "0"
            and not validator(attempt)
        ):
            return attempt
    attempt = next_attempt(stage_root)

    def heartbeat(pid: int) -> None:
        update_queue_status(state="running", active_stage=stage_id, subprocess_pid=pid)

    update_queue_status(state="starting", active_stage=stage_id)
    exit_code = runner.run_command(
        command, attempt, poll_seconds=poll_seconds, on_progress=heartbeat
    )
    errors = validator(attempt)
    if exit_code != 0 or errors:
        raise RuntimeError(
            f"{stage_id} failed with exit={exit_code}, artifact_errors={errors}"
        )
    update_queue_status(state="stage_complete", active_stage=None, last_stage=stage_id)
    return attempt


def official_submission_validator(attempt: Path) -> list[str]:
    """Validate the submission and confidence audit without touching Kaggle."""
    from scripts.validate_submission_artifacts import validate

    try:
        summary = validate(
            argparse.Namespace(
                test_csv=ROOT / "data" / "test.csv",
                submission=attempt / OFFICIAL_SUBMISSION_NAME,
                audit=attempt / OFFICIAL_AUDIT_NAME,
                expected_tta=4,
            )
        )
        if summary.get("rows") != 819:
            return [f"official submission rows {summary.get('rows')!r} != 819"]
        if summary.get("parse_failures") != 0:
            return [
                f"official submission parse failures: {summary.get('parse_failures')!r}"
            ]
        if summary.get("view_parse_failures") != 0:
            return [
                "official submission view parse failures: "
                f"{summary.get('view_parse_failures')!r}"
            ]
        audit_rows = [
            json.loads(line)
            for line in (attempt / OFFICIAL_AUDIT_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        invalid_confidence_views = sum(
            view.get("answer_confidence_valid") is not True
            for row in audit_rows
            for view in row["views"]
        )
        wrong_mode_rows = sum(
            row.get("aggregation_mode") != "confidence_tiebreak"
            for row in audit_rows
        )
        aggregation_mismatch_rows = sum(
            row.get("answer")
            != row.get("aggregations", {})
            .get("confidence_tiebreak", {})
            .get("answer")
            for row in audit_rows
        )
        if invalid_confidence_views:
            return [f"invalid confidence views: {invalid_confidence_views}"]
        if wrong_mode_rows:
            return [f"wrong aggregation mode rows: {wrong_mode_rows}"]
        if aggregation_mismatch_rows:
            return [
                "selected answer/confidence-tiebreak mismatch rows: "
                f"{aggregation_mismatch_rows}"
            ]

        first_submission = (
            ROOT / "outputs" / "submission_v5_8b_aug_checkpoint-4292_tta4.csv"
        )
        differences_from_first: int | None = None
        if first_submission.is_file():
            with first_submission.open(newline="", encoding="utf-8-sig") as handle:
                first_answers = {
                    str(row["Id"]): str(row["Answer"])
                    for row in csv.DictReader(handle)
                }
            with (attempt / OFFICIAL_SUBMISSION_NAME).open(
                newline="", encoding="utf-8-sig"
            ) as handle:
                differences_from_first = sum(
                    first_answers.get(str(row["Id"])) != str(row["Answer"])
                    for row in csv.DictReader(handle)
                )
        summary.update(
            {
                "aggregation_mismatch_rows": aggregation_mismatch_rows,
                "aggregation_mode": "confidence_tiebreak",
                "confidence_valid_views": 4 * len(audit_rows),
                "differences_from_first_hard_submission": differences_from_first,
                "invalid_confidence_views": invalid_confidence_views,
                "wrong_mode_rows": wrong_mode_rows,
            }
        )
        atomic_json(attempt / "validation.json", summary)
        return []
    except Exception as exc:
        return [f"official submission invalid: {exc}"]


def run_official_inference_stage(*, poll_seconds: float) -> Path:
    """Run or reuse the confidence-tiebreak test inference after the sweep."""
    stage_root = FOLLOWUP_ROOT / OFFICIAL_STAGE_ID
    for attempt in sorted(stage_root.glob("attempt-*"), reverse=True):
        exit_path = attempt / "exit-code.txt"
        if (
            exit_path.is_file()
            and exit_path.read_text(encoding="utf-8").strip() == "0"
            and not official_submission_validator(attempt)
        ):
            return attempt

    attempt = next_attempt(stage_root)

    def heartbeat(pid: int) -> None:
        update_queue_status(
            state="running_official_inference",
            active_stage=OFFICIAL_STAGE_ID,
            subprocess_pid=pid,
        )

    update_queue_status(state="starting_official_inference", active_stage=OFFICIAL_STAGE_ID)
    exit_code = runner.run_command(
        official_inference_command(attempt),
        attempt,
        poll_seconds=poll_seconds,
        on_progress=heartbeat,
    )
    errors = official_submission_validator(attempt)
    if exit_code != 0 or errors:
        raise RuntimeError(
            f"{OFFICIAL_STAGE_ID} failed with exit={exit_code}, "
            f"artifact_errors={errors}"
        )
    update_queue_status(
        state="official_inference_complete",
        active_stage=None,
        last_stage=OFFICIAL_STAGE_ID,
        official_submission=str(attempt / OFFICIAL_SUBMISSION_NAME),
    )
    return attempt


def run_capture(command: Sequence[str], *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a hidden non-GPU command and capture output for durable receipts."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("KAGGLE_USERNAME", None)
    environment.pop("KAGGLE_KEY", None)
    environment.pop("KAGGLE_API_TOKEN", None)
    environment["MSYS_NO_PATHCONV"] = "1"
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW
    return subprocess.run(
        list(command),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )


def list_kaggle_submissions() -> list[dict[str, str]]:
    kaggle = ROOT / ".venv" / "Scripts" / "kaggle.exe"
    result = run_capture(
        [
            str(kaggle),
            "competitions",
            "submissions",
            "-c",
            COMPETITION,
            "--csv",
        ],
        timeout=120,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(details or "Kaggle submission listing failed")
    return [dict(row) for row in csv.DictReader(io.StringIO(result.stdout))]


def find_remote_submission(
    rows: Sequence[dict[str, str]], *, file_name: str, description: str
) -> dict[str, str] | None:
    for row in rows:
        if row.get("fileName") == file_name and row.get("description") == description:
            return dict(row)
    return None


def remote_status_complete(status: str | None) -> bool:
    return str(status or "").strip().lower().endswith("complete")


def remote_status_failed(status: str | None) -> bool:
    normalized = str(status or "").strip().lower()
    return any(normalized.endswith(value) for value in ("error", "failed", "cancelled"))


def submit_official_attempt(
    attempt: Path, *, poll_seconds: float, timeout_seconds: float = 1800.0
) -> dict[str, Any]:
    """Submit exactly once, then wait for Kaggle to report a terminal result."""
    submission = attempt / OFFICIAL_SUBMISSION_NAME
    validation = load_json(attempt / "validation.json")
    submission_hash = hashlib.sha256(submission.read_bytes()).hexdigest()
    first_submission = (
        ROOT / "outputs" / "submission_v5_8b_aug_checkpoint-4292_tta4.csv"
    )
    if (
        first_submission.is_file()
        and hashlib.sha256(first_submission.read_bytes()).hexdigest() == submission_hash
    ):
        raise RuntimeError(
            "Refusing to spend the second Kaggle slot on a byte-identical copy of "
            "today's first hard-TTA submission"
        )
    receipt_path = FOLLOWUP_ROOT / "official-submission-receipt.json"
    upload_path = FOLLOWUP_ROOT / "official-submission-upload.json"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if (
            receipt.get("sha256") == submission_hash
            and remote_status_complete(receipt.get("remote", {}).get("status"))
        ):
            return receipt

    update_queue_status(
        state="checking_kaggle_submission",
        active_stage="official-kaggle-submission",
        submission_sha256=submission_hash,
    )
    remote = find_remote_submission(
        list_kaggle_submissions(),
        file_name=submission.name,
        description=OFFICIAL_DESCRIPTION,
    )
    upload = load_json(upload_path) if upload_path.is_file() else {}
    if remote is None and upload.get("sha256") != submission_hash:
        kaggle = ROOT / ".venv" / "Scripts" / "kaggle.exe"
        result = run_capture(
            [
                str(kaggle),
                "competitions",
                "submit",
                "-c",
                COMPETITION,
                "-f",
                str(submission),
                "-m",
                OFFICIAL_DESCRIPTION,
            ]
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(details or "Kaggle submission upload failed")
        upload = {
            "command_output": (result.stdout or result.stderr).strip()[-2000:],
            "description": OFFICIAL_DESCRIPTION,
            "file_name": submission.name,
            "sha256": submission_hash,
            "uploaded_at": runner.utc_now(),
        }
        atomic_json(upload_path, upload)

    deadline = time.monotonic() + timeout_seconds
    last_remote = remote
    while time.monotonic() < deadline:
        update_queue_status(
            state="waiting_for_kaggle_score",
            active_stage="official-kaggle-submission",
            kaggle_status=(last_remote or {}).get("status"),
            kaggle_public_score=(last_remote or {}).get("publicScore"),
        )
        try:
            last_remote = find_remote_submission(
                list_kaggle_submissions(),
                file_name=submission.name,
                description=OFFICIAL_DESCRIPTION,
            )
        except Exception as exc:
            update_queue_status(kaggle_poll_error=repr(exc))
            time.sleep(min(poll_seconds, 30.0))
            continue
        if last_remote is not None and remote_status_failed(last_remote.get("status")):
            raise RuntimeError(f"Kaggle submission failed: {last_remote}")
        if last_remote is not None and remote_status_complete(last_remote.get("status")):
            receipt = {
                "competition": COMPETITION,
                "description": OFFICIAL_DESCRIPTION,
                "remote": last_remote,
                "sha256": submission_hash,
                "submission_file": str(submission),
                "validated": validation,
                "verified_at": runner.utc_now(),
            }
            atomic_json(receipt_path, receipt)
            update_queue_status(
                state="official_submission_complete",
                active_stage=None,
                kaggle_status=last_remote.get("status"),
                kaggle_public_score=last_remote.get("publicScore"),
            )
            return receipt
        time.sleep(min(poll_seconds, 30.0))
    raise TimeoutError(
        "Timed out waiting for Kaggle submission completion; upload receipt was "
        "preserved to prevent a duplicate upload"
    )


def download_validator(_attempt: Path) -> list[str]:
    required = [MODEL_27B / "config.json", MODEL_27B / "model.safetensors.index.json"]
    missing = [str(path) for path in required if not path.is_file()]
    weight_bytes = sum(path.stat().st_size for path in MODEL_27B.glob("*.safetensors"))
    if weight_bytes < 50_000_000_000:
        missing.append(f"Qwen3.5 weight bytes too small: {weight_bytes}")
    return missing


def training_smoke_validator(_attempt: Path) -> list[str]:
    output = ROOT / "outputs" / "qwen3-5-27b-smoke-2step"
    required = [output / "training_summary.json", output / "model_manifest.json"]
    return [str(path) for path in required if not path.is_file()]


def qwen35_training_command(output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "snuaichal.training",
        "--model-path",
        str(MODEL_27B.relative_to(ROOT)),
        "--model-repository",
        MODEL_27B_REPOSITORY,
        "--model-family",
        "qwen3_5",
        "--model-revision",
        MODEL_27B_REVISION,
        "--model-manifest",
        str(MODEL_27B_MANIFEST.relative_to(ROOT)),
        "--output-dir",
        str(output.relative_to(ROOT)),
        "--load-in-4bit",
        "--validation-fraction",
        "0.10",
        "--image-size",
        "512",
        "--epochs",
        "6",
        "--stop-after-steps",
        "2",
        "--save-steps",
        "2",
        "--logging-steps",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        "1e-4",
        "--lora-rank",
        "8",
        "--lora-alpha",
        "32",
        "--limit",
        "48",
    ]


def run_qwen35_smoke(*, poll_seconds: float) -> dict[str, Any]:
    hf = ROOT / ".venv" / "Scripts" / "hf.exe"
    download_command = [
        str(hf),
        "download",
        "Qwen/Qwen3.5-27B",
        "--revision",
        MODEL_27B_REVISION,
        "--local-dir",
        str(MODEL_27B),
    ]
    run_generic_stage(
        "qwen35-27b-download",
        download_command,
        validator=download_validator,
        poll_seconds=poll_seconds,
    )
    output = ROOT / "outputs" / "qwen3-5-27b-smoke-2step"
    training_command = qwen35_training_command(output)
    run_generic_stage(
        "qwen35-27b-training-smoke-2step",
        training_command,
        validator=training_smoke_validator,
        poll_seconds=poll_seconds,
    )
    summary = load_json(output / "training_summary.json")
    manifest = load_json(output / "model_manifest.json")
    return {"training_summary": summary, "model_manifest": manifest}


def run_queue(*, runner_pid: int | None, poll_seconds: float) -> dict[str, Any]:
    os.chdir(ROOT)
    FOLLOWUP_ROOT.mkdir(parents=True, exist_ok=True)
    update_queue_status(
        state="waiting_for_sweep",
        queue_pid=os.getpid(),
        sweep_runner_pid=runner_pid,
        error=None,
    )
    wait_for_sweep(runner_pid=runner_pid, poll_seconds=poll_seconds)
    results = read_sweep_results()
    best = results[0]
    atomic_json(
        FOLLOWUP_ROOT / "sweep-ranking.json",
        [result.__dict__ for result in results],
    )

    official_submission: dict[str, Any] | None = None
    official_submission_error: str | None = None

    bf16 = run_inference_stage(
        f"qwen3vl8b-ckpt{best.step}-bf16-greedy-tta1-val954",
        step=best.step,
        precision="bf16",
        tta=1,
        poll_seconds=poll_seconds,
    )
    bf16_accuracy = float(bf16["exact_match"])
    coherent_accuracy = max(best.accuracy, bf16_accuracy)
    diagnostic: dict[str, Any] | None = None
    if coherent_accuracy < COHERENCE_THRESHOLD:
        diagnostic = run_inference_stage(
            "qwen3vl8b-ckpt4292-nf4-greedy-tta4-diagnostic-val954",
            step=4292,
            precision="nf4",
            tta=4,
            poll_seconds=poll_seconds,
        )
        coherent_accuracy = max(coherent_accuracy, float(diagnostic["exact_match"]))

    decision: dict[str, Any] = {
        "official_submission": official_submission,
        "official_submission_error": official_submission_error,
        "best_nf4_sweep": best.__dict__,
        "best_checkpoint_bf16_tta1": {
            "accuracy": bf16_accuracy,
            "exact_matches": bf16.get("exact_matches"),
            "peak_vram_mib": bf16.get("peak_vram_mib"),
        },
        "diagnostic_tta4": diagnostic,
        "coherence_threshold": COHERENCE_THRESHOLD,
        "coherent": coherent_accuracy >= COHERENCE_THRESHOLD,
    }
    atomic_json(FOLLOWUP_ROOT / "phase1-gate.json", decision)
    if coherent_accuracy < COHERENCE_THRESHOLD:
        update_queue_status(
            state="blocked_low_validation",
            active_stage=None,
            reason=(
                f"best validation accuracy {coherent_accuracy:.6f} is below "
                f"the {COHERENCE_THRESHOLD:.2f} safety gate"
            ),
        )
        return decision

    precision = "bf16" if bf16_accuracy > best.accuracy else "nf4"
    tta4 = run_inference_stage(
        f"qwen3vl8b-ckpt{best.step}-{precision}-greedy-tta4-val954",
        step=best.step,
        precision=precision,
        tta=4,
        poll_seconds=poll_seconds,
    )
    decision["selected_precision_before_tta4"] = precision
    decision["selected_tta4"] = {
        "accuracy": tta4.get("exact_match"),
        "exact_matches": tta4.get("exact_matches"),
        "aggregation_comparison": tta4.get("aggregation_comparison"),
        "peak_vram_mib": tta4.get("peak_vram_mib"),
        "estimated_test_seconds": tta4.get("estimated_test_seconds"),
    }
    atomic_json(FOLLOWUP_ROOT / "phase1-gate.json", decision)

    smoke = run_qwen35_smoke(poll_seconds=poll_seconds)
    decision["qwen35_smoke"] = smoke
    decision["automatic_pilot_started"] = False
    decision["automatic_pilot_reason"] = (
        "The two-step smoke is queued, but a longer pilot requires review of "
        "Qwen3.5 Gated Attention/DeltaNet LoRA targets and measured VRAM."
    )
    atomic_json(FOLLOWUP_ROOT / "phase1-gate.json", decision)
    update_queue_status(state="complete", active_stage=None)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        run_queue(runner_pid=args.runner_pid, poll_seconds=args.poll_seconds)
    except Exception as exc:
        update_queue_status(state="failed", active_stage=None, error=repr(exc))
        raise


if __name__ == "__main__":
    main()
