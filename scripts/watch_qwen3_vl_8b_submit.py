from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "outputs" / "qwen3-vl-8b-aug"
TRAIN_EXIT = RUN_DIR / "exit-code.txt"
TRAIN_SUMMARY = RUN_DIR / "training_summary.json"
CHECKPOINT = RUN_DIR / "checkpoint-4292"
INFERENCE_TASK = "SNUAICHAL_Qwen3VL8B_Inference"
INFERENCE_EXIT = ROOT / "outputs" / "submission_v5_8b_aug_checkpoint-4292_tta4.exit-code.txt"
SUBMISSION = ROOT / "outputs" / "submission_v5_8b_aug_checkpoint-4292_tta4.csv"
AUDIT = ROOT / "outputs" / "submission_v5_8b_aug_checkpoint-4292_tta4.jsonl"
STATE_PATH = RUN_DIR / "kaggle-submit-watchdog-state.json"
UPLOAD_ATTEMPT = RUN_DIR / "kaggle-upload-attempt.json"
RECEIPT = RUN_DIR / "kaggle-submission-receipt.json"
DESCRIPTION = "Qwen3-VL-8B QLoRA r16 seed42 4ep checkpoint-4292 cyclic TTA4"
COMPETITION = "snuaichallenge"
KAGGLE_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/submissions"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
KAGGLE = ROOT / ".venv" / "Scripts" / "kaggle.exe"
SCHTASKS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "schtasks.exe"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit_once(state: dict[str, Any], key: str, fingerprint: str, message: str) -> None:
    if state.get(key) == fingerprint:
        return
    state[key] = fingerprint
    write_json(STATE_PATH, state)
    print(message)


def command(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    env.pop("KAGGLE_API_TOKEN", None)
    env["MSYS_NO_PATHCONV"] = "1"
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def task_state() -> str | None:
    result = command(
        [str(SCHTASKS), "/Query", "/TN", INFERENCE_TASK, "/FO", "CSV", "/NH"]
    )
    if result.returncode != 0:
        return None
    try:
        return next(csv.reader(io.StringIO(result.stdout)))[2]
    except (IndexError, StopIteration, csv.Error):
        return None


def validate_artifacts() -> dict[str, Any]:
    result = command(
        [
            str(PYTHON),
            "scripts/validate_submission_artifacts.py",
            "--test-csv",
            "data/test.csv",
            "--submission",
            str(SUBMISSION),
            "--audit",
            str(AUDIT),
            "--expected-tta",
            "4",
        ],
        timeout=300,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(details or f"validator exited with code {result.returncode}")
    return json.loads(result.stdout)


def find_remote_submission() -> dict[str, str] | None:
    result = command(
        [str(KAGGLE), "competitions", "submissions", "-c", COMPETITION, "--csv"],
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip()[-2000:])
    for row in csv.DictReader(io.StringIO(result.stdout)):
        if row.get("fileName") == SUBMISSION.name and row.get("description") == DESCRIPTION:
            return dict(row)
    return None


def main() -> None:
    if RECEIPT.is_file():
        return

    state = read_json(STATE_PATH, {})
    if not TRAIN_EXIT.is_file():
        return
    try:
        train_exit_code = int(TRAIN_EXIT.read_text(encoding="utf-8").strip())
    except ValueError:
        emit_once(state, "training_error", "invalid-exit", "snuaichallenge 자동 제출 중단: 학습 exit code가 올바르지 않습니다.")
        return
    if train_exit_code != 0:
        emit_once(
            state,
            "training_error",
            str(train_exit_code),
            f"snuaichallenge 자동 제출 중단: 학습이 exit code {train_exit_code}로 종료됐습니다.",
        )
        return

    summary = read_json(TRAIN_SUMMARY, {})
    if int(summary.get("global_step", -1)) != 4292 or not (CHECKPOINT / "trainer_state.json").is_file():
        emit_once(
            state,
            "training_artifact_error",
            json.dumps(summary, sort_keys=True),
            "snuaichallenge 자동 제출 중단: step 4292 학습 요약 또는 체크포인트가 불완전합니다.",
        )
        return

    if not INFERENCE_EXIT.is_file():
        current_task_state = task_state()
        if current_task_state != "Running":
            result = command([str(SCHTASKS), "/Run", "/TN", INFERENCE_TASK])
            if result.returncode != 0:
                details = (result.stderr or result.stdout).strip()[-1000:]
                emit_once(
                    state,
                    "inference_start_error",
                    details,
                    f"snuaichallenge 제출 추론 시작 실패: {details}",
                )
                return
            emit_once(
                state,
                "inference_started",
                "started",
                "snuaichallenge 학습 완료를 확인해 checkpoint-4292 cyclic 4-TTA 제출 추론을 시작했습니다.",
            )
        return

    try:
        inference_exit_code = int(INFERENCE_EXIT.read_text(encoding="utf-8").strip())
    except ValueError:
        inference_exit_code = -1
    if inference_exit_code != 0:
        emit_once(
            state,
            "inference_error",
            str(inference_exit_code),
            f"snuaichallenge 자동 제출 중단: 제출 추론이 exit code {inference_exit_code}로 종료됐습니다.",
        )
        return

    try:
        validation = validate_artifacts()
    except Exception as exc:
        details = str(exc)[-1500:]
        emit_once(
            state,
            "validation_error",
            details,
            f"snuaichallenge 자동 제출 중단: CSV/audit 검증 실패: {details}",
        )
        return

    submission_hash = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    upload = read_json(UPLOAD_ATTEMPT, {})
    if upload.get("sha256") != submission_hash:
        result = command(
            [
                str(KAGGLE),
                "competitions",
                "submit",
                "-c",
                COMPETITION,
                "-f",
                str(SUBMISSION),
                "-m",
                DESCRIPTION,
            ],
            timeout=600,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()[-1500:]
            emit_once(
                state,
                "upload_error",
                details,
                "snuaichallenge 제출 파일 검증은 통과했지만 Kaggle 업로드에 실패했습니다. "
                f"credential을 갱신해 주세요: {details}",
            )
            return
        upload = {
            "command_output": (result.stdout or result.stderr).strip()[-2000:],
            "description": DESCRIPTION,
            "sha256": submission_hash,
            "uploaded_at": utc_now(),
        }
        write_json(UPLOAD_ATTEMPT, upload)

    try:
        remote = find_remote_submission()
    except Exception as exc:
        details = str(exc)[-1500:]
        emit_once(
            state,
            "verification_error",
            details,
            "snuaichallenge 파일은 Kaggle에 업로드됐지만 원격 submission 목록 검증에 실패했습니다: "
            f"{details}",
        )
        return
    if remote is None:
        emit_once(
            state,
            "verification_error",
            "not-listed",
            "snuaichallenge 파일 업로드 명령은 성공했지만 원격 submission 목록에서 동일 파일/설명을 찾지 못했습니다.",
        )
        return

    receipt = {
        "competition": COMPETITION,
        "description": DESCRIPTION,
        "remote": remote,
        "sha256": submission_hash,
        "submission_file": str(SUBMISSION),
        "submission_url": KAGGLE_URL,
        "validated": validation,
        "verified_at": utc_now(),
    }
    write_json(RECEIPT, receipt)
    status = remote.get("status", "unknown")
    score = remote.get("publicScore") or "pending"
    print(
        "snuaichallenge Kaggle 제출 완료 및 원격 확인 성공. "
        f"status={status}, publicScore={score}, sha256={submission_hash}, URL={KAGGLE_URL}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"snuaichallenge 자동 제출 watchdog 오류: {exc}")
        sys.exit(1)
