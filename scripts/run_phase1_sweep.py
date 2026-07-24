"""Durable, resumable Phase 1 GPU experiment runner.

Each configuration runs in a fresh Python subprocess and a new attempt directory.
Successful validated experiments are skipped; failures stop the plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import psutil


MODEL_8B_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_8B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_8B_MANIFEST = (
    "outputs/blocked_low_validation_resume/model-manifests/"
    f"qwen3-vl-8b-{MODEL_8B_REVISION}.json"
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    command: list[str]
    expected_rows: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_status(path: Path) -> dict:
    if not path.is_file():
        return {"updated_at": utc_now(), "experiments": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def next_attempt_dir(output_root: Path, experiment_id: str) -> tuple[int, Path]:
    experiment_root = output_root / experiment_id
    existing = []
    if experiment_root.is_dir():
        for candidate in experiment_root.glob("attempt-*"):
            try:
                existing.append(int(candidate.name.removeprefix("attempt-")))
            except ValueError:
                continue
    attempt = max(existing, default=0) + 1
    return attempt, experiment_root / f"attempt-{attempt:03d}"


def materialize_command(command: Sequence[str], attempt_dir: Path) -> list[str]:
    """Replace attempt placeholders, including placeholders embedded in paths."""
    concrete: list[str] = []
    for value in command:
        if value == "{attempt_dir}":
            concrete.append(str(attempt_dir))
        elif value.startswith("{attempt_dir}/"):
            concrete.append(str(attempt_dir / value.removeprefix("{attempt_dir}/")))
        else:
            concrete.append(value.replace("{attempt_dir}", str(attempt_dir)))
    return concrete


def validate_artifacts(attempt_dir: Path, *, expected_rows: int) -> list[str]:
    errors: list[str] = []
    predictions_path = attempt_dir / "predictions.csv"
    audit_path = attempt_dir / "audit.jsonl"
    metrics_path = attempt_dir / "metrics.json"

    try:
        with predictions_path.open(newline="", encoding="utf-8") as file:
            prediction_rows = list(csv.DictReader(file))
        if len(prediction_rows) != expected_rows:
            errors.append(
                f"predictions row count {len(prediction_rows)} != {expected_rows}"
            )
    except Exception as exc:  # Artifact diagnostics must preserve the subprocess result.
        errors.append(f"predictions invalid: {exc}")

    try:
        audit_rows = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(audit_rows) != expected_rows:
            errors.append(f"audit row count {len(audit_rows)} != {expected_rows}")
    except Exception as exc:
        errors.append(f"audit invalid: {exc}")

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("samples") != expected_rows:
            errors.append(
                f"metrics samples {metrics.get('samples')!r} != {expected_rows}"
            )
    except Exception as exc:
        errors.append(f"metrics invalid: {exc}")
    return errors


def terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a subprocess and descendants so CUDA workers cannot be orphaned."""
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in reversed(descendants):
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*descendants, parent], timeout=5)
        for item in alive:
            item.kill()
        _, still_alive = psutil.wait_procs(alive, timeout=5)
        if still_alive:
            raise RuntimeError(f"failed to terminate process tree: {still_alive}")
    except psutil.NoSuchProcess:
        return
    except Exception:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=15,
            )
        else:
            os.killpg(os.getpgid(process.pid), 9)
        process.wait(timeout=15)


def default_child_environment() -> dict[str, str]:
    """Build a credential-free runtime environment without mutating the parent."""
    allowed_names = {
        "ALL_PROXY",
        "APPDATA",
        "COMSPEC",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_DISABLE_REQUIRE",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "NVIDIA_VISIBLE_DEVICES",
        "HOMEDRIVE",
        "HOMEPATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LOCALAPPDATA",
        "NO_PROXY",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TORCH_HOME",
        "USERNAME",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_names
        and not (
            key.upper() in {"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"}
            and "@" in value.partition("://")[2].partition("/")[0]
        )
    }


def run_command(
    command: Sequence[str],
    attempt_dir: Path,
    *,
    poll_seconds: float = 30.0,
    on_progress: Callable[[int], None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    (attempt_dir / "command.txt").write_text(
        subprocess.list2cmdline(list(command)) + "\n", encoding="utf-8"
    )
    (attempt_dir / "started-at.txt").write_text(utc_now() + "\n", encoding="utf-8")
    child_environment = dict(
        environment if environment is not None else default_child_environment()
    )
    source_path = str(Path.cwd() / "src")
    child_environment["PYTHONPATH"] = source_path
    child_environment["PYTHONUNBUFFERED"] = "1"
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    process: subprocess.Popen | None = None
    exit_code = -1
    try:
        with (attempt_dir / "stdout.log").open("wb") as stdout_file, (
            attempt_dir / "stderr.log"
        ).open("wb") as stderr_file:
            process = subprocess.Popen(
                list(command),
                cwd=Path.cwd(),
                env=child_environment,
                stdout=stdout_file,
                stderr=stderr_file,
                startupinfo=startupinfo,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            (attempt_dir / "pid.txt").write_text(
                f"{process.pid}\n", encoding="utf-8"
            )
            while process.poll() is None:
                if on_progress is not None:
                    on_progress(process.pid)
                time.sleep(poll_seconds)
            exit_code = int(process.returncode)
    except BaseException:
        if process is not None:
            terminate_process_tree(process)
            process.poll()
            if process.returncode is not None:
                exit_code = int(process.returncode)
        raise
    finally:
        (attempt_dir / "exit-code.txt").write_text(
            f"{exit_code}\n", encoding="utf-8"
        )
        (attempt_dir / "ended-at.txt").write_text(
            utc_now() + "\n", encoding="utf-8"
        )
    return exit_code


def _append_registry(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def refresh_checkpoint_sweep(
    output_root: Path,
    *,
    csv_path: Path,
    json_path: Path,
) -> list[dict]:
    """Atomically rebuild checkpoint sweep summaries from durable status records."""
    status = load_status(output_root / "status.json")
    rows: list[dict] = []
    pattern = re.compile(
        r"^qwen3vl8b-ckpt(?P<step>\d+)-(?P<precision>[^-]+)-"
        r"(?P<decode>[^-]+)-tta(?P<tta>\d+)-val954$"
    )
    for experiment_id, record in status.get("experiments", {}).items():
        match = pattern.match(experiment_id)
        if match is None:
            continue
        attempt_dir = Path(record["attempt_dir"])
        metrics_path = attempt_dir / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {}
        )
        step = int(match.group("step"))
        peak_mib = metrics.get("peak_vram_mib")
        rows.append(
            {
                "checkpoint": f"checkpoint-{step}",
                "global_step": step,
                "precision": metrics.get("model_precision", match.group("precision")),
                "tta": metrics.get("tta_views", int(match.group("tta"))),
                "decode_mode": match.group("decode"),
                "exact_match_count": metrics.get("exact_matches"),
                "validation_count": metrics.get("samples"),
                "exact_match_accuracy": metrics.get("exact_match"),
                "parse_failures": metrics.get("parse_failures"),
                "seconds_per_sample": metrics.get("inference_seconds_per_sample"),
                "projected_test_seconds": metrics.get("estimated_test_seconds"),
                "peak_vram_bytes": (
                    int(round(float(peak_mib) * 1024 * 1024))
                    if peak_mib is not None
                    else None
                ),
                "visual_tokens_mean": metrics.get("visual_tokens", {}).get("mean"),
                "status": record.get("status"),
                "output_paths": {
                    "attempt_dir": str(attempt_dir),
                    "audit": str(attempt_dir / "audit.jsonl"),
                    "metrics": str(metrics_path),
                    "predictions": str(attempt_dir / "predictions.csv"),
                },
            }
        )
    rows.sort(key=lambda row: row["global_step"])
    atomic_write_json(json_path, rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    fieldnames = list(rows[0]) if rows else [
        "checkpoint",
        "global_step",
        "precision",
        "tta",
        "decode_mode",
        "exact_match_count",
        "validation_count",
        "exact_match_accuracy",
        "parse_failures",
        "seconds_per_sample",
        "projected_test_seconds",
        "peak_vram_bytes",
        "visual_tokens_mean",
        "status",
        "output_paths",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["output_paths"] = json.dumps(row["output_paths"], sort_keys=True)
            writer.writerow(csv_row)
    os.replace(temporary, csv_path)
    return rows


def run_plan(
    experiments: Sequence[Experiment],
    *,
    output_root: Path,
    poll_seconds: float = 30.0,
) -> bool:
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    registry_path = output_root / "experiment_registry.jsonl"
    status = load_status(status_path)

    for experiment in experiments:
        previous = status["experiments"].get(experiment.experiment_id)
        if previous and previous.get("status") == "succeeded":
            previous_dir = Path(previous["attempt_dir"])
            if not validate_artifacts(
                previous_dir, expected_rows=experiment.expected_rows
            ):
                refresh_checkpoint_sweep(
                    output_root,
                    csv_path=output_root.parent / "checkpoint_sweep.csv",
                    json_path=output_root.parent / "checkpoint_sweep.json",
                )
                continue

        attempt_number, attempt_dir = next_attempt_dir(
            output_root, experiment.experiment_id
        )
        concrete_command = materialize_command(experiment.command, attempt_dir)
        record = {
            "experiment_id": experiment.experiment_id,
            "attempt": attempt_number,
            "attempt_dir": str(attempt_dir),
            "command": concrete_command,
            "expected_rows": experiment.expected_rows,
            "started_at": utc_now(),
            "status": "running",
        }
        status["experiments"][experiment.experiment_id] = record
        status["active_experiment"] = experiment.experiment_id
        status["updated_at"] = utc_now()
        atomic_write_json(status_path, status)

        def update_progress(pid: int) -> None:
            record["pid"] = pid
            record["heartbeat_at"] = utc_now()
            status["updated_at"] = record["heartbeat_at"]
            atomic_write_json(status_path, status)

        caught: BaseException | None = None
        try:
            exit_code = run_command(
                concrete_command,
                attempt_dir,
                poll_seconds=poll_seconds,
                on_progress=update_progress,
            )
            errors = validate_artifacts(
                attempt_dir, expected_rows=experiment.expected_rows
            )
        except BaseException as exc:
            caught = exc
            exit_code = None
            errors = [f"{type(exc).__name__}: {exc}"]

        terminal_status = (
            "interrupted"
            if isinstance(caught, (KeyboardInterrupt, SystemExit))
            else "succeeded"
            if exit_code == 0 and not errors
            else "failed"
        )
        record.update(
            {
                "ended_at": utc_now(),
                "exit_code": exit_code,
                "artifact_errors": errors,
                "status": terminal_status,
            }
        )
        status.pop("active_experiment", None)
        status["state"] = terminal_status
        status["updated_at"] = utc_now()
        atomic_write_json(status_path, status)
        _append_registry(registry_path, record)
        if caught is not None:
            raise caught
        refresh_checkpoint_sweep(
            output_root,
            csv_path=output_root.parent / "checkpoint_sweep.csv",
            json_path=output_root.parent / "checkpoint_sweep.json",
        )
        if record["status"] != "succeeded":
            return False
    return True


def inference_command(
    *, adapter: Path, precision: str, tta: int, expected_rows: int
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "snuaichal.inference",
        "--test-csv",
        "data/train.csv",
        "--image-dir",
        "data/train",
        "--model-path",
        "models/Qwen3-VL-8B-Instruct",
        "--model-repository",
        MODEL_8B_REPOSITORY,
        "--model-family",
        "qwen3_vl",
        "--model-revision",
        MODEL_8B_REVISION,
        "--model-manifest",
        MODEL_8B_MANIFEST,
        "--adapter-path",
        str(adapter),
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
    if expected_rows != 954:
        command.extend(["--limit", str(expected_rows)])
    return command


def build_experiments(plan: str) -> list[Experiment]:
    smoke = [
        Experiment(
            "qwen3vl8b-ckpt4292-nf4-greedy-tta1-val1",
            inference_command(
                adapter=Path("outputs/qwen3-vl-8b-aug/checkpoint-4292"),
                precision="nf4",
                tta=1,
                expected_rows=1,
            ),
            1,
        ),
        Experiment(
            "qwen3vl8b-ckpt4292-bf16-greedy-tta1-val8",
            inference_command(
                adapter=Path("outputs/qwen3-vl-8b-aug/checkpoint-4292"),
                precision="bf16",
                tta=1,
                expected_rows=8,
            ),
            8,
        ),
    ]
    sweep = [
        Experiment(
            f"qwen3vl8b-ckpt{step}-nf4-greedy-tta1-val954",
            inference_command(
                adapter=Path(f"outputs/qwen3-vl-8b-aug/checkpoint-{step}"),
                precision="nf4",
                tta=1,
                expected_rows=954,
            ),
            954,
        )
        for step in (3000, 3500, 3750, 4000, 4250, 4292)
    ]
    if plan == "smoke":
        return smoke
    if plan == "sweep":
        return sweep
    return smoke + sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=("smoke", "sweep", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    success = run_plan(
        build_experiments(args.plan),
        output_root=args.output_root,
        poll_seconds=args.poll_seconds,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
