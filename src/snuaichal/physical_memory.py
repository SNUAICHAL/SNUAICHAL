"""Narrow NVML sampler for one trusted CUDA child process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command_identity(command: Sequence[str]) -> str:
    encoded = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cuda_workload_identity(pid: int) -> dict[str, Any] | None:
    """Return identity only for a credible CUDA workload, not WDDM GUI noise."""
    import psutil

    try:
        process = psutil.Process(pid)
        name = process.name()
        command = process.cmdline()
        create_time = process.create_time()
    except (psutil.Error, OSError):
        return None
    lowered_name = name.lower()
    executable_is_compute = any(
        marker in lowered_name
        for marker in ("python", "torchrun", "ollama", "llama", "vllm")
    )
    command_text = " ".join(command).lower()
    command_declares_compute = any(
        marker in command_text
        for marker in (
            "snuaichal.training",
            "snuaichal.inference",
            "run_blocked_low_validation_resume",
            "torchrun",
            "ollama",
            "llama",
            "vllm",
        )
    )
    if not executable_is_compute and not command_declares_compute:
        return None
    return {
        "pid": pid,
        "create_time": create_time,
        "command_identity": command_identity(command),
        "process_name": name,
    }


def process_identity_matches(
    process: Any,
    *,
    expected_pid: int,
    expected_create_time: float,
    expected_command_identity: str,
) -> bool:
    """Require exact PID, creation time, and command identity equality."""
    return (
        process.pid == expected_pid
        and process.create_time() == expected_create_time
        and command_identity(process.cmdline()) == expected_command_identity
    )


def select_physical_measurement(
    *,
    physical_total_vram_bytes: int,
    process_peak_bytes: int | None,
    process_sample_count: int,
    device_peak_bytes: int | None,
    device_sample_count: int,
    process_memory_unavailable: bool,
    trusted_cuda_child_seen: bool,
    process_identity_match: bool,
    unexpected_cuda_processes: list[dict[str, Any]],
    sampling_started_before_model_load: bool,
    sampling_finished_after_work: bool,
) -> dict[str, Any]:
    """Select one physical source, failing closed when device fallback is not exclusive."""
    valid_process = (
        process_peak_bytes is not None
        and process_sample_count > 0
        and not process_memory_unavailable
        and trusted_cuda_child_seen
        and process_identity_match
        and not unexpected_cuda_processes
        and sampling_started_before_model_load
        and sampling_finished_after_work
    )
    valid_device_fallback = (
        process_memory_unavailable
        and device_peak_bytes is not None
        and device_sample_count > 0
        and trusted_cuda_child_seen
        and process_identity_match
        and not unexpected_cuda_processes
        and sampling_started_before_model_load
        and sampling_finished_after_work
    )
    if valid_process:
        source = "nvml_per_process_used_bytes"
        observed = process_peak_bytes
        sample_count = process_sample_count
    elif valid_device_fallback:
        source = "nvml_device_memory_info_used"
        observed = device_peak_bytes
        sample_count = device_sample_count
    else:
        reasons: list[str] = []
        if not process_identity_match:
            reasons.append("trusted child PID/create-time/command identity mismatch")
        if not trusted_cuda_child_seen:
            reasons.append("trusted CUDA child was not observed by NVML")
        if unexpected_cuda_processes:
            reasons.append("unexpected CUDA process appeared during sampling")
        if not sampling_started_before_model_load:
            reasons.append("sampling did not start before model loading")
        if not sampling_finished_after_work:
            reasons.append("sampling did not cover the end of child work")
        if process_peak_bytes is None and not process_memory_unavailable:
            reasons.append("per-process NVML availability was not resolved")
        if device_peak_bytes is None or device_sample_count <= 0:
            reasons.append("device fallback has zero valid samples")
        return {
            "physical_measurement_status": "indeterminate",
            "physical_peak_observed_bytes": None,
            "physical_measurement_source": None,
            "sample_count": 0,
            "physical_measurement_reason": "; ".join(reasons) or "no valid source",
        }
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed < 0
        or observed > physical_total_vram_bytes
    ):
        return {
            "physical_measurement_status": "indeterminate",
            "physical_peak_observed_bytes": None,
            "physical_measurement_source": None,
            "sample_count": 0,
            "physical_measurement_reason": "observed physical peak is invalid",
        }
    return {
        "physical_measurement_status": "valid",
        "physical_peak_observed_bytes": observed,
        "physical_measurement_source": source,
        "sample_count": sample_count,
        "physical_measurement_reason": None,
    }


def _valid_used_bytes(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    # NVML_VALUE_NOT_AVAILABLE is an unsigned all-ones sentinel.
    if value >= 2**63:
        return None
    return value


def monitor_process(
    *,
    parent_pid: int,
    expected_create_time: float,
    expected_command_identity: str,
    ready_path: Path,
    done_path: Path,
    report_path: Path,
    sample_interval_seconds: float,
) -> None:
    import psutil
    import pynvml

    started_at = utc_now()
    process_peak: int | None = None
    device_peak: int | None = None
    process_samples = 0
    device_samples = 0
    process_memory_unavailable = False
    trusted_cuda_child_seen = False
    unexpected: dict[int, dict[str, Any]] = {}
    identity_match = False
    physical_total = 0
    fatal_error: str | None = None
    pynvml_initialized = False

    try:
        parent = psutil.Process(parent_pid)
        identity_match = process_identity_matches(
            parent,
            expected_pid=parent_pid,
            expected_create_time=expected_create_time,
            expected_command_identity=expected_command_identity,
        )
        if not identity_match:
            raise RuntimeError("trusted parent identity mismatch before sampling")
        pynvml.nvmlInit()
        pynvml_initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        physical_total = int(memory.total)

        def sample_once() -> None:
            nonlocal process_peak, device_peak, process_samples, device_samples
            nonlocal process_memory_unavailable, trusted_cuda_child_seen, identity_match
            current = psutil.Process(parent_pid)
            identity_match = identity_match and process_identity_matches(
                current,
                expected_pid=parent_pid,
                expected_create_time=expected_create_time,
                expected_command_identity=expected_command_identity,
            )
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_used = _valid_used_bytes(int(memory_info.used))
            if device_used is not None:
                device_peak = max(device_peak or 0, device_used)
                device_samples += 1
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            current_pids = {int(item.pid) for item in processes}
            for item in processes:
                pid = int(item.pid)
                if pid != parent_pid:
                    identity = cuda_workload_identity(pid)
                    if identity is not None:
                        unexpected[pid] = identity
                    continue
                trusted_cuda_child_seen = True
                used = _valid_used_bytes(item.usedGpuMemory)
                if used is None:
                    process_memory_unavailable = True
                else:
                    process_peak = max(process_peak or 0, used)
                    process_samples += 1
            if trusted_cuda_child_seen and parent_pid not in current_pids:
                process_memory_unavailable = True

        sample_once()
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(started_at + "\n", encoding="utf-8")
        while not done_path.exists():
            if not psutil.pid_exists(parent_pid):
                break
            time.sleep(sample_interval_seconds)
            sample_once()
        if psutil.pid_exists(parent_pid):
            sample_once()
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text(started_at + "\n", encoding="utf-8")
    finally:
        ended_at = utc_now()
        if pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        selected = select_physical_measurement(
            physical_total_vram_bytes=physical_total,
            process_peak_bytes=process_peak,
            process_sample_count=process_samples,
            device_peak_bytes=device_peak,
            device_sample_count=device_samples,
            process_memory_unavailable=process_memory_unavailable,
            trusted_cuda_child_seen=trusted_cuda_child_seen,
            process_identity_match=identity_match,
            unexpected_cuda_processes=list(unexpected.values()),
            sampling_started_before_model_load=True,
            sampling_finished_after_work=done_path.exists(),
        )
        if fatal_error is not None:
            selected.update(
                {
                    "physical_measurement_status": "indeterminate",
                    "physical_peak_observed_bytes": None,
                    "physical_measurement_source": None,
                    "sample_count": 0,
                    "physical_measurement_reason": fatal_error,
                }
            )
        report = {
            "schema_version": 1,
            **selected,
            "physical_total_vram_bytes": physical_total or None,
            "sample_interval_seconds": sample_interval_seconds,
            "sampling_started_at": started_at,
            "sampling_ended_at": ended_at,
            "sampling_started_before_model_load": True,
            "sampling_finished_after_work": done_path.exists(),
            "process_peak_bytes": process_peak,
            "process_sample_count": process_samples,
            "device_peak_bytes": device_peak,
            "device_sample_count": device_samples,
            "process_memory_unavailable": process_memory_unavailable,
            "trusted_cuda_child_seen": trusted_cuda_child_seen,
            "process_identity_match": identity_match,
            "trusted_process_identity": {
                "pid": parent_pid,
                "create_time": expected_create_time,
                "command_identity": expected_command_identity,
            },
            "unexpected_cuda_processes": list(unexpected.values()),
            "device_fallback_exclusive": (
                trusted_cuda_child_seen and identity_match and not unexpected
            ),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--expected-create-time", type=float, required=True)
    parser.add_argument("--expected-command-identity", required=True)
    parser.add_argument("--ready-path", type=Path, required=True)
    parser.add_argument("--done-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.sample_interval_seconds <= 1.0:
        raise ValueError("sample interval must be in (0, 1] seconds")
    monitor_process(
        parent_pid=args.parent_pid,
        expected_create_time=args.expected_create_time,
        expected_command_identity=args.expected_command_identity,
        ready_path=args.ready_path,
        done_path=args.done_path,
        report_path=args.report_path,
        sample_interval_seconds=args.sample_interval_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
