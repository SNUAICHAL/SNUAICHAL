"""Resume bounded diagnostics after Phase 1 fails the auto-submission gate.

This runner never contacts Kaggle and never writes inside completed Phase 1 stage
or checkpoint directories. The original 0.70 auto-submission threshold remains
unchanged; this module only performs explicitly authorized local diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import psutil

from scripts import run_phase1_sweep as durable
from snuaichal.model_manifest import (
    create_model_manifest,
    verify_model_manifest,
    write_revision_marker,
)
from snuaichal.submission import is_permutation
from snuaichal.physical_memory import cuda_workload_identity
from snuaichal.training import ALLOCATOR_MEMORY_SEMANTICS


AGGREGATION_MODES = ("hard", "confidence_tiebreak", "confidence_weighted")
ROOT = Path(__file__).resolve().parents[1]
RESUME_ROOT = ROOT / "outputs" / "blocked_low_validation_resume"
STAGES_ROOT = RESUME_ROOT / "stages"
STATUS_PATH = RESUME_ROOT / "status.json"
REGISTRY_PATH = RESUME_ROOT / "experiment_registry.jsonl"
RUNNER_LOCK_PATH = RESUME_ROOT / "runner-lock.json"
CORRECTIVE_AUDIT_PATH = (
    RESUME_ROOT
    / "corrective-audits"
    / "qwen35-27b-smoke-attempt-001-vram-audit.json"
)
LEGACY_SMOKE_ATTEMPT = (
    STAGES_ROOT / "qwen35-27b-training-smoke-2step" / "attempt-001"
)
MODEL_8B = ROOT / "models" / "Qwen3-VL-8B-Instruct"
MODEL_8B_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_27B = ROOT / "models" / "Qwen3.5-27B"
MODEL_27B_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_MANIFEST_ROOT = RESUME_ROOT / "model-manifests"
MODEL_8B_MANIFEST = MODEL_MANIFEST_ROOT / f"qwen3-vl-8b-{MODEL_8B_REVISION}.json"
MODEL_27B_MANIFEST = MODEL_MANIFEST_ROOT / f"qwen35-27b-{MODEL_27B_REVISION}.json"
VALIDATION_MANIFEST = ROOT / "outputs" / "qwen3-vl-8b-aug" / "split_manifest.json"
SWEEP_CSV = ROOT / "outputs" / "checkpoint_sweep.csv"
DIAGNOSTIC_METRICS = (
    ROOT
    / "outputs"
    / "phase1_followup"
    / "qwen3vl8b-ckpt4292-nf4-greedy-tta4-diagnostic-val954"
    / "attempt-001"
    / "metrics.json"
)
RECEIPT = ROOT / "outputs" / "phase1_followup" / "official-submission-receipt.json"
PAIR_MANIFEST = RESUME_ROOT / "paired-96-manifest.json"
BASELINE_REPORT = RESUME_ROOT / "baseline-report-v2.json"
PRESERVED_MANIFEST = RESUME_ROOT / "preserved-inputs-manifest.json"
FULL_OPTIMIZER_STEPS = 6438
MAX_PROJECTED_HOURS = 72.0
MAX_SAFE_VRAM_BYTES = 22 * 1024**3
PHYSICAL_VRAM_BYTES = 24 * 1024**3
MAX_PACKAGE_BYTES = 80_000_000_000
PROVENANCE_SCHEMA_VERSION = 2

Validator = Callable[[Path], list[str]]


def _project_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_ml_child_environment() -> dict[str, str]:
    """Return an explicit local runtime environment without inherited credentials."""
    allowed_names = {
        "ALL_PROXY",
        "APPDATA",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
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
        "ONEDRIVE",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TORCH_HOME",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
    child = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed_names
        and not (
            key.upper() in {"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"}
            and "@" in value.partition("://")[2].partition("/")[0]
        )
    }
    child["PYTHONPATH"] = str(ROOT / "src")
    child["PYTHONUNBUFFERED"] = "1"
    return child


def build_acquisition_child_environment() -> dict[str, str]:
    """Return a public-model acquisition environment isolated from user credentials."""
    child = build_ml_child_environment()
    child["HF_HOME"] = str(RESUME_ROOT / "hf-home")
    child["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return child


def atomic_status(**values: Any) -> dict[str, Any]:
    status = durable.load_status(STATUS_PATH)
    status.update(values)
    status["updated_at"] = durable.utc_now()
    durable.atomic_write_json(STATUS_PATH, status)
    return status


def build_inference_command(
    *,
    model_path: Path,
    adapter_path: Path | None,
    validation_manifest: Path,
    tta: int,
    model_family: str,
    model_revision: str,
    precision: str,
    aggregation_mode: str = "hard",
    limit: int | None = None,
) -> list[str]:
    """Build a local-only labeled-validation inference command."""
    repository, pinned_revision, pinned_family, manifest_path = (
        _model_manifest_configuration(model_path)
    )
    if model_revision != pinned_revision or model_family != pinned_family:
        raise ValueError("inference model family/revision differs from pinned model identity")
    command = [
        sys.executable,
        "-m",
        "snuaichal.inference",
        "--test-csv",
        "data/train.csv",
        "--image-dir",
        "data/train",
        "--model-path",
        _project_path(model_path),
        "--model-repository",
        repository,
        "--model-family",
        model_family,
        "--model-revision",
        model_revision,
        "--model-manifest",
        _project_path(manifest_path),
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", _project_path(adapter_path)])
    else:
        command.append("--no-adapter")
    command.extend(
        [
            "--validation-manifest",
            _project_path(validation_manifest),
            "--precision",
            precision,
            "--image-size",
            "512",
            "--tta",
            str(tta),
            "--aggregation-mode",
            aggregation_mode,
            "--seed",
            "42",
            "--output",
            "{attempt_dir}/predictions.csv",
            "--audit-log",
            "{attempt_dir}/audit.jsonl",
            "--metrics-output",
            "{attempt_dir}/metrics.json",
        ]
    )
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def build_final_test_inference_stage(
    *,
    selection: dict[str, Any],
    test_csv: Path,
    image_dir: Path,
    test_input_identity: dict[str, Any],
) -> dict[str, Any]:
    """Define the fail-closed post-selection 819-row test inference stage."""
    required = {
        "model_path",
        "model_repository",
        "model_family",
        "model_revision",
        "model_manifest",
        "verified_model_tree_sha256",
        "adapter_path",
        "precision",
        "image_size",
        "tta_orders",
        "aggregation_mode",
        "fallback_policy",
        "seed",
    }
    missing = required - selection.keys()
    if missing:
        raise ValueError(f"selected model is missing fields: {sorted(missing)}")
    if not test_csv.is_file() or not image_dir.is_dir():
        raise ValueError("final test CSV/image directory is missing")
    with test_csv.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        test_rows = list(reader)
    if reader.fieldnames is None or "Id" not in reader.fieldnames:
        raise ValueError("final test CSV must contain Id")
    expected_ids = [str(row["Id"]) for row in test_rows]
    if len(expected_ids) != 819:
        raise ValueError(f"final test CSV must contain exactly 819 rows, got {len(expected_ids)}")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("final test CSV IDs must be unique")

    model_path = Path(selection["model_path"])
    model_manifest = Path(selection["model_manifest"])
    adapter_value = selection["adapter_path"]
    adapter_path = Path(adapter_value) if adapter_value is not None else None
    tta_orders = selection["tta_orders"]
    if (
        not isinstance(tta_orders, list)
        or len(tta_orders) not in (1, 4)
        or any(not _valid_permutation(order) for order in tta_orders)
        or len({tuple(order) for order in tta_orders}) != len(tta_orders)
    ):
        raise ValueError("selected model TTA orders must be one or four unique permutations")
    if selection["fallback_policy"] != "identity":
        raise ValueError("unsupported final inference fallback policy")
    normalized_selection = {
        **selection,
        "model_path": _project_path(model_path),
        "model_manifest": _project_path(model_manifest),
        "adapter_path": _project_path(adapter_path) if adapter_path is not None else None,
    }
    command = [
        sys.executable,
        "-m",
        "snuaichal.inference",
        "--test-csv",
        _project_path(test_csv),
        "--image-dir",
        _project_path(image_dir),
        "--model-path",
        normalized_selection["model_path"],
        "--model-repository",
        str(selection["model_repository"]),
        "--model-family",
        str(selection["model_family"]),
        "--model-revision",
        str(selection["model_revision"]),
        "--model-manifest",
        normalized_selection["model_manifest"],
    ]
    if adapter_path is None:
        command.append("--no-adapter")
    else:
        command.extend(["--adapter-path", normalized_selection["adapter_path"]])
    command.extend(
        [
            "--precision",
            str(selection["precision"]),
            "--image-size",
            str(selection["image_size"]),
            "--tta",
            str(len(tta_orders)),
            "--tta-orders-json",
            json.dumps(tta_orders, separators=(",", ":")),
            "--aggregation-mode",
            str(selection["aggregation_mode"]),
            "--fallback-policy",
            str(selection["fallback_policy"]),
            "--seed",
            str(selection["seed"]),
            "--output",
            "{attempt_dir}/submission.csv",
            "--audit-log",
            "{attempt_dir}/audit.jsonl",
            "--metrics-output",
            "{attempt_dir}/metrics.json",
        ]
    )

    def validate(attempt: Path) -> list[str]:
        errors: list[str] = []
        submission_path = attempt / "submission.csv"
        audit_path = attempt / "audit.jsonl"
        metrics_path = attempt / "metrics.json"
        try:
            with submission_path.open(newline="", encoding="utf-8-sig") as file:
                submission_reader = csv.DictReader(file)
                submission_rows = list(submission_reader)
            if submission_reader.fieldnames != ["Id", "Answer"]:
                errors.append("submission CSV schema must be exactly Id,Answer")
            submission_ids = [str(row.get("Id")) for row in submission_rows]
            if len(submission_ids) != len(set(submission_ids)):
                errors.append("submission contains duplicate IDs")
            if submission_ids != expected_ids:
                errors.append("submission ordered IDs disagree with original test CSV")
            if len(submission_rows) != 819:
                errors.append(f"submission row count {len(submission_rows)} != 819")
            submission_answers: list[list[int] | None] = []
            for row in submission_rows:
                try:
                    answer = json.loads(str(row.get("Answer")))
                except (TypeError, json.JSONDecodeError):
                    answer = None
                submission_answers.append(answer)
                if not _valid_permutation(answer):
                    errors.append(f"ID {row.get('Id')!r} answer is not a permutation")

            audit_rows = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            audit_ids = [str(row.get("Id")) for row in audit_rows]
            if len(audit_ids) != len(set(audit_ids)):
                errors.append("audit contains duplicate IDs")
            if audit_ids != expected_ids:
                errors.append("audit ordered IDs disagree with original test CSV")
            parse_failures = 0
            for row in audit_rows:
                if not _valid_permutation(row.get("answer")):
                    errors.append(f"ID {row.get('Id')!r} audit answer is not a permutation")
                if not isinstance(row.get("parse_ok"), bool):
                    errors.append(f"ID {row.get('Id')!r} parse_ok is not boolean")
                elif row["parse_ok"] is False:
                    parse_failures += 1
                if row.get("valid_tta_views") not in range(len(tta_orders) + 1):
                    errors.append(f"ID {row.get('Id')!r} valid TTA view count is invalid")
                if row.get("aggregation_mode") != selection["aggregation_mode"]:
                    errors.append(f"ID {row.get('Id')!r} aggregation mode mismatch")
                if not isinstance(row.get("views"), list) or len(row["views"]) != len(
                    tta_orders
                ):
                    errors.append(f"ID {row.get('Id')!r} TTA view count mismatch")

            for submission_row, submission_answer, audit_row in zip(
                submission_rows, submission_answers, audit_rows, strict=True
            ):
                if submission_answer != audit_row.get("answer"):
                    errors.append(
                        f"ID {submission_row.get('Id')!r} submission answer disagrees with audit"
                    )

            metrics = load_json(metrics_path)
            runtime = metrics.get("runtime_state", {})
            required_metrics = {
                "samples",
                "parse_failures",
                "tta_views",
                "aggregation_mode",
                "inference_seconds_per_sample",
                "estimated_test_seconds",
                "peak_vram_mib",
                "model_precision",
                "detected_model_family",
                "declared_model_family",
                "model_revision",
                "base_model_path",
                "adapter_path",
                "runtime_state",
            }
            missing_metrics = required_metrics - metrics.keys()
            if missing_metrics:
                errors.append(f"metric schema missing fields: {sorted(missing_metrics)}")
            if metrics.get("samples") != 819:
                errors.append("metrics samples does not equal 819")
            if metrics.get("parse_failures") != parse_failures:
                errors.append("metrics parse_failures disagrees with audit")
            if metrics.get("tta_views") != len(tta_orders):
                errors.append("metrics TTA view count disagrees with selection")
            if metrics.get("aggregation_mode") != selection["aggregation_mode"]:
                errors.append("metrics aggregation mode disagrees with selection")
            if metrics.get("model_precision") != selection["precision"]:
                errors.append("metrics precision disagrees with selection")
            if metrics.get("detected_model_family") != selection["model_family"]:
                errors.append("detected model family disagrees with selection")
            if metrics.get("declared_model_family") != selection["model_family"]:
                errors.append("declared model family disagrees with selection")
            if metrics.get("model_revision") != selection["model_revision"]:
                errors.append("model revision disagrees with selection")
            if _normalized_declared_path(metrics.get("base_model_path")) != _normalized_declared_path(
                model_path
            ):
                errors.append("model path disagrees with selection")
            if _normalized_declared_path(metrics.get("adapter_path")) != _normalized_declared_path(
                adapter_path
            ):
                errors.append("adapter path disagrees with selection")
            for field in (
                "inference_seconds_per_sample",
                "estimated_test_seconds",
                "peak_vram_mib",
            ):
                if not _finite_number(metrics.get(field), positive=True):
                    errors.append(f"{field} must be finite and positive")
            if runtime.get("quantization_applied") is not (
                selection["precision"] == "nf4"
            ):
                errors.append("runtime quantization state disagrees with precision")
            if runtime.get("adapter_loaded") is not (adapter_path is not None):
                errors.append("runtime adapter state disagrees with selection")
            if runtime.get("precision") != selection["precision"]:
                errors.append("runtime precision disagrees with selection")
            if runtime.get("cuda_available") is not True:
                errors.append("CUDA is unavailable for final test inference")
            if runtime.get("model_eval") is not True:
                errors.append("model is not in eval mode")
            if runtime.get("use_cache") is not True:
                errors.append("model use_cache is not enabled")
        except Exception as exc:
            errors.append(f"final test inference artifacts invalid: {exc}")
        return errors

    provenance_context = {
        "operation": "post_selection_final_test_inference",
        "selected_model": normalized_selection,
        "selected_model_identity": {
            "model": _path_identity(model_path),
            "adapter": _path_identity(adapter_path) if adapter_path is not None else None,
        },
        "test_csv": _project_path(test_csv),
        "test_image_dir": _project_path(image_dir),
        "test_input_identity": test_input_identity,
        "expected_ordered_ids_sha256": hashlib.sha256(
            json.dumps(expected_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "expected_output_schema": {
            "submission_csv": ["Id", "Answer"],
            "rows": 819,
            "ordered_unique_ids": True,
            "answer_permutation": [1, 2, 3, 4],
            "audit_jsonl_rows": 819,
            "metrics_samples": 819,
            "immutable_provenance_and_sha256": True,
        },
        "publication_performed": False,
    }
    return {
        "stage_id": "post-selection-final-test-inference-819",
        "command": command,
        "validator": validate,
        "provenance_context": provenance_context,
        "uses_cuda": True,
    }


def select_final_model(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select one fully validated model by exact match and deterministic tie-breaks."""
    if not candidates:
        raise ValueError("at least one validated model candidate is required")
    required_candidate = {
        "name",
        "model_path",
        "model_family",
        "model_revision",
        "adapter_path",
        "precision",
        "image_size",
        "tta_orders",
        "aggregation_mode",
        "fallback_policy",
        "seed",
        "validation",
    }
    validated: list[dict[str, Any]] = []
    for candidate in candidates:
        missing = required_candidate - candidate.keys()
        if missing:
            raise ValueError(f"model candidate is missing fields: {sorted(missing)}")
        metrics = candidate["validation"]
        if metrics.get("samples") != 954:
            raise ValueError("model selection requires exactly 954 validation rows")
        exact_matches = metrics.get("exact_matches")
        non_identity_matches = metrics.get("non_identity_exact_matches")
        parse_failures = metrics.get("parse_failures")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (exact_matches, non_identity_matches, parse_failures)
        ):
            raise ValueError("model selection counts must be explicit integers")
        if not (
            0 <= exact_matches <= 954
            and 0 <= non_identity_matches <= 954
            and 0 <= parse_failures <= 954
        ):
            raise ValueError("model selection counts are outside the validation range")
        exact_rate = metrics.get("exact_match")
        if not _finite_number(exact_rate, positive=False) or not math.isclose(
            float(exact_rate), exact_matches / 954, abs_tol=1e-12
        ):
            raise ValueError("model selection exact-match rate disagrees with count")
        for field in ("non_identity_exact_match", "inference_seconds_per_sample"):
            if not _finite_number(metrics.get(field), positive=field.endswith("seconds_per_sample")):
                raise ValueError(f"model selection {field} is invalid")
        validated.append(
            {
                **candidate,
                "tta_orders": [list(order) for order in candidate["tta_orders"]],
                "validation": dict(metrics),
            }
        )
    return min(
        validated,
        key=lambda candidate: (
            -candidate["validation"]["exact_matches"],
            -candidate["validation"]["non_identity_exact_matches"],
            candidate["validation"]["parse_failures"],
            candidate["validation"]["inference_seconds_per_sample"],
            candidate["name"],
        ),
    )


def select_aggregation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select a TTA mode by EM, parse failures, then inference time."""
    parse_failures = int(metrics["parse_failures"])
    seconds = float(metrics["inference_seconds_per_sample"])
    candidates = []
    for mode in AGGREGATION_MODES:
        result = metrics["aggregation_comparison"][mode]["vs_hard"]
        candidates.append(
            {
                "mode": mode,
                "exact_matches": int(result["exact_matches"]),
                "accuracy": float(result["accuracy"]),
                "parse_failures": parse_failures,
                "inference_seconds_per_sample": seconds,
            }
        )
    return min(
        candidates,
        key=lambda item: (
            -item["exact_matches"],
            item["parse_failures"],
            item["inference_seconds_per_sample"],
            item["mode"],
        ),
    )


def validation_metrics_for_aggregation(
    metrics: dict[str, Any],
    audit_rows: Sequence[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Reconstruct selection metrics from the chosen mode's audited answers."""
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"unsupported aggregation mode: {mode}")
    samples = metrics.get("samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("aggregation selection requires a positive integer sample count")
    if len(audit_rows) != samples:
        raise ValueError("aggregation audit row count disagrees with metrics samples")

    exact_matches = 0
    non_identity_exact_matches = 0
    non_identity_samples = 0
    for row in audit_rows:
        reference = row.get("reference")
        no_ordering = row.get("no_ordering")
        aggregation = row.get("aggregations", {}).get(mode, {})
        answer = aggregation.get("answer")
        if not _valid_permutation(reference) or not _valid_permutation(answer):
            raise ValueError(f"aggregation audit row {row.get('Id')!r} is invalid")
        if not isinstance(no_ordering, bool):
            raise ValueError(f"aggregation audit row {row.get('Id')!r} lacks No_ordering")
        matched = answer == reference
        exact_matches += int(matched)
        if not no_ordering:
            non_identity_samples += 1
            non_identity_exact_matches += int(matched)

    reported = metrics.get("aggregation_comparison", {}).get(mode, {}).get("vs_hard", {})
    if reported.get("exact_matches") != exact_matches:
        raise ValueError("audited aggregation exact matches disagree with comparison metrics")
    exact_match = exact_matches / samples
    if not math.isclose(float(reported.get("accuracy", -1)), exact_match, abs_tol=1e-12):
        raise ValueError("audited aggregation accuracy disagrees with comparison metrics")

    return {
        **metrics,
        "aggregation_mode": mode,
        "exact_matches": exact_matches,
        "exact_match": exact_match,
        "accuracy": exact_match,
        "non_identity_exact_matches": non_identity_exact_matches,
        "non_identity_exact_match": (
            non_identity_exact_matches / non_identity_samples
            if non_identity_samples
            else 0.0
        ),
    }


def build_8b_candidate(
    *,
    best: dict[str, Any],
    best_adapter: Path,
    model_spec: dict[str, Any],
    tta4_metrics: dict[str, Any],
    tta4_audit: Sequence[dict[str, Any]],
    selected_mode: str,
) -> dict[str, Any]:
    validation = validation_metrics_for_aggregation(
        tta4_metrics, tta4_audit, selected_mode
    )
    return {
        "name": f"qwen3vl8b-checkpoint-{best['step']}-tta4-{selected_mode}",
        "model_path": MODEL_8B,
        "model_family": "qwen3_vl",
        "model_revision": MODEL_8B_REVISION,
        "model_repository": model_spec["repository"],
        "model_manifest": MODEL_8B_MANIFEST,
        "verified_model_tree_sha256": model_spec["verified_model_tree_sha256"],
        "adapter_path": best_adapter,
        "precision": "nf4",
        "image_size": 512,
        "tta_orders": [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        "aggregation_mode": selected_mode,
        "fallback_policy": "identity",
        "seed": 42,
        "validation": validation,
    }


def summarize_tta(metrics: dict[str, Any]) -> dict[str, Any]:
    modes = {}
    for mode in AGGREGATION_MODES:
        comparison = metrics["aggregation_comparison"][mode]["vs_hard"]
        modes[mode] = {
            "accuracy": comparison["accuracy"],
            "exact_matches": comparison["exact_matches"],
            "parse_failures": metrics["parse_failures"],
            "vs_hard": comparison,
        }
    return {
        "samples": metrics["samples"],
        "modes": modes,
        "selected": select_aggregation(metrics),
        "parse_failures": metrics["parse_failures"],
        "tta_consistency": metrics.get("tta_consistency"),
        "tta_agreement_patterns": metrics.get("tta_agreement_patterns"),
        "no_ordering_accuracy": metrics.get("no_ordering_accuracy"),
        "identity_exact_match": metrics.get("identity_exact_match"),
        "identity_samples": metrics.get("identity_samples"),
        "non_identity_exact_match": metrics.get("non_identity_exact_match"),
        "non_identity_samples": metrics.get("non_identity_samples"),
        "inference_seconds_per_sample": metrics["inference_seconds_per_sample"],
        "peak_vram_mib": metrics.get("peak_vram_mib"),
    }


def _stratum_key(row: dict[str, Any]) -> str:
    no_ordering = str(row.get("No_ordering", "false")).strip().lower()
    if no_ordering in {"1", "true"}:
        no_ordering = "true"
    elif no_ordering in {"0", "false"}:
        no_ordering = "false"
    else:
        raise ValueError(f"invalid No_ordering value: {row.get('No_ordering')!r}")
    return f"{row['Answer']}|{no_ordering}"


def build_stratified_subset_manifest(
    rows: list[dict[str, Any]],
    *,
    validation_ids: list[str],
    size: int,
    seed: int,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Select exactly ``size`` validation IDs with deterministic Hamilton quotas."""
    if not 0 < size <= len(validation_ids):
        raise ValueError("subset size must be positive and no larger than validation")
    by_id = {str(row["Id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("input rows must have unique IDs")
    missing = [sample_id for sample_id in validation_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"validation IDs missing from rows: {missing[:5]}")

    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in validation_ids:
        groups[_stratum_key(by_id[sample_id])].append(sample_id)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total = len(validation_ids)
    for key, ids in groups.items():
        ideal = size * len(ids) / total
        quotas[key] = int(ideal)
        remainders.append((ideal - quotas[key], key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : size - sum(quotas.values())
    ]:
        quotas[key] += 1

    rng = random.Random(seed)
    chosen: set[str] = set()
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        chosen.update(candidates[: quotas[key]])
    selected_ids = [sample_id for sample_id in validation_ids if sample_id in chosen]
    if len(selected_ids) != size:
        raise RuntimeError(f"stratified subset size {len(selected_ids)} != {size}")
    return {
        "schema_version": 1,
        "seed": seed,
        "size": size,
        "selection_strategy": "answer_and_no_ordering_hamilton",
        "source_manifest_sha256": source_manifest_sha256,
        "source_validation_count": total,
        "strata": {
            key: {"available": len(groups[key]), "selected": quotas[key]}
            for key in sorted(groups)
        },
        "validation_ids": selected_ids,
    }


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create JSON once and reject any later byte-changing rewrite."""
    serialized = _canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ValueError(f"immutable JSON does not match existing file: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8", newline="\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != serialized:
                raise ValueError(f"immutable JSON race at existing file: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_versioned_report(root: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write an immutable content-addressed report and return its path."""
    serialized = _canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    path = root / "reports" / name / f"report-{digest}.json"
    write_immutable_json(path, payload)
    return path


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        (ROOT / "scripts" / "run_phase1_sweep.py").resolve(),
        *[
            (ROOT / "src" / "snuaichal" / name).resolve()
            for name in (
                "training.py",
                "physical_memory.py",
                "augmentation.py",
                "modeling.py",
                "scheduling.py",
                "model_manifest.py",
                "inference.py",
                "tta.py",
                "evaluation.py",
            )
        ],
    ]
    return {_project_path(path): sha256_file(path) for path in paths}


def _path_identity(path: Path) -> dict[str, Any]:
    """Return a deterministic exact-file identity for a file or directory."""
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": _project_path(resolved),
            "kind": "file",
            "size": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    if not resolved.is_dir():
        return {"path": _project_path(resolved), "kind": "missing"}
    files = []
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": item.relative_to(resolved).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    tree_payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": _project_path(resolved),
        "kind": "directory",
        "files": files,
        "tree_sha256": hashlib.sha256(tree_payload).hexdigest(),
    }


def verify_preserved_manifest(
    manifest_path: Path, *, root: Path = ROOT
) -> dict[str, Any]:
    """Fail closed unless every unique, in-root preserved record still matches."""
    manifest = load_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list) or manifest.get("file_count") != len(records):
        raise ValueError("preserved manifest file_count mismatch")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    for record in records:
        relative = str(record.get("path", "")).replace("\\", "/")
        if not relative or relative in seen:
            raise ValueError(f"preserved manifest duplicate/empty path: {relative!r}")
        seen.add(relative)
        candidate = (root_resolved / relative).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"preserved path escapes root: {relative}") from exc
        if not candidate.is_file():
            raise ValueError(f"preserved file missing: {relative}")
        size = candidate.stat().st_size
        digest = sha256_file(candidate)
        if size != record.get("size_bytes") or digest != record.get("sha256"):
            raise ValueError(f"preserved file mismatch: {relative}")
        verified.append({"path": relative, "size": size, "sha256": digest})
    encoded = json.dumps(verified, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest": _path_identity(manifest_path),
        "verified_file_count": len(verified),
        "verified_tree_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validation_input_identity(
    csv_path: Path,
    image_dir: Path,
    validation_manifest: Path,
) -> dict[str, Any]:
    """Hash the CSV and exact ordered image bytes selected by a manifest."""
    manifest = load_json(validation_manifest)
    ids = [str(value) for value in manifest.get("validation_ids", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("validation manifest must contain nonempty unique ordered IDs")
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    by_id = {str(row["Id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("training CSV IDs are not unique")
    records: list[dict[str, Any]] = []
    image_root = image_dir.resolve()
    for sample_id in ids:
        row = by_id.get(sample_id)
        if row is None:
            raise ValueError(f"validation ID absent from CSV: {sample_id}")
        for column in ("Input_1", "Input_2", "Input_3", "Input_4"):
            relative = str(row[column]).replace("\\", "/")
            normalized_relative = f"{sample_id}/{relative}"
            image = (image_root / sample_id / relative).resolve()
            try:
                image.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"image path escapes image directory: {relative}") from exc
            if not image.is_file():
                raise ValueError(f"selected image missing: {relative}")
            records.append(
                {
                    "id": sample_id,
                    "slot": column,
                    "path": normalized_relative,
                    "size": image.stat().st_size,
                    "sha256": sha256_file(image),
                }
            )
    payload = {
        "csv": _path_identity(csv_path),
        "manifest": _path_identity(validation_manifest),
        "ordered_ids": ids,
        "images": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "tree_sha256": hashlib.sha256(encoded).hexdigest()}


def test_input_identity(csv_path: Path, image_dir: Path) -> dict[str, Any]:
    """Hash the exact ordered 819-row test inputs without reading image content semantically."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    required_columns = {"Id", "Input_1", "Input_2", "Input_3", "Input_4"}
    if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise ValueError("test CSV is missing required ID/image columns")
    ids = [str(row["Id"]) for row in rows]
    if len(ids) != 819 or len(set(ids)) != 819:
        raise ValueError("test CSV must contain exactly 819 unique ordered IDs")
    image_root = image_dir.resolve()
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["Id"])
        for column in ("Input_1", "Input_2", "Input_3", "Input_4"):
            relative = str(row[column]).replace("\\", "/")
            image = (image_root / sample_id / relative).resolve()
            try:
                image.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"test image escapes image directory: {relative}") from exc
            if not image.is_file():
                raise ValueError(f"test image missing: {sample_id}/{relative}")
            records.append(
                {
                    "id": sample_id,
                    "slot": column,
                    "path": f"{sample_id}/{relative}",
                    "size": image.stat().st_size,
                    "sha256": sha256_file(image),
                }
            )
    payload = {
        "schema_version": 1,
        "csv": _path_identity(csv_path),
        "ordered_ids": ids,
        "images": records,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    manifest_path = RESUME_ROOT / "input-identities" / "test" / f"input-{digest}.json"
    write_immutable_json(manifest_path, payload)
    return {
        **payload,
        "tree_sha256": digest,
        "content_addressed_manifest": _path_identity(manifest_path),
    }


def training_input_identity(
    csv_path: Path,
    image_dir: Path,
    *,
    limit: int | None,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    """Build a content-addressed identity for exact selected rows, images, and split."""
    from snuaichal.training import select_training_rows

    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        all_rows = list(csv.DictReader(file))
    train_rows, validation_rows = select_training_rows(
        all_rows,
        image_dir=image_dir,
        validation_fraction=validation_fraction,
        seed=seed,
        limit=limit,
        clean_validation=True,
    )
    rows = [*train_rows, *validation_rows]
    ids = [str(row["Id"]) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selected training CSV rows must contain nonempty unique IDs")
    image_root = image_dir.resolve()
    records: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["Id"])
        for column in ("Input_1", "Input_2", "Input_3", "Input_4"):
            relative = str(row[column]).replace("\\", "/")
            normalized_relative = f"{sample_id}/{relative}"
            image = (image_root / sample_id / relative).resolve()
            try:
                image.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"training image escapes image directory: {relative}") from exc
            if not image.is_file():
                raise ValueError(f"training image missing: {relative}")
            records.append(
                {
                    "id": sample_id,
                    "slot": column,
                    "path": normalized_relative,
                    "size": image.stat().st_size,
                    "sha256": sha256_file(image),
                }
            )
    payload = {
        "schema_version": 2,
        "csv": _path_identity(csv_path),
        "ordered_selected_ids": ids,
        "images": records,
        "split_manifest": {
            "train_ids": [str(row["Id"]) for row in train_rows],
            "validation_ids": [str(row["Id"]) for row in validation_rows],
        },
        "selection": {"limit": limit, "selected_rows": len(rows)},
        "split_configuration": {
            "strategy": "split_rows_without_image_overlap",
            "validation_fraction": validation_fraction,
            "seed": seed,
        },
        "augmentation": {
            "strategy": "epoch_seeded_input_permutation",
            "seed": seed,
        },
    }
    encoded = _canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    manifest_path = (
        RESUME_ROOT / "input-identities" / "training" / f"input-{digest}.json"
    )
    write_immutable_json(manifest_path, payload)
    return {
        **payload,
        "tree_sha256": digest,
        "content_addressed_manifest": _path_identity(manifest_path),
    }


def build_stage_provenance(
    stage_id: str,
    command: Sequence[str],
    provenance_context: dict[str, Any],
) -> dict[str, Any]:
    """Bind a stage to exact code, argv, preserved inputs, and declared context."""
    normalized_argv = [str(value).replace("\\", "/") for value in command]
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "stage_id": stage_id,
        "argv": normalized_argv,
        "working_directory": str(ROOT.resolve()).replace("\\", "/"),
        "source_hashes": _source_hashes(),
        "preserved_manifest": verify_preserved_manifest(PRESERVED_MANIFEST),
        "context": provenance_context,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return {**payload, "provenance_key": hashlib.sha256(encoded).hexdigest()}


def _write_attempt_artifact_manifest(attempt_dir: Path) -> dict[str, Any]:
    manifest_path = attempt_dir / "artifact-manifest.json"
    files = []
    for item in sorted(candidate for candidate in attempt_dir.rglob("*") if candidate.is_file()):
        if item == manifest_path:
            continue
        files.append(
            {
                "path": item.relative_to(attempt_dir).as_posix(),
                "size": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    payload = {"schema_version": 1, "files": files}
    write_immutable_json(manifest_path, payload)
    return payload


def _validate_attempt_artifact_manifest(
    attempt_dir: Path, *, expected_sha256: str | None
) -> list[str]:
    errors: list[str] = []
    manifest_path = attempt_dir / "artifact-manifest.json"
    try:
        if expected_sha256 is None or sha256_file(manifest_path) != expected_sha256:
            return ["artifact manifest hash differs from terminal status"]
        manifest = load_json(manifest_path)
        if manifest.get("schema_version") != 1 or not isinstance(
            manifest.get("files"), list
        ):
            return ["artifact manifest schema invalid"]
        recorded: set[str] = set()
        root = attempt_dir.resolve()
        for record in manifest["files"]:
            relative = str(record.get("path", "")).replace("\\", "/")
            if not relative or relative in recorded:
                errors.append(f"duplicate/empty artifact path: {relative!r}")
                continue
            recorded.add(relative)
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"artifact path escapes attempt: {relative}")
                continue
            if not path.is_file():
                errors.append(f"missing recorded artifact: {relative}")
                continue
            if path.stat().st_size != record.get("size"):
                errors.append(f"artifact size changed: {relative}")
            elif sha256_file(path) != record.get("sha256"):
                errors.append(f"artifact hash changed: {relative}")
        actual = {
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file() and item != manifest_path
        }
        if actual != recorded:
            errors.append("artifact inventory differs from immutable manifest")
    except Exception as exc:
        errors.append(f"artifact manifest invalid: {exc}")
    return errors


def compare_paired_audit_rows(
    rows_8b: list[dict[str, Any]], rows_27b: list[dict[str, Any]]
) -> dict[str, int]:
    """Compare 27B against 8B on the exact same ordered labeled rows."""
    ids_8b = [str(row["Id"]) for row in rows_8b]
    ids_27b = [str(row["Id"]) for row in rows_27b]
    if ids_8b != ids_27b:
        raise ValueError("paired audits must contain the same ordered IDs")
    counts = {
        "samples": len(rows_8b),
        "prediction_changes": 0,
        "corrected_by_27b": 0,
        "worsened_by_27b": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "net_gain_27b": 0,
    }
    for row_8b, row_27b in zip(rows_8b, rows_27b, strict=True):
        if row_8b["reference"] != row_27b["reference"]:
            raise ValueError(f"paired reference mismatch for ID {row_8b['Id']}")
        correct_8b = (
            row_8b.get("parse_ok") is True
            and row_8b["answer"] == row_8b["reference"]
        )
        correct_27b = (
            row_27b.get("parse_ok") is True
            and row_27b["answer"] == row_27b["reference"]
        )
        counts["prediction_changes"] += row_8b["answer"] != row_27b["answer"]
        if correct_27b and not correct_8b:
            counts["corrected_by_27b"] += 1
        elif correct_8b and not correct_27b:
            counts["worsened_by_27b"] += 1
        elif correct_8b:
            counts["both_correct"] += 1
        else:
            counts["both_wrong"] += 1
    counts["net_gain_27b"] = (
        counts["corrected_by_27b"] - counts["worsened_by_27b"]
    )
    return counts


def paired_decision(
    metrics_8b: dict[str, Any],
    metrics_27b: dict[str, Any],
    *,
    exact_match_margin: float = 0.03,
) -> dict[str, Any]:
    """Apply the predeclared paired-96 continuation gate."""
    accuracy_8b = float(metrics_8b["exact_match"])
    accuracy_27b = float(metrics_27b["exact_match"])
    exact_match_gain = accuracy_27b - accuracy_8b
    non_identity_8b = float(metrics_8b["non_identity_exact_match"])
    non_identity_27b = float(metrics_27b["non_identity_exact_match"])
    non_identity_gain = non_identity_27b - non_identity_8b
    parse_failure_gain = int(metrics_8b["parse_failures"]) - int(
        metrics_27b["parse_failures"]
    )
    epsilon = 1e-12
    if exact_match_gain + epsilon >= exact_match_margin:
        proceed = True
        reason = "exact_match_gain_at_least_0.03"
    elif (
        abs(exact_match_gain) <= exact_match_margin + epsilon
        and (non_identity_gain > 0 or parse_failure_gain > 0)
    ):
        proceed = True
        reason = "within_0.03_with_strict_secondary_improvement"
    else:
        proceed = False
        reason = "paired_gate_not_met"
    return {
        "continue_to_full_validation": proceed,
        "reason": reason,
        "exact_match_8b": accuracy_8b,
        "exact_match_27b": accuracy_27b,
        "exact_match_gain": exact_match_gain,
        "non_identity_8b": non_identity_8b,
        "non_identity_27b": non_identity_27b,
        "non_identity_gain": non_identity_gain,
        "parse_failures_8b": int(metrics_8b["parse_failures"]),
        "parse_failures_27b": int(metrics_27b["parse_failures"]),
        "parse_failure_reduction": parse_failure_gain,
    }


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return psutil.Process(int(pid)).is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def process_identity(pid: int) -> dict[str, Any]:
    """Capture mandatory PID-reuse-resistant identity."""
    process = psutil.Process(pid)
    return {
        "pid": pid,
        "create_time": process.create_time(),
        "command_line": process.cmdline(),
    }


def recorded_process_alive(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    expected = record.get("process_identity")
    if not isinstance(expected, dict) or "create_time" not in expected:
        return False
    if not process_alive(pid):
        return False
    try:
        current = process_identity(int(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False
    return (
        current.get("create_time") == expected.get("create_time")
        and current.get("command_line") == expected.get("command_line")
    )


def acquire_runner_lock(path: Path = RUNNER_LOCK_PATH) -> dict[str, Any]:
    """Atomically acquire a non-reentrant, PID-identity-safe runner lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        record = {
            "schema_version": 1,
            "state": "running",
            "started_at": durable.utc_now(),
            "pid": os.getpid(),
            "process_identity": process_identity(os.getpid()),
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                existing = load_json(path)
            except Exception as exc:
                raise RuntimeError(f"runner lock is unreadable: {exc}") from exc
            if existing.get("state") == "running" and recorded_process_alive(existing):
                raise RuntimeError(
                    f"runner already active with PID {existing.get('pid')}"
                )
            history = path.parent / "runner-lock-history"
            write_versioned_report(history, "stale-lock", existing)
            path.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(_canonical_json(record))
            file.flush()
            os.fsync(file.fileno())
        return record
    raise RuntimeError("could not acquire runner lock")


def release_runner_lock(
    path: Path,
    record: dict[str, Any],
    *,
    state: str,
) -> None:
    """Atomically leave a terminal lock record for durable diagnosis."""
    current = load_json(path)
    if (
        current.get("pid") != record.get("pid")
        or current.get("started_at") != record.get("started_at")
    ):
        raise RuntimeError("runner lock ownership changed")
    terminal = {**record, "state": state, "ended_at": durable.utc_now()}
    durable.atomic_write_json(path, terminal)


def cuda_compute_processes() -> list[dict[str, Any]]:
    completed: subprocess.CompletedProcess[str] | None = None
    for attempt in range(3):
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(1.0)
    if completed is None:  # pragma: no cover - loop either succeeds or raises
        raise RuntimeError("nvidia-smi query produced no result")
    records: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid_text, _, name = line.partition(",")
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        records.append({"pid": pid, "process_name": name.strip()})
    return records


def assert_no_foreign_cuda_processes() -> None:
    """Reject credible ML compute owners while ignoring WDDM GUI process noise."""
    conflicts = [
        identity
        for record in cuda_compute_processes()
        if record.get("pid") != os.getpid()
        if (identity := cuda_workload_identity(int(record["pid"]))) is not None
    ]
    if conflicts:
        raise RuntimeError(f"foreign CUDA compute process(es) detected: {conflicts}")


def run_stage(
    stage_id: str,
    command: Sequence[str],
    *,
    validator: Validator,
    poll_seconds: float,
    provenance_context: dict[str, Any],
    external_outputs: Sequence[Path] = (),
    uses_cuda: bool = False,
    child_environment: dict[str, str] | None = None,
    post_success: Callable[[], None] | None = None,
) -> Path:
    """Run one fail-stop stage with provenance-safe immutable attempts."""
    provenance = build_stage_provenance(stage_id, command, provenance_context)
    status = durable.load_status(STATUS_PATH)
    previous = status.get("experiments", {}).get(stage_id)
    if previous:
        previous_dir = Path(previous["attempt_dir"])
        exit_path = previous_dir / "exit-code.txt"
        external_identities = [_path_identity(path) for path in external_outputs]
        reusable = (
            previous.get("status") == "succeeded"
            and previous.get("provenance_key") == provenance["provenance_key"]
            and previous.get("external_output_identities") == external_identities
            and exit_path.is_file()
            and exit_path.read_text(encoding="utf-8").strip() == "0"
            and not _validate_attempt_artifact_manifest(
                previous_dir,
                expected_sha256=previous.get("artifact_manifest_sha256"),
            )
            and not validator(previous_dir)
        )
        if reusable:
            return previous_dir
        if previous.get("status") == "running" and recorded_process_alive(previous):
            raise RuntimeError(
                f"stage {stage_id} already has live PID {previous.get('pid')}"
            )

    attempt_number, attempt_dir = durable.next_attempt_dir(STAGES_ROOT, stage_id)
    concrete_command = durable.materialize_command(command, attempt_dir)
    record = {
        "experiment_id": stage_id,
        "attempt": attempt_number,
        "attempt_dir": str(attempt_dir),
        "command": concrete_command,
        "provenance_key": provenance["provenance_key"],
        "started_at": durable.utc_now(),
        "status": "running",
    }
    status.setdefault("experiments", {})[stage_id] = record
    status["active_experiment"] = stage_id
    status["state"] = "running"
    status["updated_at"] = durable.utc_now()
    durable.atomic_write_json(STATUS_PATH, status)

    def heartbeat(pid: int) -> None:
        record["pid"] = pid
        record.setdefault("process_identity", process_identity(pid))
        record["heartbeat_at"] = durable.utc_now()
        current = durable.load_status(STATUS_PATH)
        current.setdefault("experiments", {})[stage_id] = dict(record)
        current["active_experiment"] = stage_id
        current["state"] = "running"
        current["updated_at"] = durable.utc_now()
        durable.atomic_write_json(STATUS_PATH, current)

    exit_code: int | None = None
    errors: list[str] = []
    caught: BaseException | None = None
    terminal_status = "failed"
    exception_text: str | None = None
    try:
        if uses_cuda:
            assert_no_foreign_cuda_processes()
        exit_code = durable.run_command(
            concrete_command,
            attempt_dir,
            poll_seconds=poll_seconds,
            on_progress=heartbeat,
            environment=(
                dict(child_environment)
                if child_environment is not None
                else build_ml_child_environment()
            ),
        )
        if exit_code == 0 and post_success is not None:
            post_success()
        errors = validator(attempt_dir)
        terminal_status = "succeeded" if exit_code == 0 and not errors else "failed"
    except BaseException as exc:
        caught = exc
        terminal_status = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        exception_text = "".join(traceback.format_exception(exc))
        errors = [f"runner exception: {type(exc).__name__}: {exc}"]
    finally:
        artifact_manifest_sha256: str | None = None
        external_identities: list[dict[str, Any]] = []
        if attempt_dir.is_dir():
            try:
                write_immutable_json(attempt_dir / "provenance.json", provenance)
                if exception_text is not None:
                    (attempt_dir / "traceback.txt").write_text(
                        exception_text, encoding="utf-8", newline="\n"
                    )
                _write_attempt_artifact_manifest(attempt_dir)
                artifact_manifest_sha256 = sha256_file(
                    attempt_dir / "artifact-manifest.json"
                )
                if terminal_status == "succeeded":
                    external_identities = [
                        _path_identity(path) for path in external_outputs
                    ]
            except Exception as manifest_exc:
                errors.append(f"artifact finalization failed: {manifest_exc}")
                terminal_status = "failed"
        current = durable.load_status(STATUS_PATH)
        current_record = current.setdefault("experiments", {}).get(stage_id, record)
        current_record.update(
            {
                "ended_at": durable.utc_now(),
                "exit_code": exit_code,
                "artifact_errors": errors,
                "status": terminal_status,
                "provenance_key": provenance["provenance_key"],
                "provenance_path": str(attempt_dir / "provenance.json"),
                "traceback_path": (
                    str(attempt_dir / "traceback.txt")
                    if exception_text is not None
                    else None
                ),
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "external_output_identities": external_identities,
            }
        )
        current["experiments"][stage_id] = current_record
        current.pop("active_experiment", None)
        current["last_experiment"] = stage_id
        current["state"] = terminal_status
        current["updated_at"] = durable.utc_now()
        durable.atomic_write_json(STATUS_PATH, current)
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY_PATH.open("a", encoding="utf-8", newline="\n") as registry:
            registry.write(json.dumps(current_record, sort_keys=True) + "\n")

    if caught is not None:
        raise caught
    if terminal_status != "succeeded":
        raise RuntimeError(
            f"{stage_id} failed: exit={exit_code}, artifact_errors={errors}"
        )
    return attempt_dir


def _valid_permutation(value: Any) -> bool:
    return is_permutation(value)


def reuse_historical_or_create(
    *,
    historical_attempt: Path,
    validator: Validator,
    expected_context: dict[str, Any],
    create_new: Callable[[], Path],
) -> Path:
    """Reuse a historical attempt only when semantics and provenance are exact."""
    errors = list(validator(historical_attempt))
    provenance_path = historical_attempt / "provenance.json"
    try:
        provenance = load_json(provenance_path)
        recorded_key = provenance.pop("provenance_key")
        encoded = _canonical_json(provenance).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != recorded_key:
            errors.append("historical provenance key is invalid")
        if provenance.get("context") != expected_context:
            errors.append("historical provenance context does not match")
        if provenance.get("source_hashes") != _source_hashes():
            errors.append("historical provenance source hashes do not match")
        if provenance.get("preserved_manifest") != verify_preserved_manifest(
            PRESERVED_MANIFEST
        ):
            errors.append("historical provenance preserved manifest does not match")
    except Exception as exc:
        errors.append(f"historical provenance is invalid: {exc}")
    if errors:
        return create_new()
    return historical_attempt


def _normalized_declared_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return os.path.normcase(str(path.resolve()))


def inference_validator(
    expected_rows: int,
    *,
    adapter_loaded: bool,
    validation_manifest: Path,
    expected_tta: int,
    expected_aggregation_mode: str,
    expected_model_path: Path,
    expected_model_family: str,
    expected_model_revision: str,
    expected_adapter_path: Path | None,
    expected_precision: str,
    uses_cuda: bool,
    require_valid_parse: bool = False,
) -> Validator:
    if expected_rows <= 0 or expected_tta <= 0:
        raise ValueError("expected row and TTA counts must be positive")
    if not expected_model_family or not expected_model_revision or not expected_precision:
        raise ValueError("model family, revision, and precision must be explicit")
    if not validation_manifest.is_file():
        raise ValueError("expected validation manifest is missing")
    if adapter_loaded != (expected_adapter_path is not None):
        raise ValueError("adapter expectation and expected adapter path disagree")

    def validate(attempt: Path) -> list[str]:
        errors = durable.validate_artifacts(attempt, expected_rows=expected_rows)
        try:
            metrics = load_json(attempt / "metrics.json")
            audit_rows = [
                json.loads(line)
                for line in (attempt / "audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            with (attempt / "predictions.csv").open(
                newline="", encoding="utf-8-sig"
            ) as file:
                reader = csv.DictReader(file)
                prediction_rows = list(reader)
                if reader.fieldnames != ["Id", "Answer"]:
                    errors.append("predictions CSV schema must be exactly Id,Answer")
            manifest_ids = [
                str(value)
                for value in load_json(validation_manifest).get("validation_ids", [])
            ]
            if not manifest_ids or len(manifest_ids) != len(set(manifest_ids)):
                errors.append("validation manifest IDs must be nonempty and unique")
            expected_ids = manifest_ids[:expected_rows]
            if len(expected_ids) != expected_rows:
                errors.append("validation manifest has fewer IDs than expected rows")
            audit_ids = [str(row.get("Id")) for row in audit_rows]
            prediction_ids = [str(row.get("Id")) for row in prediction_rows]
            if len(audit_ids) != len(set(audit_ids)):
                errors.append("audit IDs are not unique")
            if len(prediction_ids) != len(set(prediction_ids)):
                errors.append("prediction IDs are not unique")
            if audit_ids != expected_ids or prediction_ids != expected_ids:
                errors.append("artifact IDs/order disagree with validation manifest")
            required_audit_fields = {
                "Id",
                "answer",
                "parse_ok",
                "valid_tta_views",
                "aggregation_mode",
                "reference",
                "no_ordering",
                "views",
                "aggregations",
            }
            for row in audit_rows:
                missing = required_audit_fields - row.keys()
                if missing:
                    errors.append(f"audit schema missing fields: {sorted(missing)}")
            required_metric_fields = {
                "samples",
                "exact_matches",
                "exact_match",
                "parse_failures",
                "non_identity_exact_matches",
                "non_identity_exact_match",
                "inference_seconds_per_sample",
                "estimated_test_seconds",
                "peak_vram_mib",
                "model_precision",
                "detected_model_family",
                "declared_model_family",
                "model_revision",
                "base_model_path",
                "adapter_path",
                "tta_views",
                "aggregation_mode",
                "runtime_state",
                "aggregation_comparison",
            }
            metric_missing = required_metric_fields - metrics.keys()
            if metric_missing:
                errors.append(f"metric schema missing fields: {sorted(metric_missing)}")
            runtime = metrics.get("runtime_state", {})
            if metrics.get("model_precision") != expected_precision:
                errors.append(
                    f"precision is {metrics.get('model_precision')!r}, not {expected_precision!r}"
                )
            if runtime.get("precision") != expected_precision:
                errors.append("runtime precision disagrees with expected precision")
            if metrics.get("detected_model_family") != expected_model_family:
                errors.append("detected model family disagrees with expected model family")
            if metrics.get("declared_model_family") != expected_model_family:
                errors.append("declared model family disagrees with expected model family")
            if metrics.get("model_family") not in (None, expected_model_family):
                errors.append("model family disagrees with expected model family")
            if metrics.get("model_revision") != expected_model_revision:
                errors.append("model revision disagrees with pinned revision")
            if _normalized_declared_path(metrics.get("base_model_path")) != _normalized_declared_path(
                expected_model_path
            ):
                errors.append("model path disagrees with expected model path")
            if _normalized_declared_path(metrics.get("adapter_path")) != _normalized_declared_path(
                expected_adapter_path
            ):
                errors.append("adapter path disagrees with expected adapter path")
            if metrics.get("tta_views") != expected_tta:
                errors.append("metrics tta_views disagrees with command")
            if metrics.get("aggregation_mode") != expected_aggregation_mode:
                errors.append("metrics aggregation_mode disagrees with command")
            if runtime.get("quantization_applied") is not True:
                errors.append("NF4 quantization was not applied")
            if runtime.get("adapter_loaded") is not adapter_loaded:
                errors.append(
                    f"adapter_loaded={runtime.get('adapter_loaded')!r}, "
                    f"expected {adapter_loaded!r}"
                )
            if uses_cuda and runtime.get("cuda_available") is not True:
                errors.append("CUDA is unavailable for a required CUDA stage")
            if runtime.get("model_eval") is not True:
                errors.append("model is not in eval mode")
            if runtime.get("use_cache") is not True:
                errors.append("model use_cache is not enabled")
            for field in (
                "inference_seconds_per_sample",
                "estimated_test_seconds",
                "peak_vram_mib",
            ):
                if not _finite_number(metrics.get(field), positive=True):
                    errors.append(f"{field} must be finite and positive")
            peak_mib = metrics.get("peak_vram_mib")
            if _finite_number(peak_mib, positive=True) and float(peak_mib) * 1024**2 > PHYSICAL_VRAM_BYTES:
                errors.append("peak_vram_mib exceeds physical VRAM")
            parse_failures = 0
            exact_matches = 0
            non_identity_total = 0
            non_identity_matches = 0
            mode_matches = {mode: 0 for mode in AGGREGATION_MODES}
            for index, row in enumerate(audit_rows):
                answer = row.get("answer")
                reference = row.get("reference")
                parse_ok = row.get("parse_ok") is True
                if not isinstance(row.get("parse_ok"), bool):
                    errors.append(f"ID {row.get('Id')!r} parse_ok is not boolean")
                if not isinstance(row.get("no_ordering"), bool):
                    errors.append(f"ID {row.get('Id')!r} no_ordering is not boolean")
                if not _valid_permutation(reference):
                    errors.append(f"ID {row.get('Id')!r} has invalid reference")
                if not _valid_permutation(answer):
                    errors.append(f"ID {row.get('Id')!r} answer is not a permutation")
                if row.get("aggregation_mode") != expected_aggregation_mode:
                    errors.append(f"ID {row.get('Id')!r} aggregation mode mismatch")
                if not isinstance(row.get("views"), list) or len(row["views"]) != expected_tta:
                    errors.append(f"ID {row.get('Id')!r} TTA view count mismatch")
                valid_views = row.get("valid_tta_views")
                if not isinstance(valid_views, int) or not 0 <= valid_views <= expected_tta:
                    errors.append(f"ID {row.get('Id')!r} valid view count impossible")
                if parse_ok and not _valid_permutation(answer):
                    errors.append(f"ID {row.get('Id')!r} parsed answer is not a permutation")
                if require_valid_parse and (not parse_ok or not _valid_permutation(answer)):
                    errors.append(f"ID {row.get('Id')!r} lacks a valid parsed permutation")
                if not parse_ok:
                    parse_failures += 1
                selected_correct = parse_ok and answer == reference
                exact_matches += selected_correct
                if row.get("no_ordering") is not True:
                    non_identity_total += 1
                    non_identity_matches += selected_correct
                try:
                    csv_answer = json.loads(prediction_rows[index]["Answer"])
                    if csv_answer != answer or not _valid_permutation(csv_answer):
                        errors.append(f"ID {row.get('Id')!r} CSV answer mismatch")
                except Exception as exc:
                    errors.append(f"ID {row.get('Id')!r} CSV answer invalid: {exc}")
                aggregations = row.get("aggregations", {})
                if not isinstance(aggregations, dict) or set(aggregations) != set(
                    AGGREGATION_MODES
                ):
                    errors.append(f"ID {row.get('Id')!r} aggregation schema mismatch")
                    aggregations = {}
                for mode in AGGREGATION_MODES:
                    aggregate = aggregations.get(mode, {})
                    mode_answer = aggregate.get("answer")
                    mode_valid = aggregate.get("valid_views")
                    if not isinstance(mode_valid, int) or not 0 <= mode_valid <= expected_tta:
                        errors.append(f"ID {row.get('Id')!r} {mode} valid_views impossible")
                    if mode_valid and not _valid_permutation(mode_answer):
                        errors.append(f"ID {row.get('Id')!r} {mode} answer invalid")
                    mode_matches[mode] += bool(mode_valid) and mode_answer == reference
            expected_exact = exact_matches / expected_rows
            expected_non_identity = (
                non_identity_matches / non_identity_total if non_identity_total else 0.0
            )
            checks = {
                "samples": expected_rows,
                "parse_failures": parse_failures,
                "exact_matches": exact_matches,
                "non_identity_exact_matches": non_identity_matches,
            }
            for field, expected in checks.items():
                if metrics.get(field) != expected:
                    errors.append(f"metrics {field} disagrees with ordered audit")
            for field, expected in (
                ("exact_match", expected_exact),
                ("non_identity_exact_match", expected_non_identity),
            ):
                value = metrics.get(field)
                if not _finite_number(value, positive=False) or not math.isclose(
                    float(value), expected, abs_tol=1e-12
                ):
                    errors.append(f"metrics {field} disagrees with ordered audit")
            comparison = metrics.get("aggregation_comparison", {})
            for mode, matches in mode_matches.items():
                values = comparison.get(mode, {}).get("vs_hard", {})
                if values.get("exact_matches") != matches or not math.isclose(
                    float(values.get("accuracy", float("nan"))),
                    matches / expected_rows,
                    abs_tol=1e-12,
                ):
                    errors.append(f"aggregation metrics for {mode} disagree with audit")
        except Exception as exc:
            errors.append(f"runtime metrics invalid: {exc}")
        return errors

    return validate


def download_validator(_attempt: Path) -> list[str]:
    required = [MODEL_27B / "config.json", MODEL_27B / "model.safetensors.index.json"]
    errors = [f"missing {path}" for path in required if not path.is_file()]
    weight_bytes = sum(path.stat().st_size for path in MODEL_27B.glob("*.safetensors"))
    if weight_bytes < 50_000_000_000:
        errors.append(f"Qwen3.5 weight bytes too small: {weight_bytes}")
    return errors


def _finite_number(value: Any, *, positive: bool, integer: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    if integer and not isinstance(value, int):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    return numeric > 0 if positive else numeric >= 0


def training_validator(
    expected_step: int,
    *,
    resume_from_checkpoint: Path | None = None,
) -> Validator:
    expected_resume_identity = (
        _path_identity(resume_from_checkpoint)
        if resume_from_checkpoint is not None
        else None
    )

    def validate(attempt: Path) -> list[str]:
        output = attempt / "training"
        checkpoint = output / f"checkpoint-{expected_step}"
        required = [
            attempt / "pid.txt",
            output / "training_summary.json",
            output / "physical_memory_measurement.json",
            output / "model_manifest.json",
            output / "schedule.json",
            output / "final" / "adapter_config.json",
            output / "final" / "adapter_model.safetensors",
            checkpoint / "trainer_state.json",
            checkpoint / "adapter_config.json",
            checkpoint / "adapter_model.safetensors",
            checkpoint / "optimizer.pt",
            checkpoint / "scheduler.pt",
            checkpoint / "rng_state.pth",
            checkpoint / "training_args.bin",
        ]
        errors = [
            f"missing or empty {path}"
            for path in required
            if not path.is_file() or path.stat().st_size <= 0
        ]
        try:
            summary = load_json(output / "training_summary.json")
            physical = load_json(output / "physical_memory_measurement.json")
            manifest = load_json(output / "model_manifest.json")
            schedule = load_json(output / "schedule.json")
            trainer_state = load_json(checkpoint / "trainer_state.json")
            if summary.get("global_step") != expected_step:
                errors.append(
                    f"global_step={summary.get('global_step')!r}, expected {expected_step}"
                )
            if trainer_state.get("global_step") != expected_step:
                errors.append(
                    "checkpoint trainer_state global_step does not match expected step"
                )
            numeric_rules = {
                "epoch": True,
                "learning_rate": expected_step <= 3,
                "training_loss": True,
                "train_runtime_seconds": True,
                "train_steps_per_second": True,
                "seconds_per_optimizer_step": True,
                "initial_global_step": False,
                "optimizer_steps_this_run": True,
                "logical_peak_allocated_bytes": False,
                "logical_peak_reserved_bytes": False,
            }
            for field, positive in numeric_rules.items():
                if not _finite_number(summary.get(field), positive=positive):
                    errors.append(
                        f"{field} must be finite and "
                        f"{'positive' if positive else 'non-negative'}"
                    )
            if summary.get("memory_schema_version") != 3:
                errors.append("memory schema version is not 3")
            allocated = summary.get("logical_peak_allocated_bytes")
            reserved = summary.get("logical_peak_reserved_bytes")
            if not _finite_number(allocated, positive=False, integer=True):
                errors.append("logical peak allocated bytes must be a non-negative integer")
            if not _finite_number(reserved, positive=False, integer=True):
                errors.append("logical peak reserved bytes must be a non-negative integer")
            if (
                _finite_number(allocated, positive=False)
                and _finite_number(reserved, positive=False)
                and float(allocated) > float(reserved)
            ):
                errors.append("logical peak allocated bytes exceeds logical peak reserved bytes")
            if summary.get("allocator_memory_semantics") != ALLOCATOR_MEMORY_SEMANTICS:
                errors.append("allocator memory semantics are missing or incorrect")
            if not isinstance(summary.get("allocator_backend"), str) or not str(
                summary.get("allocator_backend", "")
            ).strip():
                errors.append("allocator_backend must be a non-empty string")
            if physical.get("schema_version") != 1:
                errors.append("physical monitor schema version is not 1")
            trusted_identity = summary.get("trusted_process_identity")
            try:
                launched_pid = int((attempt / "pid.txt").read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                launched_pid = None
                errors.append("launched child PID record is invalid")
            if not isinstance(trusted_identity, dict):
                errors.append("trusted process identity is missing")
            else:
                trusted_pid = trusted_identity.get("pid")
                create_time = trusted_identity.get("create_time")
                command_digest = trusted_identity.get("command_identity")
                if (
                    isinstance(trusted_pid, bool)
                    or not isinstance(trusted_pid, int)
                    or trusted_pid <= 0
                    or trusted_pid != launched_pid
                ):
                    errors.append("monitor PID differs from the launched CUDA child")
                if not _finite_number(create_time, positive=True):
                    errors.append("monitor create-time identity is invalid")
                if (
                    not isinstance(command_digest, str)
                    or len(command_digest) != 64
                    or any(character not in "0123456789abcdef" for character in command_digest)
                ):
                    errors.append("monitor command identity is not a SHA-256 digest")
            for key, value in physical.items():
                if key != "schema_version" and summary.get(key) != value:
                    errors.append(f"physical monitor field {key} differs from summary")
            interval = summary.get("sample_interval_seconds")
            if (
                not _finite_number(interval, positive=True)
                or float(interval) > 1.0
            ):
                errors.append("physical sample interval must be in (0, 1] seconds")
            parsed_timestamps: list[datetime] = []
            for timestamp_field in ("sampling_started_at", "sampling_ended_at"):
                try:
                    parsed = datetime.fromisoformat(str(summary.get(timestamp_field)))
                    if parsed.tzinfo is None:
                        raise ValueError("timezone missing")
                    parsed_timestamps.append(parsed)
                except (TypeError, ValueError):
                    errors.append(f"{timestamp_field} must be a timezone-aware timestamp")
            if (
                len(parsed_timestamps) == 2
                and parsed_timestamps[1] < parsed_timestamps[0]
            ):
                errors.append("physical sampling ended before it started")
            if summary.get("sampling_started_before_model_load") is not True:
                errors.append("physical sampling did not start before model loading")
            if summary.get("sampling_finished_after_work") is not True:
                errors.append("physical sampling did not cover the end of child work")

            measurement_status = summary.get("physical_measurement_status")
            observed = summary.get("physical_peak_observed_bytes")
            physical_total = summary.get("physical_total_vram_bytes")
            source = summary.get("physical_measurement_source")
            sample_count = summary.get("sample_count")
            if measurement_status == "valid":
                for field, value, positive in (
                    ("physical_total_vram_bytes", physical_total, True),
                    ("physical_peak_observed_bytes", observed, False),
                    ("sample_count", sample_count, True),
                ):
                    if not _finite_number(value, positive=positive, integer=True):
                        errors.append(f"{field} must be an integer with valid bounds")
                if (
                    _finite_number(observed, positive=False)
                    and _finite_number(physical_total, positive=True)
                    and float(observed) > float(physical_total)
                ):
                    errors.append("observed physical peak exceeds physical total VRAM")
                if source not in {
                    "nvml_per_process_used_bytes",
                    "nvml_device_memory_info_used",
                }:
                    errors.append("physical measurement source is invalid")
                if summary.get("continuation_gate_source") != source:
                    errors.append("continuation gate does not use the selected physical source")
                if summary.get("continuation_gate_bytes") != observed:
                    errors.append("continuation gate bytes differ from physical peak")
                if summary.get("process_identity_match") is not True:
                    errors.append("trusted CUDA child identity did not match")
                if summary.get("trusted_cuda_child_seen") is not True:
                    errors.append("trusted CUDA child was not observed")
                unexpected = summary.get("unexpected_cuda_processes")
                if not isinstance(unexpected, list) or unexpected:
                    errors.append("unexpected CUDA process invalidates physical measurement")
                if source == "nvml_per_process_used_bytes":
                    if not _finite_number(
                        summary.get("process_sample_count"),
                        positive=True,
                        integer=True,
                    ):
                        errors.append("per-process source has zero valid samples")
                elif source == "nvml_device_memory_info_used":
                    if summary.get("process_memory_unavailable") is not True:
                        errors.append("device fallback used while process memory was available")
                    if summary.get("device_fallback_exclusive") is not True:
                        errors.append("device fallback workload exclusivity is unproven")
                    if not _finite_number(
                        summary.get("device_sample_count"),
                        positive=True,
                        integer=True,
                    ):
                        errors.append("device fallback has zero valid samples")
            elif measurement_status == "indeterminate":
                if observed is not None or source is not None or sample_count != 0:
                    errors.append("indeterminate measurement selected physical evidence")
                if summary.get("continuation_gate_source") is not None:
                    errors.append("indeterminate measurement selected a gate source")
                if summary.get("continuation_gate_bytes") is not None:
                    errors.append("indeterminate measurement selected gate bytes")
                if not str(summary.get("physical_measurement_reason", "")).strip():
                    errors.append("indeterminate measurement requires a reason")
            else:
                errors.append("physical measurement status is invalid")
            steps_per_second = summary.get("train_steps_per_second")
            seconds_per_step = summary.get("seconds_per_optimizer_step")
            if _finite_number(steps_per_second, positive=True) and _finite_number(
                seconds_per_step, positive=True
            ) and not math.isclose(
                float(seconds_per_step),
                1.0 / float(steps_per_second),
                rel_tol=0.02,
            ):
                errors.append("step timing metrics are internally inconsistent")
            expected_initial_step = 0
            if resume_from_checkpoint is not None:
                expected_initial_step = int(
                    load_json(resume_from_checkpoint / "trainer_state.json")[
                        "global_step"
                    ]
                )
            completed_steps = expected_step - expected_initial_step
            if summary.get("initial_global_step") != expected_initial_step:
                errors.append("training did not begin at the required checkpoint step")
            if summary.get("optimizer_steps_this_run") != completed_steps:
                errors.append("optimizer step delta does not prove exact continuation")
            runtime = summary.get("train_runtime_seconds")
            if (
                _finite_number(runtime, positive=True)
                and _finite_number(seconds_per_step, positive=True)
                and completed_steps > 0
                and not math.isclose(
                    float(seconds_per_step),
                    float(runtime) / completed_steps,
                    rel_tol=0.02,
                )
            ):
                errors.append("seconds per optimizer step uses the wrong step count")
            trainable = manifest.get("trainable_parameters")
            total = manifest.get("total_parameters")
            if not _finite_number(trainable, positive=True, integer=True):
                errors.append("trainable_parameters must be a positive integer")
            if not _finite_number(total, positive=True, integer=True):
                errors.append("total_parameters must be a positive integer")
            elif isinstance(trainable, int) and trainable > total:
                errors.append("trainable_parameters exceeds total_parameters")
            if manifest.get("load_in_4bit") is not True:
                errors.append("training was not NF4")
            if Path(str(manifest.get("model_path"))).as_posix() != _project_path(MODEL_27B):
                errors.append("model_manifest model_path does not match Qwen3.5-27B")
            if manifest.get("lora_rank") != 8:
                errors.append("model_manifest lora_rank is not 8")
            if schedule.get("stop_after_steps") != expected_step:
                errors.append("schedule stop_after_steps does not match command")
            if resume_from_checkpoint is not None:
                source_state = load_json(resume_from_checkpoint / "trainer_state.json")
                source_step = source_state.get("global_step")
                source_required = [
                    resume_from_checkpoint / "adapter_config.json",
                    resume_from_checkpoint / "adapter_model.safetensors",
                    resume_from_checkpoint / "optimizer.pt",
                    resume_from_checkpoint / "scheduler.pt",
                    resume_from_checkpoint / "rng_state.pth",
                    resume_from_checkpoint / "training_args.bin",
                ]
                if any(
                    not path.is_file() or path.stat().st_size <= 0
                    for path in source_required
                ):
                    errors.append("resume checkpoint is missing required state artifacts")
                if expected_step == 3:
                    if source_step != 2 or resume_from_checkpoint.name != "checkpoint-2":
                        errors.append("step-3 continuation must start from exact checkpoint-2")
                elif (
                    not isinstance(source_step, int)
                    or source_step <= 0
                    or source_step >= expected_step
                    or resume_from_checkpoint.name != f"checkpoint-{source_step}"
                ):
                    errors.append("resume checkpoint trainer_state/step is invalid")
                if (
                    expected_resume_identity is None
                    or _path_identity(resume_from_checkpoint) != expected_resume_identity
                ):
                    errors.append("resume checkpoint identity changed during continuation")
        except Exception as exc:
            errors.append(f"training metadata invalid: {exc}")
        return errors

    return validate


def validated_legacy_smoke_attempt() -> Path:
    """Reuse only the immutable execution-valid, measurement-indeterminate smoke."""
    audit = load_json(CORRECTIVE_AUDIT_PATH)
    attempt = LEGACY_SMOKE_ATTEMPT
    if audit.get("source_attempt", {}).get("path") != _project_path(attempt):
        raise RuntimeError("corrective audit source attempt mismatch")
    expected_manifest = audit.get("source_attempt", {}).get(
        "artifact_manifest_sha256"
    )
    manifest_errors = _validate_attempt_artifact_manifest(
        attempt, expected_sha256=expected_manifest
    )
    if manifest_errors:
        raise RuntimeError(f"legacy smoke artifact integrity failed: {manifest_errors}")
    if (attempt / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        raise RuntimeError("legacy smoke execution did not exit successfully")
    summary = load_json(attempt / "training" / "training_summary.json")
    checkpoint = attempt / "training" / "checkpoint-2"
    state = load_json(checkpoint / "trainer_state.json")
    if summary.get("global_step") != 2 or state.get("global_step") != 2:
        raise RuntimeError("legacy smoke did not complete exact optimizer step 2")
    source = audit.get("source_attempt", {})
    expected_hashes = {
        attempt / "provenance.json": source.get("provenance_sha256"),
        attempt / "training" / "training_summary.json": source.get(
            "training_summary_sha256"
        ),
        checkpoint / "adapter_model.safetensors": source.get(
            "checkpoint_2_adapter_sha256"
        ),
        checkpoint / "trainer_state.json": source.get(
            "checkpoint_2_trainer_state_sha256"
        ),
    }
    for path, expected_sha256 in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"legacy source identity mismatch: {_project_path(path)}")
    outcomes = audit.get("outcomes", {})
    reconstruction = audit.get("reconstruction", {})
    if (
        outcomes.get("training_execution") != "succeeded"
        or outcomes.get("checkpoint_creation") != "succeeded"
        or outcomes.get("artifact_integrity") != "succeeded"
        or outcomes.get("vram_measurement_validation") != "failed"
        or outcomes.get("overall_continuation_gate") != "pending_correction"
        or reconstruction.get("classification")
        != "execution_succeeded_measurement_indeterminate"
        or reconstruction.get("precise_corrected_peak_reconstructable") is not False
    ):
        raise RuntimeError("corrective audit outcome is invalid")
    return attempt


def _terminal_record_for_attempt(attempt_dir: Path) -> dict[str, Any] | None:
    if not REGISTRY_PATH.is_file():
        return None
    expected = attempt_dir.resolve()
    matched: dict[str, Any] | None = None
    try:
        with REGISTRY_PATH.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if Path(str(record.get("attempt_dir", ""))).resolve() == expected:
                    matched = record
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if matched is None or matched.get("status") not in {
        "succeeded",
        "failed",
        "interrupted",
    }:
        return None
    return matched


def _valid_provenance_compatibility(
    provenance: dict[str, Any], required_compatibility_key: str
) -> bool:
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        return False
    recorded_provenance_key = provenance.get("provenance_key")
    unsigned = {
        key: value for key, value in provenance.items() if key != "provenance_key"
    }
    if recorded_provenance_key != hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest():
        return False
    context = provenance.get("context")
    if not isinstance(context, dict):
        return False
    try:
        reconstructed = _resume_compatibility_key(context)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        context.get("resume_compatibility_key") == reconstructed
        and reconstructed == required_compatibility_key
    )


def latest_complete_checkpoint(
    stage_root: Path,
    *,
    target_step: int,
    required_compatibility_key: str,
) -> Path | None:
    """Return the highest manifest-anchored compatible checkpoint below target."""
    candidates: list[tuple[int, Path]] = []
    required_names = (
        "trainer_state.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
    )
    if not stage_root.is_dir():
        return None
    for checkpoint in stage_root.glob("attempt-*/training/checkpoint-*"):
        attempt_dir = checkpoint.parents[1]
        terminal_record = _terminal_record_for_attempt(attempt_dir)
        if terminal_record is None or _validate_attempt_artifact_manifest(
            attempt_dir,
            expected_sha256=terminal_record.get("artifact_manifest_sha256"),
        ):
            continue
        try:
            provenance = load_json(attempt_dir / "provenance.json")
            if not _valid_provenance_compatibility(
                provenance, required_compatibility_key
            ):
                continue
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            state = load_json(checkpoint / "trainer_state.json")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not 0 < step < target_step or state.get("global_step") != step:
            continue
        if all(
            (checkpoint / name).is_file()
            and (checkpoint / name).stat().st_size > 0
            for name in required_names
        ):
            candidates.append((step, checkpoint))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def read_sweep_ranking() -> list[dict[str, Any]]:
    with SWEEP_CSV.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row.get("status") == "succeeded"]
    ranking = [
        {
            "checkpoint": row["checkpoint"],
            "step": int(row["global_step"]),
            "exact_matches": int(row["exact_match_count"]),
            "accuracy": float(row["exact_match_accuracy"]),
            "parse_failures": int(row["parse_failures"]),
            "seconds_per_sample": float(row["seconds_per_sample"]),
        }
        for row in rows
    ]
    return sorted(
        ranking,
        key=lambda item: (
            -item["exact_matches"],
            item["parse_failures"],
            item["seconds_per_sample"],
            item["step"],
        ),
    )


def training_command(
    *,
    stop_after_steps: int,
    save_steps: int,
    limit: int | None,
    resume_from_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "snuaichal.training",
        "--model-path",
        _project_path(MODEL_27B),
        "--model-repository",
        "Qwen/Qwen3.5-27B",
        "--model-family",
        "qwen3_5",
        "--model-revision",
        MODEL_27B_REVISION,
        "--model-manifest",
        _project_path(MODEL_27B_MANIFEST),
        "--output-dir",
        "{attempt_dir}/training",
        "--load-in-4bit",
        "--validation-fraction",
        "0.10",
        "--image-size",
        "512",
        "--epochs",
        "6",
        "--stop-after-steps",
        str(stop_after_steps),
        "--save-steps",
        str(save_steps),
        "--logging-steps",
        "1" if limit is not None else "5",
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
        "--seed",
        "42",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
    return command


def _model_manifest_configuration(model_path: Path) -> tuple[str, str, str, Path]:
    if model_path.resolve() == MODEL_8B.resolve():
        return (
            "Qwen/Qwen3-VL-8B-Instruct",
            MODEL_8B_REVISION,
            "qwen3_vl",
            MODEL_8B_MANIFEST,
        )
    if model_path.resolve() == MODEL_27B.resolve():
        return (
            "Qwen/Qwen3.5-27B",
            MODEL_27B_REVISION,
            "qwen3_5",
            MODEL_27B_MANIFEST,
        )
    raise ValueError(f"unrecognized pinned model path: {model_path}")


def _ensure_model_manifest(model_path: Path) -> dict[str, Any]:
    repository, revision, family, manifest_path = _model_manifest_configuration(model_path)
    if not manifest_path.exists():
        create_model_manifest(
            model_path,
            manifest_path,
            repository=repository,
            revision=revision,
            model_family=family,
        )
    return verify_model_manifest(
        manifest_path,
        model_root=model_path,
        expected_repository=repository,
        expected_revision=revision,
        expected_family=family,
    )


def _model_spec(model_path: Path) -> dict[str, Any]:
    repository, revision, family, manifest_path = _model_manifest_configuration(model_path)
    verified = _ensure_model_manifest(model_path)
    return {
        "repository": repository,
        "revision": revision,
        "model_family": family,
        "local_model_path": _project_path(model_path),
        "verified_model_tree_sha256": verified["tree_sha256"],
        "model_manifest": _path_identity(manifest_path),
        "model_manifest_sha256": verified["manifest_sha256"],
    }


def inference_provenance_context(
    *,
    model_path: Path,
    adapter_path: Path | None,
    validation_manifest: Path,
    tta: int,
    aggregation_mode: str,
    expected_rows: int,
    model_spec: dict[str, Any],
    validation_inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operation": "local_labeled_validation_inference",
        "model_path": _project_path(model_path),
        "model": model_spec,
        "adapter_identity": (
            _path_identity(adapter_path) if adapter_path is not None else None
        ),
        "validation_manifest_path": _project_path(validation_manifest),
        "validation_input_identity": validation_inputs,
        "seed": 42,
        "parameters": {
            "precision": "nf4",
            "image_size": 512,
            "tta": tta,
            "aggregation_mode": aggregation_mode,
            "expected_rows": expected_rows,
        },
        "expected_output_schema": {
            "predictions_csv": ["Id", "Answer"],
            "audit_jsonl_rows": expected_rows,
            "metrics_samples": expected_rows,
            "runtime_state_required": True,
        },
    }


def _resume_compatibility_key(context: dict[str, Any]) -> str:
    compatibility_base = {
        key: context[key]
        for key in (
            "operation",
            "model",
            "training_input_identity",
            "ordered_train_ids",
            "ordered_validation_ids",
            "seed",
            "parameters",
            "recipe_source_hashes",
        )
    }
    return hashlib.sha256(_canonical_json(compatibility_base).encode("utf-8")).hexdigest()


def training_provenance_context(
    *,
    expected_step: int,
    limit: int | None,
    resume_from_checkpoint: Path | None,
    training_inputs: dict[str, Any],
    model_spec: dict[str, Any],
    material_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_manifest = training_inputs.get("split_manifest", {})
    material_parameters = {
        "limit": limit,
        "row_selection_semantics": "split_full_dataset_then_limit_each_partition_v1",
        "load_in_4bit": True,
        "precision": {
            "quantization": "nf4",
            "double_quant": True,
            "compute_dtype": "bf16",
        },
        "validation_fraction": 0.10,
        "clean_validation": True,
        "balance_inputs": True,
        "image_size": 512,
        "min_pixels": 56 * 28 * 28,
        "max_pixels": 512**2,
        "epochs": 6,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "gate_up_proj",
        ],
        "train_vision_encoder": False,
        "optimizer": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "scheduler_horizon": {"type": "epochs", "epochs": 6},
        "augmentation": training_inputs.get("augmentation"),
        "split_configuration": training_inputs.get("split_configuration"),
    }
    if material_overrides:
        unknown = set(material_overrides) - set(material_parameters)
        if unknown:
            raise ValueError(f"unknown material training override(s): {sorted(unknown)}")
        material_parameters.update(material_overrides)
    compatibility_base = {
        "operation": "local_nf4_qlora_training",
        "model": model_spec,
        "training_input_identity": training_inputs,
        "ordered_train_ids": split_manifest.get("train_ids"),
        "ordered_validation_ids": split_manifest.get("validation_ids"),
        "seed": 42,
        "parameters": material_parameters,
        "recipe_source_hashes": _source_hashes(),
    }
    compatibility_key = _resume_compatibility_key(compatibility_base)
    return {
        **compatibility_base,
        "execution_control": {
            "stop_after_steps": expected_step,
            "save_steps": expected_step if expected_step <= 3 else 1073,
        },
        "expected_output_schema": {
            "global_step": expected_step,
            "complete_checkpoint": f"checkpoint-{expected_step}",
            "training_summary_schema": 3,
            "physical_memory_schema": 1,
            "physical_gate_sources": [
                "nvml_per_process_used_bytes",
                "nvml_device_memory_info_used",
            ],
            "model_manifest_required": True,
        },
        "resume_compatibility_key": compatibility_key,
        "resume_checkpoint_identity": (
            _path_identity(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        ),
    }


def prepare_full_training_resume(
    *,
    stage_root: Path,
    report_root: Path,
    target_step: int,
    save_steps: int,
    training_inputs: dict[str, Any],
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    """Prepare and durably record one exact full-training resume decision."""
    fresh_context = training_provenance_context(
        expected_step=target_step,
        limit=None,
        resume_from_checkpoint=None,
        training_inputs=training_inputs,
        model_spec=model_spec,
    )
    compatibility_key = fresh_context["resume_compatibility_key"]
    checkpoint = latest_complete_checkpoint(
        stage_root,
        target_step=target_step,
        required_compatibility_key=compatibility_key,
    )
    command = training_command(
        stop_after_steps=target_step,
        save_steps=save_steps,
        limit=None,
        resume_from_checkpoint=checkpoint,
    )
    provenance = training_provenance_context(
        expected_step=target_step,
        limit=None,
        resume_from_checkpoint=checkpoint,
        training_inputs=training_inputs,
        model_spec=model_spec,
    )
    decision = {
        "decision": "resume" if checkpoint is not None else "fresh_start",
        "resume_compatibility_key": compatibility_key,
        "target_step": target_step,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_identity": provenance["resume_checkpoint_identity"],
        "command": command,
    }
    report_path = write_versioned_report(
        report_root,
        "full-training-resume-decision",
        decision,
    )
    provenance["resume_decision"] = {
        **decision,
        "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_status(
        full_training_resume_decision=decision,
        full_training_resume_report=str(report_path),
    )
    return {
        "checkpoint": checkpoint,
        "command": command,
        "provenance": provenance,
        "decision": decision,
        "report": str(report_path),
    }


def create_pair_manifest() -> dict[str, Any]:
    source = load_json(VALIDATION_MANIFEST)
    with (ROOT / "data" / "train.csv").open(
        newline="", encoding="utf-8-sig"
    ) as file:
        rows = list(csv.DictReader(file))
    payload = build_stratified_subset_manifest(
        rows,
        validation_ids=[str(item) for item in source["validation_ids"]],
        size=96,
        seed=42,
        source_manifest_sha256=sha256_file(VALIDATION_MANIFEST),
    )
    write_immutable_json(PAIR_MANIFEST, payload)
    return payload


def build_baseline_report() -> dict[str, Any]:
    receipt = load_json(RECEIPT)
    diagnostic = load_json(DIAGNOSTIC_METRICS)
    ranking = read_sweep_ranking()
    report = {
        "schema_version": 1,
        "existing_kaggle_public_score": float(receipt["remote"]["publicScore"]),
        "receipt_path": _project_path(RECEIPT),
        "receipt_sha256": sha256_file(RECEIPT),
        "checkpoint_ranking": ranking,
        "best_checkpoint": ranking[0],
        "checkpoint_4292_tta4": summarize_tta(diagnostic),
        "coherence_threshold_preserved_for_auto_submission": 0.70,
        "preserved_inputs_manifest": _project_path(
            RESUME_ROOT / "preserved-inputs-manifest.json"
        ),
    }
    write_immutable_json(BASELINE_REPORT, report)
    return report


def run(*, poll_seconds: float) -> dict[str, Any]:
    os.chdir(ROOT)
    RESUME_ROOT.mkdir(parents=True, exist_ok=True)
    STAGES_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline_report()
    best = baseline["best_checkpoint"]
    best_adapter = ROOT / "outputs" / "qwen3-vl-8b-aug" / best["checkpoint"]
    model_8b_spec = _model_spec(MODEL_8B)
    validation_954_inputs = validation_input_identity(
        ROOT / "data" / "train.csv",
        ROOT / "data" / "train",
        VALIDATION_MANIFEST,
    )
    best_8b_validator = inference_validator(
        954,
        adapter_loaded=True,
        validation_manifest=VALIDATION_MANIFEST,
        expected_tta=4,
        expected_aggregation_mode="hard",
        expected_model_path=MODEL_8B,
        expected_model_family="qwen3_vl",
        expected_model_revision=MODEL_8B_REVISION,
        expected_adapter_path=best_adapter,
        expected_precision="nf4",
        uses_cuda=True,
    )

    tta4_command = build_inference_command(
        model_path=MODEL_8B,
        adapter_path=best_adapter,
        validation_manifest=VALIDATION_MANIFEST,
        tta=4,
        model_family="qwen3_vl",
        model_revision=MODEL_8B_REVISION,
        precision="nf4",
        aggregation_mode="hard",
    )
    tta4_context = inference_provenance_context(
        model_path=MODEL_8B,
        adapter_path=best_adapter,
        validation_manifest=VALIDATION_MANIFEST,
        tta=4,
        aggregation_mode="hard",
        expected_rows=954,
        model_spec=model_8b_spec,
        validation_inputs=validation_954_inputs,
    )
    stage_id = f"qwen3vl8b-ckpt{best['step']}-nf4-tta4-val954"

    def create_tta4_attempt() -> Path:
        return run_stage(
            stage_id,
            tta4_command,
            validator=best_8b_validator,
            poll_seconds=poll_seconds,
            provenance_context=tta4_context,
            uses_cuda=True,
        )

    historical_attempt = (
        DIAGNOSTIC_METRICS.parent
        if best["step"] == 4292
        else RESUME_ROOT / "no-matching-historical-attempt"
    )
    tta4_attempt = reuse_historical_or_create(
        historical_attempt=historical_attempt,
        validator=best_8b_validator,
        expected_context=tta4_context,
        create_new=create_tta4_attempt,
    )
    tta4_metrics = load_json(tta4_attempt / "metrics.json")
    tta4_audit = [
        json.loads(line)
        for line in (tta4_attempt / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tta4_report = summarize_tta(tta4_metrics)
    write_versioned_report(RESUME_ROOT, "best-8b-tta4", tta4_report)

    hf = ROOT / ".venv" / "Scripts" / "hf.exe"
    download_attempt = run_stage(
        "qwen35-27b-download",
        [
            str(hf),
            "download",
            "Qwen/Qwen3.5-27B",
            "--revision",
            MODEL_27B_REVISION,
            "--local-dir",
            str(MODEL_27B),
        ],
        validator=download_validator,
        poll_seconds=poll_seconds,
        provenance_context={
            "operation": "pinned_huggingface_model_download",
            "executable_identity": _path_identity(hf),
            "model_repository": "Qwen/Qwen3.5-27B",
            "model_revision": MODEL_27B_REVISION,
            "destination": _project_path(MODEL_27B),
            "expected_output_schema": {
                "config_json": True,
                "safetensors_index": True,
                "minimum_weight_bytes": 50_000_000_000,
            },
        },
        external_outputs=(MODEL_27B,),
        uses_cuda=False,
        child_environment=build_acquisition_child_environment(),
        post_success=lambda: write_revision_marker(
            MODEL_27B,
            repository="Qwen/Qwen3.5-27B",
            revision=MODEL_27B_REVISION,
        ),
    )
    model_27b_spec = _model_spec(MODEL_27B)
    smoke_training_inputs = training_input_identity(
        ROOT / "data" / "train.csv",
        ROOT / "data" / "train",
        limit=48,
        seed=42,
        validation_fraction=0.10,
    )
    smoke_attempt = validated_legacy_smoke_attempt()
    smoke_output = smoke_attempt / "training"
    smoke_checkpoint = smoke_output / "checkpoint-2"
    smoke_adapter = smoke_checkpoint
    reload_attempt = run_stage(
        "qwen35-27b-checkpoint2-adapter-reload-inference-1row-corrected",
        build_inference_command(
            model_path=MODEL_27B,
            adapter_path=smoke_adapter,
            validation_manifest=VALIDATION_MANIFEST,
            tta=1,
            model_family="qwen3_5",
            model_revision=MODEL_27B_REVISION,
            precision="nf4",
            aggregation_mode="hard",
            limit=1,
        ),
        validator=inference_validator(
            1,
            adapter_loaded=True,
            validation_manifest=VALIDATION_MANIFEST,
            expected_tta=1,
            expected_aggregation_mode="hard",
            expected_model_path=MODEL_27B,
            expected_model_family="qwen3_5",
            expected_model_revision=MODEL_27B_REVISION,
            expected_adapter_path=smoke_adapter,
            expected_precision="nf4",
            uses_cuda=True,
            require_valid_parse=True,
        ),
        poll_seconds=poll_seconds,
        provenance_context=inference_provenance_context(
            model_path=MODEL_27B,
            adapter_path=smoke_adapter,
            validation_manifest=VALIDATION_MANIFEST,
            tta=1,
            aggregation_mode="hard",
            expected_rows=1,
            model_spec=model_27b_spec,
            validation_inputs=validation_954_inputs,
        ),
        uses_cuda=True,
    )
    resume_attempt = run_stage(
        "qwen35-27b-checkpoint2-continuation-step3-corrected",
        training_command(
            stop_after_steps=3,
            save_steps=3,
            limit=48,
            resume_from_checkpoint=smoke_checkpoint,
        ),
        validator=training_validator(
            3,
            resume_from_checkpoint=smoke_checkpoint,
        ),
        poll_seconds=poll_seconds,
        provenance_context=training_provenance_context(
            expected_step=3,
            limit=48,
            resume_from_checkpoint=smoke_checkpoint,
            training_inputs=smoke_training_inputs,
            model_spec=model_27b_spec,
        ),
        uses_cuda=True,
    )

    smoke_manifest = load_json(smoke_output / "model_manifest.json")
    resume_summary = load_json(resume_attempt / "training" / "training_summary.json")
    seconds_per_step = float(resume_summary["seconds_per_optimizer_step"])
    projected_seconds = seconds_per_step * FULL_OPTIMIZER_STEPS
    package_bytes = directory_size(MODEL_27B) + directory_size(smoke_output / "final")
    physical_valid = resume_summary.get("physical_measurement_status") == "valid"
    physical_peak = resume_summary.get("physical_peak_observed_bytes")
    smoke_report = {
        "download_attempt": str(download_attempt),
        "training_attempt": str(smoke_attempt),
        "legacy_training_execution": "succeeded",
        "legacy_physical_measurement": "indeterminate",
        "loss": resume_summary["training_loss"],
        "physical_peak_observed_bytes": physical_peak,
        "physical_measurement_source": resume_summary.get(
            "physical_measurement_source"
        ),
        "seconds_per_optimizer_step": seconds_per_step,
        "trainable_parameters": smoke_manifest["trainable_parameters"],
        "adapter_reload_success": True,
        "adapter_reload_attempt": str(reload_attempt),
        "resumable_checkpoint_success": True,
        "resume_probe_attempt": str(resume_attempt),
        "projected_full_optimizer_steps": FULL_OPTIMIZER_STEPS,
        "projected_full_training_seconds": projected_seconds,
        "projected_full_training_hours": projected_seconds / 3600,
        "projected_package_bytes": package_bytes,
        "vram_safe": (
            physical_valid
            and isinstance(physical_peak, int)
            and physical_peak <= MAX_SAFE_VRAM_BYTES
        ),
        "runtime_acceptable": projected_seconds <= MAX_PROJECTED_HOURS * 3600,
        "package_under_80gb": package_bytes <= MAX_PACKAGE_BYTES,
    }
    write_versioned_report(RESUME_ROOT, "qwen35-smoke", smoke_report)

    create_pair_manifest()
    paired_validation_inputs = validation_input_identity(
        ROOT / "data" / "train.csv",
        ROOT / "data" / "train",
        PAIR_MANIFEST,
    )
    paired_8b_attempt = run_stage(
        f"paired96-qwen3vl8b-ckpt{best['step']}-nf4-tta1",
        build_inference_command(
            model_path=MODEL_8B,
            adapter_path=best_adapter,
            validation_manifest=PAIR_MANIFEST,
            tta=1,
            model_family="qwen3_vl",
            model_revision=MODEL_8B_REVISION,
            precision="nf4",
            aggregation_mode="hard",
        ),
        validator=inference_validator(
            96,
            adapter_loaded=True,
            validation_manifest=PAIR_MANIFEST,
            expected_tta=1,
            expected_aggregation_mode="hard",
            expected_model_path=MODEL_8B,
            expected_model_family="qwen3_vl",
            expected_model_revision=MODEL_8B_REVISION,
            expected_adapter_path=best_adapter,
            expected_precision="nf4",
            uses_cuda=True,
        ),
        poll_seconds=poll_seconds,
        provenance_context=inference_provenance_context(
            model_path=MODEL_8B,
            adapter_path=best_adapter,
            validation_manifest=PAIR_MANIFEST,
            tta=1,
            aggregation_mode="hard",
            expected_rows=96,
            model_spec=model_8b_spec,
            validation_inputs=paired_validation_inputs,
        ),
        uses_cuda=True,
    )
    paired_27b_attempt = run_stage(
        "paired96-qwen35-27b-base-nf4-tta1",
        build_inference_command(
            model_path=MODEL_27B,
            adapter_path=None,
            validation_manifest=PAIR_MANIFEST,
            tta=1,
            model_family="qwen3_5",
            model_revision=MODEL_27B_REVISION,
            precision="nf4",
            aggregation_mode="hard",
        ),
        validator=inference_validator(
            96,
            adapter_loaded=False,
            validation_manifest=PAIR_MANIFEST,
            expected_tta=1,
            expected_aggregation_mode="hard",
            expected_model_path=MODEL_27B,
            expected_model_family="qwen3_5",
            expected_model_revision=MODEL_27B_REVISION,
            expected_adapter_path=None,
            expected_precision="nf4",
            uses_cuda=True,
        ),
        poll_seconds=poll_seconds,
        provenance_context=inference_provenance_context(
            model_path=MODEL_27B,
            adapter_path=None,
            validation_manifest=PAIR_MANIFEST,
            tta=1,
            aggregation_mode="hard",
            expected_rows=96,
            model_spec=model_27b_spec,
            validation_inputs=paired_validation_inputs,
        ),
        uses_cuda=True,
    )
    paired_8b = load_json(paired_8b_attempt / "metrics.json")
    paired_27b = load_json(paired_27b_attempt / "metrics.json")
    audit_8b = [
        json.loads(line)
        for line in (paired_8b_attempt / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    audit_27b = [
        json.loads(line)
        for line in (paired_27b_attempt / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    paired = paired_decision(paired_8b, paired_27b)
    paired.update(
        {
            "manifest": _project_path(PAIR_MANIFEST),
            "paired_outcomes": compare_paired_audit_rows(audit_8b, audit_27b),
            "qwen3vl8b": {
                "exact_match": paired_8b["exact_match"],
                "exact_matches": paired_8b["exact_matches"],
                "non_identity_exact_match": paired_8b["non_identity_exact_match"],
                "parse_failures": paired_8b["parse_failures"],
                "inference_seconds_per_sample": paired_8b[
                    "inference_seconds_per_sample"
                ],
                "peak_vram_mib": paired_8b["peak_vram_mib"],
            },
            "qwen35_27b": {
                "exact_match": paired_27b["exact_match"],
                "exact_matches": paired_27b["exact_matches"],
                "non_identity_exact_match": paired_27b[
                    "non_identity_exact_match"
                ],
                "parse_failures": paired_27b["parse_failures"],
                "inference_seconds_per_sample": paired_27b[
                    "inference_seconds_per_sample"
                ],
                "peak_vram_mib": paired_27b["peak_vram_mib"],
            },
            "qwen3vl8b_metrics": str(paired_8b_attempt / "metrics.json"),
            "qwen35_27b_metrics": str(paired_27b_attempt / "metrics.json"),
        }
    )
    write_versioned_report(RESUME_ROOT, "paired-96", paired)

    full_27b: dict[str, Any] | None = None
    full_training_attempt: str | None = None
    if paired["continue_to_full_validation"]:
        full_attempt = run_stage(
            "qwen35-27b-base-nf4-tta1-val954",
            build_inference_command(
                model_path=MODEL_27B,
                adapter_path=None,
                validation_manifest=VALIDATION_MANIFEST,
                tta=1,
                model_family="qwen3_5",
                model_revision=MODEL_27B_REVISION,
                precision="nf4",
                aggregation_mode="hard",
            ),
            validator=inference_validator(
                954,
                adapter_loaded=False,
                validation_manifest=VALIDATION_MANIFEST,
                expected_tta=1,
                expected_aggregation_mode="hard",
                expected_model_path=MODEL_27B,
                expected_model_family="qwen3_5",
                expected_model_revision=MODEL_27B_REVISION,
                expected_adapter_path=None,
                expected_precision="nf4",
                uses_cuda=True,
            ),
            poll_seconds=poll_seconds,
            provenance_context=inference_provenance_context(
                model_path=MODEL_27B,
                adapter_path=None,
                validation_manifest=VALIDATION_MANIFEST,
                tta=1,
                aggregation_mode="hard",
                expected_rows=954,
                model_spec=model_27b_spec,
                validation_inputs=validation_954_inputs,
            ),
            uses_cuda=True,
        )
        full_27b = load_json(full_attempt / "metrics.json")

    best_8b_accuracy = float(tta4_report["selected"]["accuracy"])
    superior = bool(full_27b and float(full_27b["exact_match"]) > best_8b_accuracy)
    training_conditions = {
        "full_validation_superior": superior,
        "vram_safe": smoke_report["vram_safe"],
        "resumable_checkpoints_work": smoke_report["resumable_checkpoint_success"],
        "runtime_acceptable": smoke_report["runtime_acceptable"],
        "package_under_80gb": smoke_report["package_under_80gb"],
    }
    start_full_training = all(training_conditions.values())
    full_training_command: list[str] | None = None
    full_training_resume_report: str | None = None
    trained_27b: dict[str, Any] | None = None
    trained_27b_attempt: Path | None = None
    if start_full_training:
        full_training_inputs = training_input_identity(
            ROOT / "data" / "train.csv",
            ROOT / "data" / "train",
            limit=None,
            seed=42,
            validation_fraction=0.10,
        )
        prepared_resume = prepare_full_training_resume(
            stage_root=STAGES_ROOT / "qwen35-27b-full-qlora",
            report_root=RESUME_ROOT,
            target_step=FULL_OPTIMIZER_STEPS,
            save_steps=1073,
            training_inputs=full_training_inputs,
            model_spec=model_27b_spec,
        )
        resume_checkpoint = prepared_resume["checkpoint"]
        full_training_command = prepared_resume["command"]
        full_training_resume_report = prepared_resume["report"]
        attempt = run_stage(
            "qwen35-27b-full-qlora",
            full_training_command,
            validator=training_validator(
                FULL_OPTIMIZER_STEPS,
                resume_from_checkpoint=resume_checkpoint,
            ),
            poll_seconds=poll_seconds,
            provenance_context=prepared_resume["provenance"],
            uses_cuda=True,
        )
        full_training_attempt = str(attempt)
        trained_adapter = attempt / "training" / "final"
        trained_27b_attempt = run_stage(
            "qwen35-27b-full-qlora-nf4-tta1-val954",
            build_inference_command(
                model_path=MODEL_27B,
                adapter_path=trained_adapter,
                validation_manifest=VALIDATION_MANIFEST,
                tta=1,
                model_family="qwen3_5",
                model_revision=MODEL_27B_REVISION,
                precision="nf4",
                aggregation_mode="hard",
            ),
            validator=inference_validator(
                954,
                adapter_loaded=True,
                validation_manifest=VALIDATION_MANIFEST,
                expected_tta=1,
                expected_aggregation_mode="hard",
                expected_model_path=MODEL_27B,
                expected_model_family="qwen3_5",
                expected_model_revision=MODEL_27B_REVISION,
                expected_adapter_path=trained_adapter,
                expected_precision="nf4",
                uses_cuda=True,
            ),
            poll_seconds=poll_seconds,
            provenance_context=inference_provenance_context(
                model_path=MODEL_27B,
                adapter_path=trained_adapter,
                validation_manifest=VALIDATION_MANIFEST,
                tta=1,
                aggregation_mode="hard",
                expected_rows=954,
                model_spec=model_27b_spec,
                validation_inputs=validation_954_inputs,
            ),
            uses_cuda=True,
        )
        trained_27b = load_json(trained_27b_attempt / "metrics.json")

    candidates = [
        build_8b_candidate(
            best=best,
            best_adapter=best_adapter,
            model_spec=model_8b_spec,
            tta4_metrics=tta4_metrics,
            tta4_audit=tta4_audit,
            selected_mode=str(tta4_report["selected"]["mode"]),
        )
    ]
    if full_27b is not None:
        candidates.append(
            {
                "name": "qwen35-27b-base-tta1-hard",
                "model_path": MODEL_27B,
                "model_family": "qwen3_5",
                "model_revision": MODEL_27B_REVISION,
                "model_repository": model_27b_spec["repository"],
                "model_manifest": MODEL_27B_MANIFEST,
                "verified_model_tree_sha256": model_27b_spec[
                    "verified_model_tree_sha256"
                ],
                "adapter_path": None,
                "precision": "nf4",
                "image_size": 512,
                "tta_orders": [[1, 2, 3, 4]],
                "aggregation_mode": "hard",
                "fallback_policy": "identity",
                "seed": 42,
                "validation": full_27b,
            }
        )
    if trained_27b is not None and trained_27b_attempt is not None:
        candidates.append(
            {
                "name": "qwen35-27b-full-qlora-tta1-hard",
                "model_path": MODEL_27B,
                "model_family": "qwen3_5",
                "model_revision": MODEL_27B_REVISION,
                "model_repository": model_27b_spec["repository"],
                "model_manifest": MODEL_27B_MANIFEST,
                "verified_model_tree_sha256": model_27b_spec[
                    "verified_model_tree_sha256"
                ],
                "adapter_path": Path(full_training_attempt) / "training" / "final",
                "precision": "nf4",
                "image_size": 512,
                "tta_orders": [[1, 2, 3, 4]],
                "aggregation_mode": "hard",
                "fallback_policy": "identity",
                "seed": 42,
                "validation": trained_27b,
            }
        )
    selected_model = select_final_model(candidates)
    selected_model_report = {
        **selected_model,
        "model_path": _project_path(Path(selected_model["model_path"])),
        "adapter_path": (
            _project_path(Path(selected_model["adapter_path"]))
            if selected_model["adapter_path"] is not None
            else None
        ),
    }
    selection_report_path = write_versioned_report(
        RESUME_ROOT, "final-model-selection", selected_model_report
    )
    final_test_inputs = test_input_identity(
        ROOT / "data" / "test.csv", ROOT / "data" / "test"
    )
    final_test_stage = build_final_test_inference_stage(
        selection=selected_model,
        test_csv=ROOT / "data" / "test.csv",
        image_dir=ROOT / "data" / "test",
        test_input_identity=final_test_inputs,
    )
    final_test_attempt = run_stage(
        final_test_stage["stage_id"],
        final_test_stage["command"],
        validator=final_test_stage["validator"],
        poll_seconds=poll_seconds,
        provenance_context=final_test_stage["provenance_context"],
        uses_cuda=final_test_stage["uses_cuda"],
    )
    recommended_action = "publish the one validated final submission after receipt checks"
    final = {
        "existing_kaggle_public_score": baseline["existing_kaggle_public_score"],
        "checkpoint_ranking": baseline["checkpoint_ranking"],
        "checkpoint_4292_tta4": baseline["checkpoint_4292_tta4"],
        "best_8b_tta4": tta4_report,
        "best_8b_checkpoint": best,
        "qwen35_smoke": smoke_report,
        "paired_96": paired,
        "qwen35_full_validation": full_27b,
        "qwen35_trained_validation": trained_27b,
        "full_training_conditions": training_conditions,
        "full_training_started": start_full_training,
        "full_training_attempt": full_training_attempt,
        "full_training_resume_report": full_training_resume_report,
        "selected_model": selected_model_report,
        "selection_report": str(selection_report_path),
        "selection_report_sha256": sha256_file(selection_report_path),
        "final_test_inference_attempt": str(final_test_attempt),
        "final_submission_csv": str(final_test_attempt / "submission.csv"),
        "final_submission_sha256": sha256_file(final_test_attempt / "submission.csv"),
        "recommended_next_action": recommended_action,
        "exact_command_for_full_training": full_training_command,
        "auto_submission_threshold": 0.70,
        "submission_performed": False,
    }
    final_report_path = write_versioned_report(RESUME_ROOT, "terminal", final)
    atomic_status(
        state="complete",
        active_experiment=None,
        terminal_report=str(final_report_path),
        terminal_report_sha256=sha256_file(final_report_path),
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    lock_record = acquire_runner_lock()
    terminal_state = "failed"
    atomic_status(
        state="running",
        runner_pid=lock_record["pid"],
        runner_process_identity=lock_record.get("process_identity"),
        runner_started_at=lock_record["started_at"],
        runner_heartbeat_at=durable.utc_now(),
    )
    try:
        result = run(poll_seconds=args.poll_seconds)
        terminal_state = "succeeded"
        atomic_status(
            state="succeeded",
            active_experiment=None,
            runner_heartbeat_at=durable.utc_now(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        terminal_state = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        trace = "".join(traceback.format_exception(exc))
        trace_path = write_versioned_report(
            RESUME_ROOT,
            "runner-traceback",
            {
                "schema_version": 1,
                "state": terminal_state,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": trace,
                "timestamp": durable.utc_now(),
            },
        )
        atomic_status(
            state=terminal_state,
            active_experiment=None,
            error=f"{type(exc).__name__}: {exc}",
            traceback_path=str(trace_path),
            traceback_sha256=sha256_file(trace_path),
            runner_heartbeat_at=durable.utc_now(),
        )
        raise
    finally:
        release_runner_lock(
            RUNNER_LOCK_PATH,
            lock_record,
            state=terminal_state,
        )


if __name__ == "__main__":
    main()
