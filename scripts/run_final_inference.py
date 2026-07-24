from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.download_final_adapter import (
    load_manifest as load_adapter_manifest,
)
from scripts.download_final_adapter import verify_adapter_directory
from scripts.validate_submission_artifacts import validate
from scripts.verify_evaluation_package import (
    sha256_file,
    verify_static_contract,
    verify_unlabeled_dataset,
)


MODEL_REPOSITORY = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
LATIN4 = [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]


def build_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir
    command = [
        sys.executable,
        "-B",
        "-m",
        "snuaichal.inference",
        "--test-csv",
        str(args.data_dir / "test.csv"),
        "--image-dir",
        str(args.data_dir / "test"),
        "--model-path",
        str(args.model_path),
        "--model-repository",
        MODEL_REPOSITORY,
        "--model-family",
        "qwen3_5",
        "--model-revision",
        MODEL_REVISION,
        "--model-manifest",
        str(args.model_manifest),
        "--adapter-path",
        str(args.adapter_path),
        "--precision",
        "nf4",
        "--dtype",
        "bfloat16",
        "--device-map",
        "auto",
        "--attn-implementation",
        "sdpa",
        "--image-size",
        "512",
        "--max-new-tokens",
        "64",
        "--tta",
        "4",
        "--tta-orders-json",
        json.dumps(LATIN4, separators=(",", ":")),
        "--aggregation-mode",
        "hard",
        "--fallback-policy",
        "identity",
        "--seed",
        "42",
        "--output",
        str(output_dir / "submission.csv"),
        "--audit-log",
        str(output_dir / "audit.jsonl"),
        "--metrics-output",
        str(output_dir / "metrics.json"),
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the final checkpoint-2726 Latin4 hard inference"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/Qwen3.6-27B")
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("configs/weights/qwen36-27b-final.manifest.json"),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("weights/qwen36-checkpoint2726"),
    )
    parser.add_argument(
        "--adapter-manifest",
        type=Path,
        default=Path("configs/weights/qwen36-checkpoint2726-adapter.manifest.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/final-inference")
    )
    return parser


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            "final output directory must be new or empty; stale rows are forbidden"
        )
    path.mkdir(parents=True, exist_ok=True)
    lease = path / ".run-lease"
    descriptor = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) != 1 or devices[0] in {"-1", "none", "None"}:
            raise RuntimeError(
                "final inference requires exactly one scheduler-visible CUDA device"
            )
    return environment


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def main() -> None:
    args = build_parser().parse_args()
    static_contract = verify_static_contract(Path("."))
    dataset_contract = verify_unlabeled_dataset(args.data_dir)
    manifest = load_adapter_manifest(args.adapter_manifest)
    verification = verify_adapter_directory(args.adapter_path, manifest)
    if (
        verification.get("checkpoint_step") != 2726
        or verification.get("adapter_sha256")
        != static_contract["adapter_sha256"]
        or manifest.get("manifest_sha256")
        != static_contract["adapter_manifest_sha256"]
    ):
        raise RuntimeError("adapter does not match the frozen checkpoint-2726 contract")
    prepare_output_directory(args.output_dir)
    environment = child_environment()
    command = build_command(args)
    subprocess.run(command, check=True, env=environment)
    artifact_validation = validate(
        argparse.Namespace(
            test_csv=args.data_dir / "test.csv",
            submission=args.output_dir / "submission.csv",
            audit=args.output_dir / "audit.jsonl",
            expected_tta=4,
            metrics=args.output_dir / "metrics.json",
            aggregation_mode="hard",
        )
    )
    post_dataset_contract = verify_unlabeled_dataset(args.data_dir)
    if post_dataset_contract != dataset_contract:
        raise RuntimeError("evaluation dataset changed during inference")
    release_sources = {
        relative: sha256_file(Path(relative))
        for relative in (
            "configs/final_inference.json",
            "configs/weights/qwen36-27b-final.manifest.json",
            "configs/weights/qwen36-checkpoint2726-adapter.manifest.json",
            "scripts/run_final_inference.py",
            "scripts/validate_submission_artifacts.py",
            "scripts/verify_evaluation_package.py",
            "src/snuaichal/inference.py",
            "src/snuaichal/modeling.py",
            "src/snuaichal/physical_memory.py",
            "src/snuaichal/submission.py",
            "src/snuaichal/tta.py",
        )
    }
    execution_contract = {
        "command": command,
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "release_source_sha256": release_sources,
    }
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "execution_contract": execution_contract,
        "execution_contract_sha256": canonical_sha256(execution_contract),
        "static_contract": static_contract,
        "unlabeled_dataset": dataset_contract,
        "adapter": verification,
        "artifacts": artifact_validation,
    }
    atomic_write_json(args.output_dir / "reproduction-receipt.json", receipt)


if __name__ == "__main__":
    main()
