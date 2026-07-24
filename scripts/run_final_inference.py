from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.download_final_adapter import (
    load_manifest as load_adapter_manifest,
)
from scripts.download_final_adapter import verify_adapter_directory


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
    if args.resume:
        command.append("--resume")
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
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_adapter_manifest(args.adapter_manifest)
    verification = verify_adapter_directory(args.adapter_path, manifest)
    if verification.get("checkpoint_step") != 2726:
        raise RuntimeError("final adapter is not checkpoint-2726")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_command(args), check=True)


if __name__ == "__main__":
    main()
