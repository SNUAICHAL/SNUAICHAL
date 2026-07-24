from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.download_final_adapter import (
    canonical_json,
    load_manifest as load_adapter_manifest,
    verify_adapter_directory,
)
from scripts.download_weights import verify_snapshot
from snuaichal.model_manifest import verify_model_manifest


FINAL_CONFIG = Path("configs/final_inference.json")
FINAL_RESULTS = Path("docs/final_results.json")
MODEL_MANIFEST = Path("configs/weights/qwen36-27b-final.manifest.json")
ADAPTER_MANIFEST = Path(
    "configs/weights/qwen36-checkpoint2726-adapter.manifest.json"
)
EXPECTED_COLUMNS = [
    "Id",
    "Input_1",
    "Input_2",
    "Input_3",
    "Input_4",
    "Sentence",
]
EXPECTED_ORDERS = [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def verify_self_hash(payload: dict[str, Any], *, label: str) -> None:
    recorded = payload.get("manifest_sha256")
    if not isinstance(recorded, str):
        raise RuntimeError(f"{label} has no manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if recorded != actual:
        raise RuntimeError(f"{label} self SHA-256 mismatch")


def _safe_component(value: object, *, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or ":" in text
        or "\x00" in text
    ):
        raise RuntimeError(f"unsafe {label}: {text!r}")
    return text


def verify_static_contract(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    config = load_json(root / FINAL_CONFIG)
    results = load_json(root / FINAL_RESULTS)
    model_manifest = load_json(root / MODEL_MANIFEST)
    adapter_manifest = load_adapter_manifest(root / ADAPTER_MANIFEST)
    verify_self_hash(model_manifest, label="model manifest")

    model = config.get("model")
    adapter = config.get("adapter")
    runtime = config.get("runtime")
    tta = config.get("tta")
    decision = results.get("final_decision")
    if not all(
        isinstance(value, dict)
        for value in (model, adapter, runtime, tta, decision)
    ):
        raise RuntimeError("final config/results contract is incomplete")

    expected = {
        "repository": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "family": "qwen3_5",
        "tree_sha256": "e4107e6508793261ca372faf4b560dcb55a5b6ba79a5ab921bfe1b25a207ec07",
    }
    if model != expected:
        raise RuntimeError("final model contract drift")
    if model_manifest.get("repository") != expected["repository"]:
        raise RuntimeError("model manifest repository drift")
    if model_manifest.get("revision") != expected["revision"]:
        raise RuntimeError("model manifest revision drift")
    if model_manifest.get("model_family") != expected["family"]:
        raise RuntimeError("model manifest family drift")
    if model_manifest.get("tree_sha256") != expected["tree_sha256"]:
        raise RuntimeError("model manifest tree drift")
    if model_manifest.get("license") != "Apache-2.0":
        raise RuntimeError("model manifest license drift")
    if model_manifest.get("public_date") != "2026-04-24":
        raise RuntimeError("model manifest public-date drift")

    files = model_manifest.get("files")
    if not isinstance(files, list) or len(files) != 29:
        raise RuntimeError("final model manifest must contain exactly 29 files")
    shards = [
        record
        for record in files
        if isinstance(record, dict)
        and str(record.get("path", "")).startswith("model-")
        and str(record.get("path", "")).endswith(".safetensors")
    ]
    if len(shards) != 15:
        raise RuntimeError("final model manifest must contain exactly 15 weight shards")

    if adapter.get("checkpoint_step") != 2726:
        raise RuntimeError("final adapter checkpoint drift")
    if adapter.get("rank") != 32 or adapter.get("alpha") != 32:
        raise RuntimeError("final adapter rank/alpha drift")
    if adapter_manifest.get("checkpoint_step") != 2726:
        raise RuntimeError("adapter manifest checkpoint drift")
    if adapter_manifest.get("adapter_rank") != 32:
        raise RuntimeError("adapter manifest rank drift")
    if adapter_manifest.get("adapter_alpha") != 32:
        raise RuntimeError("adapter manifest alpha drift")
    if adapter_manifest.get("base_revision") != expected["revision"]:
        raise RuntimeError("adapter manifest base revision drift")
    adapter_weight = next(
        (
            record
            for record in adapter_manifest.get("files", [])
            if record.get("path") == "adapter_model.safetensors"
        ),
        None,
    )
    if not isinstance(adapter_weight, dict):
        raise RuntimeError("adapter manifest has no adapter_model.safetensors")
    if adapter_weight.get("sha256") != adapter.get("sha256"):
        raise RuntimeError("adapter SHA-256 contract drift")

    if runtime != {
        "precision": "nf4",
        "compute_dtype": "bfloat16",
        "attention": "sdpa",
        "image_size": 512,
        "max_new_tokens": 64,
        "batch_size": 1,
        "seed": 42,
        "enable_thinking": False,
    }:
        raise RuntimeError("final runtime contract drift")
    if tta != {
        "orders": EXPECTED_ORDERS,
        "canonicalize_to_original_slots": True,
        "aggregation": "hard_majority",
        "tie_break": "lexicographic",
    }:
        raise RuntimeError("final TTA contract drift")

    if decision.get("model") != expected["repository"]:
        raise RuntimeError("final result model drift")
    if decision.get("revision") != expected["revision"]:
        raise RuntimeError("final result revision drift")
    if decision.get("checkpoint_step") != 2726:
        raise RuntimeError("final result checkpoint drift")
    if decision.get("adapter_sha256") != adapter.get("sha256"):
        raise RuntimeError("final result adapter drift")
    if decision.get("model_tree_sha256") != expected["tree_sha256"]:
        raise RuntimeError("final result model tree drift")
    if decision.get("tta_orders") != ["1234", "2341", "3412", "4123"]:
        raise RuntimeError("final result TTA drift")
    if decision.get("aggregation") != "hard_majority":
        raise RuntimeError("final result aggregation drift")

    source_hashes = results.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("final results have no source hashes")
    for relative, expected_sha in source_hashes.items():
        source = root / str(relative)
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise RuntimeError(f"scored source SHA-256 mismatch: {relative}")

    release_hashes = results.get("release_source_sha256")
    if not isinstance(release_hashes, dict) or not release_hashes:
        raise RuntimeError("final results have no release source hashes")
    for relative, expected_sha in release_hashes.items():
        source = root / str(relative)
        if not source.is_file() or sha256_file(source) != expected_sha:
            raise RuntimeError(f"release source SHA-256 mismatch: {relative}")

    return {
        "adapter_archive_sha256": adapter_manifest["archive_sha256"],
        "adapter_manifest_sha256": adapter_manifest["manifest_sha256"],
        "adapter_sha256": adapter["sha256"],
        "checkpoint_step": 2726,
        "model_files": len(files),
        "model_revision": expected["revision"],
        "model_shards": len(shards),
        "model_tree_sha256": expected["tree_sha256"],
        "public_score": decision.get("public_score"),
        "release_source_files": len(release_hashes),
        "scored_source_files": len(source_hashes),
        "status": "PASS",
        "tta_orders": EXPECTED_ORDERS,
    }


def verify_unlabeled_dataset(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve(strict=True)
    csv_path = data_dir / "test.csv"
    image_root = (data_dir / "test").resolve(strict=True)
    if not csv_path.is_file():
        raise RuntimeError(f"test.csv is missing: {csv_path}")
    if not image_root.is_dir():
        raise RuntimeError(f"test image root is missing: {image_root}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"test.csv columns must be exactly {EXPECTED_COLUMNS}; "
                f"got {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError("test.csv has no rows")

    seen_ids: set[str] = set()
    referenced: set[str] = set()
    referenced_bytes = 0
    image_tree_digest = hashlib.sha256()
    for index, row in enumerate(rows):
        sample_id = _safe_component(row["Id"], label=f"Id at row {index}")
        if sample_id in seen_ids:
            raise RuntimeError(f"duplicate test Id: {sample_id}")
        seen_ids.add(sample_id)
        if not str(row["Sentence"]).strip():
            raise RuntimeError(f"empty Sentence at row {index}")
        row_paths: set[str] = set()
        for slot in range(1, 5):
            filename = _safe_component(
                row[f"Input_{slot}"], label=f"Input_{slot} at row {index}"
            )
            candidate = image_root / sample_id / filename
            if candidate.is_symlink():
                raise RuntimeError(f"image symlink is forbidden: {candidate}")
            resolved = candidate.resolve(strict=True)
            if image_root not in resolved.parents or not resolved.is_file():
                raise RuntimeError(f"image escapes test root: {candidate}")
            relative = resolved.relative_to(image_root).as_posix()
            if relative in row_paths:
                raise RuntimeError(f"duplicate image within row {sample_id}: {relative}")
            row_paths.add(relative)
            if relative not in referenced:
                referenced.add(relative)
                referenced_bytes += resolved.stat().st_size
                image_tree_digest.update(relative.encode("utf-8"))
                image_tree_digest.update(b"\0")
                image_tree_digest.update(sha256_file(resolved).encode("ascii"))
                image_tree_digest.update(b"\n")

    return {
        "answer_column_present": False,
        "csv_sha256": sha256_file(csv_path),
        "referenced_image_tree_sha256": image_tree_digest.hexdigest(),
        "referenced_image_bytes": referenced_bytes,
        "referenced_images": len(referenced),
        "rows": len(rows),
        "status": "PASS",
        "unique_ids": len(seen_ids),
    }


def verify_package(
    *,
    root: Path,
    data_dir: Path | None,
    model_path: Path | None,
    adapter_path: Path | None,
    require_all: bool,
) -> dict[str, Any]:
    if require_all and any(
        value is None for value in (data_dir, model_path, adapter_path)
    ):
        raise RuntimeError(
            "--require-all needs --data-dir, --model-path, and --adapter-path"
        )
    root = root.resolve(strict=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "static_contract": verify_static_contract(root),
    }
    if data_dir is not None:
        result["unlabeled_dataset"] = verify_unlabeled_dataset(data_dir)
    if model_path is not None:
        model_manifest = load_json(root / MODEL_MANIFEST)
        snapshot = verify_snapshot(model_path, model_manifest)
        strict_manifest = verify_model_manifest(
            root / MODEL_MANIFEST,
            model_root=model_path,
            expected_repository="Qwen/Qwen3.6-27B",
            expected_revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
            expected_family="qwen3_5",
        )
        result["model_snapshot"] = {
            **snapshot,
            "indexed_weight_shards": strict_manifest["indexed_weight_shards"],
            "tree_sha256": strict_manifest["tree_sha256"],
        }
    if adapter_path is not None:
        adapter_manifest = load_adapter_manifest(root / ADAPTER_MANIFEST)
        result["adapter"] = verify_adapter_directory(adapter_path, adapter_manifest)
    result["complete"] = all(
        key in result for key in ("unlabeled_dataset", "model_snapshot", "adapter")
    )
    result["status"] = "PASS" if result["complete"] else "PARTIAL_PASS"
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only verification of the frozen final model, adapter, source, "
            "and unlabeled evaluation-data contract"
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = verify_package(
        root=args.root,
        data_dir=args.data_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        require_all=args.require_all,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
