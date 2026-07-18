"""Fail-closed, content-addressed identities for local Hugging Face model trees."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

MODEL_MANIFEST_SCHEMA_VERSION = 1
_IGNORED_ROOT_ENTRIES = {".cache", "SNAPSHOT_REVISION"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_root(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _safe_relative(value: object) -> str:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or pure.is_absolute()
        or text != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe model-manifest path: {text!r}")
    return text


def _source_tree(model_root: Path, revision: str) -> dict[str, Any]:
    path = model_root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing/invalid pinned Hugging Face tree metadata: {path}") from exc
    if payload.get("format_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("unsupported pinned Hugging Face tree metadata schema")
    return payload


def _verify_snapshot_marker(model_root: Path, repository: str, revision: str) -> None:
    marker = model_root / "SNAPSHOT_REVISION"
    if not marker.exists():
        raise ValueError("SNAPSHOT_REVISION repository/revision marker is required")
    values: dict[str, str] = {}
    for line in marker.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if values.get("repo_id") != repository or values.get("revision") != revision:
        raise ValueError("SNAPSHOT_REVISION repository/revision mismatch")


def _actual_model_files(model_root: Path) -> set[str]:
    files: set[str] = set()
    for path in model_root.rglob("*"):
        relative = path.relative_to(model_root)
        if relative.parts and relative.parts[0] in _IGNORED_ROOT_ENTRIES:
            continue
        if path.is_file():
            files.add(relative.as_posix())
    return files


def _inventory_from_source_tree(
    model_root: Path, source_tree: dict[str, Any]
) -> list[dict[str, Any]]:
    source_files = source_tree["files"]
    normalized = [_safe_relative(path) for path in source_files]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate paths in pinned Hugging Face tree metadata")
    if set(normalized) != _actual_model_files(model_root):
        raise ValueError("local model files differ from pinned Hugging Face tree inventory")

    inventory: list[dict[str, Any]] = []
    root = model_root.resolve()
    for relative in sorted(normalized):
        metadata = source_files[relative]
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid source metadata for {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"model path escapes root: {relative}") from exc
        if not path.is_file():
            raise ValueError(f"model file missing: {relative}")
        size = path.stat().st_size
        if size != metadata.get("size"):
            raise ValueError(f"pinned size mismatch: {relative}")
        sha256 = _sha256_file(path)
        expected_lfs = metadata.get("lfs_sha256")
        if expected_lfs is not None and sha256 != expected_lfs:
            raise ValueError(f"pinned LFS SHA-256 mismatch: {relative}")
        expected_blob = metadata.get("blob_id")
        if expected_lfs is None and expected_blob is not None:
            if _git_blob_sha1(path) != expected_blob:
                raise ValueError(f"pinned Git blob mismatch: {relative}")
        inventory.append({"path": relative, "size_bytes": size, "sha256": sha256})
    return inventory


def _index_identity(
    model_root: Path, inventory: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    index_records = [
        record for record in inventory if record["path"].endswith(".safetensors.index.json")
    ]
    if len(index_records) != 1:
        raise ValueError("model tree must contain exactly one safetensors index")
    index_record = index_records[0]
    try:
        index = json.loads(
            (model_root / index_record["path"]).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("model safetensors index is invalid") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model safetensors index has no weight_map")
    shards = sorted({_safe_relative(value) for value in weight_map.values()})
    inventory_paths = {record["path"] for record in inventory}
    if any(shard not in inventory_paths for shard in shards):
        raise ValueError("indexed weight shard is absent from model manifest")
    return dict(index_record), shards


def write_revision_marker(
    model_root: Path, *, repository: str, revision: str
) -> Path:
    """Persist the acquisition repository/revision before manifest creation."""
    marker = model_root / "SNAPSHOT_REVISION"
    content = f"repo_id={repository}\nrevision={revision}\n"
    if marker.exists():
        _verify_snapshot_marker(model_root, repository, revision)
        return marker
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    try:
        try:
            os.link(temporary, marker)
        except FileExistsError:
            _verify_snapshot_marker(model_root, repository, revision)
    finally:
        temporary.unlink(missing_ok=True)
    return marker


def create_model_manifest(
    model_root: Path,
    manifest_path: Path,
    *,
    repository: str,
    revision: str,
    model_family: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Create an immutable manifest after verifying pinned HF source metadata."""
    model_root = model_root.resolve()
    if not model_root.is_dir():
        raise ValueError(f"model root is missing: {model_root}")
    _verify_snapshot_marker(model_root, repository, revision)
    source_tree = _source_tree(model_root, revision)
    inventory = _inventory_from_source_tree(model_root, source_tree)
    index_identity, shards = _index_identity(model_root, inventory)
    tree_sha256 = hashlib.sha256(_canonical_json(inventory).encode()).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "repository": repository,
        "revision": revision,
        "local_model_path": _normalized_root(model_root),
        "model_family": model_family,
        "files": inventory,
        "model_index_identity": index_identity,
        "indexed_weight_shards": shards,
        "total_size_bytes": sum(record["size_bytes"] for record in inventory),
        "tree_sha256": tree_sha256,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    payload["manifest_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode()
    ).hexdigest()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing = verify_model_manifest(
            manifest_path,
            model_root=model_root,
            expected_repository=repository,
            expected_revision=revision,
            expected_family=model_family,
        )
        if existing["tree_sha256"] != payload["tree_sha256"]:
            raise ValueError("immutable model manifest differs from verified model tree")
        return existing
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=manifest_path.parent, delete=False, newline="\n"
    ) as file:
        file.write(serialized)
        temporary = Path(file.name)
    try:
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def verify_model_manifest(
    manifest_path: Path,
    *,
    model_root: Path,
    expected_repository: str,
    expected_revision: str,
    expected_family: str,
) -> dict[str, Any]:
    """Verify manifest integrity and every live model byte, failing closed."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model manifest is missing or invalid: {manifest_path}") from exc
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("model manifest schema mismatch")
    recorded_self_hash = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_self_hash = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    if recorded_self_hash != actual_self_hash:
        raise ValueError("model manifest self SHA-256 mismatch")

    model_root = model_root.resolve()
    expected_identity = {
        "repository": expected_repository,
        "revision": expected_revision,
        "local_model_path": _normalized_root(model_root),
        "model_family": expected_family,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"model manifest {key} mismatch")
    _verify_snapshot_marker(model_root, expected_repository, expected_revision)
    source_tree = _source_tree(model_root, expected_revision)

    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("model manifest inventory is empty")
    normalized: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("model manifest inventory record is invalid")
        normalized.append(_safe_relative(record.get("path")))
    if len(normalized) != len(set(normalized)) or normalized != sorted(normalized):
        raise ValueError("model manifest inventory paths are duplicate or unordered")
    if set(normalized) != _actual_model_files(model_root):
        raise ValueError("local model contains missing or unlisted files")

    pinned_inventory = _inventory_from_source_tree(model_root, source_tree)
    if records != pinned_inventory:
        raise ValueError("model manifest inventory differs from pinned source tree")
    verified = pinned_inventory

    index_identity, shards = _index_identity(model_root, verified)
    if manifest.get("model_index_identity") != index_identity:
        raise ValueError("model index identity mismatch")
    if manifest.get("indexed_weight_shards") != shards:
        raise ValueError("indexed weight shard inventory mismatch")
    if manifest.get("total_size_bytes") != sum(
        record["size_bytes"] for record in verified
    ):
        raise ValueError("model manifest total byte size mismatch")
    tree_sha256 = hashlib.sha256(_canonical_json(verified).encode()).hexdigest()
    if manifest.get("tree_sha256") != tree_sha256:
        raise ValueError("model tree SHA-256 mismatch")
    return manifest
