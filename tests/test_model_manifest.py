import hashlib
import json
from pathlib import Path

import pytest

from snuaichal.model_manifest import create_model_manifest, verify_model_manifest


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


REPOSITORY = "example/synthetic-model"
REVISION = "a" * 40
FAMILY = "qwen3_5"


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data, usedforsecurity=False
    ).hexdigest()


def _write_synthetic_model(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "model"
    root.mkdir()
    files = {
        "config.json": b'{"model_type":"qwen3_5"}\n',
        "tokenizer.json": b'{"version":"1"}\n',
        "preprocessor_config.json": b'{"size":512}\n',
        "model-00001-of-00002.safetensors": b"weight-shard-one",
        "model-00002-of-00002.safetensors": b"weight-shard-two",
    }
    index = {
        "metadata": {"total_size": 32},
        "weight_map": {
            "layer.0": "model-00001-of-00002.safetensors",
            "layer.1": "model-00002-of-00002.safetensors",
        },
    }
    files["model.safetensors.index.json"] = (
        json.dumps(index, sort_keys=True) + "\n"
    ).encode()
    source_files = {}
    for relative, data in files.items():
        path = root / relative
        path.write_bytes(data)
        record = {"size": len(data), "blob_id": _git_blob_id(data)}
        if relative.endswith(".safetensors"):
            record["lfs_sha256"] = hashlib.sha256(data).hexdigest()
            record["lfs_size"] = len(data)
        source_files[relative] = record
    tree_dir = root / ".cache" / "huggingface" / "trees"
    tree_dir.mkdir(parents=True)
    (tree_dir / f"{REVISION}.json").write_text(
        json.dumps({"format_version": 1, "files": source_files}), encoding="utf-8"
    )
    (root / "SNAPSHOT_REVISION").write_text(
        f"repo_id={REPOSITORY}\nrevision={REVISION}\n", encoding="utf-8"
    )
    return root, tmp_path / "model-manifest.json"


def _create(tmp_path: Path) -> tuple[Path, Path, dict]:
    root, manifest_path = _write_synthetic_model(tmp_path)
    manifest = create_model_manifest(
        root,
        manifest_path,
        repository=REPOSITORY,
        revision=REVISION,
        model_family=FAMILY,
        created_at="2026-07-18T00:00:00Z",
    )
    return root, manifest_path, manifest


def _verify(root: Path, manifest_path: Path, **overrides):
    return verify_model_manifest(
        manifest_path,
        model_root=root,
        expected_repository=overrides.get("repository", REPOSITORY),
        expected_revision=overrides.get("revision", REVISION),
        expected_family=overrides.get("family", FAMILY),
    )


def test_correct_pinned_model_tree_passes_with_verified_digest(tmp_path: Path) -> None:
    root, manifest_path, created = _create(tmp_path)

    verified = _verify(root, manifest_path)

    assert verified == created
    assert len(verified["tree_sha256"]) == 64
    assert len(verified["manifest_sha256"]) == 64
    assert verified["indexed_weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"revision": "b" * 40}, "revision mismatch"),
        ({"repository": "other/repository"}, "repository mismatch"),
    ],
)
def test_declared_model_identity_change_fails(
    tmp_path: Path, override: dict[str, str], message: str
) -> None:
    root, manifest_path, _ = _create(tmp_path)

    with pytest.raises(ValueError, match=message):
        _verify(root, manifest_path, **override)


@pytest.mark.parametrize(
    "relative",
    [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model-00001-of-00002.safetensors",
    ],
)
def test_required_model_file_tampering_fails(tmp_path: Path, relative: str) -> None:
    root, manifest_path, _ = _create(tmp_path)
    with (root / relative).open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(ValueError, match="pinned (size|LFS SHA-256|Git blob) mismatch"):
        _verify(root, manifest_path)


def test_missing_indexed_weight_shard_fails(tmp_path: Path) -> None:
    root, manifest_path, _ = _create(tmp_path)
    (root / "model-00002-of-00002.safetensors").unlink()

    with pytest.raises(ValueError, match="missing or unlisted"):
        _verify(root, manifest_path)


def test_model_manifest_self_hash_tampering_fails(tmp_path: Path) -> None:
    root, manifest_path, _ = _create(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["created_at"] = "tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="self SHA-256"):
        _verify(root, manifest_path)


def test_runtime_verification_rejects_consistently_rewritten_file_and_manifest(
    tmp_path: Path,
) -> None:
    root, manifest_path, _ = _create(tmp_path)
    changed = b'{"model_type":"qwen3_x"}\n'
    (root / "config.json").write_bytes(changed)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in payload["files"] if item["path"] == "config.json")
    record["size_bytes"] = len(changed)
    record["sha256"] = hashlib.sha256(changed).hexdigest()
    payload["tree_sha256"] = hashlib.sha256(
        _canonical_json(payload["files"]).encode()
    ).hexdigest()
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = hashlib.sha256(
        _canonical_json(unsigned).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned Git blob mismatch"):
        _verify(root, manifest_path)


def test_unlisted_model_file_fails(tmp_path: Path) -> None:
    root, manifest_path, _ = _create(tmp_path)
    (root / "unexpected-tokenizer.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or unlisted"):
        _verify(root, manifest_path)
