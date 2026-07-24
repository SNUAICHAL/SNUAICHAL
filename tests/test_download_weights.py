from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.download_weights import safe_relative_path, verify_snapshot


def record(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_verify_snapshot_accepts_exact_inventory(tmp_path: Path) -> None:
    content = b"frozen-weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    manifest = {
        "repository": "example/model",
        "revision": "a" * 40,
        "files": [record("model.safetensors", content)],
    }
    result = verify_snapshot(tmp_path, manifest)
    assert result["status"] == "PASS"
    assert result["verified_files"] == 1


def test_verify_snapshot_accepts_and_validates_local_revision_marker(tmp_path: Path) -> None:
    content = b"frozen-weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    marker = (
        "repo_id=example/model\n"
        f"revision={'a' * 40}\n"
        "license=Apache-2.0\n"
        "public_date=2025-10-11\n"
    )
    (tmp_path / "SNAPSHOT_REVISION").write_text(marker, encoding="utf-8")
    manifest = {
        "repository": "example/model",
        "revision": "a" * 40,
        "license": "Apache-2.0",
        "files": [record("model.safetensors", content)],
    }

    result = verify_snapshot(tmp_path, manifest)

    assert result["status"] == "PASS"
    assert result["local_revision_marker_sha256"] == hashlib.sha256(
        (tmp_path / "SNAPSHOT_REVISION").read_bytes()
    ).hexdigest()


def test_verify_snapshot_rejects_mismatched_local_revision_marker(tmp_path: Path) -> None:
    content = b"frozen-weights"
    (tmp_path / "model.safetensors").write_bytes(content)
    (tmp_path / "SNAPSHOT_REVISION").write_text(
        f"repo_id=other/model\nrevision={'b' * 40}\n",
        encoding="utf-8",
    )
    manifest = {
        "repository": "example/model",
        "revision": "a" * 40,
        "files": [record("model.safetensors", content)],
    }

    with pytest.raises(RuntimeError, match="SNAPSHOT_REVISION.*differs"):
        verify_snapshot(tmp_path, manifest)


def test_verify_snapshot_rejects_hash_drift(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    manifest = {
        "repository": "example/model",
        "revision": "a" * 40,
        "files": [record("model.safetensors", b"expected")],
    }
    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        verify_snapshot(tmp_path, manifest)


def test_verify_snapshot_rejects_unexpected_file(tmp_path: Path) -> None:
    content = b"ok"
    (tmp_path / "model.safetensors").write_bytes(content)
    (tmp_path / "extra.bin").write_bytes(b"unexpected")
    manifest = {
        "repository": "example/model",
        "revision": "a" * 40,
        "files": [record("model.safetensors", content)],
    }
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_snapshot(tmp_path, manifest)


@pytest.mark.parametrize("value", ["../escape", "/absolute", "C:/drive", "nested\\escape"])
def test_safe_relative_path_rejects_escape(value: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe"):
        safe_relative_path(value)
