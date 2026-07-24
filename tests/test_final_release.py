from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.download_final_adapter import (
    acquire_adapter,
    canonical_json,
    download_archive,
    load_manifest,
)
from scripts.run_final_inference import LATIN4, build_command


def _record(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = b'{"r":32,"lora_alpha":32}\n'
    weights = b"synthetic-adapter"
    archive = tmp_path / "adapter.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in (
            ("adapter_config.json", config),
            ("adapter_model.safetensors", weights),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    manifest = {
        "schema_version": 1,
        "download_url": archive.as_uri(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_size_bytes": archive.stat().st_size,
        "checkpoint_step": 2726,
        "files": [
            _record("adapter_config.json", config),
            _record("adapter_model.safetensors", weights),
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode()
    ).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return archive, manifest_path


def test_final_adapter_download_extract_and_verify(tmp_path: Path) -> None:
    archive, manifest_path = _fixture(tmp_path)
    output = tmp_path / "weights"

    result = acquire_adapter(manifest_path, output, archive_override=archive)

    assert result["status"] == "PASS"
    assert result["checkpoint_step"] == 2726
    assert sorted(path.name for path in output.iterdir()) == [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]


def test_adapter_manifest_rejects_self_hash_drift(tmp_path: Path) -> None:
    _, manifest_path = _fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["checkpoint_step"] = 3400
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="self SHA-256"):
        load_manifest(manifest_path)


def test_private_github_release_download_uses_authenticated_gh(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "adapter.tar.gz"
    manifest = {
        "repository": "example/private",
        "release_tag": "final-v1",
        "download_url": (
            "https://github.com/example/private/releases/download/"
            "final-v1/adapter.tar.gz"
        ),
    }
    observed: list[str] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        observed.extend(command)
        assert check is True
        destination.write_bytes(b"downloaded")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.download_final_adapter.shutil.which", lambda _: "gh")
    monkeypatch.setattr("scripts.download_final_adapter.subprocess.run", fake_run)

    download_archive(manifest, destination)

    assert destination.read_bytes() == b"downloaded"
    assert observed == [
        "gh",
        "release",
        "download",
        "final-v1",
        "--repo",
        "example/private",
        "--pattern",
        "adapter.tar.gz",
        "--output",
        str(destination),
    ]


def test_final_inference_command_is_frozen_to_latin4_hard(tmp_path: Path) -> None:
    args = argparse.Namespace(
        data_dir=Path("data"),
        model_path=Path("models/Qwen3.6-27B"),
        model_manifest=Path("configs/weights/qwen36-27b-final.manifest.json"),
        adapter_path=Path("weights/qwen36-checkpoint2726"),
        output_dir=tmp_path,
        resume=True,
    )

    command = build_command(args)

    assert command[command.index("--aggregation-mode") + 1] == "hard"
    assert command[command.index("--tta") + 1] == "4"
    assert json.loads(command[command.index("--tta-orders-json") + 1]) == LATIN4
    assert command[command.index("--max-new-tokens") + 1] == "64"
    assert command[-1] == "--resume"
