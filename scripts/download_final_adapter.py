from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "download_url",
        "archive_sha256",
        "archive_size_bytes",
        "files",
        "manifest_sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"adapter manifest missing fields: {missing}")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if payload["manifest_sha256"] != actual:
        raise RuntimeError("adapter manifest self SHA-256 mismatch")
    return payload


def safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("adapter path must be a non-empty string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or ":" in value
        or value != pure.as_posix()
    ):
        raise RuntimeError(f"unsafe adapter path: {value!r}")
    return Path(*pure.parts)


def verify_adapter_directory(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    expected: set[str] = set()
    verified_bytes = 0
    for record in manifest["files"]:
        relative = safe_relative_path(record["path"])
        relative_posix = relative.as_posix()
        if relative_posix in expected:
            raise RuntimeError(f"duplicate adapter path: {relative_posix}")
        expected.add(relative_posix)
        candidate = root / relative
        if candidate.is_symlink():
            raise RuntimeError(f"adapter symlink forbidden: {relative_posix}")
        resolved = candidate.resolve(strict=True)
        if root not in resolved.parents or not resolved.is_file():
            raise RuntimeError(f"adapter path escapes root: {relative_posix}")
        size = resolved.stat().st_size
        if size != int(record["size_bytes"]):
            raise RuntimeError(f"adapter size mismatch: {relative_posix}")
        if sha256_file(resolved) != record["sha256"]:
            raise RuntimeError(f"adapter SHA-256 mismatch: {relative_posix}")
        verified_bytes += size
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(
            f"adapter inventory mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return {
        "status": "PASS",
        "verified_files": len(expected),
        "verified_bytes": verified_bytes,
        "checkpoint_step": manifest.get("checkpoint_step"),
        "adapter_sha256": next(
            record["sha256"]
            for record in manifest["files"]
            if record["path"] == "adapter_model.safetensors"
        ),
    }


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names: set[str] = set()
        for member in members:
            path = safe_relative_path(member.name)
            normalized = path.as_posix()
            if normalized in names:
                raise RuntimeError(f"duplicate archive member: {normalized}")
            names.add(normalized)
            if not member.isfile():
                raise RuntimeError(f"only regular adapter files are allowed: {normalized}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"archive member has no readable payload: {normalized}")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)


def acquire_adapter(
    manifest_path: Path,
    output: Path,
    *,
    archive_override: Path | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if output.exists():
        return verify_adapter_directory(output, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "adapter.tar.gz"
        if archive_override is None:
            urllib.request.urlretrieve(str(manifest["download_url"]), archive)
        else:
            shutil.copyfile(archive_override, archive)
        if archive.stat().st_size != int(manifest["archive_size_bytes"]):
            raise RuntimeError("adapter archive size mismatch")
        if sha256_file(archive) != manifest["archive_sha256"]:
            raise RuntimeError("adapter archive SHA-256 mismatch")
        extracted = temporary / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        result = verify_adapter_directory(extracted, manifest)
        try:
            os.replace(extracted, output)
        except FileExistsError:
            return verify_adapter_directory(output, manifest)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the frozen checkpoint-2726 LoRA adapter"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/weights/qwen36-checkpoint2726-adapter.manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/qwen36-checkpoint2726"),
    )
    parser.add_argument("--archive", type=Path, help="Verify/extract a local archive")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_adapter_directory(args.output, load_manifest(args.manifest))
    else:
        result = acquire_adapter(
            args.manifest,
            args.output,
            archive_override=args.archive,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
