from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"repository", "revision", "files", "manifest_sha256"}
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"weight manifest missing fields: {missing}")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()
    if payload["manifest_sha256"] != actual:
        raise RuntimeError("weight manifest self SHA-256 mismatch")
    if not isinstance(payload["files"], list) or not payload["files"]:
        raise RuntimeError("weight manifest files must be a non-empty list")
    return payload


def safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("weight manifest file path must be a non-empty string")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts or "\\" in value or ":" in value:
        raise RuntimeError(f"unsafe weight manifest file path: {value!r}")
    return Path(*posix.parts)


def verify_local_revision_marker(root: Path, manifest: dict[str, object]) -> str | None:
    """Validate local provenance metadata separately from upstream snapshot bytes."""
    marker = root / "SNAPSHOT_REVISION"
    if not marker.exists():
        return None
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("SNAPSHOT_REVISION must be a regular non-symlink file")
    fields: dict[str, str] = {}
    allowed = {"repo_id", "revision", "license", "public_date"}
    for line in marker.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise RuntimeError("SNAPSHOT_REVISION contains a malformed line")
        key, value = line.split("=", 1)
        if key not in allowed or key in fields or not value:
            raise RuntimeError("SNAPSHOT_REVISION contains an invalid field")
        fields[key] = value
    if fields.get("repo_id") != manifest["repository"] or fields.get("revision") != manifest["revision"]:
        raise RuntimeError("SNAPSHOT_REVISION repository/revision differs from weight manifest")
    manifest_license = manifest.get("license")
    if "license" in fields and manifest_license is not None and fields["license"] != manifest_license:
        raise RuntimeError("SNAPSHOT_REVISION license differs from weight manifest")
    return sha256_file(marker)


def verify_snapshot(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    root = root.resolve(strict=True)
    marker_sha256 = verify_local_revision_marker(root, manifest)
    records = manifest["files"]
    assert isinstance(records, list)
    verified_bytes = 0
    expected: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("weight manifest file record must be an object")
        relative = safe_relative_path(record.get("path"))
        relative_posix = relative.as_posix()
        if relative_posix in expected:
            raise RuntimeError(f"duplicate weight manifest path: {relative_posix}")
        expected.add(relative_posix)
        candidate = root / relative
        if candidate.is_symlink():
            raise RuntimeError(f"symlink forbidden in weight snapshot: {relative_posix}")
        resolved = candidate.resolve(strict=True)
        if root not in resolved.parents:
            raise RuntimeError(f"weight file escapes snapshot root: {relative_posix}")
        if not resolved.is_file():
            raise RuntimeError(f"weight path is not a regular file: {relative_posix}")
        expected_size = int(record["size_bytes"])
        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"weight size mismatch for {relative_posix}: expected {expected_size}, got {actual_size}"
            )
        expected_sha = str(record["sha256"])
        actual_sha = sha256_file(resolved)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"weight SHA-256 mismatch for {relative_posix}: expected {expected_sha}, got {actual_sha}"
            )
        verified_bytes += actual_size
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.relative_to(root).as_posix() != "SNAPSHOT_REVISION"
    }
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected or missing:
        raise RuntimeError(f"weight inventory mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "status": "PASS",
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "verified_files": len(expected),
        "verified_bytes": verified_bytes,
        "local_revision_marker_sha256": marker_sha256,
    }


def download_snapshot(manifest: dict[str, object], output: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install requirements.txt first") from exc
    snapshot_download(
        repo_id=str(manifest["repository"]),
        revision=str(manifest["revision"]),
        local_dir=output,
    )
    marker_lines = [
        f"repo_id={manifest['repository']}",
        f"revision={manifest['revision']}",
    ]
    if manifest.get("license") is not None:
        marker_lines.append(f"license={manifest['license']}")
    if manifest.get("public_date") is not None:
        marker_lines.append(f"public_date={manifest['public_date']}")
    (output / "SNAPSHOT_REVISION").write_text(
        "\n".join(marker_lines) + "\n", encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and fail-closed verify pinned model weights")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if not args.verify_only:
        args.output.mkdir(parents=True, exist_ok=True)
        download_snapshot(manifest, args.output)
    result = verify_snapshot(args.output, manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
