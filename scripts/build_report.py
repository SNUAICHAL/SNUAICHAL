"""Build and validate the five-page LaTeX competition report."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "report" / "main.tex"
DEFAULT_BUILD_DIR = ROOT / "tmp" / "report-build"
DEFAULT_OUTPUT = ROOT / "docs" / "final_report_5page_ko.pdf"


def executable(
    explicit: str | None,
    *,
    environment_name: str,
    command_name: str,
    required: bool,
) -> str | None:
    candidate = explicit or os.environ.get(environment_name) or shutil.which(command_name)
    if candidate:
        resolved = Path(candidate).expanduser()
        if resolved.exists():
            return str(resolved.resolve())
        from_path = shutil.which(candidate)
        if from_path:
            return from_path
    if required:
        raise SystemExit(
            f"{command_name} was not found; pass --{command_name} or set "
            f"{environment_name}"
        )
    return None


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def validate_log(log_path: Path) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failures = [
        line.strip()
        for line in text.splitlines()
        if "Overfull \\hbox" in line or "Overfull \\vbox" in line
    ]
    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"report layout overflow detected:\n{joined}")


def validate_pdf(pdf_path: Path, pdfinfo: str) -> None:
    metadata = run([pdfinfo, str(pdf_path)]).stdout
    pages_match = re.search(r"^Pages:\s+(\d+)\s*$", metadata, flags=re.MULTILINE)
    if pages_match is None or int(pages_match.group(1)) != 5:
        raise SystemExit("report must contain exactly five pages")
    if not re.search(r"^Page size:.*\(A4\)\s*$", metadata, flags=re.MULTILINE):
        raise SystemExit("report must use A4 pages")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tectonic")
    parser.add_argument("--pdfinfo")
    parser.add_argument(
        "--require-pdfinfo",
        action="store_true",
        help="fail instead of skipping page-size validation when pdfinfo is absent",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="build and validate without replacing the tracked PDF",
    )
    args = parser.parse_args()

    tectonic = executable(
        args.tectonic,
        environment_name="TECTONIC",
        command_name="tectonic",
        required=True,
    )
    pdfinfo = executable(
        args.pdfinfo,
        environment_name="PDFINFO",
        command_name="pdfinfo",
        required=args.require_pdfinfo,
    )

    source = args.source.resolve()
    build_dir = args.build_dir.resolve()
    output = args.output.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    completed = run(
        [
            tectonic,
            "-X",
            "compile",
            str(source),
            "--outdir",
            str(build_dir),
            "--keep-logs",
        ]
    )
    print(completed.stdout, end="")

    built_pdf = build_dir / f"{source.stem}.pdf"
    built_log = build_dir / f"{source.stem}.log"
    if not built_pdf.is_file() or not built_log.is_file():
        raise SystemExit("Tectonic did not produce the expected PDF and log")
    validate_log(built_log)
    if pdfinfo:
        validate_pdf(built_pdf, pdfinfo)
    else:
        print("warning: pdfinfo unavailable; page-size validation was skipped")

    if not args.check_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_pdf, output)
        target = output
    else:
        target = built_pdf
    print(f"report: {target}")
    print(f"sha256: {sha256(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
