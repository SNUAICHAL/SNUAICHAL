"""Create a deterministic identity-order submission without loading a model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from snuaichal.submission import IDENTITY_ORDER, answer_to_string


def create_identity_submission(test_csv: Path, output: Path) -> int:
    """Write one valid identity-order prediction for every row in *test_csv*."""
    if not test_csv.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    with test_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "Id" not in reader.fieldnames:
            raise ValueError("test.csv must contain an Id column")
        sample_ids = [str(row["Id"]) for row in reader]

    if not sample_ids:
        raise ValueError("test.csv contains no samples")
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("test.csv contains an empty Id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("test.csv contains duplicate Id values")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["Id", "Answer"])
        writer.writeheader()
        answer = answer_to_string(IDENTITY_ORDER)
        writer.writerows({"Id": sample_id, "Answer": answer} for sample_id in sample_ids)

    return len(sample_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_submission.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = create_identity_submission(args.test_csv, args.output)
    print(f"Saved {count} identity predictions to {args.output}")


if __name__ == "__main__":
    main()