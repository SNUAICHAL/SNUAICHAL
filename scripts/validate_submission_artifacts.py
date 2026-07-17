from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a snuaichallenge submission and inference audit")
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-tta", type=int, required=True)
    return parser.parse_args()


def parse_answer(raw: str) -> list[int]:
    try:
        answer = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid Answer literal: {raw!r}") from exc
    if not isinstance(answer, list) or len(answer) != 4 or sorted(answer) != [1, 2, 3, 4]:
        raise ValueError(f"Invalid Answer permutation: {raw!r}")
    return answer


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_tta <= 0:
        raise ValueError("expected_tta must be positive")

    with args.test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        expected_ids = [row["Id"] for row in csv.DictReader(handle)]
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Test IDs must be non-empty and unique")

    with args.submission.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Id", "Answer"]:
            raise ValueError(f"Invalid submission columns: {reader.fieldnames}")
        submission_rows = list(reader)

    submission_ids = [row["Id"] for row in submission_rows]
    if submission_ids != expected_ids:
        raise ValueError("Submission IDs or row order do not match test.csv")
    answers = [parse_answer(row["Answer"]) for row in submission_rows]

    audit_rows = [
        json.loads(line)
        for line in args.audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(audit_rows) != len(expected_ids):
        raise ValueError(
            f"Audit row count mismatch: expected {len(expected_ids)}, got {len(audit_rows)}"
        )
    audit_ids = [row.get("Id") for row in audit_rows]
    if audit_ids != expected_ids:
        raise ValueError("Audit IDs or row order do not match test.csv")

    parse_failures = 0
    view_parse_failures = 0
    inconsistent_tta = 0
    for index, (audit, answer) in enumerate(zip(audit_rows, answers, strict=True)):
        if audit.get("answer") != answer:
            raise ValueError(f"Audit/submission answer mismatch at row {index}")
        views = audit.get("views")
        if not isinstance(views, list) or len(views) != args.expected_tta:
            raise ValueError(
                f"Expected {args.expected_tta} audit views at row {index}, got {views!r}"
            )
        if not isinstance(audit.get("parse_ok"), bool):
            raise ValueError(f"Invalid parse_ok at row {index}")
        if not audit["parse_ok"]:
            parse_failures += 1
        view_parse_failures += sum(view.get("parse_ok") is not True for view in views)
        if audit.get("tta_consistent") is not True:
            inconsistent_tta += 1

    return {
        "audit_rows": len(audit_rows),
        "expected_tta": args.expected_tta,
        "inconsistent_tta_rows": inconsistent_tta,
        "parse_failures": parse_failures,
        "rows": len(submission_rows),
        "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        "unique_ids": len(set(submission_ids)),
        "view_parse_failures": view_parse_failures,
    }


def main() -> None:
    print(json.dumps(validate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
