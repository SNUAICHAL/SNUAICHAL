from __future__ import annotations

import ast
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "submission_checkpoint953.csv"
AUDIT = ROOT / "outputs" / "raw_predictions_checkpoint953.jsonl"
EXIT_CODE = ROOT / "outputs" / "inference_checkpoint953_exitcode.txt"
TEST_CSV = ROOT / "data" / "test.csv"

while not EXIT_CODE.exists():
    time.sleep(15)

exit_code = int(EXIT_CODE.read_text(encoding="utf-8").strip())
if exit_code != 0:
    raise SystemExit(f"Inference failed with exit code {exit_code}")
if not OUTPUT.is_file() or not AUDIT.is_file():
    raise SystemExit("Inference exited successfully but output artifacts are missing")

with TEST_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
    expected_ids = [row["Id"] for row in csv.DictReader(handle)]
with OUTPUT.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != ["Id", "Answer"]:
        raise SystemExit(f"Invalid columns: {reader.fieldnames}")
    rows = list(reader)

ids = [row["Id"] for row in rows]
if ids != expected_ids:
    raise SystemExit("Submission IDs or row order do not match data/test.csv")
if len(ids) != 819 or len(set(ids)) != 819:
    raise SystemExit(f"Expected 819 unique IDs, got rows={len(ids)} unique={len(set(ids))}")

invalid: list[tuple[str, str]] = []
for row in rows:
    try:
        answer = ast.literal_eval(row["Answer"])
    except (SyntaxError, ValueError):
        invalid.append((row["Id"], row["Answer"]))
        continue
    if not isinstance(answer, list) or len(answer) != 4 or sorted(answer) != [1, 2, 3, 4]:
        invalid.append((row["Id"], row["Answer"]))
if invalid:
    raise SystemExit(f"Invalid answer permutations: {invalid[:5]}")

audit_rows = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line]
if len(audit_rows) != 819:
    raise SystemExit(f"Expected 819 audit rows, got {len(audit_rows)}")
parse_failures = sum(not bool(row.get("parse_ok")) for row in audit_rows)
sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
print("VALIDATION_OK")
print(f"submission={OUTPUT}")
print(f"rows={len(rows)} unique_ids={len(set(ids))} invalid_answers=0")
print(f"audit_rows={len(audit_rows)} parse_failures={parse_failures}")
print(f"sha256={sha256}")
