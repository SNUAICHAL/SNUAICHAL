import csv
from pathlib import Path

import pytest

from snuaichal.baseline import create_identity_submission


def test_create_identity_submission_preserves_ids(tmp_path: Path) -> None:
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("Id,Sentence\n001,first\n002,second\n", encoding="utf-8")
    output = tmp_path / "submission.csv"

    assert create_identity_submission(test_csv, output) == 2

    with output.open(encoding="utf-8", newline="") as submission:
        assert list(csv.DictReader(submission)) == [
            {"Id": "001", "Answer": "[1, 2, 3, 4]"},
            {"Id": "002", "Answer": "[1, 2, 3, 4]"},
        ]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("Sentence\nmissing id\n", "Id column"),
        ("Id,Sentence\n", "no samples"),
        ("Id,Sentence\n1,a\n1,b\n", "duplicate Id"),
    ],
)
def test_create_identity_submission_rejects_invalid_test_csv(
    tmp_path: Path, contents: str, message: str
) -> None:
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        create_identity_submission(test_csv, tmp_path / "submission.csv")