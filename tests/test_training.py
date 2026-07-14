from pathlib import Path

import pytest
import torch

from snuaichal.training import (
    Qwen2VLCollator,
    answer_positions_to_chronological,
    balance_training_rows,
    build_parser,
    build_training_messages,
    permute_training_row,
    split_rows,
)


def test_answer_positions_are_converted_to_chronological_labels() -> None:
    assert answer_positions_to_chronological("[3, 2, 4, 1]") == [4, 2, 1, 3]


@pytest.mark.parametrize("answer", ["not a list", "[1, 2, 3]", "[1, 1, 2, 3]"])
def test_invalid_training_answer_is_rejected(answer: str) -> None:
    with pytest.raises(ValueError, match="Invalid training answer"):
        answer_positions_to_chronological(answer)


def test_training_messages_end_with_chronological_assistant_answer(
    tmp_path: Path,
) -> None:
    row = {
        "Id": "sample",
        "Sentence": "First this, then that.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in ("one.jpg", "two.jpg", "three.jpg", "four.jpg"):
        (sample_dir / filename).touch()

    messages = build_training_messages(row, tmp_path)

    assert messages[-1] == {"role": "assistant", "content": "[4, 2, 1, 3]"}
    assert [item["type"] for item in messages[0]["content"]].count("image") == 4


def test_collator_masks_prompt_and_padding_tokens(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def __getattr__(self, name: str):
            return self[name]

    class FakeProcessor:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "full" if messages[-1]["role"] == "assistant" else "prompt"

        def __call__(self, *, text, **kwargs):
            if text == ["full"]:
                return FakeBatch(
                    input_ids=torch.tensor([[10, 11, 12, 13, 0]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
                )
            return FakeBatch(
                input_ids=torch.tensor([[10, 11, 12]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )

    row = {
        "Id": "sample",
        "Sentence": "First this, then that.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    for filename in ("one.jpg", "two.jpg", "three.jpg", "four.jpg"):
        (sample_dir / filename).touch()

    collator = Qwen2VLCollator(
        FakeProcessor(),
        tmp_path,
        process_vision_info_fn=lambda messages: (["images"], None),
    )

    batch = collator([row])

    assert batch["labels"].tolist() == [[-100, -100, -100, 13, -100]]


def test_split_rows_is_deterministic_and_disjoint() -> None:
    rows = [{"Id": str(index)} for index in range(20)]

    train_rows, validation_rows = split_rows(rows, validation_fraction=0.2, seed=7)

    assert len(train_rows) == 16
    assert len(validation_rows) == 4
    assert {row["Id"] for row in train_rows}.isdisjoint(
        row["Id"] for row in validation_rows
    )
    assert split_rows(rows, validation_fraction=0.2, seed=7) == (
        train_rows,
        validation_rows,
    )


def test_training_defaults_fit_a_single_24gb_gpu() -> None:
    args = build_parser().parse_args([])

    assert args.batch_size == 1
    assert args.gradient_accumulation_steps == 8
    assert args.max_pixels == 200704
    assert args.lora_rank == 16


def test_permute_training_row_reorders_inputs_and_target() -> None:
    row = {
        "Id": "sample",
        "Sentence": "A sequence.",
        "Input_1": "one.jpg",
        "Input_2": "two.jpg",
        "Input_3": "three.jpg",
        "Input_4": "four.jpg",
        "Answer": "[3, 2, 4, 1]",
    }

    permuted = permute_training_row(row, [2, 4, 1, 3])

    assert [permuted[f"Input_{index}"] for index in range(1, 5)] == [
        "two.jpg",
        "four.jpg",
        "one.jpg",
        "three.jpg",
    ]
    assert permuted["Answer"] == "[2, 1, 3, 4]"
    assert permuted["Id"] == row["Id"]
    assert permuted["Sentence"] == row["Sentence"]


def test_balance_training_rows_flattens_all_24_labels() -> None:
    rows = [
        {
            "Id": str(index),
            "Sentence": "A sequence.",
            "Input_1": "one.jpg",
            "Input_2": "two.jpg",
            "Input_3": "three.jpg",
            "Input_4": "four.jpg",
            "Answer": "[1, 2, 3, 4]",
        }
        for index in range(49)
    ]

    balanced = balance_training_rows(rows, seed=7)
    counts = {}
    for row in balanced:
        counts[row["Answer"]] = counts.get(row["Answer"], 0) + 1

    assert len(balanced) == len(rows)
    assert len(counts) == 24
    assert max(counts.values()) - min(counts.values()) <= 1
    assert balance_training_rows(rows, seed=7) == balanced


def test_split_rows_is_label_stratified() -> None:
    rows = [
        {"Id": f"a-{index}", "Answer": "[1, 2, 3, 4]"} for index in range(10)
    ] + [{"Id": f"b-{index}", "Answer": "[4, 3, 2, 1]"} for index in range(10)]

    train_rows, validation_rows = split_rows(rows, validation_fraction=0.2, seed=3)

    assert len(validation_rows) == 4
    assert sum(row["Answer"] == "[1, 2, 3, 4]" for row in validation_rows) == 2
    assert sum(row["Answer"] == "[4, 3, 2, 1]" for row in validation_rows) == 2
    assert len(train_rows) == 16
