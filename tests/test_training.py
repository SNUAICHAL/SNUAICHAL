from pathlib import Path

import pytest
import torch

from snuaichal.training import (
    Qwen2VLCollator,
    answer_positions_to_chronological,
    balance_training_rows,
    build_parser,
    build_training_argument_kwargs,
    build_training_messages,
    image_dhash,
    permute_training_row,
    seed_training,
    split_rows,
    split_rows_without_image_overlap,
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


def test_collator_masks_left_padded_batch_prompts(tmp_path: Path) -> None:
    class FakeBatch(dict):
        def __getattr__(self, name: str):
            return self[name]

    class LeftPaddingProcessor:
        tokenizer = type(
            "Tokenizer", (), {"pad_token_id": 0, "padding_side": "left"}
        )()

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "full" if messages[-1]["role"] == "assistant" else "prompt"

        def __call__(self, *, text, **kwargs):
            if text == ["full", "full"]:
                return FakeBatch(
                    input_ids=torch.tensor(
                        [[0, 10, 11, 12, 13], [20, 21, 22, 23, 24]]
                    ),
                    attention_mask=torch.tensor(
                        [[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
                    ),
                )
            return FakeBatch(
                input_ids=torch.tensor(
                    [[0, 0, 10, 11, 12], [0, 20, 21, 22, 23]]
                ),
                attention_mask=torch.tensor(
                    [[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]
                ),
            )

    rows = []
    for index in range(2):
        sample_id = f"sample-{index}"
        sample_dir = tmp_path / sample_id
        sample_dir.mkdir()
        row = {
            "Id": sample_id,
            "Sentence": "First this, then that.",
            "Answer": "[3, 2, 4, 1]",
        }
        for slot in range(1, 5):
            filename = f"image-{slot}.jpg"
            (sample_dir / filename).touch()
            row[f"Input_{slot}"] = filename
        rows.append(row)

    collator = Qwen2VLCollator(
        LeftPaddingProcessor(),
        tmp_path,
        process_vision_info_fn=lambda messages: (["images"], None),
    )

    batch = collator(rows)

    assert batch["labels"].tolist() == [
        [-100, -100, -100, -100, 13],
        [-100, -100, -100, -100, 24],
    ]


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

    assert args.batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.max_pixels == 200704
    assert args.lora_rank == 16
    assert args.balance_inputs is True
    assert args.clean_validation is True
    assert args.validation_fraction == 0.2

    training_kwargs = build_training_argument_kwargs(args, bf16=True)
    assert training_kwargs["eval_strategy"] == "no"
    assert training_kwargs["save_total_limit"] is None


def test_training_seed_reproduces_torch_and_python_randomness() -> None:
    import random

    seed_training(17)
    first = (random.random(), torch.rand(3).tolist())
    seed_training(17)
    second = (random.random(), torch.rand(3).tolist())

    assert second == first


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


def test_perceptual_hash_matches_recompressed_image(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (96, 64))
    draw = ImageDraw.Draw(image)
    for x in range(96):
        draw.line((x, 0, x, 63), fill=(x * 2, 255 - x * 2, x))
    draw.rectangle((20, 15, 55, 45), fill=(230, 30, 80))
    high_quality = tmp_path / "high.jpg"
    low_quality = tmp_path / "low.jpg"
    image.save(high_quality, quality=95)
    image.save(low_quality, quality=45)

    assert high_quality.read_bytes() != low_quality.read_bytes()
    assert image_dhash(high_quality) == image_dhash(low_quality)


def test_clean_split_excludes_rows_with_images_reused_elsewhere(tmp_path: Path) -> None:
    import random

    from PIL import Image

    rows = []
    labels = ["[1, 2, 3, 4]"] * 3 + ["[4, 3, 2, 1]"] * 3
    for index, label in enumerate(labels):
        sample_id = f"sample-{index}"
        sample_dir = tmp_path / sample_id
        sample_dir.mkdir()
        row = {"Id": sample_id, "Answer": label}
        for slot in range(1, 5):
            filename = f"image-{slot}.jpg"
            image_seed = index * 4 + slot
            if slot == 1 and index in {0, 3}:
                image_seed = 10_000
            image_bytes = random.Random(image_seed).randbytes(32 * 32 * 3)
            Image.frombytes("RGB", (32, 32), image_bytes).save(sample_dir / filename)
            row[f"Input_{slot}"] = filename
        rows.append(row)

    train_rows, validation_rows = split_rows_without_image_overlap(
        rows,
        image_dir=tmp_path,
        validation_fraction=1 / 3,
        seed=7,
    )

    assert len(validation_rows) == 2
    assert {row["Id"] for row in validation_rows}.isdisjoint({"sample-0", "sample-3"})
    assert {row["Answer"] for row in validation_rows} == {
        "[1, 2, 3, 4]",
        "[4, 3, 2, 1]",
    }
    assert {row["Id"] for row in train_rows}.union(
        row["Id"] for row in validation_rows
    ) == {row["Id"] for row in rows}


def test_split_rows_is_label_stratified() -> None:
    rows = [
        {"Id": f"a-{index}", "Answer": "[1, 2, 3, 4]"} for index in range(10)
    ] + [{"Id": f"b-{index}", "Answer": "[4, 3, 2, 1]"} for index in range(10)]

    train_rows, validation_rows = split_rows(rows, validation_fraction=0.2, seed=3)

    assert len(validation_rows) == 4
    assert sum(row["Answer"] == "[1, 2, 3, 4]" for row in validation_rows) == 2
    assert sum(row["Answer"] == "[4, 3, 2, 1]" for row in validation_rows) == 2
    assert len(train_rows) == 16
