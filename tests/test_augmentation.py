from types import SimpleNamespace

from snuaichal.augmentation import DatasetEpochCallback, EpochShuffleDataset


def make_rows(count: int = 12) -> list[dict[str, str]]:
    return [
        {
            "Id": f"sample-{index}",
            "Sentence": "A sequence.",
            "Input_1": "one.jpg",
            "Input_2": "two.jpg",
            "Input_3": "three.jpg",
            "Input_4": "four.jpg",
            "Answer": "[3, 2, 4, 1]",
        }
        for index in range(count)
    ]


def inputs(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[f"Input_{slot}"] for slot in range(1, 5))


def test_epoch_shuffle_is_reproducible_for_seed_epoch_and_id() -> None:
    rows = make_rows()
    first = EpochShuffleDataset(rows, seed=42, augment=True)
    second = EpochShuffleDataset(list(reversed(rows)), seed=42, augment=True)
    first.set_epoch(3)
    second.set_epoch(3)

    first_by_id = {first[index]["Id"]: first[index] for index in range(len(first))}
    second_by_id = {second[index]["Id"]: second[index] for index in range(len(second))}

    assert first_by_id == second_by_id


def test_epoch_shuffle_changes_for_most_samples_between_epochs() -> None:
    dataset = EpochShuffleDataset(make_rows(), seed=42, augment=True)
    dataset.set_epoch(2)
    epoch_two = [inputs(dataset[index]) for index in range(len(dataset))]
    dataset.set_epoch(3)
    epoch_three = [inputs(dataset[index]) for index in range(len(dataset))]

    assert sum(left != right for left, right in zip(epoch_two, epoch_three)) >= 8


def test_validation_dataset_never_applies_shuffle() -> None:
    rows = make_rows(2)
    dataset = EpochShuffleDataset(rows, seed=42, augment=False)

    for epoch in (0, 1, 5):
        dataset.set_epoch(epoch)
        assert [dataset[index] for index in range(len(dataset))] == rows


def test_epoch_callback_restores_resumed_epoch() -> None:
    dataset = EpochShuffleDataset(make_rows(2), seed=42, augment=True)
    callback = DatasetEpochCallback(dataset)
    state = SimpleNamespace(epoch=3.75)

    callback.on_train_begin(None, state, None)
    assert dataset.epoch == 3

    state.epoch = 4.0
    callback.on_epoch_begin(None, state, None)
    assert dataset.epoch == 4
