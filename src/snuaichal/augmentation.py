"""Deterministic epoch-aware input-slot augmentation."""

from __future__ import annotations

import hashlib
import random
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # Lightweight CPU CI does not install runtime model packages.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


def stable_epoch_seed(global_seed: int, epoch: int, sample_id: str) -> int:
    """Hash a training seed, epoch, and sample ID without process-randomized hash()."""
    payload = f"{global_seed}\0{epoch}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class EpochShuffleDataset:
    """Apply one deterministic input permutation per sample and epoch."""

    def __init__(self, rows: list[Any], *, seed: int, augment: bool) -> None:
        self.rows = list(rows)
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Any:
        row = self.rows[index]
        if not self.augment:
            return row
        from snuaichal.training import permute_training_row

        order = [1, 2, 3, 4]
        random.Random(
            stable_epoch_seed(self.seed, self.epoch, str(row["Id"]))
        ).shuffle(order)
        return permute_training_row(row, order)


class DatasetEpochCallback(TrainerCallback):
    """Keep dataset augmentation synchronized with Trainer state, including resume."""

    def __init__(self, dataset: EpochShuffleDataset) -> None:
        self.dataset = dataset

    def _set_from_state(self, state: Any) -> None:
        self.dataset.set_epoch(int(state.epoch or 0))

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._set_from_state(state)

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._set_from_state(state)
