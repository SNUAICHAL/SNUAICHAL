"""Print the newest complete Hugging Face Trainer checkpoint, if one exists."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REQUIRED_STATE_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
)
_WEIGHT_FILES = ("adapter_model.safetensors", "model.safetensors")
_CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def find_latest_complete_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest checkpoint with weights and resumable Trainer state."""
    candidates: list[tuple[int, Path]] = []
    for checkpoint in run_dir.glob("checkpoint-*"):
        match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if checkpoint.is_dir() and match:
            candidates.append((int(match.group(1)), checkpoint))

    for expected_step, checkpoint in sorted(candidates, reverse=True):
        if not all((checkpoint / name).is_file() for name in _REQUIRED_STATE_FILES):
            continue
        if not any((checkpoint / name).is_file() for name in _WEIGHT_FILES):
            continue
        try:
            state = json.loads(
                (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if int(state.get("global_step", -1)) == expected_step:
            return checkpoint
    return None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: find_latest_checkpoint.py RUN_DIR")
    checkpoint = find_latest_complete_checkpoint(Path(sys.argv[1]))
    if checkpoint is not None:
        print(checkpoint)


if __name__ == "__main__":
    main()
