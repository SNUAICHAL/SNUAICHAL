"""Training schedule calculations and deterministic early stopping."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from transformers import Trainer, TrainerCallback, TrainerControl, get_scheduler
except ImportError:  # Lightweight CPU CI exercises only pure schedule logic.
    class Trainer:  # type: ignore[no-redef]
        pass

    class TrainerCallback:  # type: ignore[no-redef]
        pass

    class TrainerControl:  # type: ignore[no-redef]
        pass

    def get_scheduler(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("Transformers is required to construct a scheduler")


@dataclass(frozen=True)
class SchedulePlan:
    updates_per_epoch: int
    scheduler_horizon_steps: int
    stop_after_steps: int
    checkpoint_steps: tuple[int, ...]


def build_schedule_plan(
    *,
    train_samples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    scheduler_epochs: float,
    stop_after_steps: int,
    save_steps: int,
) -> SchedulePlan:
    """Calculate a ceil-based update plan without shortening the LR horizon."""
    values = {
        "train_samples": train_samples,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "scheduler_epochs": scheduler_epochs,
        "stop_after_steps": stop_after_steps,
        "save_steps": save_steps,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"Schedule values must be positive: {values}")
    effective_batch = (
        per_device_batch_size * gradient_accumulation_steps * world_size
    )
    updates_per_epoch = math.ceil(train_samples / effective_batch)
    scheduler_horizon_steps = math.ceil(updates_per_epoch * scheduler_epochs)
    if stop_after_steps > scheduler_horizon_steps:
        raise ValueError("stop_after_steps cannot exceed the scheduler horizon")
    checkpoints = list(range(save_steps, stop_after_steps + 1, save_steps))
    if not checkpoints or checkpoints[-1] != stop_after_steps:
        checkpoints.append(stop_after_steps)
    return SchedulePlan(
        updates_per_epoch=updates_per_epoch,
        scheduler_horizon_steps=scheduler_horizon_steps,
        stop_after_steps=stop_after_steps,
        checkpoint_steps=tuple(checkpoints),
    )


class StopAtStepCallback(TrainerCallback):
    """Stop at a global step while leaving Trainer's scheduler horizon unchanged."""

    def __init__(self, stop_after_steps: int) -> None:
        if stop_after_steps <= 0:
            raise ValueError("stop_after_steps must be positive")
        self.stop_after_steps = stop_after_steps

    def _update(self, state: Any, control: TrainerControl) -> TrainerControl:
        if state.global_step >= self.stop_after_steps:
            control.should_training_stop = True
        return control

    def on_train_begin(
        self, args: Any, state: Any, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        return self._update(state, control)

    def on_step_end(
        self, args: Any, state: Any, control: TrainerControl, **kwargs: Any
    ) -> TrainerControl:
        if state.global_step >= self.stop_after_steps:
            control.should_save = True
        return self._update(state, control)


class HorizonTrainer(Trainer):
    """Trainer whose LR schedule uses an explicit full-epoch horizon."""

    def __init__(self, *args: Any, scheduler_horizon_steps: int, **kwargs: Any) -> None:
        if scheduler_horizon_steps <= 0:
            raise ValueError("scheduler_horizon_steps must be positive")
        self.scheduler_horizon_steps = scheduler_horizon_steps
        super().__init__(*args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: Any = None
    ) -> Any:
        if self.lr_scheduler is None:
            scheduler_optimizer = optimizer if optimizer is not None else self.optimizer
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=scheduler_optimizer,
                num_warmup_steps=self.args.get_warmup_steps(
                    self.scheduler_horizon_steps
                ),
                num_training_steps=self.scheduler_horizon_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        return self.lr_scheduler


def validate_resume_checkpoint(checkpoint: Path, stop_after_steps: int) -> int:
    """Reject a resume that has already reached the deterministic stop step."""
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Trainer state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    global_step = int(state.get("global_step", -1))
    if global_step < 0:
        raise ValueError(f"Invalid global_step in {state_path}")
    if global_step >= stop_after_steps:
        raise ValueError(
            f"Checkpoint global_step {global_step} has already reached "
            f"stop_after_steps {stop_after_steps}"
        )
    return global_step
