from types import SimpleNamespace

import pytest

from snuaichal.scheduling import (
    StopAtStepCallback,
    build_schedule_plan,
    validate_resume_checkpoint,
)


def make_control() -> SimpleNamespace:
    return SimpleNamespace(should_training_stop=False, should_save=False)


def test_scheduler_horizon_is_six_epochs_while_stop_is_four_epochs() -> None:
    plan = build_schedule_plan(
        train_samples=8581,
        per_device_batch_size=1,
        gradient_accumulation_steps=8,
        world_size=1,
        scheduler_epochs=6,
        stop_after_steps=4292,
        save_steps=1073,
    )

    assert plan.updates_per_epoch == 1073
    assert plan.scheduler_horizon_steps == 6438
    assert plan.stop_after_steps == 4292
    assert plan.checkpoint_steps == (1073, 2146, 3219, 4292)


def test_stop_callback_stops_exactly_at_requested_resumed_global_step() -> None:
    callback = StopAtStepCallback(4292)
    control = make_control()

    before = callback.on_step_end(
        None, SimpleNamespace(global_step=4291), control
    )
    assert before.should_training_stop is False

    at_stop = callback.on_step_end(
        None, SimpleNamespace(global_step=4292), make_control()
    )
    assert at_stop.should_training_stop is True
    assert at_stop.should_save is True


def test_stop_callback_honors_checkpoint_already_at_stop() -> None:
    callback = StopAtStepCallback(4292)

    control = callback.on_train_begin(
        None, SimpleNamespace(global_step=4292), make_control()
    )

    assert control.should_training_stop is True


def test_resume_checkpoint_is_rejected_at_stop_step(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-4292"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 4292}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="already reached"):
        validate_resume_checkpoint(checkpoint, 4292)
