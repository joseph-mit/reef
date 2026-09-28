"""PPO-TTT step preparer: schedule the step, leave the advantages to the critic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from recipes.ppottt.batches import ScheduledPolicyBatch
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch


@register_step_preparer
class PpotttPreparer(StepPreparer):
    name = "ppottt"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, ScheduledPolicyBatch):
            raise TypeError(f"{self.name} requires ScheduledPolicyBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        return StepSignal(
            "train",
            self.name,
            {"steps": steps},
            {
                "steps": steps,
                "rollouts": len(batch.samples),
                "ppo_epochs": batch.ppo_epochs,
                "minibatch_size": batch.minibatch_size or len(batch.samples),
            },
            scheduling=batch.scheduling(),
        )
