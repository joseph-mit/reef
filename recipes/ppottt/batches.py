"""A policy batch that carries its own update schedule to the preparer."""

from __future__ import annotations

from dataclasses import dataclass

from reef.train.algos import StepScheduling
from reef.train.types import PolicyBatch


@dataclass(frozen=True)
class ScheduledPolicyBatch(PolicyBatch):
    """One search step's rollouts plus how the optimizer should walk them.

    Preparers see only the batch and the committed algorithm state, never the
    recipe config, so the processor stamps the PPO update schedule here.
    ``minibatch_size`` counts rollouts per optimizer step; ``0`` means one
    step over the whole batch per epoch.
    """

    ppo_epochs: int = 1
    minibatch_size: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.ppo_epochs, bool) or self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be a positive integer")
        if isinstance(self.minibatch_size, bool) or self.minibatch_size < 0:
            raise ValueError("minibatch_size must be a non-negative integer")

    def scheduling(self) -> StepScheduling:
        return StepScheduling(
            unit="sample",
            batch_size=self.minibatch_size or "actual",
            epochs=self.ppo_epochs,
            shuffle=self.shuffle,
        )
