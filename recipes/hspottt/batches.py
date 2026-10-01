"""SPO-TTT's batch plus the replayed stepping stones that ride at its end."""

from __future__ import annotations

import math
from dataclasses import dataclass

from recipes.spottt.batches import SpotttBatch
from reef.train.algos import StepScheduling


@dataclass(frozen=True)
class HspotttBatch(SpotttBatch):
    """One search step for SPO-TTT, with earlier attempts appended for a second, credited update.

    The last ``len(replay_credits)`` samples are replays: attempts from
    earlier steps whose descendants later succeeded, each with the credit it
    earned, in units of the fresh advantages. The rest are the step's own
    attempts, which train exactly as in SPO-TTT.
    """

    replay_credits: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.replay_credits) >= len(self.samples):
            raise ValueError("a batch needs at least one fresh sample besides its replays")
        if any(not math.isfinite(credit) or credit <= 0 for credit in self.replay_credits):
            raise ValueError("replay credits must be positive and finite")

    @property
    def fresh_count(self) -> int:
        return len(self.samples) - len(self.replay_credits)

    def scheduling(self) -> StepScheduling:
        """SPO-TTT's optimizer steps, widened evenly to take the replays.

        The fresh samples set the number of optimizer steps (``ceil(fresh /
        minibatch_size)``, as SPO-TTT takes without replays) and every step
        grows by the same share of the replays, so no step is a small tail
        that would weigh a few samples as much as a full step.
        """
        scheduling = super().scheduling()
        if not self.minibatch_size or not self.replay_credits:
            return scheduling
        steps = math.ceil(self.fresh_count / self.minibatch_size)
        return StepScheduling(
            unit=scheduling.unit,
            batch_size=math.ceil(len(self.samples) / steps),
            epochs=scheduling.epochs,
            shuffle=scheduling.shuffle,
        )
