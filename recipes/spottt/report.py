"""SPO-TTT's step-grid report: PPO-TTT's coordinates plus the attempt's parent states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from recipes.ppottt.report import PPOTTTRolloutReport

__all__ = ["SPOTTTRolloutReport"]


@dataclass(frozen=True)
class SPOTTTRolloutReport(PPOTTTRolloutReport):
    """One rollout plus the archive state it improved on and that state's parent.

    The tracker keys its estimates by ``parent_id``, the archive state the
    attempt expanded, seeds included; ``grandparent_id`` is the prior a state
    seen for the first time starts from, empty for a seed or a pruned parent,
    which then falls back to the task-level estimate.
    """

    accepted_algorithms: ClassVar[tuple[str, ...]] = ("spottt", "spo-ttt")

    algorithm: str = "spottt"
    parent_id: str = ""
    grandparent_id: str = ""
