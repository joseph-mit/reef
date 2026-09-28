"""SPO-TTT's step-grid report: PPO-TTT's coordinates plus the attempt's parent states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from recipes.ppottt.report import PPOTTTRolloutReport

__all__ = ["SPOTTTRolloutReport"]


@dataclass(frozen=True)
class SPOTTTRolloutReport(PPOTTTRolloutReport):
    """One rollout plus the archive state it improved on and that state's parent.

    The tracker keys its estimates by ``parent_id``; ``grandparent_id`` is the
    prior a parent seen for the first time starts from. Both are empty for
    attempts expanded from a seed.
    """

    accepted_algorithms: ClassVar[tuple[str, ...]] = ("spottt", "spo-ttt")

    algorithm: str = "spottt"
    parent_id: str = ""
    grandparent_id: str = ""
