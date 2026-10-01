"""Entropic SPO-TTT's report: SPO-TTT's grid coordinates and parent ids, under its own tag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from recipes.spottt.report import SPOTTTRolloutReport

__all__ = ["ESPOTTTRolloutReport"]


@dataclass(frozen=True)
class ESPOTTTRolloutReport(SPOTTTRolloutReport):
    """One rollout, the archive state it improved on, and that state's parent."""

    accepted_algorithms: ClassVar[tuple[str, ...]] = ("espottt", "espo-ttt")

    algorithm: str = "espottt"
