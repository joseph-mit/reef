"""Stepping-stone SPO-TTT's report: SPO-TTT's, plus the programs that link attempts across steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from recipes.spottt.report import SPOTTTRolloutReport

__all__ = ["HSPOTTTRolloutReport"]


@dataclass(frozen=True)
class HSPOTTTRolloutReport(SPOTTTRolloutReport):
    """One rollout, its archive state and that state's parent, and two program keys.

    ``solution_sha1`` names the attempt's own program (empty when it wrote
    none) and ``parent_solution_sha1`` the program of the state it was
    expanded from (empty for a seed). A later attempt whose parent's key is
    this attempt's key descends from it.
    """

    accepted_algorithms: ClassVar[tuple[str, ...]] = ("hspottt", "hspo-ttt")

    algorithm: str = "hspottt"
    solution_sha1: str = ""
    parent_solution_sha1: str = ""
