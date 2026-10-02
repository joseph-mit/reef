"""The step-grid rollout report shared by the single-stream TTT recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from reef.core.reports import ReportValidationError, ScoredRolloutReport

__all__ = ["PPOTTTRolloutReport"]


@dataclass(frozen=True)
class PPOTTTRolloutReport(ScoredRolloutReport):
    """One rollout addressed into a search step's (parent, attempt) grid.

    The coordinates are the same as TTT-Discover's, so the same PUCT harness
    and run controller drive this recipe, but they carry no comparison-group
    meaning: every rollout is its own training sample. ``rollouts_per_group``
    may therefore be 1. Undeclared metadata the harness sends (parent ids,
    search values) passes through untouched.

    ``step_groups`` is how many grid rows this step fills when the harness
    runs a smaller step than the grid (its step-size policy); absent, the
    step fills all ``groups_per_step`` rows.
    """

    accepted_algorithms: ClassVar[tuple[str, ...]] = ("ppottt", "ppo-ttt")

    score: float
    step: int
    group: int
    rollout: int
    groups_per_step: int
    rollouts_per_group: int
    algorithm: str = "ppottt"
    step_groups: int | None = None

    @property
    def groups_in_step(self) -> int:
        """The grid rows this report's step fills."""
        return self.groups_per_step if self.step_groups is None else self.step_groups

    def validate(self) -> None:
        if self.algorithm not in self.accepted_algorithms:
            accepted = " or ".join(repr(name) for name in self.accepted_algorithms)
            raise ReportValidationError(f"metadata.algorithm must be {accepted}")
        if self.step < 0 or self.groups_per_step < 1 or self.rollouts_per_group < 1:
            raise ReportValidationError(
                "metadata grid requires step >= 0, groups_per_step >= 1, rollouts_per_group >= 1"
            )
        if self.step_groups is not None and (
            isinstance(self.step_groups, bool)
            or not isinstance(self.step_groups, int)
            or not 1 <= self.step_groups <= self.groups_per_step
        ):
            raise ReportValidationError("metadata.step_groups must be between 1 and metadata.groups_per_step")
        if not 0 <= self.group < self.groups_in_step:
            raise ReportValidationError("metadata.group must sit inside the step's rows")
        if not 0 <= self.rollout < self.rollouts_per_group:
            raise ReportValidationError("metadata.rollout must sit inside metadata.rollouts_per_group")
