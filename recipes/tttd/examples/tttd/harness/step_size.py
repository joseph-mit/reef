"""How many parents a search step expands, and when a run has spent its attempts.

TTT-Discover expands the same ``groups_per_step`` parents every step. Once the
best score stops moving, a smaller step trains the policy more often for the
same number of attempts. With ``policy: stall``, a step expands ``min_groups``
parents (``rollouts_per_group`` attempts each) while the best score has not
improved by more than ``tolerance`` for ``patience`` steps in a row, and the
full ``groups_per_step`` otherwise. ``attempt_budget`` ends the run at the
first step boundary where the attempts made reach it, so a run with smaller
steps spends the same budget over more training steps.

The choice reads only the per-step history the search archive saves with
each committed step, so a step repeated after a restart is planned the same
way, and every report of a step announces the same size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICIES = ("fixed", "stall")


@dataclass(frozen=True)
class StepRecord:
    """What one finished step did: its attempts and the best score before and after it."""

    step: int
    attempts: int
    best_before: float
    best_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "attempts": self.attempts,
            "best_before": self.best_before,
            "best_after": self.best_after,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> StepRecord:
        record = cls(int(row["step"]), int(row["attempts"]), float(row["best_before"]), float(row["best_after"]))
        if record.step < 0 or record.attempts < 0:
            raise ValueError(f"invalid step record {dict(row)!r}")
        return record


@dataclass(frozen=True)
class StepSizeSettings:
    """The config's ``search.step_size`` block; the defaults keep TTT-Discover's fixed steps."""

    policy: str = "fixed"
    patience: int = 3
    min_groups: int = 4
    tolerance: float = 1e-9
    attempt_budget: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"step_size.policy must be one of {', '.join(POLICIES)}, got {self.policy!r}")
        if self.patience < 1:
            raise ValueError("step_size.patience must be at least 1")
        if self.min_groups < 1:
            raise ValueError("step_size.min_groups must be at least 1")
        if self.tolerance < 0:
            raise ValueError("step_size.tolerance must be non-negative")
        if self.attempt_budget is not None and self.attempt_budget < 1:
            raise ValueError("step_size.attempt_budget must be positive")

    @property
    def varies(self) -> bool:
        """Whether a step can be smaller than the configured grid."""
        return self.policy != "fixed"

    @property
    def active(self) -> bool:
        return self.varies or self.attempt_budget is not None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> StepSizeSettings:
        if not mapping:
            return cls()
        known = {"policy", "patience", "min_groups", "tolerance", "attempt_budget"}
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"unknown step_size settings: {', '.join(sorted(unknown))}")
        budget = mapping.get("attempt_budget")
        return cls(
            policy=str(mapping.get("policy", "fixed")),
            patience=int(mapping.get("patience", 3)),
            min_groups=int(mapping.get("min_groups", 4)),
            tolerance=float(mapping.get("tolerance", 1e-9)),
            attempt_budget=None if budget is None else int(budget),
        )

    def identity(self) -> tuple[tuple[str, Any], ...] | None:
        """The run identity's entry: None for TTT-Discover's fixed steps, so older runs keep theirs."""
        if not self.active:
            return None
        return (
            ("policy", self.policy),
            ("patience", self.patience),
            ("min_groups", self.min_groups),
            ("tolerance", self.tolerance),
            ("attempt_budget", self.attempt_budget),
        )

    def groups_for(self, history: Sequence[StepRecord], groups_per_step: int) -> int:
        """How many parents the next step expands, from the steps finished so far."""
        if not self.varies:
            return groups_per_step
        if self.min_groups > groups_per_step:
            raise ValueError(f"step_size.min_groups {self.min_groups} exceeds groups_per_step {groups_per_step}")
        return self.min_groups if stalled_steps(history, self.tolerance) >= self.patience else groups_per_step

    def spent(self, history: Sequence[StepRecord]) -> bool:
        """Whether the attempts made so far reach the budget."""
        return self.attempt_budget is not None and sum(record.attempts for record in history) >= self.attempt_budget


def stalled_steps(history: Sequence[StepRecord], tolerance: float) -> int:
    """How many of the latest steps in a row did not raise the best score by more than ``tolerance``."""
    count = 0
    for record in reversed(history):
        if record.best_after > record.best_before + tolerance:
            break
        count += 1
    return count
