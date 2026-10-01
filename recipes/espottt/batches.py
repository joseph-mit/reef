"""The scheduled policy batch plus the history settings the entropic SPO-TTT preparer needs."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.espottt.history import DEFAULT_HISTORY_SIZE, DEFAULT_MIN_HISTORY, HistorySettings
from recipes.ppottt.batches import ScheduledPolicyBatch
from recipes.spottt.tracker import (
    DEFAULT_HALF_LIFE,
    DEFAULT_INHERIT_FRACTION,
    DEFAULT_RHO_MAX,
    DEFAULT_RHO_MIN,
    TrackerSettings,
)

BASELINES = ("node", "task", "siblings")


@dataclass(frozen=True)
class EspotttBatch(ScheduledPolicyBatch):
    """One search step for entropic SPO-TTT.

    ``baseline`` picks the comparison set. ``node`` uses the history of the
    archive state an attempt was expanded from, topped up from its parent's
    history, then the step's other attempts from the same state, then the
    task's history while it is thinner than ``min_history``. ``task`` uses
    the task's history. ``siblings`` uses only the step's other attempts from
    the same state, which is TTT-Discover's advantage; it is the control.
    ``pool_siblings`` adds the step's other attempts from the same state to
    the history even when the history is thick enough on its own. Under
    ``node`` and ``task`` the comparison set's weights are scaled to total
    ``comparison_size`` (TTT-Discover's ``group - 1`` siblings; ``0`` means
    the largest group in the batch less one, and at least 2), so the
    advantages keep TTT-Discover's scale however much history a key holds.
    Against a single outcome of weight 1 the KL target ``log 2`` is reached
    only as ``beta`` grows without bound, which is why 1 is refused. Each sample's
    ``extras`` carry its ``parent_id`` and ``grandparent_id``.
    """

    baseline: str = "node"
    pool_siblings: bool = False
    half_life: float = DEFAULT_HALF_LIFE
    rho_min: float = DEFAULT_RHO_MIN
    rho_max: float = DEFAULT_RHO_MAX
    inherit_fraction: float = DEFAULT_INHERIT_FRACTION
    history_size: int = DEFAULT_HISTORY_SIZE
    min_history: float = DEFAULT_MIN_HISTORY
    comparison_size: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.comparison_size, bool) or self.comparison_size < 0 or self.comparison_size == 1:
            raise ValueError("comparison_size must be 0 (the group size less one) or at least 2")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}")
        self.history_settings()

    def history_settings(self) -> HistorySettings:
        return HistorySettings(
            TrackerSettings(
                half_life=self.half_life,
                rho_min=self.rho_min,
                rho_max=self.rho_max,
                inherit_fraction=self.inherit_fraction,
            ),
            history_size=self.history_size,
            min_history=self.min_history,
        )
