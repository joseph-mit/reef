"""The scheduled policy batch plus the tracker settings the SPO-TTT preparer needs."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.ppottt.batches import ScheduledPolicyBatch
from recipes.spottt.tracker import (
    DEFAULT_HALF_LIFE,
    DEFAULT_INHERIT_FRACTION,
    DEFAULT_RHO_MAX,
    DEFAULT_RHO_MIN,
    TrackerSettings,
)

BASELINES = ("node", "task", "none")


@dataclass(frozen=True)
class SpotttBatch(ScheduledPolicyBatch):
    """One search step for SPO-TTT.

    ``baseline`` picks where historical outcomes are shared: ``node`` keys the
    tracker by the archive state an attempt was expanded from, ``task`` keeps
    one estimate for the whole test instance, and ``none`` subtracts nothing
    (SPO's no-baseline ablation). Each sample's ``extras`` carry its
    ``parent_id`` and ``grandparent_id`` from the report.
    """

    baseline: str = "node"
    normalize: bool = True
    half_life: float = DEFAULT_HALF_LIFE
    rho_min: float = DEFAULT_RHO_MIN
    rho_max: float = DEFAULT_RHO_MAX
    inherit_fraction: float = DEFAULT_INHERIT_FRACTION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}")
        self.tracker_settings()

    def tracker_settings(self) -> TrackerSettings:
        return TrackerSettings(
            half_life=self.half_life,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            inherit_fraction=self.inherit_fraction,
        )
