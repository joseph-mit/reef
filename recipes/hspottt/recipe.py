"""Stepping-stone SPO-TTT recipe: SPO-TTT, plus credit for attempts that later led to a success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recipes.hspottt.processor import HSPOTTTProcessor
from recipes.hspottt.report import HSPOTTTRolloutReport
from recipes.hspottt.stones import (
    DEFAULT_CREDIT,
    DEFAULT_DISCOUNT,
    DEFAULT_GENERATIONS,
    DEFAULT_KEEP_PER_PARENT,
    DEFAULT_SUCCESS,
    DEFAULT_TOLERANCE,
    DEFAULT_TOP_FRACTION,
    DEFAULT_WINDOW,
    StoneSettings,
)
from recipes.spottt.batches import BASELINES
from recipes.spottt.tracker import (
    DEFAULT_HALF_LIFE,
    DEFAULT_INHERIT_FRACTION,
    DEFAULT_RHO_MAX,
    DEFAULT_RHO_MIN,
    TrackerSettings,
)
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class HSPOTTTRecipe(WeightTrainingRecipe):
    """SPO-TTT with hindsight credit for stepping stones.

    Every step's own attempts get exactly SPO-TTT's advantages. In addition,
    an attempt that a later successful attempt descends from, and that did
    not succeed itself, is trained once more ``stone_window`` steps after it
    was sampled, with the credit it earned as its advantage. Those replays are ``stone_window``
    policy versions old, so ``max_staleness`` must be at least that: the
    trainer refuses any batch holding an older sample.
    """

    name: str = "hspottt"
    groups_per_step: int = config_field(8, env="REEF_HSPOTTT_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_HSPOTTT_ROLLOUTS_PER_GROUP")
    ppo_epochs: int = config_field(1, env="REEF_HSPOTTT_PPO_EPOCHS")
    minibatch_size: int = config_field(128, env="REEF_HSPOTTT_MINIBATCH_SIZE")
    shuffle_minibatches: bool = config_field(True, env="REEF_HSPOTTT_SHUFFLE_MINIBATCHES")
    baseline: str = config_field("node", env="REEF_HSPOTTT_BASELINE")
    normalize_advantages: bool = config_field(True, env="REEF_HSPOTTT_NORMALIZE_ADVANTAGES")
    half_life: float = config_field(DEFAULT_HALF_LIFE, env="REEF_HSPOTTT_HALF_LIFE")
    rho_min: float = config_field(DEFAULT_RHO_MIN, env="REEF_HSPOTTT_RHO_MIN")
    rho_max: float = config_field(DEFAULT_RHO_MAX, env="REEF_HSPOTTT_RHO_MAX")
    inherit_fraction: float = config_field(DEFAULT_INHERIT_FRACTION, env="REEF_HSPOTTT_INHERIT_FRACTION")
    stone_window: int = config_field(DEFAULT_WINDOW, env="REEF_HSPOTTT_STONE_WINDOW")
    stone_generations: int = config_field(DEFAULT_GENERATIONS, env="REEF_HSPOTTT_STONE_GENERATIONS")
    stone_discount: float = config_field(DEFAULT_DISCOUNT, env="REEF_HSPOTTT_STONE_DISCOUNT")
    stone_credit: float = config_field(DEFAULT_CREDIT, env="REEF_HSPOTTT_STONE_CREDIT")
    stone_success: str = config_field(DEFAULT_SUCCESS, env="REEF_HSPOTTT_STONE_SUCCESS")
    stone_top_fraction: float = config_field(DEFAULT_TOP_FRACTION, env="REEF_HSPOTTT_STONE_TOP_FRACTION")
    stone_keep_per_parent: int = config_field(DEFAULT_KEEP_PER_PARENT, env="REEF_HSPOTTT_STONE_KEEP_PER_PARENT")
    stone_tolerance: float = config_field(DEFAULT_TOLERANCE, env="REEF_HSPOTTT_STONE_TOLERANCE")
    # Where the store a step is planned from is kept, so a step built again after
    # a reload or restart is identical; empty keeps it in memory only.
    stone_dir: str = config_field("", env="REEF_HSPOTTT_STONE_DIR")

    @property
    def report_type(self) -> type[ReportBase]:
        return HSPOTTTRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="hspottt", loss_family="spottt", processor=HSPOTTTProcessor)

    def processor_config(self) -> dict[str, Any]:
        # The processor leaves out a replay the trainer would refuse, so it
        # needs the trainer's staleness bound.
        return {**super().processor_config(), "replay_max_lag": self.max_staleness}

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0 or self.rollouts_per_group <= 0:
            raise ValueError("groups_per_step and rollouts_per_group must be positive")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        step_size = self.groups_per_step * self.rollouts_per_group
        if not 0 <= self.minibatch_size <= step_size:
            raise ValueError(f"minibatch_size must lie in [0, {step_size}]")
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}")
        TrackerSettings(self.half_life, self.rho_min, self.rho_max, self.inherit_fraction)
        StoneSettings(
            window=self.stone_window,
            generations=self.stone_generations,
            discount=self.stone_discount,
            credit=self.stone_credit,
            success=self.stone_success,
            top_fraction=self.stone_top_fraction,
            keep_per_parent=self.stone_keep_per_parent,
            tolerance=self.stone_tolerance,
        )
        if self.max_staleness < self.stone_window:
            raise ValueError(
                f"replays are stone_window={self.stone_window} policy versions old; "
                f"set max_staleness to at least {self.stone_window} (it is {self.max_staleness})"
            )
