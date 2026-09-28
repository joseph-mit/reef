"""SPO-TTT recipe: TTT-Discover's search, trained against a baseline built from past outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.spottt.batches import BASELINES
from recipes.spottt.processor import SPOTTTProcessor
from recipes.spottt.report import SPOTTTRolloutReport
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
class SPOTTTRecipe(WeightTrainingRecipe):
    """Direction II of the single-stream TTT study.

    No critic and no sibling group: each attempt's baseline is a forgetting
    estimate of what attempts from the same archive state (``baseline:
    node``) or the whole task (``baseline: task``) have scored, the rule of
    Single-stream Policy Optimization with archive states in place of
    prompts. The advantages are computed on the Reef side and trained with
    Slime's clipped policy loss; the update schedule fields match PPO-TTT's.
    """

    name: str = "spottt"
    groups_per_step: int = config_field(8, env="REEF_SPOTTT_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_SPOTTT_ROLLOUTS_PER_GROUP")
    ppo_epochs: int = config_field(1, env="REEF_SPOTTT_PPO_EPOCHS")
    minibatch_size: int = config_field(128, env="REEF_SPOTTT_MINIBATCH_SIZE")
    shuffle_minibatches: bool = config_field(True, env="REEF_SPOTTT_SHUFFLE_MINIBATCHES")
    baseline: str = config_field("node", env="REEF_SPOTTT_BASELINE")
    normalize_advantages: bool = config_field(True, env="REEF_SPOTTT_NORMALIZE_ADVANTAGES")
    half_life: float = config_field(DEFAULT_HALF_LIFE, env="REEF_SPOTTT_HALF_LIFE")
    rho_min: float = config_field(DEFAULT_RHO_MIN, env="REEF_SPOTTT_RHO_MIN")
    rho_max: float = config_field(DEFAULT_RHO_MAX, env="REEF_SPOTTT_RHO_MAX")
    inherit_fraction: float = config_field(DEFAULT_INHERIT_FRACTION, env="REEF_SPOTTT_INHERIT_FRACTION")

    @property
    def report_type(self) -> type[ReportBase]:
        return SPOTTTRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="spottt", loss_family="spottt", processor=SPOTTTProcessor)

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
