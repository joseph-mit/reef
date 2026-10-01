"""Entropic SPO-TTT recipe: TTT-Discover's entropic advantage, with history in place of siblings."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.espottt.batches import BASELINES
from recipes.espottt.history import DEFAULT_HISTORY_SIZE, DEFAULT_MIN_HISTORY, HistorySettings
from recipes.espottt.processor import ESPOTTTProcessor
from recipes.espottt.report import ESPOTTTRolloutReport
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
class ESPOTTTRecipe(WeightTrainingRecipe):
    """TTT-Discover's objective with SPO-TTT's history.

    Each attempt gets TTT-Discover's entropic advantage, which favours the
    best attempts, but its comparison set is the forgetting history of
    attempts from the same archive state rather than the step's sibling group,
    so a state needs few siblings once its history is known. The loss is
    TTT-Discover's: the un-clipped importance-sampling surrogate with its
    centred KL to the frozen base, one optimizer step per search step by
    default (``minibatch_size: 0``).
    """

    name: str = "espottt"
    groups_per_step: int = config_field(8, env="REEF_ESPOTTT_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_ESPOTTT_ROLLOUTS_PER_GROUP")
    ppo_epochs: int = config_field(1, env="REEF_ESPOTTT_PPO_EPOCHS")
    minibatch_size: int = config_field(0, env="REEF_ESPOTTT_MINIBATCH_SIZE")
    shuffle_minibatches: bool = config_field(True, env="REEF_ESPOTTT_SHUFFLE_MINIBATCHES")
    baseline: str = config_field("node", env="REEF_ESPOTTT_BASELINE")
    pool_siblings: bool = config_field(False, env="REEF_ESPOTTT_POOL_SIBLINGS")
    half_life: float = config_field(DEFAULT_HALF_LIFE, env="REEF_ESPOTTT_HALF_LIFE")
    rho_min: float = config_field(DEFAULT_RHO_MIN, env="REEF_ESPOTTT_RHO_MIN")
    rho_max: float = config_field(DEFAULT_RHO_MAX, env="REEF_ESPOTTT_RHO_MAX")
    inherit_fraction: float = config_field(DEFAULT_INHERIT_FRACTION, env="REEF_ESPOTTT_INHERIT_FRACTION")
    history_size: int = config_field(DEFAULT_HISTORY_SIZE, env="REEF_ESPOTTT_HISTORY_SIZE")
    min_history: float = config_field(DEFAULT_MIN_HISTORY, env="REEF_ESPOTTT_MIN_HISTORY")
    # Total weight a history-based comparison set is scaled to; 0 means
    # rollouts_per_group - 1 (at least 2), the size of TTT-Discover's comparison set.
    comparison_size: int = config_field(0, env="REEF_ESPOTTT_COMPARISON_SIZE")

    @property
    def report_type(self) -> type[ReportBase]:
        return ESPOTTTRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="espottt", loss_family="tttd", processor=ESPOTTTProcessor)

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
        if self.comparison_size < 0 or self.comparison_size == 1:
            # Against one outcome of weight 1, KL = log 2 is reached only as beta grows without bound.
            raise ValueError("comparison_size must be 0 (rollouts_per_group - 1, at least 2) or at least 2")
        HistorySettings(
            TrackerSettings(self.half_life, self.rho_min, self.rho_max, self.inherit_fraction),
            history_size=self.history_size,
            min_history=self.min_history,
        )
