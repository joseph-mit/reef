"""PPO-TTT recipe: TTT-Discover's search, trained with a clipped policy loss and a scalar critic."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.ppottt.processor import PPOTTTProcessor
from recipes.ppottt.report import PPOTTTRolloutReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field


@dataclass(frozen=True, kw_only=True)
class PPOTTTRecipe(WeightTrainingRecipe):
    """Direction I of the single-stream TTT study.

    Reef owns the step grid and the update schedule; the objective (clip
    range, GAE parameters, critic cadence, KL coefficient) belongs to the
    Slime driver flags, as it does for SAO.

    ``groups_per_step * rollouts_per_group`` must equal the Slime driver's
    ``--global-batch-size``, and ``minibatch_size`` (rollouts per optimizer
    step, ``0`` for the whole step) must divide it for equal-sized updates.
    """

    name: str = "ppottt"
    groups_per_step: int = config_field(8, env="REEF_PPOTTT_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_PPOTTT_ROLLOUTS_PER_GROUP")
    ppo_epochs: int = config_field(1, env="REEF_PPOTTT_PPO_EPOCHS")
    minibatch_size: int = config_field(128, env="REEF_PPOTTT_MINIBATCH_SIZE")
    shuffle_minibatches: bool = config_field(True, env="REEF_PPOTTT_SHUFFLE_MINIBATCHES")

    @property
    def report_type(self) -> type[ReportBase]:
        return PPOTTTRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="ppottt", loss_family="ppottt", processor=PPOTTTProcessor)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group <= 0:
            raise ValueError("rollouts_per_group must be positive")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if self.minibatch_size < 0:
            raise ValueError("minibatch_size must be non-negative")
        step_size = self.groups_per_step * self.rollouts_per_group
        if self.minibatch_size > step_size:
            raise ValueError(f"minibatch_size {self.minibatch_size} exceeds the step's {step_size} rollouts")
