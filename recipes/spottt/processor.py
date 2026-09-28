"""SPO-TTT's processor: PPO-TTT's step barrier, with each attempt's parent ids attached."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from recipes.ppottt.processor import PPOTTTProcessor
from recipes.spottt.batches import BASELINES, SpotttBatch
from recipes.spottt.tracker import DEFAULT_HALF_LIFE, DEFAULT_INHERIT_FRACTION, DEFAULT_RHO_MAX, DEFAULT_RHO_MIN
from reef.train.types import PolicySample, ProcessorContext


class SPOTTTProcessor(PPOTTTProcessor):
    """Wait for one complete search step and hand it over with tracker settings."""

    output_schema = SpotttBatch
    batch_label = "spottt"

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.baseline = str(config.get("baseline", "node"))
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}")
        self.normalize = bool(config.get("normalize_advantages", True))
        self.half_life = float(config.get("half_life", DEFAULT_HALF_LIFE))
        self.rho_min = float(config.get("rho_min", DEFAULT_RHO_MIN))
        self.rho_max = float(config.get("rho_max", DEFAULT_RHO_MAX))
        self.inherit_fraction = float(config.get("inherit_fraction", DEFAULT_INHERIT_FRACTION))
        super().__init__(context)

    def make_row(self, sample: PolicySample, report: Any) -> PolicySample:
        return replace(
            sample,
            extras={**sample.extras, "parent_id": report.parent_id, "grandparent_id": report.grandparent_id},
        )

    def make_step_batch(self, step: Any, samples: tuple[PolicySample, ...]) -> SpotttBatch:
        return SpotttBatch(
            f"{self.scenario}:{self.batch_label}:{step}",
            samples,
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            shuffle=self.shuffle_minibatches,
            baseline=self.baseline,
            normalize=self.normalize,
            half_life=self.half_life,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            inherit_fraction=self.inherit_fraction,
        )
