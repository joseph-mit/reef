"""Entropic SPO-TTT's processor: PPO-TTT's step barrier, with each attempt's parent ids attached."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from recipes.espottt.batches import BASELINES, EspotttBatch
from recipes.espottt.history import DEFAULT_HISTORY_SIZE, DEFAULT_MIN_HISTORY
from recipes.ppottt.processor import PPOTTTProcessor
from recipes.spottt.tracker import DEFAULT_HALF_LIFE, DEFAULT_INHERIT_FRACTION, DEFAULT_RHO_MAX, DEFAULT_RHO_MIN
from reef.train.types import PolicySample, ProcessorContext


class ESPOTTTProcessor(PPOTTTProcessor):
    """Wait for one complete search step and hand it over with history settings."""

    output_schema = EspotttBatch
    batch_label = "espottt"

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.baseline = str(config.get("baseline", "node"))
        if self.baseline not in BASELINES:
            raise ValueError(f"baseline must be one of {', '.join(BASELINES)}, got {self.baseline!r}")
        self.pool_siblings = bool(config.get("pool_siblings", False))
        self.half_life = float(config.get("half_life", DEFAULT_HALF_LIFE))
        self.rho_min = float(config.get("rho_min", DEFAULT_RHO_MIN))
        self.rho_max = float(config.get("rho_max", DEFAULT_RHO_MAX))
        self.inherit_fraction = float(config.get("inherit_fraction", DEFAULT_INHERIT_FRACTION))
        self.history_size = int(config.get("history_size", DEFAULT_HISTORY_SIZE))
        self.min_history = float(config.get("min_history", DEFAULT_MIN_HISTORY))
        super().__init__(context)
        # TTT-Discover compares an attempt with its group's other siblings; at
        # least two, or the KL target is reachable only as beta grows without bound.
        self.comparison_size = int(config.get("comparison_size", 0)) or max(self.rollouts_per_group - 1, 2)

    def make_row(self, sample: PolicySample, report: Any) -> PolicySample:
        return replace(
            sample,
            extras={**sample.extras, "parent_id": report.parent_id, "grandparent_id": report.grandparent_id},
        )

    def make_step_batch(self, step: Any, samples: tuple[PolicySample, ...]) -> EspotttBatch:
        return EspotttBatch(
            f"{self.scenario}:{self.batch_label}:{step}",
            samples,
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            shuffle=self.shuffle_minibatches,
            baseline=self.baseline,
            pool_siblings=self.pool_siblings,
            half_life=self.half_life,
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            inherit_fraction=self.inherit_fraction,
            history_size=self.history_size,
            min_history=self.min_history,
            comparison_size=self.comparison_size,
        )
