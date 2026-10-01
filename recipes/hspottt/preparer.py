"""Stepping-stone SPO-TTT step preparer: SPO-TTT's advantages for the step, credit for the replays."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from recipes.hspottt.batches import HspotttBatch
from recipes.spottt.preparer import observation_for
from recipes.spottt.tracker import ForgettingTracker, assign_advantages, batch_normalize
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch

STATE_KEY = "hspottt"
LOSS_FAMILY = "spottt"


def _std(values: list[float]) -> float:
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))


@register_step_preparer
class HspotttPreparer(StepPreparer):
    """SPO-TTT on the step's own attempts; each replay's advantage is its credit.

    The fresh attempts get SPO-TTT's advantages exactly, and only they update
    the tracker: a replay's reward was observed when it was sampled. A
    replay's credit is in units of the fresh advantages: as given when they
    are normalised, times their raw spread when they are not. Replays train
    with SPO-TTT's clipped loss, which bounds how far their stale
    log-probabilities can move the policy.
    """

    name = "hspottt"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, HspotttBatch):
            raise TypeError(f"{self.name} requires HspotttBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        version = steps - 1
        fresh = batch.samples[: batch.fresh_count]
        rewards = [sample.reward for sample in fresh]
        if batch.baseline == "none":
            advantages = batch_normalize(rewards) if batch.normalize else tuple(rewards)
            tracker_state = dict(state.get(STATE_KEY) or {})
            metrics: dict[str, Any] = {"tracked_fraction": 0.0}
            spread = _std(rewards)
        else:
            tracker = ForgettingTracker(batch.tracker_settings(), state.get(STATE_KEY))
            observations = [observation_for(sample, batch.baseline) for sample in fresh]
            advantages, metrics = assign_advantages(tracker, observations, version, normalize=batch.normalize)
            tracker_state = tracker.state_dict()
            spread = float(metrics.get("raw_advantage_std", 0.0))
        scale = 1.0 if batch.normalize else spread
        replay = tuple(credit * scale for credit in batch.replay_credits)
        metrics = {
            "steps": steps,
            "rollouts": len(fresh),
            **metrics,
            "replays": len(replay),
            "replay_credit_mean": math.fsum(batch.replay_credits) / len(replay) if replay else 0.0,
        }
        return StepSignal(
            "train",
            LOSS_FAMILY,
            {"steps": steps, STATE_KEY: tracker_state},
            metrics,
            advantages=(*advantages, *replay),
            scheduling=batch.scheduling(),
        )
