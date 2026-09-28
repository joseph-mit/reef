"""SPO-TTT step preparer: advantages from the forgetting tracker, persisted with the commit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from recipes.spottt.batches import SpotttBatch
from recipes.spottt.tracker import (
    TASK_KEY,
    ForgettingTracker,
    Observation,
    assign_advantages,
    batch_normalize,
    node_key,
)
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import PolicySample, TrainingBatch

STATE_KEY = "spottt"


def observation_for(sample: PolicySample, baseline: str) -> Observation:
    """Where one attempt's reward is compared and recorded.

    Under ``node`` an attempt is judged against the attempts expanded from
    the same archive state; a state seen for the first time borrows its own
    parent's estimate, then the task's.
    """
    if baseline == "task":
        return Observation(TASK_KEY, sample.reward)
    parent_id = str(sample.extras.get("parent_id") or "")
    if not parent_id:
        return Observation(TASK_KEY, sample.reward)
    grandparent_id = str(sample.extras.get("grandparent_id") or "")
    prior = node_key(grandparent_id) if grandparent_id else None
    fallbacks = (prior, TASK_KEY) if prior is not None else (TASK_KEY,)
    return Observation(node_key(parent_id), sample.reward, prior_key=prior, fallbacks=fallbacks)


@register_step_preparer
class SpotttPreparer(StepPreparer):
    """One advantage per attempt, no siblings needed.

    The policy version is the committed step count, so the tracker's drift
    between two updates of a key is the number of policy updates in between.
    The tracker is read from and written back to the algorithm state Reef
    commits with each training step, so a restart resumes it exactly.
    """

    name = "spottt"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, SpotttBatch):
            raise TypeError(f"{self.name} requires SpotttBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        version = steps - 1
        rewards = [sample.reward for sample in batch.samples]
        if batch.baseline == "none":
            advantages = batch_normalize(rewards) if batch.normalize else tuple(rewards)
            tracker_state = dict(state.get(STATE_KEY) or {})
            metrics: dict[str, Any] = {"tracked_fraction": 0.0}
        else:
            tracker = ForgettingTracker(batch.tracker_settings(), state.get(STATE_KEY))
            observations = [observation_for(sample, batch.baseline) for sample in batch.samples]
            advantages, metrics = assign_advantages(tracker, observations, version, normalize=batch.normalize)
            tracker_state = tracker.state_dict()
        return StepSignal(
            "train",
            self.name,
            {"steps": steps, STATE_KEY: tracker_state},
            {"steps": steps, "rollouts": len(batch.samples), **metrics},
            advantages=tuple(advantages),
            scheduling=batch.scheduling(),
        )
