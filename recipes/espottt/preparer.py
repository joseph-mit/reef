"""Entropic SPO-TTT step preparer: TTT-Discover's entropic advantage against a reward history."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from recipes.espottt.batches import EspotttBatch
from recipes.espottt.history import RewardHistory, Weighted, entropic_advantage, total_weight
from recipes.spottt.preparer import observation_for
from recipes.spottt.tracker import TASK_KEY, Observation
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch

STATE_KEY = "espottt"
# The deepest source a comparison set drew on, in the order they are tried.
SOURCES = ("own", "prior", "siblings", "task", "step", "none")
LOSS_FAMILY = "tttd"


def _without_one(rewards: Sequence[float], reward: float) -> list[Weighted]:
    """``rewards`` as weight-1 outcomes with one occurrence of ``reward`` removed (leave-one-out)."""
    remaining = list(rewards)
    remaining.remove(reward)
    return [(value, 1.0) for value in remaining]


def comparison_set(
    history: RewardHistory,
    observation: Observation,
    version: int,
    *,
    siblings: Sequence[float],
    others: Callable[[], Sequence[float]],
    baseline: str,
    pool_siblings: bool,
) -> tuple[list[Weighted], str]:
    """The outcomes an attempt is compared with, and the deepest source used.

    ``siblings`` are this step's rewards from the same archive state,
    including the attempt's own; ``others`` returns this step's rewards from
    other states, and is called only when they are needed. Sources are added
    in order until the set reaches ``min_history`` effective outcomes: the
    state's history, its parent's history (with ``inherit_fraction`` of its
    weight), the step's other attempts from the same state, the task's
    history, then the rest of the step.
    """
    reward = observation.reward
    minimum = history.settings.min_history
    if baseline == "siblings":
        group = _without_one(siblings, reward)
        if group:
            return group, "siblings"
        rest = [(value, 1.0) for value in others()]
        return rest, "step" if rest else "none"

    outcomes: list[Weighted] = []
    source = "none"

    def add(extra: Sequence[Weighted], name: str) -> None:
        nonlocal source
        if extra:
            outcomes.extend(extra)
            source = name

    add(history.outcomes_at(observation.key, version), "own")
    prior = observation.prior_key
    if total_weight(outcomes) < minimum and prior is not None and prior != observation.key:
        share = history.settings.forgetting.inherit_fraction
        add([(value, weight * share) for value, weight in history.outcomes_at(prior, version)], "prior")
    if pool_siblings or total_weight(outcomes) < minimum:
        add(_without_one(siblings, reward), "siblings")
    if total_weight(outcomes) < minimum and TASK_KEY not in (observation.key, prior):
        add(history.outcomes_at(TASK_KEY, version), "task")
    if total_weight(outcomes) < minimum:
        add([(value, 1.0) for value in others()], "step")
    return outcomes, source


def rescaled(outcomes: Sequence[Weighted], size: float) -> list[Weighted]:
    """``outcomes`` with their weights scaled to total ``size``.

    TTT-Discover's advantage is a ratio of the attempt's weight to a
    comparison set of ``group - 1`` siblings, so the set's total weight sets
    the advantage's scale: one breakthrough among ``n`` equal outcomes gets
    about ``n / 2``. A history can hold far more effective outcomes than a
    sibling group (the task key gains a whole step at every version), which
    would inflate the advantages, and with them the un-clipped loss, by the
    same factor. Scaling the set to the group's size keeps the shape the
    history has learnt and TTT-Discover's scale.
    """
    total = total_weight(outcomes)
    if total <= 0 or size <= 0:
        return list(outcomes)
    factor = size / total
    return [(value, weight * factor) for value, weight in outcomes]


@register_step_preparer
class EspotttPreparer(StepPreparer):
    """One entropic advantage per attempt; the history rides the committed algorithm state.

    Every comparison set is read before the step updates any key, so an
    attempt is never compared with itself, as in TTT-Discover's leave-one-out
    normaliser and SPO-TTT's pre-update baseline. The policy version is the
    committed step count, so forgetting counts policy updates.
    """

    name = "espottt"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, EspotttBatch):
            raise TypeError(f"{self.name} requires EspotttBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        version = steps - 1
        history = RewardHistory(batch.history_settings(), state.get(STATE_KEY))
        key_mode = "task" if batch.baseline == "task" else "node"
        observations = [observation_for(sample, key_mode) for sample in batch.samples]

        by_key: dict[str, list[float]] = {}
        for observation in observations:
            by_key.setdefault(observation.key, []).append(observation.reward)
        # TTT-Discover's comparison set is a parent's other siblings.
        comparison_size = float(
            batch.comparison_size or max(max((len(rewards) for rewards in by_key.values()), default=3) - 1, 2)
        )

        advantages: list[float] = []
        betas: list[float] = []
        weights: list[float] = []
        sources = dict.fromkeys(SOURCES, 0)
        # The comparison set depends only on the attempt's key and reward (its
        # prior and fallbacks follow from the key), so attempts that share both,
        # failures above all, are computed once.
        cache: dict[tuple[str, float], tuple[float, float, float, str]] = {}
        for observation in observations:
            cached = cache.get((observation.key, observation.reward))
            if cached is None:
                key = observation.key

                def others(key: str = key) -> list[float]:
                    return [value for other, values in by_key.items() if other != key for value in values]

                outcomes, source = comparison_set(
                    history,
                    observation,
                    version,
                    siblings=by_key[key],
                    others=others,
                    baseline=batch.baseline,
                    pool_siblings=batch.pool_siblings,
                )
                weight = total_weight(outcomes)
                if weight > 0:
                    if batch.baseline != "siblings":
                        outcomes = rescaled(outcomes, comparison_size)
                    advantage, beta = entropic_advantage(observation.reward, outcomes)
                else:
                    advantage, beta = 0.0, 0.0
                cached = (advantage, beta, weight, source)
                cache[(observation.key, observation.reward)] = cached
            advantage, beta, weight, source = cached
            advantages.append(advantage)
            betas.append(beta)
            weights.append(weight)
            sources[source] += 1

        self._update(history, observations, version)
        count = max(len(advantages), 1)
        metrics: dict[str, Any] = {
            "steps": steps,
            "rollouts": len(batch.samples),
            "history_keys": float(len(history)),
            "comparison_weight_mean": math.fsum(weights) / count,
            **{f"comparison_from_{name}_fraction": sources[name] / count for name in SOURCES},
        }
        if advantages:
            metrics.update(
                advantage_mean=math.fsum(advantages) / count,
                advantage_abs_mean=math.fsum(abs(value) for value in advantages) / count,
                advantage_max=max(advantages),
                advantage_min=min(advantages),
                advantage_positive_fraction=sum(value > 0 for value in advantages) / count,
                beta_median=statistics.median(betas),
                beta_max=max(betas),
            )
        return StepSignal(
            "train",
            LOSS_FAMILY,
            {"steps": steps, STATE_KEY: history.state_dict()},
            metrics,
            advantages=tuple(advantages),
            scheduling=batch.scheduling(),
        )

    @staticmethod
    def _update(history: RewardHistory, observations: Sequence[Observation], version: int) -> None:
        """Fold the step into the history: cold keys start from their parent's history as it stood."""
        grouped: dict[str, tuple[str | None, list[float]]] = {}
        for observation in observations:
            grouped.setdefault(observation.key, (observation.prior_key, []))[1].append(observation.reward)
        priors = {
            key: prior
            for key, (prior_key, _) in grouped.items()
            if key not in history and prior_key is not None and (prior := history.get(prior_key)) is not None
        }
        for key, prior in priors.items():
            history.adopt(key, prior)
        for key, (_, rewards) in grouped.items():
            history.observe(key, rewards, version)
        if TASK_KEY not in grouped and observations:
            history.observe(TASK_KEY, [observation.reward for observation in observations], version)
        history.enforce_capacity()
