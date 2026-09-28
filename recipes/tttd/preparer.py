"""TTT-Discover step preparer: grouped adaptive-entropic advantages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import GroupedPolicyBatch, TrainingBatch


@register_step_preparer
class TttdPreparer(StepPreparer):
    name = "tttd"

    @staticmethod
    def adaptive_entropic_advantages(rewards: list[float]) -> tuple[tuple[float, ...], float]:
        """Return the reference TTT-Discover LOO advantages and solved beta.

        This is a direct port of
        ``test-time-training/discover@6c40e82/ttt_discover/rl/train.py`` for
        ``adv_estimator == "entropic_adaptive_beta"``. Beta is selected so
        ``KL(softmax(beta * rewards) || uniform) == log(2)`` using the same
        expansion bound, 60 bisection iterations, stable reward shift, and
        leave-one-out normalizer.
        """
        if not rewards:
            raise ValueError("an adaptive-entropic group cannot be empty")

        delta = math.log(2)
        beta_max = 1e6
        iterations = 60
        epsilon = 1e-12

        group_size = len(rewards)
        max_reward = max(rewards)

        beta: float | None
        if group_size < 2:
            beta = 0.0
        else:
            log_group_size = math.log(group_size)

            def estimated_kl(beta_scalar: float) -> float:
                logits = [beta_scalar * (r - max_reward) for r in rewards]
                max_logit = max(logits)
                exp_shifted = [math.exp(lg - max_logit) for lg in logits]
                sum_exp = sum(exp_shifted)
                log_sum_exp = math.log(sum_exp) + max_logit
                kl = 0.0
                for i in range(group_size):
                    prob = exp_shifted[i] / sum_exp
                    log_prob = logits[i] - log_sum_exp
                    kl += prob * (log_prob + log_group_size)
                return kl

            low, high = 0.0, 1.0
            if estimated_kl(high) < delta:
                while high < beta_max and estimated_kl(high) < delta:
                    high *= 2.0
                beta = high if estimated_kl(high) < delta else None
            else:
                beta = None

            if beta is None:
                for _ in range(iterations):
                    midpoint = 0.5 * (low + high)
                    if estimated_kl(midpoint) < delta:
                        low = midpoint
                    else:
                        high = midpoint
                beta = high

        if beta is None:
            raise RuntimeError("adaptive beta search did not produce a value")
        exponentials = [math.exp(beta * (r - max_reward)) for r in rewards]
        if group_size == 1:
            normalizers = exponentials
        else:
            total = sum(exponentials)
            normalizers = [(total - e) / (group_size - 1) for e in exponentials]
        advantages = tuple(e / (n + epsilon) - 1.0 for e, n in zip(exponentials, normalizers, strict=True))
        return advantages, beta

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, GroupedPolicyBatch):
            raise TypeError(f"{self.name} requires GroupedPolicyBatch, got {type(batch).__name__}")
        advantages: list[float] = []
        betas: list[float] = []
        for comparison_set in batch.comparison_sets:
            group_advantages, beta = self.adaptive_entropic_advantages([sample.reward for sample in comparison_set])
            advantages.extend(group_advantages)
            betas.append(beta)
        steps = next_steps(state)
        normalized = tuple(advantages)
        # The processor drops constant-reward groups but keeps one when every
        # group is constant; flag that batch (its advantages carry no signal).
        constant_groups = all(len({sample.reward for sample in group}) == 1 for group in batch.comparison_sets)
        return StepSignal(
            "train",
            self.name,
            {"steps": steps},
            {
                "advantages": normalized,
                "adaptive_betas": tuple(betas),
                "constant_groups_retained": int(constant_groups),
                "steps": steps,
            },
            normalized,
            StepScheduling(unit="sample", batch_size="actual"),
        )


@register_step_preparer
class TttdMeanBaselinePreparer(StepPreparer):
    """Group-mean baseline in place of the entropic weights, for the comparison ladder.

    Same grid, same barrier, same loss as ``tttd``; only the advantage changes
    to GRPO's ``(r_i - mean) / (std + eps)`` within each comparison set. Running
    this next to ``tttd`` separates what the entropic objective contributes
    from what the grouped baseline contributes, which is the control a
    critic-based or tracker-based method has to be read against.
    """

    name = "tttd-mean"
    epsilon = 1e-6

    @staticmethod
    def mean_baseline_advantages(rewards: list[float], epsilon: float = 1e-6) -> tuple[float, ...]:
        """GRPO-normalised advantages; a constant group carries no signal and maps to zeros."""
        if not rewards:
            raise ValueError("a mean-baseline group cannot be empty")
        mean = math.fsum(rewards) / len(rewards)
        std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
        if std == 0.0:
            return tuple(0.0 for _ in rewards)
        return tuple((r - mean) / (std + epsilon) for r in rewards)

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, GroupedPolicyBatch):
            raise TypeError(f"{self.name} requires GroupedPolicyBatch, got {type(batch).__name__}")
        advantages: list[float] = []
        for comparison_set in batch.comparison_sets:
            advantages.extend(
                self.mean_baseline_advantages([sample.reward for sample in comparison_set], self.epsilon)
            )
        steps = next_steps(state)
        constant_groups = all(len({sample.reward for sample in group}) == 1 for group in batch.comparison_sets)
        return StepSignal(
            "train",
            "tttd",
            {"steps": steps},
            {
                "advantages": tuple(advantages),
                "constant_groups_retained": int(constant_groups),
                "steps": steps,
            },
            tuple(advantages),
            StepScheduling(unit="sample", batch_size="actual"),
        )
