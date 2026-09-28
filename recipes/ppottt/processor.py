"""Step barrier for single-stream TTT: one full search step, trained as independent samples."""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from recipes.ppottt.batches import ScheduledPolicyBatch
from reef.train.processors.reported import (
    BatchUnit,
    Candidate,
    GroupDecision,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
    SampleAssembly,
)
from reef.train.types import PolicySample, ProcessorContext, policy_row_violation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepRow:
    """One accepted rollout: the assembled sample plus its producing version."""

    sample: PolicySample
    release_id: str | None


class PPOTTTProcessor(ReportedFeedbackProcessor):
    """Wait for one complete search step, then hand it over as a flat batch.

    The search still runs in steps of ``groups_per_step`` parents with
    ``rollouts_per_group`` attempts each, and the barrier stays: a step is
    trained only once every attempt is scored, and a step whose reports span
    two policy releases is discarded, because its rewards came from different
    policies. What goes away is the group: no advantage is computed from
    siblings, so constant-reward groups are kept (they still train the critic
    and still carry signal against it) and a group may hold a single attempt.

    Subclasses change the batch they emit through :meth:`make_step_batch` and
    the per-row payload through :meth:`make_row`; the barrier is shared.
    """

    output_schema = ScheduledPolicyBatch
    exclusive_sources = True
    ordered_groups = True  # steps train in order
    batch_label = "ppottt"

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.groups_per_step = int(config.get("groups_per_step", 8))
        self.rollouts_per_group = int(config.get("rollouts_per_group", 64))
        self.ppo_epochs = int(config.get("ppo_epochs", 1))
        self.minibatch_size = int(config.get("minibatch_size", 0))
        self.shuffle_minibatches = bool(config.get("shuffle_minibatches", True))
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group <= 0:
            raise ValueError("rollouts_per_group must be positive")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")
        if self.minibatch_size < 0:
            raise ValueError("minibatch_size must be non-negative")
        self._assembly = SampleAssembly.from_config(context)
        self._failed_step_versions: dict[int, tuple[str, ...]] = {}
        super().__init__(context.with_config({**config, "batch_size": 1}))

    @property
    def step_size(self) -> int:
        return self.groups_per_step * self.rollouts_per_group

    def judge(self, context: ReportContext) -> ReportDecision:
        parsed: Any = context.parsed_report
        if parsed is None:
            return ReportDecision.never(f"{type(self).__name__} requires the recipe's report schema")
        # A valid report for a different grid can never train in this scenario.
        if parsed.groups_per_step != self.groups_per_step or parsed.rollouts_per_group != self.rollouts_per_group:
            return ReportDecision.never(
                f"report announces a {parsed.groups_per_step}x{parsed.rollouts_per_group} grid; "
                f"this scenario trains on {self.groups_per_step}x{self.rollouts_per_group}"
            )
        if (gate := context.eligibility()) is not None:
            return gate
        score = context.score
        if score is None or context.inferences is None:
            raise RuntimeError("eligible report is not fully resolved")
        try:
            sample = self._assembly.build(context, score)
        except (TypeError, ValueError) as error:
            return ReportDecision.never(f"sample assembly failed: {error}")
        if sample is None or policy_row_violation(sample.tokens, sample.loss_mask, sample.rollout_log_probs):
            return ReportDecision.never("policy tensor contract violation")
        inference = context.inferences[0]
        release_id = inference.artifact_ref.release_id if inference.artifact_ref is not None else None
        return ReportDecision.train(
            StepRow(self.make_row(sample, parsed), release_id),
            group_key=parsed.step,
            slot=(parsed.group, parsed.rollout),
        )

    def make_row(self, sample: PolicySample, report: Any) -> PolicySample:
        """Per-rollout hook for subclasses that attach report metadata to the sample."""
        return sample

    def decide_group(self, key: Hashable, candidates: tuple[Candidate, ...]) -> GroupDecision:
        if len(candidates) != self.step_size:
            return GroupDecision.INCOMPLETE
        versions = {candidate.value.release_id for candidate in candidates if candidate.value.release_id is not None}
        if len(versions) <= 1:
            return GroupDecision.READY
        if not isinstance(key, int):
            raise TypeError(f"step key must be an integer, got {key!r}")
        ordered_versions = tuple(sorted(versions))
        self._failed_step_versions[key] = ordered_versions
        logger.error(
            "step %d failed because its %d reports span releases %s", key, self.step_size, list(ordered_versions)
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        """Expose steps discarded for mixing policy releases (read by the run controller)."""
        return {
            "failed_steps": [
                {"step": step, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for step, versions in sorted(self._failed_step_versions.items())
            ]
        }

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> ScheduledPolicyBatch:
        if len(units) != 1:
            raise RuntimeError(f"{type(self).__name__} creates exactly one complete step per batch")
        unit = units[0]
        by_slot = {candidate.slot: candidate.value.sample for candidate in unit.candidates}
        samples = tuple(
            by_slot[(group, rollout)]
            for group in range(self.groups_per_step)
            for rollout in range(self.rollouts_per_group)
        )
        self.experiment_logger.log(step_reward_metrics(unit.group_key, samples), namespace=self.batch_label)
        return self.make_step_batch(unit.group_key, samples)

    def make_step_batch(self, step: Any, samples: tuple[PolicySample, ...]) -> ScheduledPolicyBatch:
        return ScheduledPolicyBatch(
            f"{self.scenario}:{self.batch_label}:{step}",
            samples,
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            shuffle=self.shuffle_minibatches,
        )


def step_reward_metrics(step: Any, samples: tuple[PolicySample, ...]) -> dict[str, Any]:
    """Reward summary for one step, in the same keys the tttd processor logs."""
    rewards = [sample.reward for sample in samples]
    mean = math.fsum(rewards) / len(rewards)
    return {
        "step": step,
        "grid_rollouts": len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_mean": mean,
        "reward_std": math.sqrt(math.fsum((reward - mean) ** 2 for reward in rewards) / len(rewards)),
        "reward_zero_fraction": sum(reward == 0 for reward in rewards) / len(rewards),
    }
