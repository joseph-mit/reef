"""Critic cadence for PPO-TTT: warm-up steps and critic updates per actor update."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CRITIC_STEPS_PER_ACTOR = 2


@dataclass(frozen=True)
class CriticStepPlan:
    critic_updates: int
    train_actor: bool


@dataclass(frozen=True)
class CriticSchedule:
    """How one training step splits between the value model and the policy.

    The value head starts from zero, so the first ``critic_only_steps`` steps
    fit the critic alone; the search still runs during them, only the policy
    is held. After that every step makes ``critic_steps_per_actor`` critic
    updates before the single policy update, so the baseline tracks a policy
    that keeps moving.
    """

    critic_steps_per_actor: int = DEFAULT_CRITIC_STEPS_PER_ACTOR
    critic_only_steps: int = 0

    def __post_init__(self) -> None:
        if self.critic_steps_per_actor < 1:
            raise ValueError("critic_steps_per_actor must be >= 1")
        if self.critic_only_steps < 0:
            raise ValueError("critic_only_steps must be >= 0")

    def plan(self, rollout_index: int) -> CriticStepPlan:
        if rollout_index < 0:
            raise ValueError("rollout_index must be >= 0")
        return CriticStepPlan(self.critic_steps_per_actor, rollout_index >= self.critic_only_steps)
