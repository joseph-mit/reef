"""Slime implementation of PPO-TTT: a scalar critic in place of TTT-Discover's sibling group."""

from __future__ import annotations

import math
from argparse import Namespace
from numbers import Integral, Real
from typing import Any

from recipes.ppottt.slime.utils.schedule import DEFAULT_CRITIC_STEPS_PER_ACTOR, CriticSchedule
from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult, register_loss_family


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


@register_loss_family
class PpotttAlgorithm(SlimeAlgorithm):
    """Standard PPO on the TTT-Discover search: clipped surrogate, learned value baseline.

    Everything numerical is Slime's except the advantage hook: the policy loss
    is Slime's clipped ``policy_loss`` with the sampler's log-probabilities as
    the old policy, the critic loss is Slime's clipped value loss, and
    ``ppottt_advantages`` builds GAE targets for one scored program per
    rollout. Reef ships no advantages; the critic produces them in the backend.
    """

    loss_family = "ppottt"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "forbidden"
    forbidden_advantages_message = (
        "ppottt advantages are computed from the value model in the training backend; the Reef payload must omit them"
    )
    allows_slime_advantage_computation = True
    # The critic's scalar value head has no counterpart in the HF checkpoint.
    critic_value_head_zero_init = True
    required_objective_hooks = ("custom_advantage_function_path",)

    _schedule: CriticSchedule

    def configure_backend_args(self, args: Namespace) -> None:
        # The advantage hook needs Slime's pre-train pass: it runs after the
        # critic has produced values and the KL to the frozen base.
        args.compute_advantages_and_returns = True

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if not getattr(args, "use_critic", False):
            raise RuntimeError(f"{source} requires a value model; pass --use-critic on the Slime driver")
        if getattr(args, "advantage_estimator", None) != "ppo":
            raise RuntimeError(f"{source} requires --advantage-estimator ppo")
        # Late in a run valid attempts differ by 1e-4 while invalid ones
        # score 0, and PPO is not invariant to reward scale: whiten each step.
        if not getattr(args, "normalize_advantages", False):
            raise RuntimeError(f"{source} requires --normalize-advantages")

        eps_clip = getattr(args, "eps_clip", None)
        if not _is_real(eps_clip) or not 0 < float(eps_clip) < 1:
            raise RuntimeError(f"{source} requires --eps-clip in (0, 1)")
        # Unset --eps-clip-high falls back to --eps-clip in Slime, which is
        # the symmetric band of standard PPO, so unlike SAO it may stay unset.
        eps_clip_high = getattr(args, "eps_clip_high", None)
        if eps_clip_high is not None and (not _is_real(eps_clip_high) or eps_clip_high <= 0):
            raise RuntimeError(f"{source} requires --eps-clip-high > 0 when set")
        value_clip = getattr(args, "value_clip", None)
        if not _is_real(value_clip) or value_clip <= 0:
            raise RuntimeError(f"{source} requires --value-clip > 0")

        gamma = getattr(args, "gamma", None)
        if not _is_real(gamma) or not 0 < float(gamma) <= 1:
            raise RuntimeError(f"{source} requires --gamma in (0, 1]")
        lambd = getattr(args, "lambd", None)
        if not _is_real(lambd) or not 0 <= float(lambd) <= 1:
            raise RuntimeError(f"{source} requires --lambd in [0, 1]")
        kl_coef = getattr(args, "kl_coef", None)
        if not _is_real(kl_coef) or kl_coef < 0:
            raise RuntimeError(f"{source} requires a finite, non-negative --kl-coef")

        steps = getattr(args, "critic_steps_per_actor", None)
        if steps is not None and (not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 1):
            raise RuntimeError(f"{source} requires --critic-steps-per-actor >= 1")
        warmup = getattr(args, "num_critic_only_steps", 0)
        if not isinstance(warmup, Integral) or isinstance(warmup, bool) or warmup < 0:
            raise RuntimeError(f"{source} requires --num-critic-only-steps >= 0")

    def configure_critic_args(self, critic_args: Namespace) -> None:
        """Keep the KL term on the actor side only.

        The critic has no frozen-base log-probabilities to compare against,
        and its regression target should be the verifier's score, not the
        score net of a policy-dependent penalty. Slime's own role overrides
        zero the critic's KL coefficient for the same reason; Reef derives the
        critic namespace by copy, so the family does it here.
        """
        critic_args.kl_coef = 0.0

    def bind(
        self,
        config: object | None = None,
        *,
        critic_steps_per_actor: int | None = None,
        critic_only_steps: int = 0,
    ) -> PpotttAlgorithm:
        if config is not None:
            raise TypeError("ppottt takes no bridge algorithm config; its settings are Slime flags")
        bound = PpotttAlgorithm()
        bound._schedule = CriticSchedule(
            DEFAULT_CRITIC_STEPS_PER_ACTOR if critic_steps_per_actor is None else critic_steps_per_actor,
            critic_only_steps,
        )
        return bound

    def train(self, rollout_id, rollout_data_refs, *, actor_group, critic_group, resolve) -> TrainResult:
        if critic_group is None:
            raise RuntimeError("ppottt requires a value model, but the bridge was booted without one")
        plan = self._schedule.plan(rollout_id)
        # Each critic pass reads values before it updates. The first pass
        # reads the critic as it stood before this step's rewards, which is
        # PPO's V_old; later passes have already fit those rewards, and a
        # baseline that has seen the reward it is subtracted from shrinks the
        # advantage toward zero. The actor gets the first pass's values.
        critic_values = None
        for update in range(plan.critic_updates):
            values = resolve(critic_group.async_train(rollout_id, rollout_data_refs))
            if update == 0:
                critic_values = values
        actor_results: list[Any] = []
        if plan.train_actor:
            actor_results = list(
                resolve(actor_group.async_train(rollout_id, rollout_data_refs, external_data=critic_values)) or ()
            )
        return TrainResult(
            actor_results,
            {"ppottt/critic_updates": plan.critic_updates, "ppottt/actor_trained": int(plan.train_actor)},
        )

    def rollout_metrics(self, rollout_data: dict[str, Any], serving_version: str) -> dict[str, Any]:
        total = sum(rollout_data.get("response_lengths") or [])
        if total <= 0:
            return {}
        trained = sum(sum(mask) for mask in rollout_data.get("loss_masks") or [])
        return {"ppottt/effective_token_rate": trained / total}


__all__ = ["PpotttAlgorithm"]
