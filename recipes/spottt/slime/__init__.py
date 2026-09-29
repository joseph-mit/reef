"""Slime implementation of SPO-TTT: Reef-side advantages on Slime's clipped policy loss."""

from __future__ import annotations

import math
from argparse import Namespace
from numbers import Real

from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family


@register_loss_family
class SpotttAlgorithm(SlimeAlgorithm):
    """Clipped PPO surrogate on advantages the SPO-TTT preparer ships.

    The tracker lives on the Reef side, so there is no critic. The one
    backend hook is TTT-Discover's centred KL to the frozen base, applied to
    the shipped advantages exactly as the tttd family applies it, so the two
    recipes differ in the baseline and the loss, not in how they anchor to
    the base model.
    """

    loss_family = "spottt"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "required"
    allows_slime_advantage_computation = True
    required_objective_hooks = ("custom_advantage_function_path",)

    def configure_backend_args(self, args: Namespace) -> None:
        # The KL hook runs in Slime's pre-train advantage pass.
        args.compute_advantages_and_returns = True

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        if getattr(args, "use_critic", False):
            raise RuntimeError(f"{source} has no value model; drop --use-critic")
        if getattr(args, "normalize_advantages", False):
            # The preparer already whitens each step's advantages per attempt;
            # Slime would whiten again, per token, after the KL hook.
            raise RuntimeError(f"{source} whitens advantages on the Reef side; drop --normalize-advantages")
        estimator = getattr(args, "advantage_estimator", "grpo")
        if estimator in ("ppo", "gspo", "cispo"):
            raise RuntimeError(
                f"{source} trains the token-level clipped loss; leave --advantage-estimator at its default, "
                f"not {estimator!r}"
            )
        eps_clip = getattr(args, "eps_clip", None)
        if not isinstance(eps_clip, Real) or isinstance(eps_clip, bool) or not 0 < float(eps_clip) < 1:
            raise RuntimeError(f"{source} requires --eps-clip in (0, 1)")
        kl_coef = getattr(args, "kl_coef", None)
        # A zero coefficient would also drop the frozen-base log-probabilities
        # the KL hook reads, as for tttd.
        if (
            not isinstance(kl_coef, Real)
            or isinstance(kl_coef, bool)
            or not math.isfinite(float(kl_coef))
            or kl_coef <= 0
        ):
            raise RuntimeError(f"{source} requires a positive finite --kl-coef (tttd uses 0.1)")


__all__ = ["SpotttAlgorithm"]
