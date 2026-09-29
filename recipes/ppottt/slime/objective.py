"""Tensor implementation of the PPO-TTT advantage estimator.

Loaded lazily by the package wrappers in ``recipes.ppottt.slime`` so importing
the package never requires torch. Slime's Megatron workers resolve the hook by
path (``--custom-advantage-function-path``); the actor and the critic run the
same hook, each with its own values, so the critic's regression targets and the
actor's advantages come from one definition. CPU parity tests compare this
transcription against an independent pure-Python oracle
(``tests/reef_service/reference_algorithms/ppottt.py``).
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch

from reef.train.slime_backend.algorithm import objective


_SCAN_CHUNK = 256


def discounted_reverse_cumsum(values: torch.Tensor, discount: float) -> torch.Tensor:
    """``y_t = sum_{s >= t} discount^(s - t) * values_s`` over a 1-D tensor, without a per-token loop.

    GAE is this scan over the TD errors with ``discount = gamma * lambd``. A
    discount of 1 is a reversed cumulative sum. Otherwise the sequence is cut
    into chunks from the end: inside a chunk one small triangular matrix of
    powers applies the discount, and the chunk's first value carries into the
    chunk before it, so no power ever exceeds the chunk length and nothing
    overflows or underflows on long responses.
    """
    if discount == 1.0:
        return values.flip(0).cumsum(0).flip(0)
    if discount == 0.0:
        return values.clone()
    length = values.numel()
    width = min(_SCAN_CHUNK, length)
    offsets = torch.arange(width, device=values.device, dtype=values.dtype)
    lags = offsets.unsqueeze(0) - offsets.unsqueeze(1)
    powers = torch.where(lags >= 0, discount ** lags.clamp(min=0), torch.zeros_like(lags))
    scanned = torch.empty_like(values)
    carry = values.new_zeros(())
    for end in range(length, 0, -width):
        start = max(0, end - width)
        size = end - start
        chunk = powers[:size, :size] @ values[start:end] + carry * discount ** (size - offsets[:size])
        scanned[start:end] = chunk
        carry = chunk[0]
    return scanned


def get_terminal_reward_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    reward: float,
    per_token_penalty: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE for a single-turn rollout scored once at the end.

    The verifier scores the whole program, so the scalar ``reward`` lands on
    the last response token; ``per_token_penalty`` (typically ``-kl_coef * kl``,
    zeros when the KL term is off) is added at every token, which is where
    Slime's own ``ppo`` estimator puts it. The recurrence is the standard one,
    ``delta_t = r_t + gamma * V_{t+1} - V_t`` with ``V`` past the last token
    equal to zero, and ``A_t = delta_t + gamma * lambd * A_{t+1}``.

    With ``gamma = lambd = 1`` and no penalty every token carries
    ``A_t = reward - V_t`` and the return target is ``reward``: the one-step
    episode of test-time discovery, where each rollout is a whole attempt.
    The discount and trace parameters stay configurable so a variant that
    treats a chain of archive states as a multi-step trajectory can reuse the code.

    Values are gathered across context-parallel ranks before the recurrence
    and the results sliced back afterwards, matching the SAO helper. The
    recurrence runs as one vectorised scan (``discounted_reverse_cumsum``), not
    a Python loop over tokens. Returned advantages are detached; returns are
    ``advantages + values`` and keep the value graph for the critic loss.
    """
    from megatron.core import mpu

    if per_token_penalty.shape != values.shape:
        raise ValueError(
            "per-token penalty must align with values: "
            f"penalty={tuple(per_token_penalty.shape)}, values={tuple(values.shape)}"
        )

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_values = all_gather_with_cp(values, total_len, response_len)
        full_penalty = all_gather_with_cp(per_token_penalty, total_len, response_len)
    else:
        full_values = values
        full_penalty = per_token_penalty

    rewards = full_penalty.to(dtype=full_values.dtype).clone()
    if response_len > 0:
        rewards[response_len - 1] = rewards[response_len - 1] + float(reward)

    advantages = torch.zeros_like(full_values)
    if response_len > 0:
        values_seen = full_values[:response_len]
        next_values = torch.cat((values_seen[1:], values_seen.new_zeros(1)))
        deltas = rewards[:response_len] + gamma * next_values - values_seen
        advantages[:response_len] = discounted_reverse_cumsum(deltas, gamma * lambd)
    returns = torch.zeros_like(full_values)
    if response_len > 0:
        returns[:response_len] = advantages[:response_len] + full_values[:response_len]

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(returns, total_len, response_len)

    return advantages.detach(), returns


@objective("custom_advantage_function_path")
def ppottt_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Populate PPO-TTT advantages and returns through Slime's custom-advantage hook.

    Slime computes ``rollout_data["kl"]`` (against the frozen base, zeros when
    ``--kl-coef`` is 0) immediately before calling this hook. The penalty
    ``-kl_coef * kl`` enters the per-token reward, the InstructGPT placement
    that Slime's built-in ``ppo`` estimator also uses; the TTT-Discover recipe
    centres a KL on the advantages instead, and the README records the
    difference. Sets ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]`` in place, one tensor per sample.
    """
    values = rollout_data.get("values")
    if values is None:
        raise ValueError("ppottt advantage computation requires critic values in rollout_data")
    rewards = rollout_data["rewards"]
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    kls = rollout_data.get("kl")
    kl_coef = float(getattr(args, "kl_coef", 0.0) or 0.0)

    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for index, value in enumerate(values):
        if kls is not None and kl_coef != 0.0:
            penalty = -kl_coef * kls[index].to(dtype=value.dtype)
        else:
            penalty = torch.zeros_like(value)
        advantage, ret = get_terminal_reward_advantages_and_returns(
            int(total_lengths[index]),
            int(response_lengths[index]),
            value,
            float(rewards[index]),
            penalty,
            float(args.gamma),
            float(args.lambd),
        )
        advantages.append(advantage)
        returns.append(ret)
    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
