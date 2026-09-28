"""Dependency-free PPO-TTT numerics used as a parity-test oracle.

Written from the estimator's definition (Schulman et al. 2016, GAE; the
terminal-reward placement of a single-turn rollout), not from the tensor code
it checks, so the two can disagree.
"""

from __future__ import annotations

DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 1.0


def terminal_rewards(response_len: int, reward: float, per_token_penalty: list[float] | None = None) -> list[float]:
    """Per-token rewards: the penalty everywhere, the verifier score on the last token."""
    penalty = [0.0] * response_len if per_token_penalty is None else list(per_token_penalty)
    if len(penalty) != response_len:
        raise ValueError(f"penalty length {len(penalty)} does not match response length {response_len}")
    if response_len > 0:
        penalty[-1] += reward
    return penalty


def gae(
    values: list[float],
    rewards: list[float],
    gamma: float = DEFAULT_GAMMA,
    lambd: float = DEFAULT_LAMBDA,
) -> tuple[list[float], list[float]]:
    """Generalised advantage estimation with a zero value past the last token."""
    if len(values) != len(rewards):
        raise ValueError(f"values/rewards length mismatch: {len(values)} vs {len(rewards)}")
    advantages = [0.0] * len(values)
    returns = [0.0] * len(values)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(values))):
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lambd * last_gae
        advantages[t] = last_gae
        returns[t] = last_gae + values[t]
        next_value = values[t]
    return advantages, returns


def one_step_advantages(values: list[float], reward: float) -> list[float]:
    """The closed form at ``gamma = lambd = 1`` without a penalty: ``reward - V_t`` at every token."""
    return [reward - value for value in values]
