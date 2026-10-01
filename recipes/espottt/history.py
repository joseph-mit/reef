"""Reward history per archive state, and TTT-Discover's entropic advantage over it.

TTT-Discover scores each attempt against its siblings: with ``beta`` chosen so
that ``KL(softmax(beta * r) || uniform) = log 2`` over the sibling group, an
attempt's advantage is ``exp(beta * r_i) / mean_{j != i} exp(beta * r_j) - 1``.
That favours the best attempts, but it needs a group of siblings from the same
parent at every step.

Entropic SPO-TTT keeps the advantage and takes the comparison set from
history instead: the rewards of earlier attempts from the same archive state,
down-weighted as the policy moves, as SPO-TTT's tracker does for its mean.
Because ``beta`` and the normaliser both depend on the whole comparison set,
not only its mean, the history keeps the rewards themselves (with weights)
rather than a running mean. When the history and its comparison set are the
siblings of a single step, the advantage is exactly TTT-Discover's.

The module is torch-free and JSON-serialisable: the preparer runs it with only
the batch and the committed algorithm state, and Reef commits that state with
every training step, so a restart resumes the history exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from recipes.spottt.tracker import TASK_KEY, TrackerSettings

DEFAULT_HISTORY_SIZE = 128
DEFAULT_MIN_HISTORY = 8.0

# TTT-Discover's adaptive-beta search (recipes/tttd/preparer.py, ported from
# test-time-training/discover@6c40e82): target KL, upper bound, bisection steps
# and the normaliser's epsilon.
TARGET_KL = math.log(2)
BETA_MAX = 1e6
BISECTION_STEPS = 60
NORMALISER_EPSILON = 1e-12

Weighted = tuple[float, float]
"""One past outcome: ``(reward, weight)``."""


@dataclass(frozen=True)
class HistorySettings:
    """Forgetting, sharing and size limits of the history.

    ``forgetting`` holds SPO-TTT's settings: the half-life in committed policy
    steps, SPO's bounds on the per-update discount, the share of a parent's
    history a new state starts from, and the key cap. ``history_size`` caps the
    distinct outcomes kept per key (equal rewards share one entry, so this
    loses nothing until a key has seen more distinct rewards); past it the
    history is reduced by :func:`compress`. ``min_history`` is the effective
    number of outcomes a comparison set needs before it is used on its own; a
    thinner set is topped up from the next source.
    """

    forgetting: TrackerSettings = field(default_factory=TrackerSettings)
    history_size: int = DEFAULT_HISTORY_SIZE
    min_history: float = DEFAULT_MIN_HISTORY

    def __post_init__(self) -> None:
        if isinstance(self.history_size, bool) or self.history_size < 2:
            raise ValueError("history_size must be at least 2")
        if not math.isfinite(self.min_history) or self.min_history <= 0:
            raise ValueError("min_history must be a positive finite number")


@dataclass
class KeyHistory:
    """The weighted outcomes kept for one key, as of the policy version that last updated it."""

    outcomes: list[Weighted]
    version: int

    def total_weight(self) -> float:
        return math.fsum(weight for _, weight in self.outcomes)


def merge_equal(outcomes: Iterable[Weighted]) -> list[Weighted]:
    """Equal rewards as one outcome carrying their summed weight, sorted by reward.

    Lossless for everything computed from the history (``beta``, the
    normaliser, effective counts): each depends on the outcomes only through
    the total weight at each reward. On the packing tasks most outcomes are
    the failure score or a repeat of the parent's score, so this is where most
    of the history's size goes.
    """
    merged: dict[float, float] = {}
    for reward, weight in outcomes:
        if weight > 0:
            merged[reward] = merged.get(reward, 0.0) + weight
    return sorted(merged.items())


def compress(outcomes: Sequence[Weighted], size: int) -> list[Weighted]:
    """At most ``size`` outcomes with the same total weight, the same top and the same quantiles.

    Equal rewards merge first, which loses nothing. Past ``size`` distinct
    rewards the best one is kept exactly, with its weight, because the
    entropic weights depend most on the top of the distribution; the rest are
    reduced to ``size - 1`` weighted quantiles of equal weight: sorted by
    reward, the k-th is the reward at cumulative weight ``(k + 1/2) / (size -
    1)`` of their total. O(n log n) in the number of outcomes.
    """
    if size < 2:
        raise ValueError("compress keeps at least two outcomes")
    merged = merge_equal(outcomes)
    if len(merged) <= size:
        return merged
    top, rest = merged[-1], merged[:-1]
    total = math.fsum(weight for _, weight in rest)
    parts = size - 1
    kept: list[Weighted] = []
    cumulative = 0.0
    index = 0
    for k in range(parts):
        target = (k + 0.5) / parts * total
        while index < len(rest) - 1 and cumulative + rest[index][1] < target:
            cumulative += rest[index][1]
            index += 1
        kept.append((rest[index][0], total / parts))
    return [*merge_equal(kept), top]


class RewardHistory:
    """Weighted reward outcomes per key, forgotten as the policy moves.

    The update is SPO-TTT's tracker applied to the outcomes themselves: when a
    key is updated at a policy version ``D`` steps after its last update, its
    past weights are multiplied by ``rho = 2 ** (-D / half_life)``, clipped to
    SPO's bounds, and the new outcomes join with weight 1, once per version
    rather than once per outcome. A key's total weight is therefore SPO's
    effective count and, until compression, its weighted mean is SPO's
    estimate exactly. A read at a later version applies the forgetting that
    version would, so an old history counts for less against fresh outcomes.
    """

    def __init__(self, settings: HistorySettings | None = None, state: Mapping[str, Any] | None = None) -> None:
        self.settings = settings or HistorySettings()
        self._keys: dict[str, KeyHistory] = {}
        if state:
            self._load(state)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def get(self, key: str) -> KeyHistory | None:
        return self._keys.get(key)

    def _drift(self, key: str, history: KeyHistory, version: int) -> int:
        drift = version - history.version
        if drift < 0:
            raise ValueError(f"key {key!r} was updated by a later version {history.version}")
        return drift

    def outcomes_at(self, key: str, version: int) -> list[Weighted]:
        """``key``'s outcomes with the forgetting that ``version`` would apply; empty when cold."""
        history = self._keys.get(key)
        if history is None:
            return []
        drift = self._drift(key, history, version)
        factor = self.settings.forgetting.forgetting_factor(float(drift)) if drift > 0 else 1.0
        return [(reward, weight * factor) for reward, weight in history.outcomes]

    def observe(self, key: str, rewards: Sequence[float], version: int) -> None:
        """Fold one policy version's rewards for ``key`` into its history, as SPO-TTT's tracker does."""
        if not rewards:
            raise ValueError("observe requires at least one reward")
        if version < 0:
            raise ValueError("version must be non-negative")
        new = [float(reward) for reward in rewards]
        if any(not math.isfinite(reward) for reward in new):
            raise ValueError("rewards must be finite")
        history = self._keys.get(key)
        past: list[Weighted] = []
        if history is not None:
            # SPO's rule, including its rho_max cap when no version has passed.
            rho = self.settings.forgetting.forgetting_factor(float(self._drift(key, history, version)))
            past = [(reward, weight * rho) for reward, weight in history.outcomes]
        outcomes = compress([*past, *((reward, 1.0) for reward in new)], self.settings.history_size)
        self._keys[key] = KeyHistory(outcomes, version)

    def adopt(self, key: str, prior: KeyHistory) -> None:
        """Start a cold key from ``prior``'s outcomes, keeping ``inherit_fraction`` of their weight."""
        if key in self._keys:
            raise ValueError(f"key {key!r} already has a history")
        share = self.settings.forgetting.inherit_fraction
        self._keys[key] = KeyHistory([(reward, weight * share) for reward, weight in prior.outcomes], prior.version)

    def enforce_capacity(self) -> None:
        """Drop the keys updated longest ago until at most ``max_keys`` remain; the task key stays."""
        excess = len(self._keys) - self.settings.forgetting.max_keys
        if excess <= 0:
            return
        candidates = sorted(
            (key for key in self._keys if key != TASK_KEY), key=lambda key: (self._keys[key].version, key)
        )
        for key in candidates[:excess]:
            self._keys.pop(key, None)

    def state_dict(self) -> dict[str, Any]:
        """JSON-compatible snapshot: ``{key: [version, [[reward, weight], ...]]}``."""
        return {
            key: [history.version, [[reward, weight] for reward, weight in history.outcomes]]
            for key, history in self._keys.items()
        }

    def _load(self, state: Mapping[str, Any]) -> None:
        for key, row in state.items():
            if not isinstance(key, str) or not isinstance(row, Sequence) or len(row) != 2:
                raise ValueError("history state rows must be {key: [version, [[reward, weight], ...]]}")
            version, outcomes = int(row[0]), row[1]
            if version < 0 or not isinstance(outcomes, Sequence):
                raise ValueError(f"history state row for {key!r} is invalid")
            pairs: list[Weighted] = []
            for pair in outcomes:
                reward, weight = float(pair[0]), float(pair[1])
                if not math.isfinite(reward) or not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"history state row for {key!r} has an invalid outcome")
                pairs.append((reward, weight))
            self._keys[key] = KeyHistory(pairs, version)


def adaptive_beta(rewards: Sequence[float], weights: Sequence[float]) -> float:
    """TTT-Discover's ``beta`` with weighted outcomes: ``KL(q || p) = log 2``.

    ``p`` is the weights normalised and ``q_j`` is proportional to
    ``p_j * exp(beta * r_j)``. With equal weights this is the reference search
    exactly: the same expansion from ``beta = 1``, the same bound and the same
    60 bisection steps.
    """
    if len(rewards) != len(weights) or not rewards:
        raise ValueError("adaptive_beta needs one weight per reward")
    if len(rewards) < 2:
        return 0.0
    total = math.fsum(weights)
    if total <= 0:
        raise ValueError("weights must have a positive sum")
    probabilities = [weight / total for weight in weights]
    top = max(rewards)

    def kl(beta: float) -> float:
        logits = [beta * (reward - top) for reward in rewards]
        scaled = [p * math.exp(logit) for p, logit in zip(probabilities, logits, strict=True)]
        mass = math.fsum(scaled)
        log_mass = math.log(mass)
        return math.fsum(
            (value / mass) * (logit - log_mass) for value, logit in zip(scaled, logits, strict=True) if value > 0
        )

    low, high = 0.0, 1.0
    beta: float | None = None
    if kl(high) < TARGET_KL:
        while high < BETA_MAX and kl(high) < TARGET_KL:
            high *= 2.0
        beta = high if kl(high) < TARGET_KL else None
    if beta is None:
        for _ in range(BISECTION_STEPS):
            midpoint = 0.5 * (low + high)
            if kl(midpoint) < TARGET_KL:
                low = midpoint
            else:
                high = midpoint
        beta = high
    return beta


def entropic_advantage(reward: float, comparison: Sequence[Weighted], own_weight: float = 1.0) -> tuple[float, float]:
    """One attempt's entropic advantage against a weighted comparison set, and its ``beta``.

    ``beta`` is solved over the attempt and the comparison set together, the
    attempt carrying ``own_weight`` (one outcome); the normaliser is the
    weighted mean of ``exp(beta * r)`` over the comparison set alone, which
    leaves the attempt out exactly as TTT-Discover's leave-one-out normaliser
    does. With the other siblings of a group as the comparison set, each with
    weight 1, the result equals TTT-Discover's advantage for that attempt.
    """
    if not comparison:
        raise ValueError("an entropic advantage needs a non-empty comparison set")
    rewards = [reward, *(value for value, _ in comparison)]
    weights = [own_weight, *(weight for _, weight in comparison)]
    beta = adaptive_beta(rewards, weights)
    top = max(rewards)
    total = math.fsum(weight for _, weight in comparison)
    if total <= 0:
        raise ValueError("the comparison set must have a positive total weight")
    normaliser = math.fsum(weight * math.exp(beta * (value - top)) for value, weight in comparison) / total
    return math.exp(beta * (reward - top)) / (normaliser + NORMALISER_EPSILON) - 1.0, beta


def total_weight(outcomes: Iterable[Weighted]) -> float:
    return math.fsum(weight for _, weight in outcomes)
