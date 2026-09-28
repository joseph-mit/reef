"""Forgetting value tracker for single-stream test-time training.

Single-stream Policy Optimization (Xu and Ding, arXiv:2509.13232, Section 4.1)
keeps one running estimate of expected reward per prompt and discounts old
evidence as the policy moves, which lets a single rollout carry a low-variance
advantage without sibling samples. Test-time discovery breaks the one
assumption that rule leans on: a scientific state almost never repeats, so a
prompt-indexed table is empty at the moment it is needed. The tracker here
keeps SPO's update rule and changes what a key is: a whole task (one running
estimate for the test instance) or a node of the search archive (the parent a
rollout was expanded from), with a fallback chain from a cold node to its own
parent, then to the task.

The module is torch-free and JSON-serialisable on purpose. The step preparer
runs it inside the training backend with only the batch and the persisted
algorithm state in hand, and Reef commits that state with every training step,
so a restart resumes the tracker where the last commit left it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TASK_KEY = "task"
NODE_KEY_PREFIX = "node:"

DEFAULT_HALF_LIFE = 8.0
DEFAULT_RHO_MIN = 0.875
DEFAULT_RHO_MAX = 0.96
DEFAULT_INHERIT_FRACTION = 0.5
DEFAULT_MAX_KEYS = 4096
NORMALIZATION_EPSILON = 1e-6


def node_key(parent_id: str) -> str:
    return f"{NODE_KEY_PREFIX}{parent_id}"


@dataclass(frozen=True)
class TrackerSettings:
    """Forgetting and sharing parameters.

    ``half_life`` is measured in units of the drift the caller reports; with
    the default drift (committed policy steps since the key was last updated)
    a key not touched for ``half_life`` steps keeps half of its evidence.
    ``rho_min`` and ``rho_max`` are SPO's bounds on the per-update discount,
    so no single update forgets more than ``1 - rho_min`` or less than
    ``1 - rho_max`` of what the key knew. ``inherit_fraction`` is how much of a
    parent's effective count a newly seen archive node starts with when it
    copies the parent's estimate as its prior.
    """

    half_life: float = DEFAULT_HALF_LIFE
    rho_min: float = DEFAULT_RHO_MIN
    rho_max: float = DEFAULT_RHO_MAX
    inherit_fraction: float = DEFAULT_INHERIT_FRACTION
    max_keys: int = DEFAULT_MAX_KEYS

    def __post_init__(self) -> None:
        if not math.isfinite(self.half_life) or self.half_life <= 0:
            raise ValueError("half_life must be a positive finite number")
        if not 0 < self.rho_min <= self.rho_max <= 1:
            raise ValueError("rho bounds must satisfy 0 < rho_min <= rho_max <= 1")
        if not 0 <= self.inherit_fraction <= 1:
            raise ValueError("inherit_fraction must lie in [0, 1]")
        if self.max_keys < 1:
            raise ValueError("max_keys must be positive")

    def forgetting_factor(self, drift: float) -> float:
        """SPO's ``rho = 2 ** (-D / D_half)`` with the drift in place of the KL, clipped to the bounds."""
        if drift < 0 or not math.isfinite(drift):
            raise ValueError("drift must be a non-negative finite number")
        return min(self.rho_max, max(self.rho_min, 2.0 ** (-drift / self.half_life)))


@dataclass
class KeyEstimate:
    """Running estimate for one key: mean, effective sample count, last policy version."""

    mean: float
    count: float
    version: int

    def as_list(self) -> list[float | int]:
        return [self.mean, self.count, self.version]


class ForgettingTracker:
    """Per-key expected-reward estimates with policy-drift forgetting.

    The update is SPO's Beta-posterior rule written for arbitrary real rewards
    and for a batch of ``k`` sibling observations produced by one policy
    version::

        n' = rho * n + k
        m' = m + (k / n') * (mean(rewards) - m)

    For rewards in {0, 1} this is exactly the posterior mean of
    ``Beta(rho * alpha + successes, rho * beta + failures)``, and for one
    observation it is SPO's adaptive EMA with step ``1 / (rho * n + 1)``. The
    discount is applied once per policy version rather than once per
    observation: siblings sampled in the same step come from the same policy,
    so per-observation discounting would forget evidence the policy change
    never touched.
    """

    def __init__(self, settings: TrackerSettings | None = None, state: Mapping[str, Any] | None = None) -> None:
        self.settings = settings or TrackerSettings()
        self._estimates: dict[str, KeyEstimate] = {}
        if state:
            self._load(state)

    def __len__(self) -> int:
        return len(self._estimates)

    def __contains__(self, key: str) -> bool:
        return key in self._estimates

    def estimate(self, key: str) -> KeyEstimate | None:
        return self._estimates.get(key)

    def baseline(self, key: str, fallbacks: Iterable[str] = ()) -> float | None:
        """The first warm estimate along ``key`` then ``fallbacks``, or ``None`` when every key is cold."""
        for candidate in (key, *fallbacks):
            estimate = self._estimates.get(candidate)
            if estimate is not None:
                return estimate.mean
        return None

    def observe(
        self,
        key: str,
        rewards: Sequence[float],
        version: int,
        *,
        prior_key: str | None = None,
    ) -> KeyEstimate:
        """Fold one policy version's observations of ``key`` into its estimate.

        A key seen for the first time starts from ``prior_key``'s estimate when
        that key is warm, with ``inherit_fraction`` of its count, so a fresh
        archive node begins where its parent's evidence points instead of at
        the batch mean. ``version`` is the policy version that produced the
        rewards; the drift is the version gap since the key's last update.
        """
        if not rewards:
            raise ValueError("observe requires at least one reward")
        if version < 0:
            raise ValueError("version must be non-negative")
        finite = [float(reward) for reward in rewards]
        if any(not math.isfinite(reward) for reward in finite):
            raise ValueError("rewards must be finite")

        estimate = self._estimates.get(key)
        if estimate is None and prior_key is not None:
            prior = self._estimates.get(prior_key)
            if prior is not None:
                estimate = KeyEstimate(prior.mean, prior.count * self.settings.inherit_fraction, prior.version)

        batch_mean = math.fsum(finite) / len(finite)
        if estimate is None:
            updated = KeyEstimate(batch_mean, float(len(finite)), version)
        else:
            if version < estimate.version:
                raise ValueError(f"key {key!r} was already updated by a later version {estimate.version}")
            rho = self.settings.forgetting_factor(float(version - estimate.version))
            count = rho * estimate.count + len(finite)
            mean = estimate.mean + (len(finite) / count) * (batch_mean - estimate.mean)
            updated = KeyEstimate(mean, count, version)
        self._estimates[key] = updated
        self._evict()
        return updated

    def state_dict(self) -> dict[str, Any]:
        """JSON-compatible snapshot: ``{key: [mean, count, version]}``."""
        return {key: estimate.as_list() for key, estimate in self._estimates.items()}

    def _load(self, state: Mapping[str, Any]) -> None:
        for key, row in state.items():
            if not isinstance(key, str) or not isinstance(row, Sequence) or len(row) != 3:
                raise ValueError("tracker state rows must be {key: [mean, count, version]}")
            mean, count, version = float(row[0]), float(row[1]), int(row[2])
            if not math.isfinite(mean) or not math.isfinite(count) or count < 0 or version < 0:
                raise ValueError(f"tracker state row for {key!r} is invalid")
            self._estimates[key] = KeyEstimate(mean, count, version)

    def _evict(self) -> None:
        # The archive itself is capped, so the table stays small; when it does
        # overflow, drop the keys whose evidence is oldest. The task key is
        # the fallback of last resort and is never evicted.
        excess = len(self._estimates) - self.settings.max_keys
        if excess <= 0:
            return
        candidates = sorted(
            (key for key in self._estimates if key != TASK_KEY),
            key=lambda key: (self._estimates[key].version, key),
        )
        for key in candidates[:excess]:
            self._estimates.pop(key, None)


@dataclass(frozen=True)
class Observation:
    """One scored rollout as the tracker sees it."""

    key: str
    reward: float
    prior_key: str | None = None
    fallbacks: tuple[str, ...] = ()


def batch_normalize(values: Sequence[float], epsilon: float = NORMALIZATION_EPSILON) -> tuple[float, ...]:
    """SPO's global normalisation: centre and scale across the whole batch.

    A batch with fewer than two values, or one where every value is equal,
    carries no relative signal and normalises to zeros.
    """
    if len(values) < 2:
        return tuple(0.0 for _ in values)
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return tuple(0.0 for _ in values)
    return tuple((value - mean) / (std + epsilon) for value in values)


def assign_advantages(
    tracker: ForgettingTracker,
    observations: Sequence[Observation],
    version: int,
    *,
    normalize: bool = True,
) -> tuple[tuple[float, ...], dict[str, float]]:
    """Advantages for one step, then the tracker update with that step's rewards.

    Every baseline is read before any key is updated (SPO's ``v_{-1}``), which
    keeps the baseline independent of the reward it is subtracted from and the
    gradient unbiased. Observations whose whole fallback chain is cold use the
    step's mean reward, so the first step of a run trains on centred rewards
    rather than on nothing. Returns the advantages in observation order and a
    few diagnostics for the step's metrics.
    """
    if not observations:
        return (), {"tracked_fraction": 0.0, "tracker_keys": float(len(tracker))}
    step_mean = math.fsum(observation.reward for observation in observations) / len(observations)
    raw: list[float] = []
    warm = 0
    for observation in observations:
        baseline = tracker.baseline(observation.key, observation.fallbacks)
        if baseline is None:
            baseline = step_mean
        else:
            warm += 1
        raw.append(observation.reward - baseline)

    grouped: dict[str, tuple[str | None, list[float]]] = {}
    for observation in observations:
        grouped.setdefault(observation.key, (observation.prior_key, []))[1].append(observation.reward)
    for key, (prior_key, rewards) in grouped.items():
        tracker.observe(key, rewards, version, prior_key=prior_key)

    advantages = batch_normalize(raw) if normalize else tuple(raw)
    metrics = {
        "tracked_fraction": warm / len(observations),
        "raw_advantage_abs_mean": math.fsum(abs(value) for value in raw) / len(raw),
        "advantage_abs_mean": math.fsum(abs(value) for value in advantages) / len(advantages),
        "tracker_keys": float(len(tracker)),
    }
    return advantages, metrics
