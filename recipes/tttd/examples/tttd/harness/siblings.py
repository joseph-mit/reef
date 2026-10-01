"""How many attempts each parent gets in a search step.

TTT-Discover expands every selected parent with the same number of attempts
(64 on the shipped grid). Those siblings do two jobs: they try to improve on
the parent, and they measure what is normal from it, which TTT-Discover's
group advantage needs. A method that learns what is normal from history
(SPO-TTT, entropic SPO-TTT, PPO-TTT's critic) needs the second job done only
where history is thin. ``adaptive`` spends the same step budget accordingly:
a parent gets enough attempts to bring what is known about it up to
``target`` outcomes, at least ``min_siblings`` and at most ``max_siblings``,
so more parents are expanded per step once their outcomes are known.

What is known about a parent is a forgetting count of the attempts made from
it, kept with the archive so a restart resumes it. A parent never expanded
before starts from ``inherit_fraction`` of what is known about its own
parent, as the Reef-side trackers start a new state from its parent's
history. A parent whose latest
attempts score far from what was known (more than ``surprise_z`` standard
errors) is treated as unknown again, since its old outcomes no longer
describe it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

POLICIES = ("fixed", "adaptive")


@dataclass(frozen=True)
class SiblingSettings:
    """``fixed`` is TTT-Discover's grid; ``adaptive`` spends the same budget by what is already known."""

    policy: str = "fixed"
    min_siblings: int = 8
    max_siblings: int = 64
    target: float = 64.0
    max_parents: int = 32
    half_life: float = 8.0
    surprise_z: float = 3.0
    inherit_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"sibling policy must be one of {', '.join(POLICIES)}, got {self.policy!r}")
        if isinstance(self.min_siblings, bool) or self.min_siblings < 1:
            raise ValueError("min_siblings must be a positive integer")
        if isinstance(self.max_siblings, bool) or self.max_siblings < self.min_siblings:
            raise ValueError("max_siblings must be at least min_siblings")
        if isinstance(self.max_parents, bool) or self.max_parents < 1:
            raise ValueError("max_parents must be a positive integer")
        for name in ("target", "half_life", "surprise_z"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not 0 <= self.inherit_fraction <= 1:
            raise ValueError("inherit_fraction must lie in [0, 1]")

    @property
    def adaptive(self) -> bool:
        return self.policy == "adaptive"

    def identity(self) -> tuple[tuple[str, Any], ...] | None:
        """What a run's state file must agree on: nothing for the fixed grid, every setting otherwise."""
        if not self.adaptive:
            return None
        return tuple((name, getattr(self, name)) for name in self.__dataclass_fields__)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> SiblingSettings:
        """Read a deployment config's ``search.siblings`` block; absent means ``fixed``."""
        if not values:
            return cls()
        known = {name: values[name] for name in cls.__dataclass_fields__ if name in values}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown search.siblings settings: {', '.join(unknown)}")
        return cls(**known)


@dataclass
class ParentOutcomes:
    """Forgetting count, mean and squared deviations of one parent's attempt scores."""

    count: float
    mean: float
    squares: float
    step: int

    def decayed(self, step: int, half_life: float) -> ParentOutcomes:
        factor = 2.0 ** (-max(step - self.step, 0) / half_life)
        return ParentOutcomes(self.count * factor, self.mean, self.squares * factor, step)

    def variance(self) -> float:
        return self.squares / self.count if self.count > 0 else 0.0


class AttemptStats:
    """What the search knows about each parent's attempts, saved with the archive."""

    def __init__(self, state: Mapping[str, Any] | None = None) -> None:
        self._parents: dict[str, ParentOutcomes] = {}
        for key, row in (state or {}).items():
            count, mean, squares, step = (float(row[0]), float(row[1]), float(row[2]), int(row[3]))
            if not all(math.isfinite(value) for value in (count, mean, squares)) or count < 0 or squares < 0:
                raise ValueError(f"invalid attempt statistics for {key!r}")
            self._parents[str(key)] = ParentOutcomes(count, mean, squares, step)

    def known(self, candidate_id: str, step: int, settings: SiblingSettings, parent_id: str | None = None) -> float:
        """The forgetting count of attempts from ``candidate_id`` as of ``step``.

        A candidate never expanded counts ``inherit_fraction`` of its parent's.
        """
        outcomes = self._parents.get(candidate_id)
        if outcomes is not None:
            return outcomes.decayed(step, settings.half_life).count
        parent = self._parents.get(parent_id) if parent_id else None
        return 0.0 if parent is None else settings.inherit_fraction * parent.decayed(step, settings.half_life).count

    def get(self, candidate_id: str) -> ParentOutcomes | None:
        return self._parents.get(candidate_id)

    def record(self, candidate_id: str, rewards: Sequence[float], step: int, settings: SiblingSettings) -> bool:
        """Fold one step's scores for a parent in; return whether they surprised what was known."""
        if not rewards:
            return False
        count = float(len(rewards))
        mean = math.fsum(rewards) / count
        squares = math.fsum((reward - mean) ** 2 for reward in rewards)
        previous = self._parents.get(candidate_id)
        surprised = False
        if previous is not None:
            past = previous.decayed(step, settings.half_life)
            if past.count >= 2 and count >= 2:
                variance = max(past.variance(), squares / count, 1e-12)
                standard_error = math.sqrt(variance / count + variance / past.count)
                surprised = abs(mean - past.mean) > settings.surprise_z * standard_error
            if not surprised and past.count > 0:
                total = past.count + count
                delta = mean - past.mean
                self._parents[candidate_id] = ParentOutcomes(
                    total,
                    past.mean + delta * count / total,
                    past.squares + squares + delta * delta * past.count * count / total,
                    step,
                )
                return False
        self._parents[candidate_id] = ParentOutcomes(count, mean, squares, step)
        return surprised

    def state_dict(self, keep: Iterable[str] | None = None) -> dict[str, list[float | int]]:
        """``{candidate_id: [count, mean, squares, step]}``, limited to ``keep`` when given."""
        kept = None if keep is None else set(keep)
        return {
            key: [value.count, value.mean, value.squares, value.step]
            for key, value in self._parents.items()
            if kept is None or key in kept
        }


def allocate_siblings(known: Sequence[float], budget: int, settings: SiblingSettings) -> list[int]:
    """Attempts per parent, in the order given (PUCT's), summing to ``budget``.

    Each parent gets what brings its known outcomes up to ``target``, clipped
    to ``[min_siblings, max_siblings]``, until the budget cannot cover another
    parent's minimum. What is left goes round the chosen parents one attempt
    at a time, starting from the first, so the budget is always spent.
    """
    if budget < 1:
        raise ValueError("the step budget must be positive")
    allocations: list[int] = []
    remaining = budget
    for count in known:
        if remaining < settings.min_siblings:
            break
        need = math.ceil(settings.target - count)
        share = min(max(need, settings.min_siblings), settings.max_siblings, remaining)
        allocations.append(share)
        remaining -= share
    if not allocations:
        return [budget] if known else []
    index = 0
    while remaining > 0:
        allocations[index % len(allocations)] += 1
        remaining -= 1
        index += 1
    return allocations
