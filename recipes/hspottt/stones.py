"""Stepping-stone credit: which recent attempts later led to a success, and what they earn.

An attempt that scores modestly can still be the program a top solution later
grows from. The search records that link: every attempt names the program of
the archive state it was expanded from (``parent_solution_sha1``), and an
attempt that enters the archive is the state its program names. Following
those links upwards from a successful attempt reaches the earlier attempts it
descends from.

This module keeps, for a few steps, the attempts that could become archive
states (the best ``keep_per_parent`` of each parent's attempts in a step: the
archive keeps at most two children per state, so no archived attempt is
missed). When a later attempt succeeds, each kept attempt above it that did
not itself succeed is credited ``credit * discount ** (k - 1)`` at ``k``
generations up. When an attempt has waited ``window`` steps it leaves the
store, and if it earned credit its sample is trained once more with that
credit as its advantage. This is self-imitation learning (Oh et al., 2018):
a positive-only update on a past sample whose hindsight return beat what it
was credited with at the time, here with descendants' success as the return.

Everything here is a pure function of the state before a step and the step's
attempts: :meth:`StoneStore.plan` returns the next state instead of changing
this one, so a batch that is built again after a failed attempt to train it
comes out identical. :func:`save_store` and :func:`load_store` keep that
state on disk between processes.
"""

from __future__ import annotations

import math
import os
import pickle
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from reef.train.types import PolicySample

SUCCESS_RULES = ("top", "best")

DEFAULT_WINDOW = 3
DEFAULT_GENERATIONS = 3
DEFAULT_DISCOUNT = 0.5
DEFAULT_CREDIT = 1.0
DEFAULT_SUCCESS = "top"
DEFAULT_TOP_FRACTION = 0.05
DEFAULT_KEEP_PER_PARENT = 2
DEFAULT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StoneSettings:
    """How success is judged and how far its credit reaches.

    ``success`` is ``top`` (an attempt among the best ``top_fraction`` of the
    step's attempts with a program) or ``best`` (an attempt that beats the best
    score seen before its step by more than ``tolerance``, relative).
    ``credit`` is the advantage a success's parent earns, in units of the
    step's fresh advantages (their standard deviation when they are
    normalised); each generation further up keeps ``discount`` of it, up to
    ``generations`` up. An attempt waits ``window`` steps for descendants
    before its replay, so replays are exactly ``window`` policy versions old.
    """

    window: int = DEFAULT_WINDOW
    generations: int = DEFAULT_GENERATIONS
    discount: float = DEFAULT_DISCOUNT
    credit: float = DEFAULT_CREDIT
    success: str = DEFAULT_SUCCESS
    top_fraction: float = DEFAULT_TOP_FRACTION
    keep_per_parent: int = DEFAULT_KEEP_PER_PARENT
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self) -> None:
        for name in ("window", "generations", "keep_per_parent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not math.isfinite(self.credit) or self.credit <= 0:
            raise ValueError("credit must be a positive finite number")
        if self.success not in SUCCESS_RULES:
            raise ValueError(f"success must be one of {', '.join(SUCCESS_RULES)}, got {self.success!r}")
        if not 0 < self.top_fraction < 1:
            raise ValueError("top_fraction must lie in (0, 1)")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be a non-negative finite number")


@dataclass(frozen=True)
class PackedSample:
    """A policy sample with its per-token sequences in compact arrays.

    Tokens take 4 bytes each, log-probabilities 8 (exact float64) and masks 1,
    against about 36 for a Python int and 32 for a float in a tuple. A kept
    attempt holds its whole prompt and response, so this is most of the
    store's memory.
    """

    shell: PolicySample
    tokens: array
    loss_mask: bytes | tuple[int, ...]
    rollout_log_probs: array
    action_mask: bytes | tuple[int, ...]

    @classmethod
    def pack(cls, sample: PolicySample) -> PackedSample:
        return cls(
            replace(sample, tokens=(), loss_mask=(), rollout_log_probs=(), action_mask=()),
            _int_array(sample.tokens),
            _mask_bytes(sample.loss_mask),
            array("d", sample.rollout_log_probs),
            _mask_bytes(sample.action_mask),
        )

    def unpack(self) -> PolicySample:
        return replace(
            self.shell,
            tokens=tuple(self.tokens),
            loss_mask=tuple(self.loss_mask),
            rollout_log_probs=tuple(self.rollout_log_probs),
            action_mask=tuple(self.action_mask),
        )


def _int_array(values: Sequence[int]) -> array:
    try:
        return array("i", values)
    except OverflowError:
        return array("q", values)


def _mask_bytes(mask: Sequence[int]) -> bytes | tuple[int, ...]:
    try:
        return bytes(mask)
    except (TypeError, ValueError):
        return tuple(mask)


@dataclass(frozen=True)
class Stone:
    """A kept attempt: its program, its parent's program, and what it has earned so far."""

    key: str
    parent_key: str
    step: int
    order: int
    reward: float
    success: bool
    credit: float
    sample: PackedSample


@dataclass(frozen=True)
class Attempt:
    """One fresh attempt as the store reads it."""

    key: str
    parent_key: str
    parent_id: str
    reward: float


@dataclass(frozen=True)
class Replay:
    """A kept attempt that earned credit and whose wait is over."""

    sample: PolicySample
    credit: float
    step: int


@dataclass(frozen=True)
class Plan:
    """What one step does to the store: the next store, the replays it releases, and counts."""

    store: StoneStore
    replays: tuple[Replay, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class StoneStore:
    """The kept attempts by program, and the best score seen; immutable."""

    settings: StoneSettings = field(default_factory=StoneSettings)
    stones: Mapping[str, Stone] = field(default_factory=dict)
    best: float | None = None

    def plan(self, step: int, attempts: Sequence[Attempt], samples: Sequence[PolicySample]) -> Plan:
        """Fold one step's attempts in; O(n log n + successes * generations) for n attempts."""
        if len(attempts) != len(samples):
            raise ValueError("plan needs one sample per attempt")
        if any(stone.step >= step for stone in self.stones.values()):
            raise ValueError(f"step {step} is not after every kept attempt")
        settings = self.settings
        successes = self._successes(attempts)

        # Credit flows up each success's chain of parents to kept attempts that
        # did not succeed themselves; an attempt keeps the largest credit offered.
        credits = {key: stone.credit for key, stone in self.stones.items()}
        for index in successes:
            key, generation = attempts[index].parent_key, 1
            while key and generation <= settings.generations and key in self.stones:
                stone = self.stones[key]
                if not stone.success:
                    offered = settings.credit * settings.discount ** (generation - 1)
                    credits[key] = max(credits[key], offered)
                key, generation = stone.parent_key, generation + 1

        kept: dict[str, Stone] = {}
        replays: list[Replay] = []
        for key, stone in sorted(self.stones.items(), key=lambda item: (item[1].step, item[1].order)):
            if step - stone.step >= settings.window:
                if credits[key] > 0:
                    replays.append(Replay(stone.sample.unpack(), credits[key], stone.step))
            else:
                kept[key] = replace(stone, credit=credits[key]) if credits[key] != stone.credit else stone

        added = 0
        success_set = set(successes)
        for index in self._candidates(attempts):
            attempt = attempts[index]
            if attempt.key in kept:
                continue  # the archive keeps a program once, under its first attempt
            kept[attempt.key] = Stone(
                attempt.key,
                attempt.parent_key,
                step,
                index,
                attempt.reward,
                index in success_set,
                0.0,
                PackedSample.pack(samples[index]),
            )
            added += 1

        rewards = [attempt.reward for attempt in attempts]
        best = max(rewards, default=self.best) if self.best is None else max([self.best, *rewards])
        credited = sum(1 for key in self.stones if credits[key] > self.stones[key].credit)
        metrics = {
            "successes": float(len(successes)),
            "newly_credited": float(credited),
            "kept": float(len(kept)),
            "added": float(added),
            "replays": float(len(replays)),
        }
        return Plan(StoneStore(settings, kept, best), tuple(replays), metrics)

    def before(self, step: int) -> StoneStore:
        """This store without attempts from ``step`` or later (a step trained again starts from here)."""
        if all(stone.step < step for stone in self.stones.values()):
            return self
        kept = {key: stone for key, stone in self.stones.items() if stone.step < step}
        return StoneStore(self.settings, kept, self.best)

    def _successes(self, attempts: Sequence[Attempt]) -> list[int]:
        """Indices of the step's successful attempts, in batch order."""
        settings = self.settings
        programs = [index for index, attempt in enumerate(attempts) if attempt.key]
        if not programs:
            return []
        if settings.success == "best":
            if self.best is None:
                return []  # nothing to beat yet: the first step only sets the bar
            margin = settings.tolerance * max(1.0, abs(self.best))
            return [index for index in programs if attempts[index].reward > self.best + margin]
        ranked = sorted((attempts[index].reward for index in programs), reverse=True)
        threshold = ranked[max(1, math.ceil(settings.top_fraction * len(ranked))) - 1]
        return [index for index in programs if attempts[index].reward >= threshold]

    def _candidates(self, attempts: Sequence[Attempt]) -> Iterable[int]:
        """The best ``keep_per_parent`` attempts with a program from each parent, in batch order."""
        by_parent: dict[str, list[int]] = {}
        for index, attempt in enumerate(attempts):
            # An attempt that returned its parent's program unchanged is no new
            # archive state; leaving it out keeps the slot for one that is.
            if attempt.key and attempt.key != attempt.parent_key:
                by_parent.setdefault(attempt.parent_id, []).append(index)
        chosen: set[int] = set()
        for indices in by_parent.values():
            ranked = sorted(indices, key=lambda index: (-attempts[index].reward, index))
            chosen.update(ranked[: self.settings.keep_per_parent])
        return sorted(chosen)


def save_store(path: Path, store: StoneStore) -> None:
    """Write ``store`` to ``path`` atomically.

    Pickle keeps the samples' arrays and dataclasses exactly and compactly.
    The file is the processor's own state under the run's directory, read back
    only by :func:`load_store`; never point it at a file from elsewhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_store(path: Path) -> StoneStore:
    """The store :func:`save_store` wrote to ``path``."""
    with path.open("rb") as handle:
        # The processor's own file under the run's directory (see save_store).
        store = pickle.load(handle)
    if not isinstance(store, StoneStore):
        raise TypeError(f"{path} does not hold a stone store")
    return store


def runtime_version(value: str | None) -> tuple[str, int] | None:
    """``(incarnation, sequence)`` from a runtime load ID ``<incarnation>:<sequence>``; None when absent or malformed."""
    if not isinstance(value, str):
        return None
    incarnation, separator, sequence = value.rpartition(":")
    if not separator or not incarnation or not sequence.isascii() or not sequence.isdecimal():
        return None
    return incarnation, int(sequence)


def admissible(replays: Iterable[Replay], fresh: Sequence[PolicySample], max_lag: int) -> tuple[list[Replay], int]:
    """The replays the trainer's staleness check will admit with the fresh samples, and how many it would not.

    The trainer refuses a whole batch if any sample comes from another
    incarnation of the serving runtime or more than ``max_staleness`` versions
    back, so a replay is kept only when the fresh samples share one version
    and the replay is from the same incarnation, between 1 and ``max_lag``
    versions older.
    """
    candidates = list(replays)
    versions = {runtime_version(sample.runtime_load_id) for sample in fresh}
    if len(versions) != 1 or None in versions:
        return [], len(candidates)
    (current,) = versions
    if current is None:
        return [], len(candidates)
    kept: list[Replay] = []
    for replay in candidates:
        version = runtime_version(replay.sample.runtime_load_id)
        if version is not None and version[0] == current[0] and 1 <= current[1] - version[1] <= max_lag:
            kept.append(replay)
    return kept, len(candidates) - len(kept)


def settings_from(config: Mapping[str, Any]) -> StoneSettings:
    """Stone settings from a processor config's ``stone_*`` keys, defaults otherwise."""
    return StoneSettings(
        window=int(config.get("stone_window", DEFAULT_WINDOW)),
        generations=int(config.get("stone_generations", DEFAULT_GENERATIONS)),
        discount=float(config.get("stone_discount", DEFAULT_DISCOUNT)),
        credit=float(config.get("stone_credit", DEFAULT_CREDIT)),
        success=str(config.get("stone_success", DEFAULT_SUCCESS)),
        top_fraction=float(config.get("stone_top_fraction", DEFAULT_TOP_FRACTION)),
        keep_per_parent=int(config.get("stone_keep_per_parent", DEFAULT_KEEP_PER_PARENT)),
        tolerance=float(config.get("stone_tolerance", DEFAULT_TOLERANCE)),
    )


__all__ = [
    "SUCCESS_RULES",
    "Attempt",
    "PackedSample",
    "Plan",
    "Replay",
    "Stone",
    "StoneSettings",
    "StoneStore",
    "admissible",
    "load_store",
    "runtime_version",
    "save_store",
    "settings_from",
]
