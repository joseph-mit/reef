"""Stepping-stone SPO-TTT's processor: SPO-TTT's step barrier, plus the store of recent attempts."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from recipes.hspottt.batches import HspotttBatch
from recipes.hspottt.stones import (
    Attempt,
    StoneSettings,
    StoneStore,
    admissible,
    load_store,
    save_store,
    settings_from,
)
from recipes.spottt.processor import SPOTTTProcessor
from reef.train.types import PolicySample, ProcessorContext

logger = logging.getLogger(__name__)

_STORE_FILE = re.compile(r"(base|result)-(\d+)\.pickle")


class HSPOTTTProcessor(SPOTTTProcessor):
    """Hand over each complete step with the stepping stones whose wait ended at it.

    A step's batch is a pure function of its attempts and of the store as it
    stood before the step (the step's *base*). The base is fixed the first
    time the step is built and kept until the step is built again or the next
    step begins, so a batch built again (after a failed training attempt, a
    scenario reload, or a restart of the Reef service) is identical, and the
    trainer can recognise a training job it already ran. The store advances
    to the step's result when the trainer acknowledges the batch. With
    ``stone_dir`` set, the base is written there before the batch leaves and
    the result when it is acknowledged, which carries both across a reload or
    a restart; without it the store lives in memory only.

    ``replay_max_lag`` is the recipe's ``max_staleness``: a replay the
    trainer's staleness check would refuse (and with it the whole step) is
    left out instead.
    """

    output_schema = HspotttBatch
    batch_label = "hspottt"

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        settings = settings_from(config)
        self.replay_max_lag = int(config.get("replay_max_lag", settings.window))
        if self.replay_max_lag < settings.window:
            raise ValueError(
                f"replays are stone_window={settings.window} versions old; "
                f"max_staleness must be at least that, got {self.replay_max_lag}"
            )
        super().__init__(context)
        directory = str(config.get("stone_dir") or "")
        self._stone_dir = Path(directory) if directory else None
        self._store = StoneStore(settings)
        self._base: tuple[int, StoneStore] | None = None
        self._pending_store: StoneStore | None = None
        self._restore(settings)

    # ---------------------------------------------------------------- store
    #
    # Two files per scenario carry the store across a reload or restart:
    # ``base-N``, the store step N is planned from, written at its first build,
    # and ``result-N``, the store after step N, written when the trainer
    # acknowledges it. A rebuilt step N plans from base-N; step N + 1 plans
    # from result-N. Both stay until base-(N + 1) is written.

    def _file(self, kind: str, step: int) -> Path:
        if self._stone_dir is None:
            raise RuntimeError("no stone_dir configured")
        scenario = re.sub(r"[^A-Za-z0-9_.-]", "_", self.scenario)
        return self._stone_dir / f"{scenario}.{kind}-{step:06d}.pickle"

    def _saved(self) -> list[tuple[int, str, Path]]:
        """The files written for this scenario as ``(step, kind, path)``, oldest step first."""
        if self._stone_dir is None or not self._stone_dir.is_dir():
            return []
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", self.scenario) + "."
        found = []
        for path in self._stone_dir.iterdir():
            match = _STORE_FILE.fullmatch(path.name[len(prefix) :]) if path.name.startswith(prefix) else None
            if match is not None:
                found.append((int(match.group(2)), match.group(1), path))
        return sorted(found)

    def _load(self, path: Path, settings: StoneSettings) -> StoneStore | None:
        try:
            store = load_store(path)
        except Exception as exc:  # a corrupt or incompatible file costs credit, never the run
            logger.warning("could not read the stepping-stone store %s (%s); starting empty", path, exc)
            return None
        if store.settings != settings:
            logger.warning("stepping-stone settings changed since %s was written; starting empty", path)
            return None
        return store

    def _restore(self, settings: StoneSettings) -> None:
        """Pick up what a previous process wrote: the newest base, and the result after it if any."""
        saved = self._saved()
        bases = [(step, path) for step, kind, path in saved if kind == "base"]
        if not bases:
            return
        step, path = bases[-1]
        base = self._load(path, settings)
        if base is None:
            return
        self._base = (step, base)
        self._store = base
        result = next((path for saved_step, kind, path in saved if kind == "result" and saved_step == step), None)
        if result is not None and (store := self._load(result, settings)) is not None:
            self._store = store

    def _base_for(self, step: int) -> StoneStore:
        """The store this step is planned from: fixed at its first build, then reused."""
        if self._base is not None and self._base[0] == step:
            return self._base[1]
        base = self._store.before(step)
        if self._stone_dir is not None:
            save_store(self._file("base", step), base)
            for saved_step, _, path in self._saved():
                if saved_step != step:
                    path.unlink(missing_ok=True)
        self._base = (step, base)
        return base

    # ---------------------------------------------------------------- batch

    def make_row(self, sample: PolicySample, report: Any) -> PolicySample:
        sample = super().make_row(sample, report)
        return replace(
            sample,
            extras={
                **sample.extras,
                "solution_sha1": report.solution_sha1,
                "parent_solution_sha1": report.parent_solution_sha1,
            },
        )

    def make_step_batch(self, step: Any, samples: tuple[PolicySample, ...]) -> HspotttBatch:
        if not isinstance(step, int):
            raise TypeError(f"step key must be an integer, got {step!r}")
        attempts = [
            Attempt(
                str(sample.extras.get("solution_sha1") or ""),
                str(sample.extras.get("parent_solution_sha1") or ""),
                str(sample.extras.get("parent_id") or ""),
                sample.reward,
            )
            for sample in samples
        ]
        plan = self._base_for(step).plan(step, attempts, samples)
        replays, refused = admissible(plan.replays, samples, self.replay_max_lag)
        self._pending_store = plan.store
        self.experiment_logger.log(
            {"step": step, **plan.metrics, "replays_admitted": len(replays), "replays_refused": refused},
            namespace=self.batch_label,
        )
        base = super().make_step_batch(step, samples)
        return HspotttBatch(
            base.batch_id,
            samples + tuple(replay.sample for replay in replays),
            ppo_epochs=base.ppo_epochs,
            minibatch_size=base.minibatch_size,
            shuffle=base.shuffle,
            baseline=base.baseline,
            normalize=base.normalize,
            half_life=base.half_life,
            rho_min=base.rho_min,
            rho_max=base.rho_max,
            inherit_fraction=base.inherit_fraction,
            replay_credits=tuple(replay.credit for replay in replays),
        )

    def _consume_pending(self) -> frozenset[str]:
        consumed = super()._consume_pending()
        if self._pending_store is not None:
            self._store = self._pending_store
            self._pending_store = None
            if self._stone_dir is not None and self._base is not None:
                save_store(self._file("result", self._base[0]), self._store)
        return consumed
