"""One line per committed step of a run: progress, speed and health.

Reads the state directory a job writes (``events.jsonl`` and Reef's
``agent-record/*.commits.jsonl``), so it works on the login node while the
job runs, with any Python 3:

    python3 run_status.py ~/orcd/scratch/reef-work/runs/spottt/circle_packing_26

Columns:

- ``mean`` and ``best``: the step's mean attempt reward and the archive's best.
- ``step min``, ``tok/s``: wall time of the step and training throughput.
- ``retries``: CUDA allocator retries on training rank 0 so far; each is a
  device-wide stall, so a count that grows by hundreds per step means the
  trainer is short of GPU memory.
- ``mismatch``: ``train/ppo_kl`` of the step's first optimizer step, before
  any update, i.e. how far the trainer's log-probabilities are from the
  sampler's. It should stay near its step-1 value; a jump or a steady climb
  means the engine is not serving the weights being trained.
- ``kl base``: the sampled policy's KL to the frozen base; it grows as
  training moves the policy and stays flat if the engine serves the base.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

COLUMNS = ("step", "mean", "best", "step min", "tok/s", "retries", "mismatch", "kl base")


def _jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def step_rows(state: Path) -> list[dict[str, float | None]]:
    """Committed steps, 1-based, joining the harness's events with Reef's commits."""
    best: dict[int, float] = {}
    events = state / "events.jsonl"
    if events.is_file():
        for event in _jsonl([events]):
            if event.get("event") == "tttd_step_committed":
                best[int(event["step"]) + 1] = float(event["archive_best_reward"])
    commits: dict[int, Mapping[str, Any]] = {}
    for record in _jsonl(sorted((state / "agent-record").glob("*.commits.jsonl"))):
        if isinstance(record.get("step"), int) and record.get("operation", "training") == "training":
            commits[record["step"]] = record.get("metrics") or {}
    rows = []
    for step in sorted(set(best) | set(commits)):
        metrics = commits.get(step, {})
        steps = metrics.get("train_steps") or [{}]
        seconds = _number(metrics.get("perf/step_time"))
        rows.append(
            {
                "step": float(step),
                "mean": _number(metrics.get("rollout/rewards")),
                "best": best.get(step),
                "step min": None if seconds is None else seconds / 60,
                "tok/s": _number(metrics.get("perf/actor_train_tok_per_s")),
                "retries": _number(metrics.get("memory/alloc_retries")),
                "mismatch": _number(steps[0].get("train/ppo_kl")),
                "kl base": _number(metrics.get("rollout/kl")),
            }
        )
    return rows


def _cell(name: str, value: float | None) -> str:
    if value is None:
        return ""
    if name in ("step", "retries", "tok/s"):
        return f"{value:.0f}"
    if name == "best":
        return f"{value:.6f}"
    if name in ("mismatch", "kl base"):
        return f"{value:.5f}"
    return f"{value:.2f}"


def table(rows: list[dict[str, float | None]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + " --- |" * len(COLUMNS)]
    lines += ["| " + " | ".join(_cell(name, row[name]) for name in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1].strip())
        return 2
    rows = step_rows(Path(argv[0]).expanduser())
    if not rows:
        print(f"no committed steps under {argv[0]}")
        return 1
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
