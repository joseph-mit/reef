"""Per-step statistics of a run, per-attempt scores saved to CSV, and what is recorded.

    python3 run_report.py ~/orcd/scratch/reef-work/runs/spottt/circle_packing_26

It reads only files, so it runs on the login node while the job runs (a run
kept on a node's local disk is read on that node through srun). It prints:

- a table per committed step: mean score, share of failed attempts (score 0),
  mean of the valid attempts, median, mean of the top 10%, the step's best
  attempt, the best so far, step time, and training signals (policy entropy,
  KL to the base model, clip fraction, gradient norm) where Reef recorded them,
  and for PPO-TTT the value loss at the step's first and last critic update;
- from the harness's step files (``attempts/step-NNNN.jsonl.gz``, written by
  runs started after they were added): how often attempts beat their parent
  and the best so far, how varied the programs were, why attempts failed, and
  what each step cost in tokens and time;
- the training signals: how well SPO-TTT's baseline and PPO-TTT's critic
  predicted each step's scores (explained variance: 1 is perfect, 0 no
  better than a constant);
- what the run recorded: the metric names in Reef's commits, and how many
  per-attempt records Reef still holds and when it starts deleting them.

Every column is also written to ``report/steps.csv`` for plotting.

By default Reef deletes an attempt's record 7 days after a training step
consumed it (``reef.agent_record_retention_days``; the TTT-Discover configs
now keep 90). Every run of this script writes all
attempts it can still read to ``report/attempts.csv`` in the state directory,
keeping rows it wrote before, so the per-attempt scores outlive that window.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

RETENTION_DAYS = 7.0
ATTEMPT_FIELDS = ("record_id", "step", "group", "rollout", "parent_id", "grandparent_id", "score", "feedback")
# Commit metrics shown per step when present, by the end of their key.
TRAINING_COLUMNS = {
    "entropy": "entropy_loss",
    "kl base": "rollout/kl",
    "clip frac": "pg_clipfrac",
    "grad norm": "grad_norm",
}
CRITIC_STEPS_KEY = "critic_train_steps"
CRITIC_LOSS = "train/critic-value_loss"
# Summary numbers the harness adds to each step's committed event.
SEARCH_COLUMNS = {
    "beat parent %": ("beat_parent_fraction", 100.0),
    "new bests": ("beat_best_count", 1.0),
    "unique %": ("unique_solution_fraction", 100.0),
    "parent reward": ("parent_reward_mean", 1.0),
    "parent depth": ("parent_depth_mean", 1.0),
}
COST_COLUMNS = {
    "search min": ("search_seconds", 1 / 60),
    "train wait min": ("train_wait_seconds", 1 / 60),
    "tokens": ("completion_tokens_mean", 1.0),
    "tokens p90": ("completion_tokens_p90", 1.0),
    "truncated %": ("truncated_fraction", 100.0),
    "gen s": ("generation_seconds_mean", 1.0),
    "eval s": ("evaluation_seconds_mean", 1.0),
}
# Commit metrics, by exact key: how well a baseline predicted the step's scores.
FIT_COLUMNS = {
    "baseline EV": "baseline_explained_variance",
    "baseline corr": "baseline_reward_correlation",
    "value EV": "ppottt/value_explained_variance",
    "value corr": "ppottt/value_reward_correlation",
    "value failed": "ppottt/value_mean_failed",
    "value valid": "ppottt/value_mean_valid",
    "critic EV after fit": "critic/ppottt/value_explained_variance",
}
TABLES = {
    "scores": (
        "step",
        "attempts",
        "mean",
        "fail %",
        "valid mean",
        "median",
        "top 10%",
        "step best",
        "best so far",
        *SEARCH_COLUMNS,
    ),
    "cost": ("step", "minutes", *COST_COLUMNS),
    "training": (
        "step",
        *TRAINING_COLUMNS,
        *FIT_COLUMNS,
        "critic loss start",
        "critic loss end",
    ),
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metric(metrics: Mapping[str, Any], suffix: str) -> float | None:
    for key, value in metrics.items():
        if key == suffix or key.endswith("/" + suffix):
            return _number(value)
    return None


def read_attempts(state: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every scored attempt Reef still holds, and counts of what it holds."""
    attempts: list[dict[str, Any]] = []
    held = {"inference": 0, "report": 0, "compacted": 0, "oldest_compacted": None, "repeats": 0}
    # Reef trains on the first report at each (step, group, rollout) slot and
    # rejects later ones, which an interrupted and resumed step produces.
    slots: set[tuple[Any, Any, Any]] = set()
    for database in sorted((state / "agent-record").glob("*.sqlite3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT agent_record_id, request_type, payload_json, compacted_at FROM agent_record ORDER BY sequence"
            ).fetchall()
        finally:
            connection.close()
        for record_id, request_type, payload_json, compacted_at in rows:
            held[request_type] = held.get(request_type, 0) + 1
            if compacted_at is not None:
                held["compacted"] += 1
                oldest = held["oldest_compacted"]
                held["oldest_compacted"] = compacted_at if oldest is None else min(oldest, compacted_at)
            if request_type != "report":
                continue
            payload = json.loads(payload_json)
            metadata = payload.get("metadata") or {}
            score = _number(payload.get("score"))
            step = metadata.get("step")
            if score is None or not isinstance(step, int):
                continue
            slot = (step, metadata.get("group"), metadata.get("rollout"))
            if slot in slots:
                held["repeats"] += 1
                continue
            slots.add(slot)
            attempts.append(
                {
                    "record_id": record_id,
                    "step": step + 1,
                    "group": metadata.get("group"),
                    "rollout": metadata.get("rollout"),
                    "parent_id": metadata.get("parent_id"),
                    "grandparent_id": metadata.get("grandparent_id"),
                    "score": score,
                    "feedback": str(payload.get("feedback") or "")[:200],
                }
            )
    return attempts, held


def _slot(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["step"]), str(row["group"]), str(row["rollout"]))


def save_attempts(path: Path, attempts: Iterable[Mapping[str, Any]]) -> int:
    """Merge attempts into the CSV, one row per slot; return how many rows it holds.

    Attempts read from Reef replace saved rows at the same slot, so a CSV
    written before repeats were left out is corrected while Reef still holds
    the records.
    """
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if path.is_file():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.setdefault(_slot(row), row)
    for attempt in attempts:
        rows[_slot(attempt)] = {field: attempt.get(field) for field in ATTEMPT_FIELDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: (int(row["step"]), str(row["record_id"]))))
    temporary.replace(path)
    return len(rows)


def load_saved_scores(path: Path) -> dict[int, list[float]]:
    scores: dict[int, list[float]] = {}
    if path.is_file():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                scores.setdefault(int(row["step"]), []).append(float(row["score"]))
    return scores


def _fail_percent(values: list[float], valid: list[float], zero_fraction: float | None) -> float | None:
    if values:
        return 100 * (len(values) - len(valid)) / len(values)
    return None if zero_fraction is None else 100 * zero_fraction


def load_step_file_scores(state: Path) -> dict[int, list[float]]:
    """Scores from the harness's own step files, which outlive Reef's records."""
    scores: dict[int, list[float]] = {}
    for path in sorted((state / "attempts").glob("step-*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                lines = [json.loads(line) for line in handle if line.strip()]
        except (OSError, EOFError, ValueError):
            continue  # a file being written right now
        if lines and isinstance(lines[0].get("step"), int):
            scores[lines[0]["step"]] = [float(line["reward"]) for line in lines[1:]]
    return scores


def step_table(state: Path, scores: Mapping[int, list[float]]) -> list[dict[str, float | None]]:
    events = _jsonl(state / "events.jsonl")
    started: dict[int, float] = {}
    committed: dict[int, dict[str, Any]] = {}
    for event in events:
        step = event.get("step")
        if not isinstance(step, int):
            continue
        if event.get("event") == "tttd_step_started" and _number(event.get("time")) is not None:
            started[step + 1] = float(event["time"])
        if event.get("event") == "tttd_step_committed":
            committed[step + 1] = event
    commits: dict[int, Mapping[str, Any]] = {}
    for path in sorted((state / "agent-record").glob("*.commits.jsonl")):
        for record in _jsonl(path):
            if isinstance(record.get("step"), int) and record.get("operation", "training") == "training":
                commits[record["step"]] = record.get("metrics") or {}

    rows = []
    for step in sorted(committed):
        event = committed[step]
        values = sorted(scores.get(step, []))
        valid = [value for value in values if value != 0]
        top = values[-max(1, len(values) // 10) :] if values else []
        metrics = commits.get(step, {})
        finished = _number(event.get("time"))
        row: dict[str, float | None] = {
            "step": float(step),
            "attempts": float(len(values)) if values else None,
            "mean": statistics.mean(values) if values else _number(event.get("reward_mean")),
            "fail %": _fail_percent(values, valid, _number(event.get("reward_zero_fraction"))),
            "valid mean": statistics.mean(valid) if valid else None,
            "median": statistics.median(values) if values else None,
            "top 10%": statistics.mean(top) if top else None,
            "step best": max(values) if values else _number(event.get("reward_max")),
            "best so far": _number(event.get("archive_best_reward")),
            "minutes": None if finished is None or step not in started else (finished - started[step]) / 60,
        }
        for columns in (SEARCH_COLUMNS, COST_COLUMNS):
            for name, (key, scale) in columns.items():
                value = _number(event.get(key))
                row[name] = None if value is None else value * scale
        for name, suffix in TRAINING_COLUMNS.items():
            row[name] = _metric(metrics, suffix)
        for name, key in FIT_COLUMNS.items():
            row[name] = _number(metrics.get(key))
        # The first update sees the step's rewards before the critic fits them.
        losses = [_number(entry.get(CRITIC_LOSS)) for entry in metrics.get(CRITIC_STEPS_KEY) or []]
        losses = [loss for loss in losses if loss is not None]
        row["critic loss start"] = losses[0] if losses else None
        row["critic loss end"] = losses[-1] if losses else None
        rows.append(row)
    return rows


def _cell(name: str, value: float | None) -> str:
    if value is None:
        return ""
    if name in ("step", "attempts"):
        return f"{value:.0f}"
    if name in ("step best", "best so far"):
        return f"{value:.6f}"
    if name in ("fail %", "minutes", "new bests", "tokens", "tokens p90") or name.endswith(" %"):
        return f"{value:.0f}"
    if name in ("search min", "train wait min", "gen s", "eval s", "parent depth"):
        return f"{value:.1f}"
    if name in ("kl base", "clip frac", "critic loss start", "critic loss end"):
        return f"{value:.5f}"
    return f"{value:.3f}"


def render(rows: list[dict[str, float | None]], names: Iterable[str] | None = None) -> str:
    """A markdown table of the columns that hold a value in some row; ``names`` picks and orders them."""
    wanted = list(rows[0]) if rows and names is None else list(names or ())
    columns = [name for name in wanted if any(row.get(name) is not None for row in rows)]
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(_cell(name, row.get(name)) for name in columns) + " |" for row in rows]
    return "\n".join(lines)


def failure_table(state: Path) -> str:
    """How many of each step's attempts failed for each reason, from the harness's step summaries."""
    per_step: dict[int, Mapping[str, Any]] = {}
    for event in _jsonl(state / "events.jsonl"):
        failures = event.get("failures")
        if event.get("event") == "tttd_step_committed" and isinstance(failures, Mapping):
            per_step[int(event["step"]) + 1] = failures
    if not per_step:
        return ""
    totals: dict[str, int] = {}
    for failures in per_step.values():
        for kind, count in failures.items():
            totals[kind] = totals.get(kind, 0) + int(count)
    kinds = sorted(totals, key=lambda kind: -totals[kind])[:8]
    lines = ["| step | " + " | ".join(kinds) + " |", "|" + " --- |" * (len(kinds) + 1)]
    for step, failures in sorted(per_step.items()):
        attempts = sum(int(count) for count in failures.values())
        cells = [str(int(failures.get(kind, 0))) for kind in kinds]
        lines.append(f"| {step} | " + " | ".join(cells) + f" |  ({attempts} failed)")
    return "failed attempts by reason:\n" + "\n".join(lines)


def save_steps(path: Path, rows: list[dict[str, float | None]]) -> None:
    columns = [name for name in rows[0] if any(row.get(name) is not None for row in rows)] if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([["" if row.get(name) is None else row[name] for name in columns] for row in rows])


def inventory(state: Path, held: Mapping[str, Any], saved: int) -> str:
    keys: set[str] = set()
    step_keys: set[str] = set()
    critic_keys: set[str] = set()
    for path in sorted((state / "agent-record").glob("*.commits.jsonl")):
        for record in _jsonl(path):
            metrics = record.get("metrics") or {}
            keys.update(key for key in metrics if key not in ("train_steps", CRITIC_STEPS_KEY))
            for entry in metrics.get("train_steps") or []:
                step_keys.update(entry)
            for entry in metrics.get(CRITIC_STEPS_KEY) or []:
                critic_keys.update(entry)
    lines = [
        f"commit metrics: {', '.join(sorted(keys)) or 'none'}",
        f"per-update training metrics: {', '.join(sorted(step_keys)) or 'none'}",
        f"per-update critic metrics: {', '.join(sorted(critic_keys)) or 'none'}",
        f"records Reef holds: {held.get('report', 0)} scored attempts, {held.get('inference', 0)} generations, "
        f"{held.get('compacted', 0)} already consumed by training",
        f"repeated attempts left out (a resumed step's later report at a slot Reef had already filled;"
        f" Reef trains on the first): {held.get('repeats', 0)}",
        f"attempts saved in report/attempts.csv: {saved}",
    ]
    oldest = held.get("oldest_compacted")
    if oldest is not None:
        left = RETENTION_DAYS - (time.time() - float(oldest)) / 86400
        lines.append(
            f"with Reef's default 7-day retention, deleting consumed records starts in about {max(left, 0):.1f} days"
            " (the configs keep 90 days from a stack's next start)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("states", nargs="+", type=Path, help="state directories of runs")
    options = parser.parse_args(argv)
    status = 0
    for state in (path.expanduser() for path in options.states):
        print(f"######## {state}")
        if not (state / "events.jsonl").is_file():
            print("no events.jsonl here")
            status = 1
            continue
        attempts, held = read_attempts(state)
        csv_path = state / "report" / "attempts.csv"
        saved = save_attempts(csv_path, attempts) if attempts or csv_path.is_file() else 0
        scores = {**load_saved_scores(csv_path), **load_step_file_scores(state)}
        rows = step_table(state, scores)
        if rows:
            for title, names in TABLES.items():
                print(f"{title}:\n{render(rows, names)}\n")
            failures = failure_table(state)
            if failures:
                print(failures + "\n")
            save_steps(state / "report" / "steps.csv", rows)
        else:
            print("no committed steps")
        print(inventory(state, held, saved))
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
