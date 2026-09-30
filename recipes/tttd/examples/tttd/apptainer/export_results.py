"""Copy everything a run has recorded into plain CSV files, ready to plot or share.

    python3 export_results.py OUT_DIR STATE_DIR [STATE_DIR ...]

For each run state directory it writes ``OUT_DIR/<method>/``:

- ``steps.csv``: one row per committed step. Scores from every attempt Reef
  trained on (the first report at each slot, as Reef trains), the harness's
  own step summary, step and cumulative time, and every numeric metric of
  Reef's commit, with per-optimizer-step metrics averaged over the step
  (``upd/`` prefix, plus ``upd_first/`` for the first update, where the
  trainer and sampler still share weights). A column is kept only when it
  has a value for every committed step, or, for training metrics of a
  method with critic-only warm-up steps, for every step that trained the
  policy; ``dropped_columns.txt`` lists the rest and why.
- ``updates.csv``: every optimizer step's training metrics, in order.
- ``attempts.csv``: every attempt: step, group, rollout, score, parent
  state, the parent's score when the harness recorded it, and the first
  200 characters of the feedback.
- ``archive.csv`` and ``best_program.py``: the search archive as the run
  left it, and the best program in it.
- ``run.json``: method, task, step count, best score, and where it came from.

It only reads the state directory, so it runs next to a live job. A run kept
on a node's local disk is read on that node through ``srun``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_report import _jsonl, _number

FEEDBACK_CHARS = 200
# Commit keys that hold lists or run-level settings rather than a step's measurement.
LIST_KEYS = ("train_steps", "critic_train_steps")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in columns})


def read_attempts(state: Path) -> list[dict[str, Any]]:
    """Every scored attempt Reef holds, keeping the first report at each slot (what Reef trained on)."""
    attempts: list[dict[str, Any]] = []
    slots: set[tuple[Any, Any, Any]] = set()
    for database in sorted((state / "agent-record").glob("*.sqlite3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT agent_record_id, payload_json, created_at FROM agent_record"
                " WHERE request_type = 'report' ORDER BY sequence"
            ).fetchall()
        finally:
            connection.close()
        for record_id, payload_json, created_at in rows:
            payload = json.loads(payload_json)
            metadata = payload.get("metadata") or {}
            score = _number(payload.get("score"))
            step = metadata.get("step")
            if score is None or not isinstance(step, int):
                continue
            slot = (step, metadata.get("group"), metadata.get("rollout"))
            if slot in slots:
                continue
            slots.add(slot)
            attempts.append(
                {
                    "step": step + 1,
                    "group": metadata.get("group"),
                    "rollout": metadata.get("rollout"),
                    "score": score,
                    "parent_id": metadata.get("parent_id"),
                    "grandparent_id": metadata.get("grandparent_id"),
                    "parent_reward": _number(metadata.get("parent_reward")),
                    "search_value": _number(metadata.get("search_value")),
                    "created_at": _number(created_at),
                    "record_id": record_id,
                    "feedback": str(payload.get("feedback") or "")[:FEEDBACK_CHARS],
                }
            )
    attempts.sort(key=lambda row: (row["step"], row["group"] or 0, row["rollout"] or 0))
    return attempts


def _score_stats(scores: list[float]) -> dict[str, float]:
    ordered = sorted(scores)
    valid = [score for score in ordered if score != 0]
    top = ordered[-max(1, len(ordered) // 10) :]
    stats = {
        "attempts": float(len(ordered)),
        "mean": statistics.mean(ordered),
        "fail_pct": 100 * (len(ordered) - len(valid)) / len(ordered),
        "median": statistics.median(ordered),
        "std": statistics.pstdev(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "top10_mean": statistics.mean(top),
        "step_best": ordered[-1],
    }
    if valid:
        stats["valid_mean"] = statistics.mean(valid)
        stats["valid_median"] = statistics.median(valid)
    return stats


def _finite(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and math.isfinite(number) else None


def _flat_numbers(metrics: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        number = _finite(value)
        if key not in LIST_KEYS and number is not None:
            flat[key] = number
    return flat


def _update_means(entries: Iterable[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    columns: dict[str, list[float]] = {}
    for entry in entries:
        for key, value in entry.items():
            number = _finite(value)
            if number is not None and key != "train/step":
                columns.setdefault(key, []).append(number)
    return {f"{prefix}{key}": statistics.mean(values) for key, values in columns.items()}


def step_rows(state: Path, attempts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One row per committed step, and one row per optimizer step."""
    started: dict[int, float] = {}
    committed: dict[int, dict[str, Any]] = {}
    for event in _jsonl(state / "events.jsonl"):
        step = event.get("step")
        if not isinstance(step, int):
            continue
        if event.get("event") == "tttd_step_started" and _number(event.get("time")) is not None:
            started[step + 1] = float(event["time"])
        elif event.get("event") == "tttd_step_committed":
            committed[step + 1] = event
    commits: dict[int, Mapping[str, Any]] = {}
    for path in sorted((state / "agent-record").glob("*.commits.jsonl")):
        for record in _jsonl(path):
            if isinstance(record.get("step"), int) and record.get("operation", "training") == "training":
                commits[record["step"]] = record.get("metrics") or {}
    scores: dict[int, list[float]] = {}
    for attempt in attempts:
        scores.setdefault(int(attempt["step"]), []).append(float(attempt["score"]))

    rows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    elapsed = 0.0
    for step in sorted(committed):
        event = committed[step]
        metrics = commits.get(step, {})
        row: dict[str, Any] = {"step": step}
        finished = _number(event.get("time"))
        if finished is not None:
            row["committed_at"] = finished
            if step in started:
                row["step_minutes"] = (finished - started[step]) / 60
                elapsed += row["step_minutes"]
                row["cumulative_hours"] = elapsed / 60
        row["best_so_far"] = _number(event.get("archive_best_reward"))
        row["archive_size"] = _number(event.get("archive_size"))
        row["search_mean"] = _number(event.get("reward_mean"))
        skip = ("step", "next_step", "time", "reward_mean", "archive_best_reward", "archive_size")
        for key, value in event.items():
            number = _finite(value)
            if key not in skip and number is not None:
                row[f"search/{key}"] = number
        if scores.get(step):
            row.update(_score_stats(scores[step]))
        row.update({f"commit/{key}": value for key, value in _flat_numbers(metrics).items()})
        train_steps = [entry for entry in metrics.get("train_steps") or [] if isinstance(entry, Mapping)]
        row.update(_update_means(train_steps, "upd/"))
        row.update(_update_means(train_steps[:1], "upd_first/"))
        critic_steps = [entry for entry in metrics.get("critic_train_steps") or [] if isinstance(entry, Mapping)]
        row.update(_update_means(critic_steps, "critic_upd/"))
        for index, entry in enumerate(train_steps):
            updates.append({"step": step, "update": index, **_flat_numbers(entry)})
        rows.append(row)
    return rows, updates


def complete_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """Columns with a value on every committed step; training columns may skip critic-only steps."""
    policy_steps = {
        row["step"] for row in rows if row.get("commit/ppottt/actor_trained", 1) != 0
    }  # PPO-TTT's warm-up steps train only the critic
    all_steps = {row["step"] for row in rows}
    order: list[str] = []
    for row in rows:
        order.extend(key for key in row if key not in order)
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for column in order:
        present = {row["step"] for row in rows if row.get(column) is not None}
        if present == all_steps or (present and present == policy_steps):
            kept.append(column)
        else:
            missing = sorted(all_steps - present)
            dropped[column] = f"missing on steps {missing[:10]}{' ...' if len(missing) > 10 else ''}"
    return kept, dropped


def archive_rows(state: Path) -> tuple[list[dict[str, Any]], str, Path | None]:
    """The search archive's states (without their programs) and the best program."""
    paths = sorted(state.glob("*-search-state.json"))
    if not paths:
        return [], "", None
    snapshot = json.loads(paths[-1].read_text())
    candidates = (snapshot.get("archive") or {}).get("candidates") or []
    rows = [
        {
            "candidate_id": item.get("candidate_id"),
            "parent_id": item.get("parent_id"),
            "reward": item.get("reward"),
            "visits": item.get("visits"),
            "seed": item.get("seed"),
            "children": len(item.get("children") or []),
        }
        for item in candidates
    ]
    best = max(candidates, key=lambda item: float(item.get("reward") or 0), default=None)
    program = str(best.get("solution") or "") if best else ""
    return rows, program, paths[-1]


def export_run(state: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    attempts = read_attempts(state)
    rows, updates = step_rows(state, attempts)
    columns, dropped = complete_columns(rows)
    _write_csv(out / "steps.csv", rows, columns)
    (out / "dropped_columns.txt").write_text("".join(f"{name}: {why}\n" for name, why in dropped.items()))
    update_columns: list[str] = []
    for row in updates:
        update_columns.extend(key for key in row if key not in update_columns)
    _write_csv(out / "updates.csv", updates, update_columns)
    _write_csv(
        out / "attempts.csv",
        attempts,
        (
            "step",
            "group",
            "rollout",
            "score",
            "parent_id",
            "grandparent_id",
            "parent_reward",
            "search_value",
            "created_at",
            "record_id",
            "feedback",
        ),
    )
    archive, program, archive_path = archive_rows(state)
    _write_csv(out / "archive.csv", archive, ("candidate_id", "parent_id", "reward", "visits", "seed", "children"))
    if program:
        (out / "best_program.py").write_text(program + "\n")
    committed = [row["step"] for row in rows]
    info = {
        "state_dir": str(state),
        "method": state.parent.name,
        "task": state.name,
        "committed_steps": len(committed),
        "last_step": max(committed, default=0),
        "best": max((row["best_so_far"] for row in rows if row.get("best_so_far") is not None), default=None),
        "attempts": len(attempts),
        "optimizer_updates": len(updates),
        "archive_file": None if archive_path is None else archive_path.name,
        "columns_kept": len(columns),
        "columns_dropped": len(dropped),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory to write into; one subdirectory per run")
    parser.add_argument("states", nargs="+", type=Path, help="state directories of runs")
    options = parser.parse_args(argv)
    status = 0
    for state in (path.expanduser() for path in options.states):
        if not (state / "events.jsonl").is_file():
            print(f"{state}: no events.jsonl here")
            status = 1
            continue
        info = export_run(state, options.out.expanduser() / state.parent.name)
        print(json.dumps(info))
    return status


if __name__ == "__main__":
    sys.exit(main())
