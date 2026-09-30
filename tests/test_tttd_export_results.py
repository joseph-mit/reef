"""export_results.py: a run's records as plain CSV files, keeping only complete columns."""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/apptainer/export_results.py"
sys.path.insert(0, str(SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("export_results", SCRIPT)
export_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_results)


def _ppo_state(tmp_path: Path) -> Path:
    state = tmp_path / "ppottt" / "circle_packing_26"
    records = state / "agent-record"
    records.mkdir(parents=True)
    events = []
    for step in range(2):
        events.append({"event": "tttd_step_started", "step": step, "time": 1000.0 + 7200 * step})
        events.append(
            {
                "event": "tttd_step_committed",
                "step": step,
                "time": 4600.0 + 7200 * step,
                "archive_best_reward": 2.5 + step,
                "archive_size": 10,
                "reward_mean": 1.0,
                "reward_max": 2.5,
            }
        )
    (state / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    commits = [
        # A critic-only warm-up step: no policy update, no training metrics.
        {"step": 1, "metrics": {"ppottt/actor_trained": 0, "steps": 1}},
        {
            "step": 2,
            "metrics": {
                "ppottt/actor_trained": 1,
                "steps": 2,
                "rollout/kl": 0.001,
                "only_once": 5.0,
                "train_steps": [{"train/pg_loss": 0.2, "train/step": 0}, {"train/pg_loss": 0.4, "train/step": 1}],
            },
        },
    ]
    (records / "s.commits.jsonl").write_text("".join(json.dumps(commit) + "\n" for commit in commits))
    connection = sqlite3.connect(records / "s.sqlite3")
    connection.execute(
        "CREATE TABLE agent_record (sequence INTEGER PRIMARY KEY, agent_record_id TEXT, scenario TEXT,"
        " request_type TEXT, payload_json TEXT, created_at REAL, references_json TEXT, artifact_json TEXT,"
        " compacted_at REAL, body_bytes INTEGER)"
    )
    rows = [(0, 0, 0, 0.0), (0, 0, 1, 2.0), (1, 0, 0, 3.0), (1, 0, 1, 1.0), (1, 0, 0, 9.0)]  # last repeats a slot
    for sequence, (step, group, rollout, score) in enumerate(rows):
        payload = {
            "score": score,
            "feedback": "" if score else "packing is not valid",
            "metadata": {"step": step, "group": group, "rollout": rollout, "parent_id": "state-1"},
        }
        connection.execute(
            "INSERT INTO agent_record VALUES (?, ?, 's', 'report', ?, 0, '[]', NULL, NULL, 0)",
            (sequence, f"r{sequence}", json.dumps(payload)),
        )
    connection.commit()
    connection.close()
    archive = {
        "archive": {
            "candidates": [
                {
                    "candidate_id": "state-0",
                    "parent_id": None,
                    "reward": 0.0,
                    "visits": 3,
                    "seed": True,
                    "children": ["state-1"],
                    "solution": "",
                },
                {
                    "candidate_id": "state-1",
                    "parent_id": "state-0",
                    "reward": 3.5,
                    "visits": 1,
                    "seed": False,
                    "children": [],
                    "solution": "```python\nbest()\n```",
                },
            ]
        }
    }
    (state / "ppottt-search-state.json").write_text(json.dumps(archive))
    return state


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.unit
def test_a_run_exports_complete_columns_every_attempt_and_its_archive(tmp_path: Path) -> None:
    state = _ppo_state(tmp_path)
    out = tmp_path / "out"

    assert export_results.main([str(out), str(state)]) == 0

    steps = _csv(out / "ppottt" / "steps.csv")
    assert [row["step"] for row in steps] == ["1", "2"]
    assert steps[1]["mean"] == "2.0" and steps[1]["attempts"] == "2.0"  # the repeated slot is left out
    assert steps[0]["fail_pct"] == "50.0" and steps[1]["step_minutes"] == "60.0"
    assert steps[1]["cumulative_hours"] == "2.0"
    # Training metrics exist only on the step that trained the policy, which is complete for PPO-TTT.
    assert steps[0]["upd/train/pg_loss"] == "" and float(steps[1]["upd/train/pg_loss"]) == pytest.approx(0.3)
    assert steps[1]["upd_first/train/pg_loss"] == "0.2"
    assert "commit/rollout/kl" in steps[0]
    dropped = (out / "ppottt" / "dropped_columns.txt").read_text()
    assert "search/reward_max" not in dropped
    assert len(_csv(out / "ppottt" / "updates.csv")) == 2
    attempts = _csv(out / "ppottt" / "attempts.csv")
    assert [row["score"] for row in attempts] == ["0.0", "2.0", "3.0", "1.0"]
    assert attempts[0]["feedback"] == "packing is not valid"
    assert (out / "ppottt" / "best_program.py").read_text() == "```python\nbest()\n```\n"
    assert len(_csv(out / "ppottt" / "archive.csv")) == 2
    info = json.loads((out / "ppottt" / "run.json").read_text())
    assert info["best"] == 3.5 and info["committed_steps"] == 2 and info["attempts"] == 4


@pytest.mark.unit
def test_a_column_missing_on_some_steps_is_dropped_and_listed() -> None:
    rows = [
        {"step": 1, "mean": 1.0, "partial": 2.0},
        {"step": 2, "mean": 1.5},
        {"step": 3, "mean": 1.7, "partial": 2.0},
    ]

    kept, dropped = export_results.complete_columns(rows)

    assert kept == ["step", "mean"]
    assert dropped == {"partial": "missing on steps [2]"}
