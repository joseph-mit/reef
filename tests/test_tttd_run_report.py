"""run_report.py: per-step statistics from events, commits and Reef's records."""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/apptainer/run_report.py"
_spec = importlib.util.spec_from_file_location("run_report", SCRIPT)
run_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_report)


def _state(tmp_path: Path, scores: list[float], *, compacted_days_ago: float | None = None) -> Path:
    state = tmp_path / "state"
    record_dir = state / "agent-record"
    record_dir.mkdir(parents=True)
    (state / "events.jsonl").write_text(
        json.dumps({"event": "tttd_step_started", "step": 0, "time": 1000.0})
        + "\n"
        + json.dumps(
            {
                "event": "tttd_step_committed",
                "step": 0,
                "archive_best_reward": 2.63,
                "reward_mean": 9.9,
                "reward_zero_fraction": 0.9,
                "reward_max": 9.9,
                "time": 4000.0,
            }
        )
        + "\n"
    )
    (record_dir / "s.commits.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "metrics": {
                    "rollout/kl": 0.001,
                    "train/entropy_loss": 0.4,
                    "train_steps": [{"train/pg_clipfrac": 0.01, "train/ppo_kl": 0.0007}],
                },
            }
        )
        + "\n"
    )
    connection = sqlite3.connect(record_dir / "s.sqlite3")
    connection.execute(
        "CREATE TABLE agent_record (sequence INTEGER PRIMARY KEY, agent_record_id TEXT, scenario TEXT,"
        " request_type TEXT, payload_json TEXT, created_at REAL, references_json TEXT, artifact_json TEXT,"
        " compacted_at REAL, body_bytes INTEGER)"
    )
    compacted = None if compacted_days_ago is None else time.time() - compacted_days_ago * 86400
    for index, score in enumerate(scores):
        payload = {
            "score": score,
            "feedback": "" if score else "overlap",
            "metadata": {"step": 0, "group": 0, "rollout": index, "parent_id": "p"},
        }
        connection.execute(
            "INSERT INTO agent_record VALUES (?, ?, 's', 'report', ?, 0, '[]', NULL, ?, 0)",
            (index, f"r{index}", json.dumps(payload), compacted),
        )
        connection.execute(
            "INSERT INTO agent_record VALUES (?, ?, 's', 'inference', '{}', 0, '[]', NULL, ?, 0)",
            (100 + index, f"i{index}", compacted),
        )
    connection.commit()
    connection.close()
    return state


@pytest.mark.unit
def test_statistics_come_from_every_attempt_and_are_saved(tmp_path: Path, capsys) -> None:
    scores = [0.0, 0.0, 2.0, 2.5] + [1.0] * 6
    state = _state(tmp_path, scores, compacted_days_ago=2.0)

    assert run_report.main([str(state)]) == 0

    (row,) = run_report.step_table(state, run_report.load_saved_scores(state / "report" / "attempts.csv"))
    assert row["attempts"] == 10
    assert row["mean"] == pytest.approx(sum(scores) / 10)
    assert row["fail %"] == pytest.approx(20)
    assert row["valid mean"] == pytest.approx(sum(scores) / 8)
    assert row["top 10%"] == 2.5 and row["step best"] == 2.5 and row["best so far"] == 2.63
    assert row["minutes"] == pytest.approx(50)
    assert row["entropy"] == 0.4 and row["kl base"] == 0.001
    output = capsys.readouterr().out
    assert "10 scored attempts, 10 generations, 20 already consumed" in output
    assert "deleting consumed records starts in about 5.0 days" in output
    assert "per-update training metrics: train/pg_clipfrac, train/ppo_kl" in output


@pytest.mark.unit
def test_saved_attempts_survive_reef_deleting_its_records(tmp_path: Path) -> None:
    state = _state(tmp_path, [1.0, 2.0])
    run_report.main([str(state)])
    connection = sqlite3.connect(state / "agent-record" / "s.sqlite3")
    connection.execute("DELETE FROM agent_record")
    connection.commit()
    connection.close()

    run_report.main([str(state)])

    with (state / "report" / "attempts.csv").open(newline="") as handle:
        assert [row["score"] for row in csv.DictReader(handle)] == ["1.0", "2.0"]
    (row,) = run_report.step_table(state, run_report.load_saved_scores(state / "report" / "attempts.csv"))
    assert row["mean"] == 1.5


@pytest.mark.unit
def test_without_records_the_step_falls_back_to_the_harness_summary(tmp_path: Path) -> None:
    state = _state(tmp_path, [])
    (row,) = run_report.step_table(state, {})
    assert row["mean"] == 9.9 and row["fail %"] == pytest.approx(90) and row["attempts"] is None


@pytest.mark.unit
def test_a_resumed_steps_repeated_slots_keep_the_first_report(tmp_path: Path, capsys) -> None:
    # Slots 0-1 reported by an interrupted job (scores 0), then by the resumed one (2.0).
    state = _state(tmp_path, [0.0, 0.0])
    connection = sqlite3.connect(state / "agent-record" / "s.sqlite3")
    for index in range(2):
        payload = {"score": 2.0, "metadata": {"step": 0, "group": 0, "rollout": index}}
        connection.execute(
            "INSERT INTO agent_record VALUES (?, ?, 's', 'report', ?, 0, '[]', NULL, NULL, 0)",
            (200 + index, f"late{index}", json.dumps(payload)),
        )
    connection.commit()
    connection.close()

    run_report.main([str(state)])

    (row,) = run_report.step_table(state, run_report.load_saved_scores(state / "report" / "attempts.csv"))
    assert row["attempts"] == 2 and row["mean"] == 0.0
    assert "Reef trains on the first): 2" in capsys.readouterr().out


@pytest.mark.unit
def test_a_csv_saved_with_repeats_is_corrected_from_reefs_records(tmp_path: Path) -> None:
    state = _state(tmp_path, [0.0, 1.0])
    path = state / "report" / "attempts.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_report.ATTEMPT_FIELDS)
        writer.writeheader()
        writer.writerow({"record_id": "late0", "step": 1, "group": 0, "rollout": 0, "score": 2.0})
        writer.writerow({"record_id": "r0", "step": 1, "group": 0, "rollout": 0, "score": 0.0})

    run_report.main([str(state)])

    (row,) = run_report.step_table(state, run_report.load_saved_scores(path))
    assert row["attempts"] == 2 and row["mean"] == 0.5
