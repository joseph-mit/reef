"""run_status.py: one row per committed step from a run's state directory."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/apptainer/run_status.py"
_spec = importlib.util.spec_from_file_location("run_status", SCRIPT)
run_status = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_status)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.unit
def test_rows_join_events_and_commits(tmp_path, capsys) -> None:
    _write(
        tmp_path / "events.jsonl",
        [
            {"event": "tttd_step_started", "step": 0},
            {"event": "tttd_step_committed", "step": 0, "archive_best_reward": 2.6241961485315453},
            {"event": "tttd_step_committed", "step": 1, "archive_best_reward": 2.63},
        ],
    )
    _write(
        tmp_path / "agent-record" / "abc.commits.jsonl",
        [
            {
                "step": 1,
                "metrics": {
                    "rollout/rewards": 0.388,
                    "perf/step_time": 12076.9,
                    "perf/actor_train_tok_per_s": 908.49,
                    "memory/alloc_retries": 1200,
                    "rollout/kl": 0.0007,
                    "train_steps": [{"train/ppo_kl": 0.00072}, {"train/ppo_kl": 0.0011}],
                },
            },
            {"step": 2, "operation": "rollback", "metrics": {"rollout/rewards": 9.0}},
        ],
    )
    rows = run_status.step_rows(tmp_path)
    assert [row["step"] for row in rows] == [1.0, 2.0]
    first = rows[0]
    assert first["best"] == pytest.approx(2.6241961485315453)
    assert first["step min"] == pytest.approx(201.28, abs=0.01)
    assert first["mismatch"] == pytest.approx(0.00072), "the first optimizer step, before any update"
    assert rows[1]["mean"] is None, "a rollback is not a training step"
    assert run_status.main([str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "| 1 | 0.39 | 2.624196 | 201.28 | 908 | 1200 | 0.00072 | 0.00070 |" in printed


@pytest.mark.unit
def test_empty_state_says_so(tmp_path, capsys) -> None:
    assert run_status.main([str(tmp_path)]) == 1
    assert "no committed steps" in capsys.readouterr().out
