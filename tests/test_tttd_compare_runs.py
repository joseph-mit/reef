"""compare_runs.py: step curves from run events, the published reference, and the table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes.tttd.examples.tttd import compare_runs
from recipes.tttd.examples.tttd.harness.run_controller import step_reward_summary
from recipes.tttd.examples.tttd.harness.search import RolloutResult


def _committed(step: int, best: float, mean: float | None = None) -> dict:
    event = {"event": "tttd_step_committed", "step": step, "archive_best_reward": best}
    if mean is not None:
        event["reward_mean"] = mean
    return event


def _write_events(work: Path, method: str, task: str, events: list[dict]) -> None:
    path = compare_runs.events_path(method, task, work)
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.unit
def test_step_curve_is_one_based_and_a_restarts_repeat_wins() -> None:
    events = [
        {"event": "tttd_step_started", "step": 0},
        _committed(0, 1.0, 0.4),
        _committed(1, 2.0),
        _committed(1, 2.5, 0.9),
    ]

    assert compare_runs.step_curve(events) == {1: {"best": 1.0, "mean": 0.4}, 2: {"best": 2.5, "mean": 0.9}}


@pytest.mark.unit
def test_tttd_keeps_its_original_state_layout(tmp_path) -> None:
    assert compare_runs.events_path("tttd", "t", tmp_path) == tmp_path / "t" / "events.jsonl"
    assert compare_runs.events_path("spottt", "t", tmp_path) == tmp_path / "spottt" / "t" / "events.jsonl"


@pytest.mark.unit
def test_published_packing_run_is_the_reference() -> None:
    curve = compare_runs.published_curve("circle_packing_26")

    assert len(curve) == 50
    assert curve[1]["mean"] == pytest.approx(0.4979191490795271)
    assert curve[1]["best"] == pytest.approx(2.6359742)
    assert compare_runs.published_curve("erdos_min_overlap") == {}


@pytest.mark.unit
def test_main_tabulates_methods_up_to_the_steps_they_reached(tmp_path, capsys) -> None:
    _write_events(tmp_path, "spottt", "circle_packing_26", [_committed(0, 2.6, 0.5), _committed(1, 2.62, 0.8)])
    _write_events(tmp_path, "search-only", "circle_packing_26", [_committed(0, 2.6, 0.5)])

    assert compare_runs.main(["--work", str(tmp_path), "--methods", "search-only", "spottt"]) == 0
    output = capsys.readouterr().out.splitlines()

    assert output[0].startswith("| step | search-only mean | search-only best | spottt mean |")
    assert output[3].startswith("| 2 |  |  | 0.800 | 2.620000 | ")
    assert len(output) == 4  # header, rule, steps 1 and 2 only


@pytest.mark.unit
def test_main_reports_when_nothing_has_run(tmp_path, capsys) -> None:
    assert compare_runs.main(["--work", str(tmp_path), "--no-reference"]) == 1
    assert "no committed steps" in capsys.readouterr().out


@pytest.mark.unit
def test_plot_draws_both_panels(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    _write_events(tmp_path, "spottt", "circle_packing_26", [_committed(0, 2.6, 0.5)])

    figure = tmp_path / "comparison.png"
    assert compare_runs.main(["--work", str(tmp_path), "--methods", "spottt", "--plot", str(figure)]) == 0
    assert figure.stat().st_size > 0


@pytest.mark.unit
def test_step_reward_summary_describes_the_steps_attempts() -> None:
    results = [RolloutResult("p", "code", reward, "a") for reward in (0.0, 1.0, 2.0, 0.0)]

    assert step_reward_summary(results) == {"reward_mean": 0.75, "reward_max": 2.0, "reward_zero_fraction": 0.5}
    assert step_reward_summary(["stand-in"]) == {}
