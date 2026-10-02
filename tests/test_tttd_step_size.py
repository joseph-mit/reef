"""Smaller steps once the best score stalls, and an attempt budget that ends the run."""

from __future__ import annotations

from dataclasses import replace

import pytest

from recipes.tttd.examples.tttd.harness.agent import ReefTTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.run_controller import TTTDRunController, TTTDRunIdentity, TTTDRunStateStore
from recipes.tttd.examples.tttd.harness.search import PUCTArchive, ScoredSolution, TTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.step_size import StepRecord, StepSizeSettings, stalled_steps

STALL = StepSizeSettings(policy="stall", patience=2, min_groups=1)


class _Model:
    """Answers every request with the same program, so the best score never moves after step 0."""

    def __call__(self, payload):
        return {"choices": [{"message": {"content": "```python\nprint(1)\n```"}}]}


def _score(solution: str) -> ScoredSolution:
    return ScoredSolution(solution, 1.0, 1.0)


def _harness(step_size: StepSizeSettings = STALL) -> TTTDiscoverHarness:
    return TTTDiscoverHarness(
        _Model(),
        _score,
        "task",
        model="m",
        groups_per_step=3,
        rollouts_per_group=2,
        max_workers=1,
        step_size=step_size,
    )


def _identity(**overrides) -> TTTDRunIdentity:
    identity = TTTDRunIdentity(
        scenario="s",
        model="m",
        recipe="search-only",
        inference_path="/v1/chat/completions",
        instruction_sha256="hash",
        groups_per_step=3,
        rollouts_per_group=2,
        max_new_tokens=64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        enable_thinking=True,
        exploration=1.0,
        invalid_reward=0.0,
    )
    return replace(identity, **overrides)


@pytest.mark.unit
def test_settings_default_to_fixed_steps_and_keep_the_old_identity() -> None:
    settings = StepSizeSettings.from_mapping(None)
    assert not settings.active and settings.identity() is None
    assert settings.groups_for([StepRecord(0, 6, 0.0, 0.0)] * 10, 8) == 8
    assert "step_size" not in _identity().as_dict()
    with pytest.raises(ValueError, match="unknown step_size"):
        StepSizeSettings.from_mapping({"policy": "stall", "groups": 2})
    with pytest.raises(ValueError, match="policy"):
        StepSizeSettings.from_mapping({"policy": "shrink"})


@pytest.mark.unit
def test_a_step_shrinks_after_patience_steps_without_improvement_and_grows_back() -> None:
    improving = StepRecord(0, 6, 0.0, 1.0)
    flat = StepRecord(1, 6, 1.0, 1.0)
    assert stalled_steps([improving, flat], 1e-9) == 1
    assert STALL.groups_for([improving, flat], 3) == 3
    assert STALL.groups_for([improving, flat, flat], 3) == 1
    assert STALL.groups_for([improving, flat, flat, StepRecord(3, 2, 1.0, 1.5)], 3) == 3
    # A rise within the tolerance is no improvement.
    assert STALL.groups_for([flat, StepRecord(2, 6, 1.0, 1.0 + 1e-12)], 3) == 1


@pytest.mark.unit
def test_the_search_runs_smaller_steps_and_saves_the_history_it_planned_from() -> None:
    harness = _harness()
    sizes = [len(harness.run_step(step)) for step in range(5)]
    # Step 0 finds 1.0; steps 1 and 2 do not improve on it, so steps 3 and 4 are one row.
    assert sizes == [6, 6, 6, 2, 2]
    assert harness.last_step_groups == 1

    restored = PUCTArchive()
    restored.load_state_dict(harness.archive.state_dict())
    assert restored.step_history == harness.archive.step_history
    assert [record.attempts for record in restored.step_history] == sizes
    # A snapshot written before step records existed loads with an empty history.
    snapshot = harness.archive.state_dict()
    snapshot.pop("step_history")
    restored.load_state_dict(snapshot)
    assert restored.step_history == []


@pytest.mark.unit
def test_min_groups_cannot_exceed_the_grid() -> None:
    with pytest.raises(ValueError, match="exceeds groups_per_step"):
        _harness(StepSizeSettings(policy="stall", min_groups=4))


@pytest.mark.unit
def test_the_run_stops_at_the_step_boundary_where_the_budget_is_spent(tmp_path) -> None:
    settings = StepSizeSettings(policy="stall", patience=2, min_groups=1, attempt_budget=20)
    harness = _harness(settings)
    events: list[dict] = []
    store = TTTDRunStateStore(tmp_path / "state.json", _identity(step_size=settings.identity()))
    controller = TTTDRunController(harness, None, store, emit=events.append, wait_for_training=False)

    outcome = controller.run(50)

    # 6 + 6 + 6 attempts, then one 2-attempt step reaches 20.
    assert len(outcome.results) == 20 and outcome.next_step == 4
    assert events[-1] == {"event": "tttd_attempt_budget_spent", "step": 4, "attempts": 20}

    # A restart picks up the spent budget from the saved history and runs nothing.
    resumed = TTTDRunController(_harness(settings), None, store, emit=events.append, wait_for_training=False)
    assert resumed.run(50).results == ()


class _Client:
    def __init__(self) -> None:
        self.reports: list[dict] = []

    def inference_with_record(self, scenario, path, payload, *, recipe=None, extra_headers=None):
        return {"choices": [{"message": {"content": "```python\nprint(1)\n```"}}]}, "record"

    def report(self, scenario, payload, *, extra_headers=None):
        self.reports.append(payload)


@pytest.mark.unit
def test_reef_reports_announce_the_step_size_only_when_steps_can_shrink() -> None:
    for step_size, expected in ((STALL, 3), (StepSizeSettings(), None)):
        client = _Client()
        harness = ReefTTTDiscoverHarness(
            client,
            _score,
            "task",
            scenario="s",
            model="m",
            groups_per_step=3,
            rollouts_per_group=2,
            max_workers=1,
            algorithm="spottt",
            step_size=step_size,
        )
        harness.run_step(0)
        assert {report["metadata"].get("step_groups") for report in client.reports} == {expected}

    with pytest.raises(ValueError, match="full grid"):
        ReefTTTDiscoverHarness(
            _Client(), _score, "task", scenario="s", model="m", algorithm="ttt-discover", step_size=STALL
        )
