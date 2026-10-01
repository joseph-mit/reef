"""Adaptive siblings: how a search step's attempts are spread over parents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes.tttd.examples.tttd.harness.search import PUCTArchive, ScoredSolution, TTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.siblings import AttemptStats, SiblingSettings, allocate_siblings

EXAMPLE = Path(__file__).resolve().parents[1] / "recipes" / "tttd" / "examples" / "tttd"


def _adaptive(**overrides) -> SiblingSettings:
    values = {"policy": "adaptive", "min_siblings": 8, "max_siblings": 64, "target": 64, "max_parents": 32}
    values.update(overrides)
    return SiblingSettings(**values)


@pytest.mark.unit
def test_unknown_parents_get_the_full_group_and_known_ones_the_minimum() -> None:
    settings = _adaptive()

    assert allocate_siblings([0.0] * 8, 512, settings) == [64] * 8
    # Well-known parents need only the minimum, so the budget reaches more of
    # them; what is left goes round them again.
    assert allocate_siblings([64.0] * 32, 512, settings) == [16] * 32
    assert allocate_siblings([64.0] * 64, 512, settings) == [8] * 64
    # Partly known: a parent with 40 known outcomes needs 24 more.
    assert allocate_siblings([40.0, 0.0], 100, settings) == [24 + 6, 64 + 6]


@pytest.mark.unit
def test_the_budget_is_always_spent_exactly() -> None:
    settings = _adaptive(min_siblings=3, max_siblings=5, target=5)
    for known in ([0.0], [0.0, 2.0, 5.0], [5.0] * 10, [1.0] * 3):
        for budget in (1, 2, 7, 16, 40):
            allocations = allocate_siblings(known, budget, settings)
            assert sum(allocations) == budget and len(allocations) <= len(known)
            assert all(count >= 1 for count in allocations)


@pytest.mark.unit
def test_settings_reject_bad_values_and_unknown_keys() -> None:
    with pytest.raises(ValueError, match="policy"):
        SiblingSettings(policy="greedy")
    with pytest.raises(ValueError, match="max_siblings"):
        SiblingSettings(min_siblings=8, max_siblings=4)
    with pytest.raises(ValueError, match="unknown"):
        SiblingSettings.from_mapping({"policy": "adaptive", "budget": 3})
    assert SiblingSettings.from_mapping(None) == SiblingSettings()


@pytest.mark.unit
def test_known_outcomes_fade_and_new_parents_inherit_half_their_parents() -> None:
    settings = _adaptive(half_life=8.0)
    stats = AttemptStats()
    stats.record("p", [1.0] * 64, step=0, settings=settings)

    assert stats.known("p", 0, settings) == pytest.approx(64.0)
    assert stats.known("p", 8, settings) == pytest.approx(32.0)
    assert stats.known("child", 8, settings, parent_id="p") == pytest.approx(16.0)
    assert stats.known("child", 8, settings) == 0.0


@pytest.mark.unit
def test_a_surprising_step_resets_what_was_known() -> None:
    settings = _adaptive(surprise_z=3.0)
    stats = AttemptStats()
    stats.record("p", [1.0, 1.1, 0.9, 1.0] * 8, step=0, settings=settings)

    assert stats.record("p", [1.05, 0.95, 1.0, 1.0], step=1, settings=settings) is False
    assert stats.get("p").count > 32
    assert stats.record("p", [2.5, 2.6, 2.4, 2.5], step=2, settings=settings) is True
    assert stats.get("p").count == 4 and stats.get("p").mean == pytest.approx(2.5)


def _harness(siblings: SiblingSettings, groups: int = 2, rollouts: int = 4) -> TTTDiscoverHarness:
    calls = {"n": 0}

    def generate(payload):
        calls["n"] += 1
        return {"choices": [{"message": {"content": f"```python\nprint({calls['n']})\n```"}}]}

    def score(solution: str) -> ScoredSolution:
        value = int(solution.split("print(")[1].split(")")[0]) % 7 / 10
        return ScoredSolution(solution, value, value)

    return TTTDiscoverHarness(
        generate, score, "pack", model="m", groups_per_step=groups, rollouts_per_group=rollouts, siblings=siblings
    )


@pytest.mark.unit
def test_an_adaptive_step_fills_the_grid_once_and_records_each_parent() -> None:
    harness = _harness(_adaptive(min_siblings=1, max_siblings=4, target=4, max_parents=4))

    first = harness.run_step(0)
    second = harness.run_step(1)

    for results in (first, second):
        slots = [(result.group, result.rollout) for result in results]
        assert sorted(slots) == [(group, rollout) for group in range(2) for rollout in range(4)]
    assert sum(parent.attempts for parent in harness.last_selection) == 8
    # Step 0 spent all 8 attempts on the 2 unknown seeds.
    seeds = [candidate.candidate_id for candidate in harness.archive.candidates if candidate.seed]
    assert all(harness.archive.attempt_stats.get(seed) is not None for seed in seeds)

    restored = PUCTArchive()
    restored.load_state_dict(json.loads(json.dumps(harness.archive.state_dict())))
    assert restored.attempt_stats.state_dict() == harness.archive.attempt_stats.state_dict(
        keep=[candidate.candidate_id for candidate in harness.archive.candidates]
    )


@pytest.mark.unit
def test_a_fixed_step_is_unchanged_and_an_old_snapshot_loads() -> None:
    harness = _harness(SiblingSettings())

    results = harness.run_step(0)

    assert [(result.group, result.rollout) for result in results] == [(g, r) for g in range(2) for r in range(4)]
    assert [parent.attempts for parent in harness.last_selection] == [4, 4]
    snapshot = harness.archive.state_dict()
    snapshot.pop("attempt_stats")
    PUCTArchive().load_state_dict(snapshot)


@pytest.mark.unit
def test_ttt_discover_cannot_run_with_adaptive_siblings() -> None:
    from recipes.tttd.examples.tttd.harness.agent import ReefTTTDiscoverHarness

    with pytest.raises(ValueError, match="sibling group"):
        ReefTTTDiscoverHarness(object(), lambda s: s, "pack", scenario="s", model="m", siblings=_adaptive())
    ReefTTTDiscoverHarness(
        object(), lambda s: s, "pack", scenario="s", model="m", algorithm="espottt", siblings=_adaptive()
    )


@pytest.mark.unit
def test_configs_choose_the_sibling_policy() -> None:
    from recipes.tttd.examples.tttd.harness.session import StackSettings

    assert StackSettings.load(EXAMPLE / "serve.espottt-adaptive.yaml").siblings == _adaptive()
    assert StackSettings.load(EXAMPLE / "serve.spottt-adaptive.yaml").siblings.adaptive
    assert not StackSettings.load(EXAMPLE / "serve.spottt.yaml").siblings.adaptive
    smoke = StackSettings.load(EXAMPLE / "serve.espottt-adaptive-smoke.yaml")
    assert smoke.siblings.max_siblings <= smoke.rollouts_per_group * smoke.groups_per_step


@pytest.mark.unit
def test_only_adaptive_siblings_enter_the_run_identity(tmp_path: Path) -> None:
    from recipes.tttd.examples.tttd.harness.run_controller import TTTDRunIdentity, TTTDRunStateError, TTTDRunStateStore

    base = {
        "scenario": "s",
        "model": "m",
        "recipe": "spottt",
        "inference_path": "/v1",
        "instruction_sha256": "h",
        "groups_per_step": 8,
        "rollouts_per_group": 64,
        "max_new_tokens": 10,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "enable_thinking": True,
        "exploration": 1.0,
        "invalid_reward": 0.0,
    }
    fixed = TTTDRunIdentity(**base, siblings=SiblingSettings().identity())
    assert "siblings" not in fixed.as_dict()
    # A state file from before adaptive siblings existed still resumes.
    TTTDRunStateStore(tmp_path / "fixed.json", TTTDRunIdentity(**base)).save_committed(
        next_step=1, runtime_load_id="v1", archive={}
    )
    assert TTTDRunStateStore(tmp_path / "fixed.json", fixed).load()["next_step"] == 1

    adaptive = TTTDRunIdentity(**base, siblings=_adaptive().identity())
    assert adaptive.as_dict()["siblings"]["policy"] == "adaptive"
    with pytest.raises(TTTDRunStateError, match="identity"):
        TTTDRunStateStore(tmp_path / "fixed.json", adaptive).load()
    store = TTTDRunStateStore(tmp_path / "adaptive.json", adaptive)
    store.save_committed(next_step=2, runtime_load_id="v2", archive={})
    assert store.load()["next_step"] == 2
    changed = TTTDRunIdentity(**base, siblings=_adaptive(min_siblings=4).identity())
    with pytest.raises(TTTDRunStateError, match="identity"):
        TTTDRunStateStore(tmp_path / "adaptive.json", changed).load()
