"""Stepping-stone SPO-TTT: the store of recent attempts, the replays, the processor, preparer and recipe."""

from __future__ import annotations

import math

import pytest

from recipes.hspottt import HspotttBatch, HSPOTTTProcessor, HSPOTTTRecipe, HSPOTTTRolloutReport
from recipes.hspottt.stones import (
    Attempt,
    PackedSample,
    Replay,
    StoneSettings,
    StoneStore,
    admissible,
    runtime_version,
)
from recipes.spottt import SpotttBatch
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train import ProcessorContext
from reef.train.algos.registry import resolve_preparer
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicySample


def _sample(reward: float, *, version: str | None = "inc:0", name: str = "s") -> PolicySample:
    return PolicySample(name, (151_000, 7, 9), (1, 0), (-0.25, -1.5), reward, runtime_load_id=version)


def _attempt(key: str, parent_key: str, reward: float, parent_id: str = "p") -> Attempt:
    return Attempt(key, parent_key, parent_id, reward)


def _plan(store: StoneStore, step: int, attempts: list[Attempt], version: str = "inc:0"):
    return store.plan(step, attempts, [_sample(a.reward, version=version, name=a.key or "none") for a in attempts])


# --- the store ---


@pytest.mark.unit
def test_packed_samples_round_trip_exactly_and_take_less_memory() -> None:
    sample = PolicySample(
        "s",
        tuple(range(150_000, 154_000)),
        (1,) * 2_000 + (0,) * 2_000,
        tuple(-1.0 / (index + 3) for index in range(4_000)),
        2.5,
        runtime_load_id="inc:3",
        action_mask=(1,) * 4_000,
        extras={"parent_id": "p"},
    )

    packed = PackedSample.pack(sample)

    assert packed.unpack() == sample
    assert packed.tokens.itemsize == 4 and packed.rollout_log_probs.itemsize == 8
    assert isinstance(packed.loss_mask, bytes)


@pytest.mark.unit
def test_credit_reaches_the_stones_above_a_success_and_skips_successful_ones() -> None:
    settings = StoneSettings(window=5, generations=3, discount=0.5, credit=1.0, success="best")
    # Step 0 sets the bar at 2.5.
    store = _plan(StoneStore(settings), 0, [_attempt("a", "", 2.0), _attempt("b", "", 2.5, "q")]).store
    # Step 1: "c" from "a" stays below it; "h" from "b" beats it, so "b" earns credit.
    step1 = _plan(store, 1, [_attempt("c", "a", 2.1, "state-a"), _attempt("h", "b", 2.6, "state-b")])
    assert step1.store.stones["b"].credit == 1.0 and step1.store.stones["h"].success
    # Step 2: "d" from "c" and "e" from "h" both beat 2.6.
    plan = _plan(step1.store, 2, [_attempt("d", "c", 2.9, "state-c"), _attempt("e", "h", 2.95, "state-h")])

    stones = plan.store.stones
    assert stones["c"].credit == 1.0  # the success's parent
    assert stones["a"].credit == 0.5  # its grandparent, one generation further up
    assert stones["h"].credit == 0.0  # a success itself: no stepping-stone credit
    assert stones["b"].credit == 1.0  # offered 0.5 through "h", keeps the 1.0 it had
    assert plan.metrics["successes"] == 2 and plan.metrics["newly_credited"] == 2


@pytest.mark.unit
def test_generations_bound_how_far_credit_reaches() -> None:
    store = StoneStore(StoneSettings(window=9, generations=2, success="top", top_fraction=0.5))
    for step, (key, parent) in enumerate((("a", ""), ("b", "a"), ("c", "b"))):
        # Each modest stone shares its step with a better attempt, so it is no success itself.
        attempts = [_attempt(key, parent, 1.0, f"p{step}"), _attempt(f"top{step}", "", 2.0, "q")]
        store = _plan(store, step, attempts).store
    plan = _plan(store, 3, [_attempt("d", "c", 3.0, "p3"), _attempt("", "", 0.0)])

    assert [plan.store.stones[key].credit for key in "abc"] == [0.0, 0.5, 1.0]


@pytest.mark.unit
def test_a_stone_is_replayed_once_its_window_ends_and_only_with_credit() -> None:
    settings = StoneSettings(window=2, success="top", top_fraction=0.3)
    store = _plan(
        StoneStore(settings), 0, [_attempt("a", "", 1.0), _attempt("b", "", 0.5, "q"), _attempt("t", "", 2.0, "r")]
    ).store
    store = _plan(store, 1, [_attempt("c", "a", 3.0, "state-a"), _attempt("", "a", 0.0, "state-a")]).store
    assert store.stones["a"].credit == 1.0 and store.stones["b"].credit == 0.0

    plan = _plan(store, 2, [_attempt("e", "", 0.1, "r"), _attempt("", "", 0.0, "r")])

    assert [(replay.sample.source_agent_record_id, replay.credit, replay.step) for replay in plan.replays] == [
        ("a", 1.0, 0)
    ]
    assert "a" not in plan.store.stones and "b" not in plan.store.stones and "t" not in plan.store.stones
    assert "c" in plan.store.stones


@pytest.mark.unit
def test_only_the_best_attempts_of_each_parent_are_kept_once_per_program() -> None:
    store = StoneStore(StoneSettings(keep_per_parent=2))
    attempts = [
        _attempt("a1", "", 1.0, "a"),
        _attempt("a2", "", 3.0, "a"),
        _attempt("a3", "", 2.0, "a"),
        _attempt("", "", 9.0, "a"),  # no program: never a parent
        _attempt("b1", "", 0.5, "b"),
        _attempt("a2", "", 3.0, "b"),  # the same program again
    ]

    plan = _plan(store, 0, attempts)

    assert sorted(plan.store.stones) == ["a2", "a3", "b1"]
    assert plan.store.stones["a2"].order == 1


@pytest.mark.unit
def test_best_success_needs_a_bar_and_a_margin() -> None:
    store = StoneStore(StoneSettings(success="best", tolerance=1e-9))
    first = _plan(store, 0, [_attempt("a", "", 2.0)])
    assert first.metrics["successes"] == 0 and first.store.best == 2.0

    second = _plan(first.store, 1, [_attempt("b", "a", 2.0 + 1e-12), _attempt("c", "a", 2.1)])
    assert second.metrics["successes"] == 1
    assert second.store.stones["a"].credit == 1.0 and second.store.best == 2.1


@pytest.mark.unit
def test_planning_is_pure_and_steps_must_advance() -> None:
    store = _plan(StoneStore(), 0, [_attempt("a", "", 1.0)]).store
    attempts = [_attempt("b", "a", 5.0), _attempt("", "", 0.0)]

    first = _plan(store, 1, attempts)
    again = _plan(store, 1, attempts)

    assert store.stones["a"].credit == 0.0
    assert first.store.stones.keys() == again.store.stones.keys() and first.metrics == again.metrics
    with pytest.raises(ValueError, match="not after"):
        _plan(first.store, 1, attempts)


@pytest.mark.unit
def test_replays_the_trainer_would_refuse_are_left_out() -> None:
    fresh = [_sample(1.0, version="inc:5"), _sample(2.0, version="inc:5")]
    replays = [
        Replay(_sample(1.0, version="inc:2"), 1.0, 0),  # 3 versions back: admitted
        Replay(_sample(1.0, version="inc:1"), 1.0, 0),  # 4 back: refused
        Replay(_sample(1.0, version="old:4"), 1.0, 0),  # another incarnation: refused
        Replay(_sample(1.0, version=None), 1.0, 0),  # unknown: refused
    ]

    kept, refused = admissible(replays, fresh, max_lag=3)

    assert [replay.sample.runtime_load_id for replay in kept] == ["inc:2"] and refused == 3
    assert admissible(replays, [_sample(1.0, version="inc:5"), _sample(1.0, version="inc:4")], 3) == ([], 4)
    assert runtime_version("checkpoint-incarnation:12") == ("checkpoint-incarnation", 12)
    assert runtime_version("bad") is None


@pytest.mark.unit
def test_settings_reject_bad_values() -> None:
    for field, value in (
        ("window", 0),
        ("generations", 0),
        ("discount", 0.0),
        ("credit", -1.0),
        ("success", "median"),
        ("top_fraction", 1.0),
        ("keep_per_parent", 0),
    ):
        with pytest.raises(ValueError, match=field):
            StoneSettings(**{field: value})


# --- batch and preparer ---


def _spo_sample(reward: float, parent: str, index: int) -> PolicySample:
    return PolicySample(f"s{index}", (5, 1), (1,), (-0.1,), reward, extras={"parent_id": parent, "grandparent_id": ""})


@pytest.mark.unit
def test_fresh_attempts_train_exactly_as_spo_and_replays_get_their_credit() -> None:
    fresh = tuple(
        _spo_sample(reward, "a" if index < 4 else "b", index) for index, reward in enumerate([0, 1, 2, 3] * 2)
    )
    replays = (_spo_sample(1.5, "z", 90), _spo_sample(0.5, "z", 91))

    spo = resolve_preparer("spottt")(SpotttBatch("b", fresh), {})
    signal = resolve_preparer("hspottt")(HspotttBatch("b", fresh + replays, replay_credits=(1.0, 0.25)), {})

    assert signal.advantages[:8] == pytest.approx(spo.advantages, rel=1e-12)
    assert signal.advantages[8:] == (1.0, 0.25)
    # Only the fresh attempts reach the tracker.
    assert signal.next_algorithm_state["hspottt"] == spo.next_algorithm_state["spottt"]
    assert signal.loss_family == "spottt" and signal.metrics["replays"] == 2


@pytest.mark.unit
def test_without_normalisation_credit_is_scaled_by_the_raw_spread() -> None:
    fresh = tuple(_spo_sample(reward, "a", index) for index, reward in enumerate([0.0, 2.0, 0.0, 2.0]))
    signal = resolve_preparer("hspottt")(
        HspotttBatch("b", (*fresh, _spo_sample(1.0, "z", 9)), replay_credits=(1.0,), baseline="none", normalize=False),
        {},
    )

    assert signal.advantages[-1] == pytest.approx(1.0)  # the std of 0, 2, 0, 2


@pytest.mark.unit
def test_replays_widen_each_optimizer_step_evenly() -> None:
    fresh = tuple(_spo_sample(1.0, "a", index) for index in range(512))
    replays = tuple(_spo_sample(1.0, "z", 1000 + index) for index in range(18))
    batch = HspotttBatch("discovery:hspottt:4", fresh + replays, replay_credits=(1.0,) * 18, minibatch_size=128)

    assert batch.scheduling().batch_size == 133  # four steps of 133, 133, 133, 131 rather than 128 x 4 + 18
    prepared = prepare_slime_step(batch, "hspottt", {"steps": 4})
    assert prepared.payload["loss"] == "spottt" and len(prepared.payload["advantages"]) == 530
    assert prepared.metrics["optimizer_steps"] == 4
    assert HspotttBatch("b", fresh, minibatch_size=128).scheduling().batch_size == 128


@pytest.mark.unit
def test_batch_rejects_bad_credits() -> None:
    with pytest.raises(ValueError, match="fresh sample"):
        HspotttBatch("b", (_spo_sample(1.0, "a", 0),), replay_credits=(1.0,))
    with pytest.raises(ValueError, match="positive"):
        HspotttBatch("b", (_spo_sample(1.0, "a", 0), _spo_sample(1.0, "z", 1)), replay_credits=(0.0,))
    with pytest.raises(TypeError, match="HspotttBatch"):
        resolve_preparer("hspottt")(SpotttBatch("b", (_spo_sample(1.0, "a", 0),)), {})


# --- report, processor and recipe ---

GROUPS, ROLLOUTS = 2, 2


class _Logger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def log(self, metrics, *, namespace):
        self.events.append(dict(metrics))


def _inference(name: str, version: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.INFERENCE,
        agent_record_id=name,
        payload={
            "input_ids": [10, 11],
            "runtime_load_id": version,
            "response": {"training": {"tokens": [10, 11, 12], "loss_mask": [1], "rollout_log_probs": [-0.2]}},
        },
    )


def _report(step: int, group: int, rollout: int, score: float, key: str, parent_key: str) -> AgentRecord:
    reference = f"i-{step}-{group}-{rollout}"
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.REPORT,
        agent_record_id=f"r-{step}-{group}-{rollout}",
        references=(reference,),
        payload={
            "score": score,
            "references": [reference],
            "metadata": {
                "algorithm": "hspottt",
                "step": step,
                "group": group,
                "rollout": rollout,
                "groups_per_step": GROUPS,
                "rollouts_per_group": ROLLOUTS,
                "parent_id": f"state-{parent_key or 'seed'}-{group}",
                "grandparent_id": "",
                "solution_sha1": key,
                "parent_solution_sha1": parent_key,
            },
        },
    )


def _step(processor, step: int, rows: list[tuple[float, str, str]]) -> HspotttBatch:
    """Ingest one 2x2 step (score, program key, parent's key) produced by version ``inc:<step>``."""
    for index, (score, key, parent_key) in enumerate(rows):
        group, rollout = divmod(index, ROLLOUTS)
        processor.ingest(_inference(f"i-{step}-{group}-{rollout}", f"inc:{step}"))
        processor.ingest(_report(step, group, rollout, score, key, parent_key))
    return processor.build_batch()


def _processor(logger=None, **config) -> HSPOTTTProcessor:
    settings = {
        "groups_per_step": GROUPS,
        "rollouts_per_group": ROLLOUTS,
        "minibatch_size": 0,
        "stone_window": 2,
        "stone_top_fraction": 0.25,
        "replay_max_lag": 2,
    }
    settings.update(config)
    return HSPOTTTProcessor(
        ProcessorContext(
            "discovery", settings, report_type=HSPOTTTRolloutReport, experiment_logger=logger or _Logger()
        )
    )


@pytest.mark.unit
def test_processor_replays_a_stepping_stone_after_its_window_and_rebuilds_identically() -> None:
    logger = _Logger()
    processor = _processor(logger)
    # Step 0: "t" is the step's top attempt, "a" a modest one.
    first = _step(processor, 0, [(1.0, "a", ""), (0.5, "b", ""), (5.0, "t", ""), (0.0, "", "")])
    assert first.replay_credits == ()
    processor.acknowledge(first.batch_id)
    # Step 1: "d", grown from "a", is the step's top attempt: "a" earns credit.
    second = _step(processor, 1, [(3.0, "d", "a"), (0.1, "e", "a"), (0.0, "", ""), (0.0, "", "")])
    processor.acknowledge(second.batch_id)

    third = _step(processor, 2, [(0.3, "f", "e"), (0.2, "g", "e"), (0.0, "", ""), (0.0, "", "")])

    assert len(third.samples) == 5 and third.replay_credits == (1.0,)
    assert third.samples[-1].source_agent_record_id == "i-0-0-0"
    assert third.samples[-1].runtime_load_id == "inc:0"
    # A failed training attempt hands the batch back; building it again gives the same batch.
    processor.release_batch(third.batch_id)
    rebuilt = processor.build_batch()
    assert rebuilt == third
    processor.acknowledge(rebuilt.batch_id)
    assert logger.events[-1]["replays_admitted"] == 1


@pytest.mark.unit
def test_processor_needs_max_staleness_to_cover_the_window() -> None:
    with pytest.raises(ValueError, match="max_staleness"):
        _processor(stone_window=3, replay_max_lag=2)


@pytest.mark.unit
def test_report_accepts_only_its_tag_and_carries_program_keys() -> None:
    report = HSPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, solution_sha1="ab", parent_solution_sha1="cd")
    assert (report.algorithm, report.solution_sha1, report.parent_solution_sha1) == ("hspottt", "ab", "cd")
    with pytest.raises(ReportValidationError, match="algorithm"):
        HSPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, algorithm="spottt")


@pytest.mark.unit
def test_recipe_requires_max_staleness_for_its_replays() -> None:
    from reef_service.runtime_stubs import StubTrainingRuntime

    from reef.recipe import RecipeConfigError, build_recipe

    spec = HSPOTTTRecipe.training_spec()
    assert (spec.step_preparer, spec.loss_family, spec.processor) == ("hspottt", "spottt", HSPOTTTProcessor)
    recipe = build_recipe(
        "recipes.hspottt.recipe:HSPOTTTRecipe",
        {"REEF_MAX_STALENESS": "3", "REEF_HSPOTTT_STONE_SUCCESS": "best"},
        runtime=StubTrainingRuntime(max_staleness=3),
    )
    assert isinstance(recipe, HSPOTTTRecipe) and recipe.report_type is HSPOTTTRolloutReport
    assert recipe.processor_config()["replay_max_lag"] == 3 and recipe.stone_success == "best"
    with pytest.raises(RecipeConfigError, match="max_staleness"):
        build_recipe("recipes.hspottt.recipe:HSPOTTTRecipe", {}, runtime=StubTrainingRuntime())
    assert math.isfinite(recipe.stone_tolerance)


@pytest.mark.unit
def test_a_reloaded_processor_rebuilds_the_same_batch_from_the_stored_base(tmp_path) -> None:
    rows = [
        [(1.0, "a", ""), (0.5, "b", ""), (5.0, "t", ""), (0.0, "", "")],
        [(3.0, "d", "a"), (0.1, "e", "a"), (0.0, "", ""), (0.0, "", "")],
        [(0.3, "f", "e"), (0.2, "g", "e"), (0.0, "", ""), (0.0, "", "")],
    ]
    processor = _processor(stone_dir=str(tmp_path))
    for step in (0, 1):
        processor.acknowledge(_step(processor, step, rows[step]).batch_id)
    third = _step(processor, 2, rows[2])
    assert third.replay_credits == (1.0,)
    # The trainer acknowledges, then the commit fails and the scenario reloads:
    # a new processor sees step 2's records again and must build the same batch.
    processor.acknowledge(third.batch_id)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "discovery.base-000002.pickle",
        "discovery.result-000002.pickle",
    ]

    reloaded = _processor(stone_dir=str(tmp_path))
    rebuilt = _step(reloaded, 2, rows[2])

    assert rebuilt == third
    # Without the stored base the store would be empty and the replay gone.
    assert _step(_processor(), 2, rows[2]).replay_credits == ()


@pytest.mark.unit
def test_an_unreadable_or_changed_store_starts_empty(tmp_path) -> None:
    (tmp_path / "discovery.base-000003.pickle").write_bytes(b"not a pickle")
    assert _processor(stone_dir=str(tmp_path))._store.stones == {}
    processor = _processor(stone_dir=str(tmp_path / "x"))
    processor.acknowledge(
        _step(processor, 0, [(1.0, "a", ""), (0.5, "b", ""), (5.0, "t", ""), (0.0, "", "")]).batch_id
    )
    _step(processor, 1, [(1.0, "c", "a"), (0.5, "h", ""), (0.0, "", ""), (0.0, "", "")])
    assert _processor(stone_dir=str(tmp_path / "x"), stone_window=1)._store.stones == {}


@pytest.mark.unit
def test_an_attempt_returning_its_parents_program_is_not_kept() -> None:
    plan = _plan(StoneStore(), 1, [_attempt("p", "p", 2.0, "state-p"), _attempt("q", "p", 1.0, "state-p")])

    assert sorted(plan.store.stones) == ["q"]


@pytest.mark.unit
def test_replays_keep_spo_ttts_number_of_optimizer_steps() -> None:
    fresh = tuple(_spo_sample(1.0, "a", index) for index in range(512))
    replays = tuple(_spo_sample(1.0, "z", 1000 + index) for index in range(6))
    batch = HspotttBatch("discovery:hspottt:5", fresh + replays, replay_credits=(1.0,) * 6, minibatch_size=100)

    prepared = prepare_slime_step(batch, "hspottt", {"steps": 5})

    assert prepared.metrics["optimizer_steps"] == 6  # as SPO-TTT's 512 / 100


@pytest.mark.unit
def test_a_restart_after_a_commit_resumes_from_that_steps_result(tmp_path) -> None:
    rows = [
        [(1.0, "a", ""), (0.5, "b", ""), (5.0, "t", ""), (0.0, "", "")],
        [(3.0, "d", "a"), (0.1, "e", "a"), (0.0, "", ""), (0.0, "", "")],
    ]
    processor = _processor(stone_dir=str(tmp_path))
    for step in (0, 1):
        processor.acknowledge(_step(processor, step, rows[step]).batch_id)
    expected = processor._store

    restarted = _processor(stone_dir=str(tmp_path))

    # Step 1 committed; the next step plans from its result, not its base.
    assert restarted._store.stones.keys() == expected.stones.keys()
    assert restarted._store.stones["a"].credit == 1.0 and restarted._store.best == expected.best
    third = _step(restarted, 2, [(0.3, "f", "e"), (0.2, "g", "e"), (0.0, "", ""), (0.0, "", "")])
    assert third.replay_credits == (1.0,)
