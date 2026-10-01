"""Entropic SPO-TTT: the weighted entropic advantage, the reward history, the preparer and the recipe."""

from __future__ import annotations

import json
import math
import random

import pytest

from recipes.espottt import EspotttBatch, ESPOTTTProcessor, ESPOTTTRecipe, ESPOTTTRolloutReport
from recipes.espottt.history import (
    HistorySettings,
    RewardHistory,
    adaptive_beta,
    compress,
    entropic_advantage,
    merge_equal,
)
from recipes.espottt.preparer import STATE_KEY, rescaled
from recipes.spottt.tracker import TASK_KEY, TrackerSettings, node_key
from recipes.tttd.preparer import TttdPreparer
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train import ProcessorContext
from reef.train.algos.registry import resolve_preparer
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicySample

GROUPS, ROLLOUTS = 2, 2


def _sample(reward: float, parent: str = "", grandparent: str = "", index: int = 0) -> PolicySample:
    return PolicySample(
        f"s-{parent}-{reward}-{index}",
        (5, 1),
        (1,),
        (-0.1,),
        reward,
        extras={"parent_id": parent, "grandparent_id": grandparent},
    )


def _batch(samples, **fields) -> EspotttBatch:
    return EspotttBatch("discovery:espottt:0", tuple(samples), **fields)


def _reference(rewards: list[float]) -> tuple[tuple[float, ...], float]:
    return TttdPreparer.adaptive_entropic_advantages(rewards)


# --- the advantage ---


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(6))
def test_with_siblings_as_the_comparison_set_the_advantage_is_ttt_discovers(seed: int) -> None:
    generator = random.Random(seed)
    # Packing-like groups: many failures at 0, valid scores bunched near the top.
    rewards = [0.0 if generator.random() < 0.4 else 2.6 + generator.random() * 0.03 for _ in range(16)]
    expected, expected_beta = _reference(rewards)

    for index, reward in enumerate(rewards):
        others = [(value, 1.0) for j, value in enumerate(rewards) if j != index]
        advantage, beta = entropic_advantage(reward, others)
        assert beta == pytest.approx(expected_beta, rel=1e-12)
        assert advantage == pytest.approx(expected[index], rel=1e-9, abs=1e-9)


@pytest.mark.unit
def test_equal_weights_give_the_reference_beta_and_weights_scale_out() -> None:
    rewards = [0.0, 1.0, 2.0, 2.5]
    assert adaptive_beta(rewards, [1.0] * 4) == pytest.approx(_reference(rewards)[1], rel=1e-12)
    # Only relative weights matter.
    assert adaptive_beta(rewards, [3.0] * 4) == pytest.approx(adaptive_beta(rewards, [1.0] * 4), rel=1e-12)
    # Doubling an outcome's weight is the same as listing it twice.
    assert adaptive_beta([0.0, 1.0], [2.0, 1.0]) == pytest.approx(adaptive_beta([0.0, 0.0, 1.0], [1.0] * 3), rel=1e-12)


@pytest.mark.unit
def test_a_breakthrough_against_a_failed_history_stays_finite() -> None:
    # Every past attempt failed; the attempt scores. beta is bounded by the
    # KL target, so the advantage is large but finite, as TTT-Discover's
    # would be for one success among 64 failures.
    history = [(0.0, 1.0)] * 63
    advantage, beta = entropic_advantage(2.6, history)
    expected, _ = _reference([2.6] + [0.0] * 63)

    assert math.isfinite(advantage) and math.isfinite(beta)
    assert advantage == pytest.approx(expected[0], rel=1e-9)
    assert entropic_advantage(0.0, history)[0] == pytest.approx(0.0, abs=1e-9)


# --- the history ---


@pytest.mark.unit
def test_compress_keeps_total_weight_the_top_and_the_distribution() -> None:
    outcomes = [(float(value), 1.0) for value in range(1001)]

    kept = compress(outcomes, 101)

    assert len(kept) == 101
    assert sum(weight for _, weight in kept) == pytest.approx(1001.0)
    # The best outcome stays exact, with its own weight.
    assert kept[-1] == (1000.0, 1.0)
    # The rest: the outcome covering the middle of each 1/100 share of their weight.
    assert [value for value, _ in kept][:3] == [4.0, 14.0, 24.0]
    assert kept[-2][0] == 994.0
    assert compress(outcomes[:10], 100) == outcomes[:10]


@pytest.mark.unit
def test_equal_rewards_merge_without_loss() -> None:
    outcomes = [(0.0, 1.0)] * 40 + [(2.6, 0.5)] * 3 + [(2.5, 1.0)]

    assert compress(outcomes, 2) == [(0.0, 41.0), (2.6, 1.5)]
    assert merge_equal(outcomes) == [(0.0, 40.0), (2.5, 1.0), (2.6, 1.5)]
    # Merging changes nothing the advantage reads.
    assert entropic_advantage(2.55, merge_equal(outcomes)) == pytest.approx(
        entropic_advantage(2.55, outcomes), rel=1e-12
    )


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(4))
def test_history_weight_and_mean_are_spo_trackers_exactly(seed: int) -> None:
    from recipes.spottt.tracker import ForgettingTracker

    generator = random.Random(seed)
    forgetting = TrackerSettings(half_life=4.0)
    history = RewardHistory(HistorySettings(forgetting, history_size=10_000))
    tracker = ForgettingTracker(forgetting)
    version = 0
    for _ in range(12):
        version += generator.choice((0, 1, 1, 3))
        rewards = [
            generator.choice((0.0, 2.5, 2.6)) + generator.random() * 1e-3 for _ in range(generator.randint(1, 9))
        ]
        if "child" not in history and "parent" in history and generator.random() < 0.5:
            history.adopt("child", history.get("parent"))
            tracker.adopt("child", tracker.estimate("parent"))
        key = generator.choice(("parent", "child")) if "child" in history else "parent"
        history.observe(key, rewards, version)
        tracker.observe(key, rewards, version)
    for key in ("parent", "child"):
        if key not in history:
            continue
        outcomes = history.get(key).outcomes
        weight = sum(w for _, w in outcomes)
        assert weight == pytest.approx(tracker.estimate(key).count, rel=1e-12)
        assert sum(r * w for r, w in outcomes) / weight == pytest.approx(tracker.estimate(key).mean, rel=1e-12)


@pytest.mark.unit
def test_history_forgets_once_per_policy_version_and_survives_json() -> None:
    settings = HistorySettings(TrackerSettings(half_life=8.0), history_size=16)
    history = RewardHistory(settings)
    history.observe("k", [1.0, 2.0], version=0)
    history.observe("k", [3.0], version=2)

    factor = settings.forgetting.forgetting_factor(2.0)
    assert history.get("k").outcomes == pytest.approx([(1.0, factor), (2.0, factor), (3.0, 1.0)])
    # Reading at a later version applies the forgetting that version would.
    assert history.outcomes_at("k", 3)[-1][1] == pytest.approx(settings.forgetting.forgetting_factor(1.0))

    restored = RewardHistory(settings, json.loads(json.dumps(history.state_dict())))
    assert restored.get("k").outcomes == pytest.approx(history.get("k").outcomes)
    assert restored.get("k").version == 2
    with pytest.raises(ValueError, match="later version"):
        history.outcomes_at("k", 1)


# --- the preparer ---


@pytest.mark.unit
def test_siblings_baseline_reproduces_ttt_discovers_advantages_per_group() -> None:
    group_a, group_b = [0.0, 2.6, 2.61, 0.0], [1.0, 1.5, 2.0, 2.0]
    samples = [_sample(r, "a", index=i) for i, r in enumerate(group_a)]
    samples += [_sample(r, "b", index=i) for i, r in enumerate(group_b)]

    signal = resolve_preparer("espottt")(_batch(samples, baseline="siblings"), {})

    expected = _reference(group_a)[0] + _reference(group_b)[0]
    assert signal.advantages == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert signal.loss_family == "tttd"
    assert signal.metrics["comparison_from_siblings_fraction"] == 1.0


@pytest.mark.unit
def test_first_step_falls_back_to_siblings_then_history_takes_over() -> None:
    first_rewards = [0.0, 2.0, 2.5, 1.0] * 4  # 16 attempts from state "a"
    first = resolve_preparer("espottt")(
        _batch([_sample(r, "a", index=i) for i, r in enumerate(first_rewards)], min_history=8.0), {}
    )
    # A cold state with no parent history: the step's other attempts from it.
    assert first.metrics["comparison_from_siblings_fraction"] == 1.0
    state = json.loads(json.dumps(first.next_algorithm_state))
    # Sixteen outcomes, four distinct rewards: equal ones share one entry.
    assert state[STATE_KEY][node_key("a")][1] == [[0.0, 4.0], [1.0, 4.0], [2.0, 4.0], [2.5, 4.0]]
    assert sum(weight for _, weight in state[STATE_KEY][TASK_KEY][1]) == 16.0

    # Next step: two attempts from "a" are compared with its history alone,
    # scaled to a 15-sibling comparison set.
    second = resolve_preparer("espottt")(
        _batch([_sample(2.5, "a", index=0), _sample(0.0, "a", index=1)], min_history=8.0, comparison_size=15),
        state,
    )
    assert second.metrics["comparison_from_own_fraction"] == 1.0
    history = RewardHistory(HistorySettings(min_history=8.0), state[STATE_KEY]).outcomes_at(node_key("a"), 1)
    expected = entropic_advantage(2.5, rescaled(history, 15))[0]
    assert second.advantages[0] == pytest.approx(expected, rel=1e-12)
    assert second.advantages[0] > 0 > second.advantages[1]


@pytest.mark.unit
def test_a_thick_history_keeps_ttt_discovers_advantage_scale() -> None:
    # One breakthrough against 63 equal siblings, and against a history of
    # thousands of the same outcomes: the same advantage, about 63 / 2.
    siblings = [_sample(2.6, "a", index=0)] + [_sample(2.5, "a", index=i) for i in range(1, 64)]
    against_siblings = resolve_preparer("espottt")(_batch(siblings, baseline="siblings"), {})
    state = {STATE_KEY: {node_key("a"): [0, [[2.5, 6000.0]]]}, "steps": 1}
    against_history = resolve_preparer("espottt")(
        _batch([_sample(2.6, "a")], comparison_size=63, min_history=8.0), state
    )

    assert against_history.advantages[0] == pytest.approx(against_siblings.advantages[0], rel=1e-9)
    assert 20 < against_history.advantages[0] < 40


@pytest.mark.unit
def test_cached_advantages_equal_computing_every_attempt_on_its_own() -> None:
    from recipes.espottt.preparer import comparison_set
    from recipes.spottt.preparer import observation_for

    generator = random.Random(7)
    first = resolve_preparer("espottt")(
        _batch([_sample(generator.choice((0.0, 2.5, 2.6)), "a", index=i) for i in range(32)], min_history=8.0), {}
    )
    state = json.loads(json.dumps(first.next_algorithm_state))
    parents = ["a"] * 20 + ["b"] * 12
    samples = [
        _sample(generator.choice((0.0, 2.5, 2.6, 2.61)), parent, "" if parent == "a" else "a", index=i)
        for i, parent in enumerate(parents)
    ]
    signal = resolve_preparer("espottt")(_batch(samples, min_history=8.0, comparison_size=63), state)

    history = RewardHistory(HistorySettings(min_history=8.0), state[STATE_KEY])
    observations = [observation_for(sample, "node") for sample in samples]
    for index, observation in enumerate(observations):
        key = observation.key
        outcomes, _ = comparison_set(
            history,
            observation,
            1,
            siblings=[o.reward for o in observations if o.key == key],
            others=lambda key=key: [o.reward for o in observations if o.key != key],
            baseline="node",
            pool_siblings=False,
        )
        expected = entropic_advantage(observation.reward, rescaled(outcomes, 63))[0]
        assert signal.advantages[index] == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_a_new_state_starts_from_its_parents_history() -> None:
    first = resolve_preparer("espottt")(
        _batch([_sample(r, "a", index=i) for i, r in enumerate([0.0, 2.0, 2.5, 1.0] * 8)], min_history=8.0), {}
    )
    state = json.loads(json.dumps(first.next_algorithm_state))

    # "c" is new; its parent "a" has 32 outcomes, half of which it inherits.
    second = resolve_preparer("espottt")(_batch([_sample(2.5, "c", "a")], min_history=8.0), state)

    assert second.metrics["comparison_from_prior_fraction"] == 1.0
    assert second.metrics["comparison_weight_mean"] == pytest.approx(
        32 * TrackerSettings().forgetting_factor(1.0) * TrackerSettings().inherit_fraction
    )
    stored = second.next_algorithm_state[STATE_KEY][node_key("c")]
    inherited = 32 * TrackerSettings().inherit_fraction * TrackerSettings().forgetting_factor(1.0)
    assert stored[0] == 1 and sum(weight for _, weight in stored[1]) == pytest.approx(inherited + 1)


@pytest.mark.unit
def test_pooling_siblings_adds_them_to_a_thick_history() -> None:
    first = resolve_preparer("espottt")(
        _batch([_sample(r, "a", index=i) for i, r in enumerate([0.0, 2.0] * 8)], min_history=4.0), {}
    )
    state = json.loads(json.dumps(first.next_algorithm_state))
    samples = [_sample(2.5, "a", index=0), _sample(1.0, "a", index=1)]

    alone = resolve_preparer("espottt")(_batch(samples, min_history=4.0), state)
    pooled = resolve_preparer("espottt")(_batch(samples, min_history=4.0, pool_siblings=True), state)

    assert alone.metrics["comparison_from_own_fraction"] == 1.0
    assert pooled.metrics["comparison_from_siblings_fraction"] == 1.0
    assert pooled.metrics["comparison_weight_mean"] == pytest.approx(alone.metrics["comparison_weight_mean"] + 1)


@pytest.mark.unit
def test_a_lone_attempt_with_nothing_to_compare_gets_no_advantage() -> None:
    signal = resolve_preparer("espottt")(_batch([_sample(2.0, "a")]), {})

    assert signal.advantages == (0.0,)
    assert signal.metrics["comparison_from_none_fraction"] == 1.0


@pytest.mark.unit
def test_prepared_payload_is_one_tttd_step_over_the_whole_batch() -> None:
    samples = tuple(_sample(r, "a", index=i) for i, r in enumerate([0.0, 1.0, 2.0, 2.5]))

    prepared = prepare_slime_step(EspotttBatch("discovery:espottt:3", samples), "espottt", {"steps": 3})

    assert prepared.payload["loss"] == "tttd"
    assert len(prepared.payload["advantages"]) == 4


@pytest.mark.unit
def test_preparer_and_batch_reject_wrong_inputs() -> None:
    from recipes.ppottt import ScheduledPolicyBatch

    with pytest.raises(TypeError, match="EspotttBatch"):
        resolve_preparer("espottt")(ScheduledPolicyBatch("b", (_sample(1.0),)), {})
    with pytest.raises(ValueError, match="baseline"):
        _batch([_sample(1.0)], baseline="none")
    with pytest.raises(ValueError, match="history_size"):
        _batch([_sample(1.0)], history_size=1)
    with pytest.raises(ValueError, match="min_history"):
        _batch([_sample(1.0)], min_history=0.0)


# --- report, processor and recipe ---


class _ExperimentLogger:
    def log(self, metrics, *, namespace):
        pass


def _inference(agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        payload={
            "input_ids": [10, 11],
            "response": {"training": {"tokens": [10, 11, 12], "loss_mask": [1], "rollout_log_probs": [-0.2]}},
        },
    )


def _report(step: int, group: int, rollout: int, score: float, parent: str) -> AgentRecord:
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
                "algorithm": "espottt",
                "step": step,
                "group": group,
                "rollout": rollout,
                "groups_per_step": GROUPS,
                "rollouts_per_group": ROLLOUTS,
                "parent_id": parent,
                "grandparent_id": "state-0",
            },
        },
    )


@pytest.mark.unit
def test_report_accepts_only_its_own_tag() -> None:
    assert ESPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1).algorithm == "espottt"
    with pytest.raises(ReportValidationError, match="algorithm"):
        ESPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, algorithm="spottt")


@pytest.mark.unit
def test_processor_builds_a_batch_with_parent_ids_and_history_settings() -> None:
    processor = ESPOTTTProcessor(
        ProcessorContext(
            "discovery",
            {"groups_per_step": GROUPS, "rollouts_per_group": ROLLOUTS, "min_history": 4.0, "pool_siblings": True},
            report_type=ESPOTTTRolloutReport,
            experiment_logger=_ExperimentLogger(),
        )
    )
    for group in range(GROUPS):
        for rollout in range(ROLLOUTS):
            processor.ingest(_inference(f"i-0-{group}-{rollout}"))
            processor.ingest(_report(0, group, rollout, float(rollout), f"state-{group}"))

    batch = processor.build_batch()

    assert isinstance(batch, EspotttBatch) and batch.batch_id == "discovery:espottt:0"
    assert [sample.extras["parent_id"] for sample in batch.samples] == ["state-0", "state-0", "state-1", "state-1"]
    assert (batch.min_history, batch.pool_siblings, batch.minibatch_size) == (4.0, True, 0)
    # TTT-Discover's comparison set is a group's other siblings, here raised to the least allowed, 2.
    assert batch.comparison_size == max(ROLLOUTS - 1, 2)
    with pytest.raises(ValueError, match="baseline"):
        ESPOTTTProcessor(ProcessorContext("discovery", {"baseline": "none"}, report_type=ESPOTTTRolloutReport))


@pytest.mark.unit
def test_recipe_trains_with_ttt_discovers_loss_family() -> None:
    from reef_service.runtime_stubs import StubTrainingRuntime

    from reef.recipe import RecipeConfigError, build_recipe

    spec = ESPOTTTRecipe.training_spec()
    assert (spec.step_preparer, spec.loss_family, spec.processor) == ("espottt", "tttd", ESPOTTTProcessor)

    recipe = build_recipe(
        "recipes.espottt.recipe:ESPOTTTRecipe",
        {"REEF_ESPOTTT_MIN_HISTORY": "4", "REEF_ESPOTTT_POOL_SIBLINGS": "true"},
        runtime=StubTrainingRuntime(),
    )
    assert isinstance(recipe, ESPOTTTRecipe) and recipe.report_type is ESPOTTTRolloutReport
    assert (recipe.min_history, recipe.pool_siblings, recipe.minibatch_size) == (4.0, True, 0)
    with pytest.raises(RecipeConfigError, match="baseline"):
        build_recipe(
            "recipes.espottt.recipe:ESPOTTTRecipe", {"REEF_ESPOTTT_BASELINE": "none"}, runtime=StubTrainingRuntime()
        )


@pytest.mark.unit
def test_a_two_attempt_group_still_gets_bounded_advantages() -> None:
    # Two attempts per state: the comparison set is raised to a weight of 2,
    # where KL = log 2 has a finite beta (at weight 1 it has none).
    processor = ESPOTTTProcessor(
        ProcessorContext(
            "discovery", {"groups_per_step": 2, "rollouts_per_group": 2}, report_type=ESPOTTTRolloutReport
        )
    )
    assert processor.comparison_size == 2
    samples = [_sample(r, p, index=i) for i, (r, p) in enumerate([(0.9, "a"), (0.1, "a"), (0.5, "b"), (0.2, "b")])]

    signal = resolve_preparer("espottt")(_batch(samples, comparison_size=2), {})

    # About 14 for the better of two, as for one success against two equal outcomes; 1e12 at weight 1.
    assert max(signal.advantages) < 20 and all(math.isfinite(value) for value in signal.advantages)
    with pytest.raises(ValueError, match="comparison_size"):
        _batch(samples, comparison_size=1)
