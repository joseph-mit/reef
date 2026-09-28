"""SPO-TTT: parent ids on the batch, tracker-based advantages, persisted state, loss-family contract."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from recipes.spottt import SpotttBatch, SPOTTTProcessor, SPOTTTRecipe, SPOTTTRolloutReport
from recipes.spottt.preparer import STATE_KEY, observation_for
from recipes.spottt.tracker import TASK_KEY, ForgettingTracker, node_key
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train import ProcessorContext
from reef.train.algos.registry import resolve_preparer
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicySample

GROUPS, ROLLOUTS = 2, 2


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


def _report(step: int, group: int, rollout: int, score: float, parent: str, grandparent: str = "") -> AgentRecord:
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
                "algorithm": "spottt",
                "step": step,
                "group": group,
                "rollout": rollout,
                "groups_per_step": GROUPS,
                "rollouts_per_group": ROLLOUTS,
                "parent_id": parent,
                "grandparent_id": grandparent,
            },
        },
    )


def _processor(**config) -> SPOTTTProcessor:
    settings = {"groups_per_step": GROUPS, "rollouts_per_group": ROLLOUTS, "minibatch_size": 2}
    settings.update(config)
    return SPOTTTProcessor(
        ProcessorContext("discovery", settings, report_type=SPOTTTRolloutReport, experiment_logger=_ExperimentLogger())
    )


def _sample(reward: float, parent: str = "", grandparent: str = "") -> PolicySample:
    return PolicySample(
        f"s-{parent}-{reward}",
        (5, 1),
        (1,),
        (-0.1,),
        reward,
        extras={"parent_id": parent, "grandparent_id": grandparent},
    )


def _batch(samples, **fields) -> SpotttBatch:
    return SpotttBatch("discovery:spottt:0", tuple(samples), minibatch_size=0, **fields)


# --- report and processor ---


@pytest.mark.unit
def test_report_carries_lineage_and_rejects_other_recipes_tags() -> None:
    report = SPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, parent_id="state-3", grandparent_id="state-1")
    assert (report.algorithm, report.parent_id, report.grandparent_id) == ("spottt", "state-3", "state-1")

    with pytest.raises(ReportValidationError, match="algorithm"):
        SPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, algorithm="ppottt")


@pytest.mark.unit
def test_processor_attaches_each_attempts_lineage_and_tracker_settings() -> None:
    processor = _processor(baseline="task", half_life=4.0)
    for group in range(GROUPS):
        for rollout in range(ROLLOUTS):
            processor.ingest(_inference(f"i-0-{group}-{rollout}"))
            processor.ingest(_report(0, group, rollout, float(rollout), f"state-{group}", "state-0"))

    batch = processor.build_batch()

    assert isinstance(batch, SpotttBatch)
    assert batch.batch_id == "discovery:spottt:0"
    assert [sample.extras["parent_id"] for sample in batch.samples] == ["state-0", "state-0", "state-1", "state-1"]
    assert {sample.extras["grandparent_id"] for sample in batch.samples} == {"state-0"}
    assert (batch.baseline, batch.half_life, batch.minibatch_size) == ("task", 4.0, 2)


@pytest.mark.unit
def test_processor_and_batch_reject_an_unknown_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        _processor(baseline="critic")
    with pytest.raises(ValueError, match="baseline"):
        _batch([_sample(1.0)], baseline="critic")
    with pytest.raises(ValueError, match="rho"):
        _batch([_sample(1.0)], rho_min=0.99, rho_max=0.9)


# --- observation keys ---


@pytest.mark.unit
def test_node_baseline_keys_by_parent_with_grandparent_prior_and_task_fallback() -> None:
    observation = observation_for(_sample(2.0, "state-7", "state-2"), "node")

    assert observation.key == node_key("state-7")
    assert observation.prior_key == node_key("state-2")
    assert observation.fallbacks == (node_key("state-2"), TASK_KEY)


@pytest.mark.unit
def test_attempts_without_a_parent_and_task_baseline_use_the_task_key() -> None:
    assert observation_for(_sample(2.0, ""), "node").key == TASK_KEY
    assert observation_for(_sample(2.0, "state-7"), "task").key == TASK_KEY
    assert observation_for(_sample(2.0, "state-7"), "node").fallbacks == (TASK_KEY,)


# --- preparer ---


@pytest.mark.unit
def test_first_step_centres_on_the_step_mean_and_persists_the_tracker() -> None:
    batch = _batch([_sample(1.0, "a"), _sample(3.0, "a"), _sample(2.0, "b"), _sample(6.0, "b")], normalize=False)

    signal = resolve_preparer("spottt")(batch, {})

    assert signal.loss_family == "spottt"
    assert signal.advantages == pytest.approx((-2.0, 0.0, -1.0, 3.0))  # step mean is 3
    assert signal.next_algorithm_state["steps"] == 1
    tracker_state = signal.next_algorithm_state[STATE_KEY]
    json.dumps(tracker_state)  # must survive Reef's durable algorithm state
    assert tracker_state[node_key("a")][:2] == pytest.approx([2.0, 2.0])
    assert tracker_state[node_key("b")][:2] == pytest.approx([4.0, 2.0])
    assert tracker_state[TASK_KEY][:2] == pytest.approx([3.0, 4.0])
    assert signal.metrics["tracked_fraction"] == 0.0


@pytest.mark.unit
def test_later_steps_use_the_committed_tracker_and_its_version() -> None:
    first = resolve_preparer("spottt")(_batch([_sample(1.0, "a"), _sample(3.0, "a")], normalize=False), {})
    state = json.loads(json.dumps(first.next_algorithm_state))

    second = resolve_preparer("spottt")(_batch([_sample(4.0, "a"), _sample(0.0, "new", "a")], normalize=False), state)

    # "a" has mean 2; "new" is cold and falls back to its parent "a".
    assert second.advantages == pytest.approx((2.0, -2.0))
    assert second.metrics["tracked_fraction"] == 1.0
    assert second.next_algorithm_state["steps"] == 2
    tracker = ForgettingTracker(state=second.next_algorithm_state[STATE_KEY])
    assert tracker.estimate(node_key("a")).version == 1
    # The new node started from half of "a"'s count as its prior.
    assert tracker.estimate(node_key("new")).count == pytest.approx(
        ForgettingTracker().settings.forgetting_factor(1.0) * 1.0 + 1.0
    )


@pytest.mark.unit
def test_no_baseline_ablation_whitens_raw_rewards_and_keeps_the_tracker() -> None:
    prior = {"steps": 3, STATE_KEY: {TASK_KEY: [1.0, 2.0, 2]}}

    signal = resolve_preparer("spottt")(_batch([_sample(1.0), _sample(3.0)], baseline="none"), prior)

    assert signal.advantages == pytest.approx((-1.0, 1.0), rel=1e-5)
    assert signal.next_algorithm_state == {"steps": 4, STATE_KEY: {TASK_KEY: [1.0, 2.0, 2]}}


@pytest.mark.unit
def test_prepared_payload_ships_one_advantage_per_row_in_schedule_order() -> None:
    batch = SpotttBatch(
        "discovery:spottt:5",
        (_sample(1.0, "a"), _sample(3.0, "a"), _sample(2.0, "b"), _sample(6.0, "b")),
        minibatch_size=2,
        shuffle=False,
    )

    prepared = prepare_slime_step(batch, "spottt", {"steps": 5})

    assert prepared.payload["loss"] == "spottt"
    assert prepared.payload["external_step_sizes"] == [2, 2]
    assert len(prepared.payload["advantages"]) == 4
    assert sum(prepared.payload["advantages"]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_preparer_rejects_a_batch_without_tracker_settings() -> None:
    from recipes.ppottt import ScheduledPolicyBatch

    with pytest.raises(TypeError, match="SpotttBatch"):
        resolve_preparer("spottt")(ScheduledPolicyBatch("b", (_sample(1.0),)), {})


# --- recipe and loss family ---


@pytest.mark.unit
def test_recipe_binds_its_parts() -> None:
    spec = SPOTTTRecipe.training_spec()

    assert (spec.step_preparer, spec.loss_family, spec.processor) == ("spottt", "spottt", SPOTTTProcessor)


def _backend_args(**overrides):
    values = {"use_critic": False, "advantage_estimator": "grpo", "eps_clip": 0.2, "kl_coef": 0.1}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_loss_family_takes_shipped_advantages_on_the_clipped_loss() -> None:
    family = resolve_loss_family("spottt")

    assert (family.loss_type, family.advantages, family.requires_rollout_logprobs) == ("policy_loss", "required", True)
    family.validate_specific_args(_backend_args(), "reef.recipe=spottt")
    args = SimpleNamespace(compute_advantages_and_returns=False)
    family.configure_backend_args(args)
    assert args.compute_advantages_and_returns is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_critic": True}, "no value model"),
        ({"advantage_estimator": "ppo"}, "advantage-estimator"),
        ({"advantage_estimator": "cispo"}, "advantage-estimator"),
        ({"eps_clip": 0.0}, "eps-clip"),
        ({"kl_coef": 0.0}, "kl-coef"),
        ({"kl_coef": float("inf")}, "kl-coef"),
    ],
)
def test_loss_family_rejects_objective_drift(overrides, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("spottt").validate_specific_args(_backend_args(**overrides), "reef.recipe=spottt")


@pytest.mark.unit
def test_loss_family_reuses_tttds_frozen_base_kl_hook(monkeypatch) -> None:
    pytest.importorskip("torch")
    from recipes.spottt.slime import objective
    from reef.train.slime_backend.algorithm import resolve_objective_paths

    args = SimpleNamespace(loss_family="spottt", compute_advantages_and_returns=True)
    resolve_objective_paths(args)
    assert args.custom_advantage_function_path.endswith("recipes.spottt.slime.objective.spottt_advantages")

    calls = []
    monkeypatch.setattr(objective, "tttd_advantages", lambda a, data: calls.append((a, data)))
    objective.spottt_advantages("args", {"advantages": []})
    assert calls == [("args", {"advantages": []})]
