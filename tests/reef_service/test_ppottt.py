"""PPO-TTT: step barrier, flat scheduled batch, loss-family contract and critic cadence."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from recipes.ppottt import PPOTTTProcessor, PPOTTTRecipe, PPOTTTRolloutReport, ScheduledPolicyBatch
from reef.artifact import ArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train import ProcessorContext
from reef.train.algos.registry import resolve_preparer
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicySample

GROUPS, ROLLOUTS = 2, 3


class _ExperimentLogger:
    def __init__(self) -> None:
        self.events = []

    def log(self, metrics, *, namespace):
        self.events.append((namespace, dict(metrics)))


def _inference(agent_record_id: str, token: int, *, artifact_ref: ArtifactRef | None = None) -> AgentRecord:
    return AgentRecord.create(
        scenario="discovery",
        request_type=RequestType.INFERENCE,
        agent_record_id=agent_record_id,
        artifact_ref=artifact_ref,
        payload={
            "input_ids": [10, 11],
            "response": {"training": {"tokens": [10, 11, token], "loss_mask": [1], "rollout_log_probs": [-0.2]}},
        },
    )


def _report(
    step: int,
    group: int,
    rollout: int,
    score: float,
    *,
    groups: int = GROUPS,
    rollouts: int = ROLLOUTS,
    algorithm: str = "ppottt",
) -> AgentRecord:
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
                "algorithm": algorithm,
                "step": step,
                "group": group,
                "rollout": rollout,
                "groups_per_step": groups,
                "rollouts_per_group": rollouts,
                "parent_id": f"state-{group}",
            },
        },
    )


def _processor(experiment_logger=None, **config) -> PPOTTTProcessor:
    settings = {"groups_per_step": GROUPS, "rollouts_per_group": ROLLOUTS, "minibatch_size": 3, "ppo_epochs": 2}
    settings.update(config)
    return PPOTTTProcessor(
        ProcessorContext(
            "discovery",
            settings,
            report_type=PPOTTTRolloutReport,
            experiment_logger=experiment_logger or _ExperimentLogger(),
        )
    )


def _fill_step(processor, rewards, *, step=0, artifact_ref=None, skip=None):
    for group, group_rewards in enumerate(rewards):
        for rollout, reward in enumerate(group_rewards):
            processor.ingest(_inference(f"i-{step}-{group}-{rollout}", 100 + rollout, artifact_ref=artifact_ref))
            if (group, rollout) != skip:
                processor.ingest(_report(step, group, rollout, reward))


# --- report contract ---


@pytest.mark.unit
def test_report_accepts_single_attempt_grids_and_rejects_foreign_tags() -> None:
    report = PPOTTTRolloutReport(score=1.0, step=0, group=0, rollout=0, groups_per_step=512, rollouts_per_group=1)
    assert report.algorithm == "ppottt"

    with pytest.raises(ReportValidationError, match="algorithm"):
        PPOTTTRolloutReport(1.0, 0, 0, 0, 1, 1, algorithm="ttt-discover")
    with pytest.raises(ReportValidationError, match="rollouts_per_group >= 1"):
        PPOTTTRolloutReport(1.0, 0, 0, 0, 1, 0)
    with pytest.raises(ReportValidationError, match="group must sit"):
        PPOTTTRolloutReport(1.0, 0, 2, 0, 2, 1)
    with pytest.raises(ReportValidationError, match="rollout must sit"):
        PPOTTTRolloutReport(1.0, 0, 0, 3, 1, 3)


# --- processor ---


@pytest.mark.unit
def test_processor_waits_for_the_whole_step_then_flattens_it_in_grid_order() -> None:
    processor = _processor()
    rewards = ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))
    _fill_step(processor, rewards, skip=(1, 2))

    assert not processor.ready()

    processor.ingest(_report(0, 1, 2, 5.0))

    batch = processor.build_batch()
    assert isinstance(batch, ScheduledPolicyBatch)
    assert batch.batch_id == "discovery:ppottt:0"
    assert [sample.reward for sample in batch.samples] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.unit
def test_processor_keeps_constant_reward_groups() -> None:
    # Group-relative methods drop these; against a critic they still carry
    # signal, and they are the critic's training data either way.
    processor = _processor()
    _fill_step(processor, ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))

    batch = processor.build_batch()

    assert [sample.reward for sample in batch.samples] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


@pytest.mark.unit
def test_processor_accepts_one_attempt_per_parent() -> None:
    processor = _processor(rollouts_per_group=1, minibatch_size=0)
    for group in range(GROUPS):
        processor.ingest(_inference(f"i-0-{group}-0", 100))
        processor.ingest(_report(0, group, 0, float(group), rollouts=1))

    batch = processor.build_batch()

    assert [sample.reward for sample in batch.samples] == [0.0, 1.0]


@pytest.mark.unit
def test_processor_stamps_the_update_schedule_on_the_batch() -> None:
    processor = _processor(minibatch_size=3, ppo_epochs=2, shuffle_minibatches=False)
    _fill_step(processor, ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)))

    scheduling = processor.build_batch().scheduling()

    assert (scheduling.unit, scheduling.batch_size, scheduling.epochs, scheduling.shuffle) == ("sample", 3, 2, False)


@pytest.mark.unit
def test_processor_does_not_mix_steps() -> None:
    processor = _processor()
    for step, group in ((0, 0), (1, 1)):
        for rollout in range(ROLLOUTS):
            processor.ingest(_inference(f"i-{step}-{group}-{rollout}", 100 + rollout))
            processor.ingest(_report(step, group, rollout, float(rollout)))

    assert not processor.ready()


@pytest.mark.unit
def test_processor_discards_a_step_spanning_release_ids(caplog) -> None:
    processor = _processor()
    with caplog.at_level("ERROR", logger="recipes.ppottt.processor"):
        for group in range(GROUPS):
            for rollout in range(ROLLOUTS):
                version = "checkpoint-2" if (group, rollout) == (1, 2) else "checkpoint-1"
                processor.ingest(
                    _inference(
                        f"i-0-{group}-{rollout}",
                        100 + rollout,
                        artifact_ref=ArtifactRef(f"a-{version}", version, None),
                    )
                )
                processor.ingest(_report(0, group, rollout, 1.0))

    assert not processor.ready()
    assert "span releases" in caplog.text
    assert processor.status() == {
        "failed_steps": [{"step": 0, "reason": "mixed_release_ids", "release_ids": ["checkpoint-1", "checkpoint-2"]}]
    }


@pytest.mark.unit
def test_processor_rejects_a_report_for_another_grid() -> None:
    processor = _processor()
    report = _report(0, 0, 0, 1.0, groups=8, rollouts=64)

    processor.ingest(report)

    assert any("8x64 grid" in reason for reason in processor.never_reasons)
    assert report.agent_record_id in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_processor_logs_the_step_reward_summary() -> None:
    experiment_logger = _ExperimentLogger()
    processor = _processor(experiment_logger)
    _fill_step(processor, ((0.0, 2.0, 2.0), (2.0, 2.0, 4.0)))

    processor.build_batch()

    namespace, metrics = experiment_logger.events[-1]
    assert namespace == "ppottt"
    assert metrics["grid_rollouts"] == 6
    assert metrics["reward_mean"] == pytest.approx(2.0)
    assert metrics["reward_std"] == pytest.approx((8 / 6) ** 0.5)
    assert metrics["reward_zero_fraction"] == pytest.approx(1 / 6)


# --- batch and preparer ---


def _sample(record_id: str, reward: float) -> PolicySample:
    return PolicySample(record_id, (5, 1), (1,), (-0.1,), reward)


@pytest.mark.unit
def test_scheduled_batch_validates_its_schedule() -> None:
    with pytest.raises(ValueError, match="ppo_epochs"):
        ScheduledPolicyBatch("b", (_sample("a", 1.0),), ppo_epochs=0)
    with pytest.raises(ValueError, match="minibatch_size"):
        ScheduledPolicyBatch("b", (_sample("a", 1.0),), minibatch_size=-1)
    assert ScheduledPolicyBatch("b", (_sample("a", 1.0),)).scheduling().batch_size == "actual"


@pytest.mark.unit
def test_preparer_ships_no_advantages_and_passes_the_schedule_through() -> None:
    batch = ScheduledPolicyBatch(
        "discovery:ppottt:0", tuple(_sample(f"s{i}", float(i)) for i in range(4)), ppo_epochs=1, minibatch_size=2
    )

    signal = resolve_preparer("ppottt")(batch, {})

    assert signal.action == "train"
    assert signal.loss_family == "ppottt"
    assert signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 1}
    assert signal.metrics["minibatch_size"] == 2


@pytest.mark.unit
def test_prepared_payload_cuts_the_step_into_minibatches_without_advantages() -> None:
    batch = ScheduledPolicyBatch(
        "discovery:ppottt:3", tuple(_sample(f"s{i}", float(i)) for i in range(4)), ppo_epochs=2, minibatch_size=2
    )

    prepared = prepare_slime_step(batch, "ppottt", {"steps": 3})

    assert prepared.action == "train"
    assert "advantages" not in prepared.payload
    assert prepared.payload["loss"] == "ppottt"
    assert prepared.payload["external_step_sizes"] == [2, 2, 2, 2]
    assert len(prepared.payload["samples"]) == 8
    assert prepared.next_algorithm_state == {"steps": 4}


@pytest.mark.unit
def test_preparer_rejects_an_unscheduled_batch() -> None:
    from reef.train.types import PolicyBatch

    with pytest.raises(TypeError, match="ScheduledPolicyBatch"):
        resolve_preparer("ppottt")(PolicyBatch("b", (_sample("a", 1.0),)), {})


# --- recipe ---


@pytest.mark.unit
def test_recipe_binds_its_parts_and_exposes_the_schedule_to_the_processor() -> None:
    spec = PPOTTTRecipe.training_spec()

    assert (spec.step_preparer, spec.loss_family, spec.processor) == ("ppottt", "ppottt", PPOTTTProcessor)
    fields = PPOTTTRecipe.__dataclass_fields__
    for name in ("groups_per_step", "rollouts_per_group", "ppo_epochs", "minibatch_size", "shuffle_minibatches"):
        assert name in fields


# --- loss family ---


def _backend_args(**overrides):
    values = {
        "use_critic": True,
        "advantage_estimator": "ppo",
        "normalize_advantages": True,
        "eps_clip": 0.2,
        "eps_clip_high": None,
        "value_clip": 0.2,
        "gamma": 1.0,
        "lambd": 1.0,
        "kl_coef": 0.1,
        "critic_steps_per_actor": 2,
        "num_critic_only_steps": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_loss_family_declares_the_stock_clipped_loss_and_a_critic() -> None:
    family = resolve_loss_family("ppottt")

    assert family.loss_type == "policy_loss"
    assert family.advantages == "forbidden"
    assert family.requires_rollout_logprobs is True
    assert family.critic_value_head_zero_init is True
    assert family.uses_pg_loss_primitive is False
    family.validate_specific_args(_backend_args(), "reef.recipe=ppottt")


@pytest.mark.unit
def test_loss_family_resolves_its_advantage_hook() -> None:
    pytest.importorskip("torch")
    from reef.train.slime_backend.algorithm import resolve_objective_paths

    args = SimpleNamespace(loss_family="ppottt", compute_advantages_and_returns=True)
    resolve_objective_paths(args)

    assert args.custom_advantage_function_path.endswith("recipes.ppottt.slime.objective.ppottt_advantages")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_critic": False}, "value model"),
        ({"advantage_estimator": "grpo"}, "advantage-estimator ppo"),
        ({"normalize_advantages": False}, "normalize-advantages"),
        ({"eps_clip": 0.0}, "eps-clip"),
        ({"eps_clip": 1.0}, "eps-clip"),
        ({"eps_clip": True}, "eps-clip"),
        ({"eps_clip_high": -0.1}, "eps-clip-high"),
        ({"value_clip": 0.0}, "value-clip"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": 1.5}, "gamma"),
        ({"lambd": -0.1}, "lambd"),
        ({"kl_coef": -0.1}, "kl-coef"),
        ({"kl_coef": float("nan")}, "kl-coef"),
        ({"critic_steps_per_actor": 0}, "critic-steps-per-actor"),
        ({"num_critic_only_steps": -1}, "num-critic-only-steps"),
    ],
)
def test_loss_family_rejects_objective_drift(overrides, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("ppottt").validate_specific_args(_backend_args(**overrides), "reef.recipe=ppottt")


@pytest.mark.unit
def test_loss_family_allows_the_kl_term_to_be_switched_off() -> None:
    resolve_loss_family("ppottt").validate_specific_args(_backend_args(kl_coef=0.0), "reef.recipe=ppottt")


@pytest.mark.unit
def test_critic_role_drops_the_kl_term() -> None:
    critic_args = SimpleNamespace(kl_coef=0.1)

    resolve_loss_family("ppottt").configure_critic_args(critic_args)

    assert critic_args.kl_coef == 0.0


@pytest.mark.unit
def test_backend_args_turn_on_slimes_advantage_pass() -> None:
    args = SimpleNamespace(compute_advantages_and_returns=False)

    resolve_loss_family("ppottt").configure_backend_args(args)

    assert args.compute_advantages_and_returns is True


# --- critic cadence through the bridge ---


def _bridge_harness(tmp_path, *, critic_only_steps=0):
    pytest.importorskip("ray")
    from tests.reef_service.test_sao_bridge import _FakeRolloutManager, _RecordingGroup
    from reef.train.slime_backend.reef_adapters import bridge

    class _RefittingCritic(_RecordingGroup):
        """Returns different values on each pass, as a critic that trains between passes would."""

        def async_train(self, rollout_id, rollout_data_ref, external_data=None):
            super().async_train(rollout_id, rollout_data_ref, external_data)
            return [{"values": [0.25 * len(self.train_calls)]}]

    template = str(tmp_path / "checkpoint-{rollout_id}")
    actor_group = _RecordingGroup(template)
    critic_group = _RefittingCritic(template, critic=True)
    actor = bridge.TrainBridgeActorImpl(
        actor_group,
        _FakeRolloutManager(["packed"]),
        save_hf_template=template,
        critic_group=critic_group,
        critic_steps_per_actor=None,
        critic_only_steps=critic_only_steps,
        loss_family="ppottt",
    )
    payload = {
        "samples": [["a", [9, 1, 2], [1, 1], [-0.1, -0.2], 0.5], ["b", [9, 1, 2], [1, 1], [-0.1, -0.2], 1.0]],
        "rollout_ids": [0, 1],
        "loss": "ppottt",
        "rollout_id": 0,
        "expected_runtime_load_id": "inc:5",
    }
    return actor, actor_group, critic_group, payload


@pytest.fixture
def _local_ray_get(monkeypatch):
    pytest.importorskip("ray")
    from reef.train.slime_backend.reef_adapters import bridge

    monkeypatch.setattr(bridge.ray, "get", lambda value, **kwargs: value)


def _run_job(actor, payload):
    result = actor.execute_training_job(payload)
    if result.outcome != "checkpoint":
        return result
    return actor.update_serving_weights(result.training_job_id)


@pytest.mark.unit
def test_training_step_fits_the_critic_twice_then_moves_the_policy(tmp_path, _local_ray_get) -> None:
    actor, actor_group, critic_group, payload = _bridge_harness(tmp_path)

    result = _run_job(actor, payload)

    assert result.outcome == "complete"
    assert len(critic_group.train_calls) == 2
    assert len(actor_group.train_calls) == 1
    # The actor's advantages use the values read before the critic's first
    # update on this step (PPO's V_old), not the refitted second pass.
    assert actor_group.external_data == [[{"values": [0.25]}]]
    assert result.metrics["ppottt/critic_updates"] == 2
    assert result.metrics["ppottt/actor_trained"] == 1
    assert result.metrics["ppottt/effective_token_rate"] == 1.0


@pytest.mark.unit
def test_critic_warmup_step_leaves_the_policy_alone(tmp_path, _local_ray_get) -> None:
    actor, actor_group, critic_group, payload = _bridge_harness(tmp_path, critic_only_steps=1)

    result = _run_job(actor, payload)

    assert result.outcome == "complete"
    assert len(critic_group.train_calls) == 2
    assert actor_group.train_calls == []
    assert result.metrics["ppottt/actor_trained"] == 0


@pytest.mark.unit
def test_bind_rejects_bridge_config() -> None:
    with pytest.raises(TypeError, match="no bridge algorithm config"):
        resolve_loss_family("ppottt").bind(object())


@pytest.mark.unit
def test_shipped_configs_live_next_to_the_tttd_example() -> None:
    example = Path(__file__).resolve().parents[2] / "recipes" / "tttd" / "examples" / "tttd"
    for name in ("serve.ppottt.yaml", "serve.ppottt-smoke.yaml"):
        text = (example / name).read_text()
        assert "recipes.ppottt.recipe:PPOTTTRecipe" in text
        assert "--use-critic" in text and "--advantage-estimator=ppo" in text
