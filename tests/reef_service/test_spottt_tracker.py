"""The SPO-TTT forgetting tracker against the SPO update rule it generalises."""

from __future__ import annotations

import json
import math

import pytest

from recipes.spottt.tracker import (
    TASK_KEY,
    ForgettingTracker,
    Observation,
    TrackerSettings,
    assign_advantages,
    batch_normalize,
    node_key,
)


def _beta_posterior_mean(alpha: float, beta: float, rewards: list[int], rho: float) -> tuple[float, float]:
    """SPO Eq. 5 with one discount for the batch: returns (mean, alpha + beta)."""
    successes = sum(rewards)
    alpha = rho * alpha + successes
    beta = rho * beta + (len(rewards) - successes)
    return alpha / (alpha + beta), alpha + beta


@pytest.mark.unit
def test_binary_rewards_follow_the_discounted_beta_posterior() -> None:
    settings = TrackerSettings(half_life=8.0)
    tracker = ForgettingTracker(settings)
    tracker.observe("q", [1, 0, 1, 1], version=0)  # alpha=3, beta=1

    alpha, beta = 3.0, 1.0
    for version, rewards in ((1, [0, 0, 1]), (3, [1]), (4, [1, 1, 0, 0, 0])):
        estimate = tracker.estimate("q")
        rho = settings.forgetting_factor(version - estimate.version)
        expected_mean, expected_count = _beta_posterior_mean(alpha, beta, rewards, rho)
        alpha = rho * alpha + sum(rewards)
        beta = rho * beta + len(rewards) - sum(rewards)

        updated = tracker.observe("q", rewards, version=version)

        assert updated.mean == pytest.approx(expected_mean)
        assert updated.count == pytest.approx(expected_count)


@pytest.mark.unit
def test_single_observation_is_spo_adaptive_ema() -> None:
    settings = TrackerSettings(half_life=8.0)
    tracker = ForgettingTracker(settings)
    tracker.observe("q", [0.4, 0.6], version=0)
    before = tracker.estimate("q")

    updated = tracker.observe("q", [1.0], version=1)

    # SPO Eq. 7: v = v_-1 + eta (r - v_-1) with eta = 1 / (rho N + 1).
    rho = settings.forgetting_factor(1.0)
    eta = 1.0 / (rho * before.count + 1.0)
    assert updated.mean == pytest.approx(before.mean + eta * (1.0 - before.mean))


@pytest.mark.unit
def test_forgetting_factor_is_clipped_to_spo_bounds() -> None:
    settings = TrackerSettings(half_life=8.0, rho_min=0.875, rho_max=0.96)

    assert settings.forgetting_factor(0.0) == pytest.approx(0.96)  # 2**0 = 1, capped above
    assert settings.forgetting_factor(1.0) == pytest.approx(2 ** (-1 / 8))
    assert settings.forgetting_factor(100.0) == pytest.approx(0.875)  # floor
    with pytest.raises(ValueError):
        settings.forgetting_factor(-1.0)


@pytest.mark.unit
def test_longer_gaps_forget_more_within_the_bounds() -> None:
    def mean_after_gap(gap: int) -> float:
        tracker = ForgettingTracker(TrackerSettings(half_life=8.0, rho_min=0.01, rho_max=1.0))
        tracker.observe("q", [0.0] * 8, version=0)
        return tracker.observe("q", [1.0], version=gap).mean

    assert mean_after_gap(1) < mean_after_gap(4) < mean_after_gap(16)


@pytest.mark.unit
def test_cold_node_inherits_its_parents_estimate_as_prior() -> None:
    settings = TrackerSettings(inherit_fraction=0.5)
    tracker = ForgettingTracker(settings)
    tracker.observe(node_key("state-1"), [2.0, 2.0, 2.0, 2.0], version=0)

    child = tracker.observe(node_key("state-7"), [3.0], version=1, prior_key=node_key("state-1"))

    # Prior: mean 2 with count 4 * 0.5 = 2, discounted by one step, then one reward of 3.
    rho = settings.forgetting_factor(1.0)
    count = rho * 2.0 + 1.0
    assert child.count == pytest.approx(count)
    assert child.mean == pytest.approx(2.0 + (1.0 / count) * (3.0 - 2.0))


@pytest.mark.unit
def test_baseline_walks_the_fallback_chain() -> None:
    tracker = ForgettingTracker()
    assert tracker.baseline(node_key("x"), (TASK_KEY,)) is None

    tracker.observe(TASK_KEY, [1.5], version=0)
    assert tracker.baseline(node_key("x"), (TASK_KEY,)) == pytest.approx(1.5)

    tracker.observe(node_key("x"), [2.5], version=1)
    assert tracker.baseline(node_key("x"), (TASK_KEY,)) == pytest.approx(2.5)


@pytest.mark.unit
def test_state_round_trips_through_json() -> None:
    tracker = ForgettingTracker(TrackerSettings(half_life=5.0))
    tracker.observe(TASK_KEY, [0.2, 0.9], version=0)
    tracker.observe(node_key("a"), [1.0], version=2)

    restored = ForgettingTracker(TrackerSettings(half_life=5.0), json.loads(json.dumps(tracker.state_dict())))

    assert restored.state_dict() == tracker.state_dict()
    assert (
        restored.observe(node_key("a"), [0.0], version=3).mean == tracker.observe(node_key("a"), [0.0], version=3).mean
    )


@pytest.mark.unit
@pytest.mark.parametrize("bad_state", [{"k": [0.0, 1.0]}, {"k": [float("nan"), 1.0, 0]}, {"k": [0.0, -1.0, 0]}])
def test_invalid_state_is_rejected(bad_state) -> None:
    with pytest.raises(ValueError):
        ForgettingTracker(state=bad_state)


@pytest.mark.unit
def test_eviction_drops_the_oldest_keys_and_keeps_the_task() -> None:
    tracker = ForgettingTracker(TrackerSettings(max_keys=3))
    tracker.observe(TASK_KEY, [1.0], version=0)
    tracker.observe(node_key("old"), [1.0], version=1)
    tracker.observe(node_key("mid"), [1.0], version=2)
    tracker.observe(node_key("new"), [1.0], version=3)

    assert TASK_KEY in tracker
    assert node_key("old") not in tracker
    assert node_key("mid") in tracker and node_key("new") in tracker


@pytest.mark.unit
def test_observe_rejects_regressions_and_bad_rewards() -> None:
    tracker = ForgettingTracker()
    tracker.observe("q", [1.0], version=5)
    with pytest.raises(ValueError, match="later version"):
        tracker.observe("q", [1.0], version=4)
    with pytest.raises(ValueError):
        tracker.observe("q", [], version=6)
    with pytest.raises(ValueError):
        tracker.observe("q", [math.inf], version=6)


@pytest.mark.unit
def test_batch_normalize_centres_and_scales() -> None:
    normalized = batch_normalize([1.0, 2.0, 3.0, 4.0])

    assert math.fsum(normalized) == pytest.approx(0.0, abs=1e-12)
    assert math.sqrt(math.fsum(value**2 for value in normalized) / 4) == pytest.approx(1.0, rel=1e-5)
    assert batch_normalize([2.0, 2.0, 2.0]) == (0.0, 0.0, 0.0)
    assert batch_normalize([7.0]) == (0.0,)


@pytest.mark.unit
def test_advantages_use_the_pre_update_baseline() -> None:
    tracker = ForgettingTracker()
    tracker.observe(node_key("p"), [1.0, 1.0], version=0)  # mean 1.0 before this step
    observations = [Observation(node_key("p"), 3.0), Observation(node_key("p"), 1.0)]

    advantages, metrics = assign_advantages(tracker, observations, version=1, normalize=False)

    # 3 - 1 and 1 - 1 against the baseline as it stood, not against the
    # post-update mean the same rewards move it to.
    assert advantages == pytest.approx((2.0, 0.0))
    assert tracker.estimate(node_key("p")).mean > 1.0
    assert metrics["tracked_fraction"] == 1.0


@pytest.mark.unit
def test_first_step_falls_back_to_the_step_mean() -> None:
    tracker = ForgettingTracker()
    observations = [
        Observation(node_key("a"), 2.0, prior_key=None, fallbacks=(TASK_KEY,)),
        Observation(node_key("b"), 4.0, prior_key=None, fallbacks=(TASK_KEY,)),
    ]

    advantages, metrics = assign_advantages(tracker, observations, version=0, normalize=False)

    assert advantages == pytest.approx((-1.0, 1.0))
    assert metrics["tracked_fraction"] == 0.0
    assert node_key("a") in tracker and node_key("b") in tracker


@pytest.mark.unit
def test_siblings_update_their_key_once_per_step() -> None:
    settings = TrackerSettings()
    tracker = ForgettingTracker(settings)
    tracker.observe(node_key("p"), [1.0], version=0)
    observations = [Observation(node_key("p"), reward) for reward in (0.0, 2.0, 4.0)]

    assign_advantages(tracker, observations, version=1)

    # One discount for the step, then the batch of three, not three discounts.
    rho = settings.forgetting_factor(1.0)
    assert tracker.estimate(node_key("p")).count == pytest.approx(rho * 1.0 + 3.0)


@pytest.mark.unit
def test_normalized_advantages_are_centred_across_the_step() -> None:
    tracker = ForgettingTracker()
    observations = [Observation(TASK_KEY, reward) for reward in (2.62, 2.63, 0.0, 2.61)]

    advantages, _ = assign_advantages(tracker, observations, version=0)

    assert math.fsum(advantages) == pytest.approx(0.0, abs=1e-12)
    assert advantages[2] < min(advantages[0], advantages[1], advantages[3])


@pytest.mark.unit
def test_node_steps_keep_the_task_estimate_current() -> None:
    tracker = ForgettingTracker()
    observations = [
        Observation(node_key("a"), 1.0, fallbacks=(TASK_KEY,)),
        Observation(node_key("b"), 3.0, fallbacks=(TASK_KEY,)),
    ]

    assign_advantages(tracker, observations, version=0)

    assert tracker.estimate(TASK_KEY).mean == pytest.approx(2.0)
    assert tracker.estimate(TASK_KEY).count == pytest.approx(2.0)


@pytest.mark.unit
def test_task_steps_update_the_task_key_once() -> None:
    tracker = ForgettingTracker()

    assign_advantages(tracker, [Observation(TASK_KEY, 1.0), Observation(TASK_KEY, 3.0)], version=0)

    assert tracker.estimate(TASK_KEY).count == pytest.approx(2.0)


@pytest.mark.unit
def test_a_new_node_inherits_its_parents_start_of_step_estimate() -> None:
    tracker = ForgettingTracker()
    tracker.observe(node_key("p"), [2.0, 2.0], version=0)
    parent_first = [Observation(node_key("p"), 10.0), Observation(node_key("c"), 0.0, prior_key=node_key("p"))]
    child_first = list(reversed(parent_first))
    other = ForgettingTracker(state=tracker.state_dict())

    assign_advantages(tracker, parent_first, version=1)
    assign_advantages(other, child_first, version=1)

    # Both orders give the child the parent's pre-step estimate (mean 2, count 1).
    assert tracker.state_dict() == other.state_dict()
    rho = tracker.settings.forgetting_factor(1.0)
    assert tracker.estimate(node_key("c")).count == pytest.approx(rho * 1.0 + 1.0)
    assert tracker.estimate(node_key("c")).mean == pytest.approx(2.0 + (0.0 - 2.0) / (rho + 1.0))


@pytest.mark.unit
def test_adopt_refuses_to_overwrite_an_estimate() -> None:
    tracker = ForgettingTracker()
    tracker.observe("q", [1.0], version=0)
    with pytest.raises(ValueError, match="already has"):
        tracker.adopt("q", tracker.estimate("q"))


@pytest.mark.unit
def test_a_step_at_the_key_cap_keeps_what_it_adopted_and_updated() -> None:
    # Eviction runs once, after the step: a cold key adopted from its parent
    # this step must not be dropped (and reset) because the parent's old
    # version made it look stale, whatever order the keys are visited in.
    tracker = ForgettingTracker(TrackerSettings(max_keys=3, inherit_fraction=0.5))
    tracker.observe(TASK_KEY, [0.5], version=0)
    tracker.observe(node_key("a"), [0.9, 0.9], version=0)
    tracker.observe(node_key("b"), [0.1], version=1)
    observations = [
        Observation(node_key("d"), 1.0, prior_key=None, fallbacks=(TASK_KEY,)),
        Observation(node_key("c"), 1.0, prior_key=node_key("a"), fallbacks=(node_key("a"), TASK_KEY)),
    ]
    assign_advantages(tracker, observations, version=2, normalize=False)

    kept = tracker.state_dict()
    assert len(kept) == 3 and TASK_KEY in kept
    assert node_key("c") in kept and node_key("d") in kept, "both keys this step touched survive"
    mean, count, version = kept[node_key("c")]
    rho = tracker.settings.forgetting_factor(2.0)
    inherited = 2.0 * 0.5
    assert version == 2
    assert count == pytest.approx(rho * inherited + 1.0)
    assert mean == pytest.approx(0.9 + (1.0 / (rho * inherited + 1.0)) * (1.0 - 0.9))
