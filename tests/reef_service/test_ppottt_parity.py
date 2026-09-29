"""Pin the PPO-TTT tensor estimator to the pure-Python oracle.

The oracle in ``reference_algorithms/ppottt.py`` is the source of truth for the
estimator's numerics; ``recipes/ppottt/slime/objective.py`` is the tensor
transcription Slime reaches through ``--custom-advantage-function-path``. Both
run on the same inputs here. Like the SAO parity tests these need CPU torch and
a single-rank ``megatron.core.mpu`` shim, never Megatron itself.
"""

from __future__ import annotations

import random
import sys
import time
import types
from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")


def _install_single_rank_mpu() -> None:
    try:
        from megatron.core import mpu as real_mpu  # noqa: F401
    except ImportError:
        pass
    else:
        return
    core = types.ModuleType("megatron.core")
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    sys.modules["megatron.core"] = core


_install_single_rank_mpu()

from recipes.ppottt.slime.objective import get_terminal_reward_advantages_and_returns, ppottt_advantages

from .reference_algorithms import ppottt as reference


def _tensor_estimate(values, reward, penalty, gamma, lambd):
    values_t = torch.tensor(values, dtype=torch.float64)
    penalty_t = torch.tensor(penalty, dtype=torch.float64)
    return get_terminal_reward_advantages_and_returns(
        total_len=len(values) + 3,
        response_len=len(values),
        values=values_t,
        reward=reward,
        per_token_penalty=penalty_t,
        gamma=gamma,
        lambd=lambd,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "reward", "penalty", "gamma", "lambd"),
    [
        ([1.0, 2.0, 3.0], 5.0, [0.0, 0.0, 0.0], 1.0, 1.0),
        ([0.4, -0.2, 2.5, 1.0], 2.625, [-0.01, 0.02, 0.0, -0.03], 1.0, 1.0),
        ([1.0, 1.0, 1.0], 8.0, [0.0, 0.0, 0.0], 0.5, 0.5),
        ([0.5, -0.5, 2.0, -1.0, 0.0], -2.0, [0.1, 0.0, -0.1, 0.0, 0.05], 0.99, 0.95),
        ([3.0], 1.0, [-0.5], 0.9, 0.0),
    ],
)
def test_terminal_reward_gae_matches_reference(values, reward, penalty, gamma, lambd) -> None:
    adv_t, ret_t = _tensor_estimate(values, reward, penalty, gamma, lambd)

    ref_rewards = reference.terminal_rewards(len(values), reward, penalty)
    ref_adv, ref_ret = reference.gae(values, ref_rewards, gamma=gamma, lambd=lambd)

    assert adv_t.tolist() == pytest.approx(ref_adv, abs=1e-9)
    assert ret_t.tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
def test_terminal_reward_gae_matches_reference_on_random_inputs() -> None:
    rng = random.Random(20260928)
    for _ in range(200):
        length = rng.randint(1, 12)
        values = [rng.uniform(-3.0, 3.0) for _ in range(length)]
        penalty = [rng.uniform(-0.2, 0.2) for _ in range(length)]
        reward = rng.uniform(-5.0, 5.0)
        gamma = rng.choice([1.0, 0.99, 0.9, 0.5])
        lambd = rng.choice([1.0, 0.97, 0.95, 0.5, 0.0])

        adv_t, ret_t = _tensor_estimate(values, reward, penalty, gamma, lambd)
        ref_adv, ref_ret = reference.gae(values, reference.terminal_rewards(length, reward, penalty), gamma, lambd)

        assert adv_t.tolist() == pytest.approx(ref_adv, abs=1e-9)
        assert ret_t.tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("length", [255, 256, 257, 700, 3000])
@pytest.mark.parametrize(("gamma", "lambd"), [(1.0, 1.0), (0.99, 0.95), (0.9, 0.0), (1.0, 0.97)])
def test_long_responses_match_reference_across_scan_chunks(length, gamma, lambd) -> None:
    # Real responses run to ~30k tokens; the scan works in chunks, so lengths
    # at and around the chunk size, and several chunks long, must match too.
    rng = random.Random(length)
    values = [rng.uniform(-3.0, 3.0) for _ in range(length)]
    penalty = [rng.uniform(-0.2, 0.2) for _ in range(length)]
    reward = rng.uniform(-5.0, 5.0)

    adv_t, ret_t = _tensor_estimate(values, reward, penalty, gamma, lambd)
    ref_adv, ref_ret = reference.gae(values, reference.terminal_rewards(length, reward, penalty), gamma, lambd)

    assert adv_t.tolist() == pytest.approx(ref_adv, abs=1e-8)
    assert ret_t.tolist() == pytest.approx(ref_ret, abs=1e-8)


@pytest.mark.unit
def test_scan_takes_no_per_token_steps() -> None:
    from recipes.ppottt.slime.objective import discounted_reverse_cumsum

    deltas = torch.linspace(-1.0, 1.0, 30_000, dtype=torch.float64)
    started = time.perf_counter()
    for discount in (1.0, 0.95):
        discounted_reverse_cumsum(deltas, discount)
    assert time.perf_counter() - started < 1.0


@pytest.mark.unit
def test_one_step_episode_reduces_to_reward_minus_value() -> None:
    # The discovery setting: gamma = lambd = 1, no penalty, one attempt per rollout.
    values = [0.7, 2.1, -0.3, 2.6]
    adv_t, ret_t = _tensor_estimate(values, 2.625, [0.0] * 4, 1.0, 1.0)

    assert adv_t.tolist() == pytest.approx(reference.one_step_advantages(values, 2.625), abs=1e-12)
    assert ret_t.tolist() == pytest.approx([2.625] * 4, abs=1e-12)


@pytest.mark.unit
def test_advantages_are_detached_and_returns_keep_the_value_graph() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    penalty = torch.zeros(3)

    adv, ret = get_terminal_reward_advantages_and_returns(3, 3, values, 5.0, penalty, 1.0, 1.0)

    assert not adv.requires_grad
    assert ret.requires_grad


@pytest.mark.unit
def test_penalty_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="penalty must align"):
        get_terminal_reward_advantages_and_returns(3, 3, torch.zeros(3), 1.0, torch.zeros(2), 1.0, 1.0)


def _rollout_data(values, rewards, kls=None):
    data = {
        "values": [torch.tensor(v, dtype=torch.float64) for v in values],
        "rewards": list(rewards),
        "response_lengths": [len(v) for v in values],
        "total_lengths": [len(v) + 2 for v in values],
    }
    if kls is not None:
        data["kl"] = [torch.tensor(k, dtype=torch.float64) for k in kls]
    return data


@pytest.mark.unit
def test_hook_folds_the_kl_penalty_into_per_token_rewards() -> None:
    values = [[0.5, 0.5, 0.5], [1.0, 2.0]]
    kls = [[0.1, 0.2, 0.3], [0.0, 0.4]]
    data = _rollout_data(values, [3.0, 1.0], kls)

    ppottt_advantages(Namespace(kl_coef=0.1, gamma=1.0, lambd=1.0), data)

    for index, sample_values in enumerate(values):
        penalty = [-0.1 * k for k in kls[index]]
        ref_adv, ref_ret = reference.gae(
            sample_values, reference.terminal_rewards(len(sample_values), data["rewards"][index], penalty)
        )
        assert data["advantages"][index].tolist() == pytest.approx(ref_adv, abs=1e-9)
        assert data["returns"][index].tolist() == pytest.approx(ref_ret, abs=1e-9)


@pytest.mark.unit
def test_hook_ignores_kl_when_the_coefficient_is_zero() -> None:
    values = [[0.5, 0.5, 0.5]]
    data = _rollout_data(values, [3.0], kls=[[9.0, 9.0, 9.0]])

    ppottt_advantages(Namespace(kl_coef=0.0, gamma=1.0, lambd=1.0), data)

    assert data["advantages"][0].tolist() == pytest.approx([2.5, 2.5, 2.5], abs=1e-12)


@pytest.mark.unit
def test_hook_without_values_raises() -> None:
    data = _rollout_data([[1.0]], [1.0])
    data.pop("values")

    with pytest.raises(ValueError, match="critic values"):
        ppottt_advantages(Namespace(kl_coef=0.1, gamma=1.0, lambd=1.0), data)
