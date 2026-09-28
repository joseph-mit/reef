"""Torch hooks for SPO-TTT: TTT-Discover's frozen-base KL on the shipped advantages."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from recipes.tttd.slime.objective import tttd_advantages
from reef.train.slime_backend.algorithm import objective


@objective("custom_advantage_function_path")
def spottt_advantages(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Centre the frozen-base KL onto the tracker's advantages, as tttd does.

    The reference implementation adds ``kl_coef * (mean_diff - diff_t)`` per
    token, where ``diff_t`` is the sampled-minus-base log-probability; reusing
    tttd's hook keeps that anchor identical across the compared recipes.
    """
    tttd_advantages(args, rollout_data)
