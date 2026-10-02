"""The opt-in repeat rule: a known score found again is trained on the parent's reward."""

from __future__ import annotations

import pytest

from recipes.tttd.examples.tttd.harness.agent import ReefTTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.search import ScoredSolution


def _harness(tolerance: float | None) -> ReefTTTDiscoverHarness:
    return ReefTTTDiscoverHarness(
        object(),
        lambda solution: ScoredSolution(solution, reward=0.0, value=0.0),
        "x",
        scenario="s",
        model="m",
        groups_per_step=1,
        rollouts_per_group=1,
        repeat_tolerance=tolerance,
    )


@pytest.mark.unit
def test_a_known_score_found_again_trains_on_the_parent_reward() -> None:
    harness = _harness(1e-9)
    parent = harness.archive.add_seed("```python\nparent\n```", 2.0)
    harness.archive.add_seed("```python\nbest\n```", 2.6)
    assert harness.training_score(2.6, parent) == 2.0  # the archive's best, found again
    assert harness.training_score(2.7, parent) == 2.7  # new: trained as is
    assert harness.training_score(1.5, parent) == 1.5  # no better than the parent: unchanged
    assert harness.training_score(0.0, parent) == 0.0  # invalid: unchanged


@pytest.mark.unit
def test_without_a_tolerance_every_score_trains_as_is() -> None:
    harness = _harness(None)
    parent = harness.archive.add_seed("```python\nparent\n```", 2.0)
    harness.archive.add_seed("```python\nbest\n```", 2.6)
    assert harness.training_score(2.6, parent) == 2.6
    with pytest.raises(ValueError):
        _harness(-1.0)
