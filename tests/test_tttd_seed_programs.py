"""Seed programs: a known program scored once and added to the search archive as one more root."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from recipes.tttd.examples.tttd.harness.run_controller import TTTDRunIdentity
from recipes.tttd.examples.tttd.harness.search import PUCTArchive, ScoredSolution, TTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.session import read_seed_program, seed_programs_from

KNOWN = "```python\ndef run():\n    return 5\n```"


def _scorer(solution: str) -> ScoredSolution:
    body = solution.removeprefix("```python\n").removesuffix("\n```")
    reward = float(body.rsplit("return", 1)[1]) if "return" in body else 0.0
    return ScoredSolution(solution=solution, reward=reward, value=reward, output=f"scored {reward}")


class _Model:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.prompts.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": "```python\ndef run():\n    return 6\n```"}}]}


def _harness(model: _Model, groups: int = 2) -> TTTDiscoverHarness:
    return TTTDiscoverHarness(
        model, _scorer, "Improve the number.", model="m", groups_per_step=groups, rollouts_per_group=1, max_workers=1
    )


@pytest.mark.unit
def test_a_seed_is_scored_once_and_searched_from_first() -> None:
    model = _Model()
    harness = _harness(model)
    seed = harness.add_known_seed(KNOWN)

    assert seed.seed and seed.reward == 5.0 and seed.output == "scored 5.0"
    # The empty roots stay, so a step still has enough independent parents.
    assert len(harness.archive.candidates) == 3
    harness.run_step(0)
    # PUCT ranks the scored seed first, so one of the step's prompts improves on it.
    assert any("reward=5.000000" in prompt and "return 5" in prompt for prompt in model.prompts)
    assert harness.archive.best().reward == 6.0


@pytest.mark.unit
def test_a_rejected_seed_raises_instead_of_entering_the_archive() -> None:
    harness = _harness(_Model())
    with pytest.raises(ValueError, match=r"scored 0\.0"):
        harness.add_known_seed("```python\ndef run():\n    pass\n```")
    assert len(harness.archive.candidates) == 2


@pytest.mark.unit
def test_a_seed_survives_the_saved_search_state() -> None:
    harness = _harness(_Model())
    harness.add_known_seed(KNOWN)
    restored = PUCTArchive()
    restored.load_state_dict(harness.archive.state_dict())
    seeds = [item for item in restored.candidates if item.solution]
    assert len(seeds) == 1 and seeds[0].seed and seeds[0].reward == 5.0


@pytest.mark.unit
def test_seed_paths_come_from_the_config_or_the_environment(tmp_path: Path) -> None:
    stack = {"search": {"seed_programs": ["a.py", "b.py"]}}
    assert seed_programs_from(stack, {}) == ("a.py", "b.py")
    assert seed_programs_from({"search": {"seed_programs": "a.py"}}, {}) == ("a.py",)
    assert seed_programs_from(stack, {"TTTD_SEED_PROGRAMS": "c.py:d.py"}) == ("c.py", "d.py")
    assert seed_programs_from({}, {}) == ()

    program = tmp_path / "best.py"
    program.write_text("def run():\n    return 5\n\n")
    assert read_seed_program(program) == KNOWN
    (tmp_path / "empty.py").write_text("\n")
    with pytest.raises(ValueError):
        read_seed_program(tmp_path / "empty.py")


@pytest.mark.unit
def test_seeds_enter_the_run_identity_only_when_set() -> None:
    fields: dict[str, Any] = {
        "scenario": "s",
        "model": "m",
        "recipe": "r",
        "inference_path": "p",
        "instruction_sha256": "i",
        "groups_per_step": 8,
        "rollouts_per_group": 64,
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "enable_thinking": True,
        "exploration": 1.0,
        "invalid_reward": 0.0,
    }
    assert "seed_programs" not in TTTDRunIdentity(**fields).as_dict()
    assert TTTDRunIdentity(**fields, seed_programs=("abc",)).as_dict()["seed_programs"] == ["abc"]
