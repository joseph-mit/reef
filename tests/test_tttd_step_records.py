"""The harness's per-step attempt files and the summary numbers in each step's event."""

from __future__ import annotations

import pytest

from recipes.tttd.examples.tttd.harness.run_controller import TTTDRunController, TTTDRunIdentity, TTTDRunStateStore
from recipes.tttd.examples.tttd.harness.search import RolloutResult, ScoredSolution, TTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.step_records import StepRecorder, failure_kind, read_step

# Four attempts per step: a program the judge accepts, one it rejects, one it
# accepts with a higher score, and a response without code.
_RESPONSES = [
    "<think>plan</think>```python\nprint(1)\n```",
    "```python\nprint(2)\n```",
    "```python\nprint(3)\n```",
    "no code here",
]
_SCORES = {"print(1)": (1.5, "sum of radii = 1.5"), "print(2)": (0.0, "packing is not valid"), "print(3)": (2.5, "ok")}


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, payload):
        text = _RESPONSES[self.calls % len(_RESPONSES)]
        self.calls += 1
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "length" if "no code" in text else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100 + self.calls},
        }


def _score(solution: str) -> ScoredSolution:
    body = solution.split("\n")[1]
    reward, reason = _SCORES[body]
    return ScoredSolution(solution, reward, reward, output=reason)


def _identity() -> TTTDRunIdentity:
    return TTTDRunIdentity(
        scenario="smoke",
        model="qwen",
        recipe="search-only",
        inference_path="/v1/chat/completions",
        instruction_sha256="hash",
        groups_per_step=2,
        rollouts_per_group=2,
        max_new_tokens=64,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        enable_thinking=True,
        exploration=1.0,
        invalid_reward=0.0,
    )


@pytest.mark.unit
def test_every_attempt_is_written_and_summarised(tmp_path) -> None:
    harness = TTTDiscoverHarness(
        _Model(), _score, "pack circles", model="qwen", groups_per_step=2, rollouts_per_group=2
    )
    recorder = StepRecorder(tmp_path / "attempts")
    events: list[dict] = []
    controller = TTTDRunController(
        harness,
        status_reader=None,
        state_store=TTTDRunStateStore(tmp_path / "state.json", _identity()),
        emit=events.append,
        wait_for_training=False,
        step_records=recorder,
    )

    controller.run(2)

    header, attempts = read_step(tmp_path / "attempts" / "step-0001.jsonl.gz")
    assert header["step"] == 1 and len(header["selection"]) == 2 and header["best_before"] == 0.0
    assert header["archive_best_reward"] == 2.5
    assert [(attempt["group"], attempt["rollout"]) for attempt in attempts] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    first, rejected, best, empty = attempts
    assert first["think_chars"] == len("<think>plan") and first["completion_tokens"] == 101
    assert first["beat_parent"] and first["improvement"] == 1.5 and first["generation_seconds"] >= 0
    assert rejected["failure"] == "packing is not valid" and not rejected["valid"]
    assert best["beat_best"] and best["solution"].startswith("```python")
    assert empty["failure"] == "no code block" and empty["finish_reason"] == "length"

    committed = [event for event in events if event["event"] == "tttd_step_committed"]
    first_step = committed[0]
    assert first_step["valid_fraction"] == 0.5 and first_step["valid_mean"] == 2.0
    assert first_step["beat_best_count"] == 2 and first_step["truncated_fraction"] == 0.25
    assert first_step["failures"] == {"packing is not valid": 1, "no code block": 1}
    assert first_step["parent_seed_count"] == 2 and first_step["search_seconds"] >= 0
    # The second step starts from the programs the first one archived.
    second, _ = read_step(tmp_path / "attempts" / "step-0002.jsonl.gz")
    assert second["best_before"] == 2.5 and second["summary"]["parent_reward_mean"] > 0


@pytest.mark.unit
def test_a_recording_failure_is_reported_and_the_run_goes_on(tmp_path) -> None:
    class _Broken:
        def record(self, *args, **kwargs):
            raise OSError("disk full")

    harness = TTTDiscoverHarness(
        _Model(), _score, "pack circles", model="qwen", groups_per_step=2, rollouts_per_group=2
    )
    events: list[dict] = []
    controller = TTTDRunController(
        harness,
        status_reader=None,
        state_store=TTTDRunStateStore(tmp_path / "state.json", _identity()),
        emit=events.append,
        wait_for_training=False,
        step_records=_Broken(),
    )

    controller.run(1)

    assert [event["event"] for event in events] == [
        "tttd_step_started",
        "tttd_step_record_failed",
        "tttd_step_committed",
    ]
    assert events[1]["error"] == "OSError: disk full"


@pytest.mark.unit
def test_failure_kinds_group_numbers_and_exceptions() -> None:
    def result(reward=0.0, error=None, output=""):
        return RolloutResult("p", "", reward, "", error=error, output=output)

    assert failure_kind(result(output="program exceeded 1000s"), 0.0) == "program exceeded Ns"
    assert failure_kind(result(error="HTTPError: 502"), 0.0) == "HTTPError"
    assert failure_kind(result(reward=2.0), 0.0) == ""
