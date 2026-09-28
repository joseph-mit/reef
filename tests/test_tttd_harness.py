from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.tttd.examples.tttd.harness.agent import ReefTTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.sandbox import execute_program
from recipes.tttd.examples.tttd.harness.search import (
    Candidate,
    PUCTArchive,
    ScoredSolution,
    TTTDChatRequestBuilder,
    TTTDiscoverHarness,
    build_prompt,
    extract_solution,
)


def test_puct_records_top_children_and_backpropagates_visits():
    archive = PUCTArchive(max_size=10, top_children=2)
    root = archive.add_seed("root", 1.0)
    children = archive.record_expansion(root.candidate_id, [("low", 2.0, 2.0), ("best", 4.0, 4.0), ("mid", 3.0, 3.0)])

    assert [child.reward for child in children] == [4.0, 3.0]
    assert root.best_child_reward == 4.0
    assert root.q == 4.0
    assert root.visits == 3
    assert archive.total_expansions == 3

    archive.record_expansion(children[0].candidate_id, [("grandchild", 5.0, 5.0)])
    assert root.visits == 4
    assert children[0].visits == 1


def test_puct_archive_snapshot_round_trip_preserves_search_state():
    archive = PUCTArchive(max_size=10, top_children=2)
    root = archive.add_seed("seed-solution", 1.0)
    archive.record_expansion(
        root.candidate_id,
        [
            ("first", 3.0, 3.0),
            ("second", 2.0, 2.0),
        ],
        failed_rollouts=1,
    )
    snapshot = archive.state_dict()

    restored = PUCTArchive()
    restored.load_state_dict(snapshot)

    assert restored.state_dict() == snapshot
    assert restored.scores() == archive.scores()
    assert restored.best().solution == "first"


def test_puct_pruning_always_retains_seeds():
    archive = PUCTArchive(max_size=2)
    seed = archive.add_seed("seed", -10.0)
    archive.record_expansion(seed.candidate_id, [("one", 1.0, 1.0), ("two", 2.0, 2.0)])

    assert seed.candidate_id in {candidate.candidate_id for candidate in archive.candidates}
    assert len(archive.candidates) == 2
    assert archive.best().solution == "two"


def test_parent_prompt_reuses_the_normalized_code_block_without_nesting():
    solution = "```python\nprint('parent')\n```"
    parent = Candidate("state-1", solution, 2.5, 2.5, output="finished")

    prompt = build_prompt("Solve the task.", parent)[0]["content"]

    assert "## Selected parent (reward=2.500000)" in prompt
    assert prompt.count("```python") == 1
    assert "```\n```python" not in prompt
    assert "--- Previous Program Output ---\nfinished\n--- End Output ---" in prompt


def test_solution_extraction_matches_official_unclosed_final_block_behavior():
    response = "Reasoning first.\n```python\nprint('complete even without a closing fence')"

    assert extract_solution(response) == "```python\nprint('complete even without a closing fence')\n```"


@pytest.mark.parametrize("task", ["circle_packing_26", "circle_packing_32"])
def test_packing_instruction_does_not_claim_the_dynamic_parent_is_empty(task):
    instruction = (REPO_ROOT / f"recipes/tttd/examples/tttd/harbor/{task}/instruction.md").read_text()

    assert "No previous code available" not in instruction
    assert "Current sum of radii (higher is better): 0.000000" not in instruction


def test_erdos_instruction_and_judge_use_the_same_program_budget():
    task = REPO_ROOT / "recipes/tttd/examples/tttd/harbor/erdos_min_overlap"
    instruction = (task / "instruction.md").read_text()
    scorer = (task / "environment/score.py").read_text()

    assert "budget_s=1000" in instruction
    assert "budget_s=1000" in scorer
    assert "budget_s=900" not in scorer


def _erdos_reward(result):
    _h_values, c5_bound, n_points = result
    c5_bound = float(c5_bound)
    if c5_bound <= 0 or math.isnan(c5_bound) or math.isinf(c5_bound):
        raise ValueError("C5 bound must be positive and finite")
    reward = 1.0 / (1e-8 + c5_bound)
    return reward, -c5_bound, {"c5_bound": c5_bound, "n_points": int(n_points)}


def test_erdos_scorer_scores_valid_construction():
    from recipes.tttd.examples.tttd.harness.scorer import ProgramScorer

    scorer = ProgramScorer(_erdos_reward, eval_timeout_s=30, num_cpus=1, prelude="import numpy as np")

    values = [0.4] * 20 + [0.6] * 20
    solution = (
        "```python\n"
        "def run(seed=42, budget_s=1000, **kwargs):\n"
        f"    h_values = np.array({values!r})\n"
        "    dx = 2.0 / len(h_values)\n"
        "    c5_bound = float(np.max(np.correlate(h_values, 1-h_values, mode='full') * dx))\n"
        "    return h_values, c5_bound, len(h_values)\n"
        "```"
    )
    scored = scorer(solution)

    assert scored.reward == pytest.approx(1.0 / (1e-8 + scored.metrics["c5_bound"]))
    assert scored.value == pytest.approx(-scored.metrics["c5_bound"])
    assert scored.metrics["n_points"] == 40


def test_erdos_scorer_rejects_invalid_solution():
    from recipes.tttd.examples.tttd.harness.scorer import ProgramScorer

    scorer = ProgramScorer(_erdos_reward, eval_timeout_s=30)
    with pytest.raises(ValueError, match="cannot extract Python code"):
        scorer("no code here")


def _scorer():
    """A simple scorer for harness tests: extracts a number from a code block."""

    def score(solution: str) -> ScoredSolution:
        match = re.search(r"```python\s+(\d+)\s*```", solution)
        if not match:
            raise ValueError("no number in code block")
        value = int(match.group(1))
        return ScoredSolution(solution=solution, reward=float(value), value=float(value))

    return score


class _Model:
    def __init__(self):
        self.index = 0
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        self.index += 1
        content = "bad" if self.index == 2 else f"```python\n{self.index}\n```"
        return {"choices": [{"message": {"content": content}}]}


def test_ordinary_harness_calls_model_evaluates_and_archives():
    model = _Model()
    harness = TTTDiscoverHarness(
        model,
        _scorer(),
        "Improve the number.",
        model="test-model",
        groups_per_step=1,
        rollouts_per_group=3,
        max_workers=1,
    )

    results = harness.run_step(0)

    assert [result.reward for result in results] == [1.0, 0.0, 3.0]
    assert results[1].solution == ""
    assert len(model.payloads) == 3
    assert model.payloads[0]["model"] == "test-model"
    assert harness.archive.best().reward == 3.0


class _Client:
    def __init__(self):
        self.index = 0
        self.reports = []

    def inference_with_record(self, scenario, path, payload, *, recipe=None, extra_headers=None):
        assert (scenario, path, recipe, extra_headers) == (
            "discovery",
            "/v1/chat/completions",
            None,
            {"x-reef-release-id": "checkpoint-v1"},
        )
        self.index += 1
        content = "bad" if self.index == 2 else f"```python\n{self.index}\n```"
        return {"choices": [{"message": {"content": content}}]}, f"agent-record-{self.index}"

    def report(self, scenario, payload, *, recipe=None, extra_headers=None):
        self.reports.append((scenario, payload, recipe, extra_headers))
        return {}


def test_reef_harness_reports_each_result_against_exact_inference():
    client = _Client()
    harness = ReefTTTDiscoverHarness(
        client,
        _scorer(),
        "Improve the number.",
        scenario="discovery",
        release_id="checkpoint-v1",
        model="reef",
        groups_per_step=1,
        rollouts_per_group=3,
        max_workers=1,
    )

    results = harness.run_step(2)

    assert [result.reward for result in results] == [1.0, 0.0, 3.0]
    assert [report[1]["references"] for report in client.reports] == [
        ["agent-record-1"],
        ["agent-record-2"],
        ["agent-record-3"],
    ]
    assert {report[1]["metadata"]["comparison_set"] for report in client.reports} == {"tttd-step-2-group-0"}
    assert [report[1]["metadata"]["rollout"] for report in client.reports] == [0, 1, 2]
    assert all(report[1]["metadata"]["groups_per_step"] == 1 for report in client.reports)
    assert all(report[1]["metadata"]["rollouts_per_group"] == 3 for report in client.reports)
    assert client.reports[1][1]["feedback"] == "ValueError: response does not contain a Python code block"


def test_tttd_chat_request_builder_leaves_token_capture_to_backend():
    builder = TTTDChatRequestBuilder(max_new_tokens=64)

    payload = builder("qwen", [{"role": "user", "content": "discover"}], {})

    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "discover"}]
    assert payload["max_completion_tokens"] == 64
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == -1
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    # Reef's weight surface names the served adapter; the harness never does.
    assert "lora_path" not in payload


def test_tttd_chat_request_builder_rejects_adapter_override():
    builder = TTTDChatRequestBuilder()

    with pytest.raises(ValueError, match="weight surface"):
        builder("model", [{"role": "user", "content": "hi"}], {"lora_path": None})


def test_sandbox_environment_excludes_operator_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "leak-me")
    source = "import os\ndef entrypoint():\n    return {'token': os.environ.get('HF_TOKEN'), 'path': bool(os.environ.get('PATH'))}\n"
    result = execute_program(source, entrypoint="entrypoint", timeout_s=30, max_cpus=1)
    assert result.result["token"] is None
    assert result.result["path"] is True


def test_reef_harness_tags_reports_for_the_scenarios_recipe_and_runs_single_attempt_groups():
    client = _Client()
    harness = ReefTTTDiscoverHarness(
        client,
        _scorer(),
        "Improve the number.",
        scenario="discovery",
        release_id="checkpoint-v1",
        model="reef",
        groups_per_step=1,
        rollouts_per_group=1,
        max_workers=1,
        algorithm="ppottt",
    )

    results = harness.run_step(0)

    assert len(results) == 1
    assert [report[1]["metadata"]["algorithm"] for report in client.reports] == ["ppottt"]
    assert client.reports[0][1]["metadata"]["rollouts_per_group"] == 1


def test_harness_still_rejects_empty_grids():
    for groups, rollouts in ((0, 1), (1, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            TTTDiscoverHarness(
                lambda payload: payload,
                _scorer(),
                "x",
                model="m",
                groups_per_step=groups,
                rollouts_per_group=rollouts,
            )


def test_methods_keep_tttd_paths_and_separate_every_other_method(tmp_path):
    from recipes.tttd.examples.tttd.harness.methods import METHODS, method_named, scenario_name, state_dir

    tttd = method_named("tttd")
    assert scenario_name(tttd, "erdos_min_overlap") == "tttd-erdos-min-overlap"
    assert state_dir(tmp_path, tttd, "erdos_min_overlap") == tmp_path / "work" / "erdos_min_overlap"

    ppottt = method_named("ppottt")
    assert scenario_name(ppottt, "erdos_min_overlap") == "ppottt-erdos-min-overlap"
    assert state_dir(tmp_path, ppottt, "erdos_min_overlap") == tmp_path / "work" / "ppottt" / "erdos_min_overlap"
    assert len({scenario_name(method, "t") for method in METHODS.values()}) == len(METHODS)

    with pytest.raises(ValueError, match="unknown TTTD_METHOD"):
        method_named("grpo")


def test_entrypoints_derive_the_same_state_root_as_the_harness(monkeypatch, tmp_path):
    import runpy
    import types

    from recipes.tttd.examples.tttd.harness.methods import method_named, state_dir

    example_dir = REPO_ROOT / "recipes" / "tttd" / "examples" / "tttd"
    run_script = (example_dir / "run.sh").read_text()
    assert 'TTTD_STATE_DIR="$PWD/work/$TTTD_TASK"' in run_script
    assert 'TTTD_STATE_DIR="$PWD/work/$TTTD_METHOD/$TTTD_TASK"' in run_script

    monkeypatch.delenv("TTTD_TASK", raising=False)
    monkeypatch.setitem(sys.modules, "reef_eval", types.SimpleNamespace(Lab=None))
    monkeypatch.setattr("asyncio.run", lambda coroutine: coroutine.close())
    for name in ("tttd", "ppottt"):
        monkeypatch.setenv("TTTD_METHOD", name)
        namespace = runpy.run_path(str(example_dir / "run.py"))
        assert namespace["STATE_DIR"] == state_dir(example_dir, method_named(name), "erdos_min_overlap")


@pytest.mark.parametrize(
    "method_name", ["tttd", "tttd-mean", "ppottt", "ppottt-smoke", "spottt", "spottt-smoke", "search-only"]
)
def test_every_method_config_exists_and_its_grid_matches_the_driver_batch(method_name):
    import yaml

    from recipes.tttd.examples.tttd.harness.methods import method_named

    method = method_named(method_name)
    example_dir = REPO_ROOT / "recipes" / "tttd" / "examples" / "tttd"
    config = yaml.safe_load((example_dir / method.config).read_text())
    run_script = (example_dir / "run.sh").read_text()
    driver = config["services"][0]["command"]
    global_batch = int(re.search(r"--global-batch-size=(\d+)", driver).group(1))
    grid = config["reef"]["groups_per_step"] * config["reef"]["rollouts_per_group"]

    # Reef trains only once the whole grid has reported, as one Slime batch.
    assert grid == global_batch
    assert f"config={method.config}" in run_script
    minibatch = config["reef"].get("minibatch_size", 0)
    if minibatch:
        assert grid % minibatch == 0


def test_reef_harness_sends_lineage_and_can_opt_out_of_training():
    client = _Client()
    harness = ReefTTTDiscoverHarness(
        client,
        _scorer(),
        "Improve the number.",
        scenario="discovery",
        release_id="checkpoint-v1",
        model="reef",
        groups_per_step=1,
        rollouts_per_group=1,
        max_workers=1,
        train=False,
    )
    seed = harness.archive.candidates[0]

    harness.run_step(0)

    metadata = client.reports[0][1]["metadata"]
    assert metadata["parent_id"] == seed.candidate_id
    assert metadata["grandparent_id"] == ""
    assert metadata["training"] == {"eligible": False}


def test_trained_reports_carry_no_training_opt_out():
    client = _Client()
    harness = ReefTTTDiscoverHarness(
        client,
        _scorer(),
        "x",
        scenario="discovery",
        release_id="checkpoint-v1",
        model="reef",
        groups_per_step=1,
        rollouts_per_group=1,
        max_workers=1,
    )

    harness.run_step(0)

    assert "training" not in client.reports[0][1]["metadata"]


def test_only_search_only_skips_training():
    from recipes.tttd.examples.tttd.harness.methods import METHODS

    assert [name for name, method in METHODS.items() if not method.train] == ["search-only"]
