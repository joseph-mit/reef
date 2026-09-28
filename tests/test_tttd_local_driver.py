"""The Harbor-free TTT-Discover driver, the smoke checker, and the shared run session.

``run_local.py`` runs against the task's real judge here, with a fake Reef
that commits one training step per complete grid through Reef's own
``CommitLog``, so the checker reads exactly the file format a real run writes.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import logging
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from reef.core import ArtifactRef
from reef.scenario.state import CommitRecord
from reef.storage.commit_log import CommitLog

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "recipes" / "tttd" / "examples" / "tttd"
TASK = "circle_packing_26"
GRID_PACKING = """import numpy as np

def run_packing():
    centers = [(0.1 + 0.16 * (i % 6), 0.1 + 0.16 * (i // 6)) for i in range(26)]
    radii = [0.05] * 26
    return np.array(centers), np.array(radii), sum(radii)
"""
_EXAMPLE_MODULES = ("harness", "run_local", "check_smoke")


def _is_example_module(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _EXAMPLE_MODULES)


@pytest.fixture
def example(monkeypatch):
    """Import run_local and check_smoke as scripts do: the example directory first on sys.path."""
    for name in [name for name in sys.modules if _is_example_module(name)]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(EXAMPLE))
    run_local = importlib.import_module("run_local")
    spec = importlib.util.spec_from_file_location("check_smoke", EXAMPLE / "apptainer" / "check_smoke.py")
    check_smoke = importlib.util.module_from_spec(spec)
    sys.modules["check_smoke"] = check_smoke
    spec.loader.exec_module(check_smoke)
    yield SimpleNamespace(run_local=run_local, check_smoke=check_smoke, local_judge=run_local.local_judge)
    for name in [name for name in sys.modules if _is_example_module(name)]:
        del sys.modules[name]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _FakeReef:
    """One Reef session: answers every inference with a packing program, commits each full grid."""

    def __init__(self, state: Path, scenario: str, step_size: int, *, session: str, committed: int = 0) -> None:
        self.state = state
        self.scenario = scenario
        self.step_size = step_size
        self.session = session
        self.committed = committed
        self.pending_reports = 0
        self.inferences = 0
        (state / "agent-record").mkdir(parents=True, exist_ok=True)
        self.log = CommitLog(state / "agent-record" / "scenario.commits.jsonl")

    # ReefClient
    def inference_with_record(self, scenario, path, payload, **_):
        assert scenario == self.scenario and path == "/v1/chat/completions"
        self.inferences += 1
        content = f"```python\n{GRID_PACKING}```"
        return {"choices": [{"message": {"content": content}}]}, f"{self.session}-record-{self.inferences}"

    def report(self, scenario, payload, **_):
        assert scenario == self.scenario and payload["references"]
        self.pending_reports += 1
        if self.pending_reports == self.step_size:
            self.pending_reports = 0
            self._commit()
        return {}

    def _commit(self) -> None:
        self.committed += 1
        step = self.committed
        state = {"steps": step, "spottt": {"task": [1.3, float(step), step - 1]}}
        self.log.append(
            CommitRecord(
                scenario=self.scenario,
                step=step,
                artifact_ref=ArtifactRef(f"artifact-{step}", f"checkpoint-{step}", None),
                checkpoint=True,
                algorithm_state=state,
                high_water_sequence=step * self.step_size,
                high_water_offset=step * self.step_size,
                metrics={"train/ppottt/critic_updates": 2, "train/ppottt/actor_trained": 1, "train/pg_clipfrac": 0.0},
            )
        )
        megatron = self.state / "checkpoints" / "megatron"
        megatron.mkdir(parents=True, exist_ok=True)
        (megatron / "latest_checkpointed_iteration.txt").write_text(str(step))
        critic = self.state / "checkpoints" / "critic" / f"iter_{step:07d}"
        critic.mkdir(parents=True, exist_ok=True)

    # ReefTrainingStatusClient
    def scenario_status(self, scenario):
        from recipes.tttd.examples.tttd.harness.run_controller import ScenarioTrainingStatus

        assert scenario == self.scenario
        return ScenarioTrainingStatus(self.committed, f"{self.session}:{self.committed}", False)


def _run_phase(example, monkeypatch, reef: _FakeReef, **env: str) -> None:
    run_local = example.run_local
    monkeypatch.setattr(
        run_local, "local_judge", lambda task_dir, data_dir: example.local_judge(task_dir, data_dir, port=_free_port())
    )
    monkeypatch.setattr(run_local, "ReefClient", lambda *args, **kwargs: reef)
    monkeypatch.setattr(run_local, "ReefTrainingStatusClient", lambda *args, **kwargs: reef)
    monkeypatch.setattr(run_local, "scenario_status", lambda scenario: {"scenario_step": reef.committed})
    monkeypatch.delenv("TTTD_RUN_STEPS", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    run_local.main()


@pytest.mark.unit
@pytest.mark.parametrize("method", ["ppottt-smoke", "spottt-smoke"])
def test_local_driver_resumes_after_a_restart_and_the_checker_passes_every_stage(
    example, monkeypatch, tmp_path, method
) -> None:
    state = tmp_path / method / TASK
    scenario = f"{method}-circle-packing-26"
    monkeypatch.setenv("TTTD_TASK", TASK)
    monkeypatch.setenv("TTTD_METHOD", method)
    monkeypatch.setenv("TTTD_STATE_DIR", str(state))

    first = _FakeReef(state, scenario, 4, session="session-a")
    _run_phase(example, monkeypatch, first, TTTD_RUN_STEPS="1")
    # The crash: a new Reef session restores the committed step.
    second = _FakeReef(state, scenario, 4, session="session-b", committed=first.committed)
    _run_phase(example, monkeypatch, second)

    summaries = [json.loads(line) for line in (state / "run-summaries.jsonl").read_text().splitlines()]
    assert [(item["start_step"], item["next_step"]) for item in summaries] == [(0, 1), (1, 2)]
    assert [item["judge_submissions"] for item in summaries] == [4, 4]
    assert summaries[-1]["best_reward"] == pytest.approx(1.3)
    assert "import numpy as np" in (state / "best_solution.py").read_text()

    stages = example.check_smoke.check(method, TASK, state)
    assert {stage.name: stage.result for stage in stages} == {
        "rollout": "passed",
        "evaluation": "passed",
        "training update": "passed",
        "checkpoint save": "passed",
        "resume after SIGKILL": "passed",
    }
    report = example.check_smoke.render(method, TASK, stages)
    assert "rebound" in report
    if method == "ppottt-smoke":
        assert "2 critic update(s), actor trained = 1" in report


@pytest.mark.unit
def test_checker_fails_a_run_that_did_not_resume(example, tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "run-summaries.jsonl").write_text(
        json.dumps({"start_step": 0, "next_step": 1, "rollouts": 4, "judge_submissions": 4}) + "\n"
    )

    stages = example.check_smoke.check("ppottt-smoke", TASK, state)
    results = {stage.name: stage.result for stage in stages}

    assert results["resume after SIGKILL"] == "FAILED"
    assert results["checkpoint save"] == "FAILED"
    assert example.check_smoke.main(["--method", "ppottt-smoke", "--task", TASK, "--state", str(state)]) == 1


@pytest.mark.unit
def test_checker_lists_unreported_metrics_without_failing(example) -> None:
    assert example.check_smoke._metric({"train/ppottt/actor_trained": 1}, "ppottt/actor_trained") == 1
    assert example.check_smoke._metric({"other/actor_trained": 1}, "ppottt/actor_trained") is None


@pytest.mark.unit
def test_local_judge_listens_on_loopback_only_and_runs_the_tasks_scorer(example, tmp_path) -> None:
    port = _free_port()
    with example.run_local.local_judge(EXAMPLE / "harbor" / TASK, tmp_path, port=port) as url:
        request = urllib.request.Request(url + "/submit", data=GRID_PACKING.encode(), method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            assert json.loads(response.read())["score"] == pytest.approx(1.3)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
        if not address.startswith("127."):
            with pytest.raises(urllib.error.URLError):
                urllib.request.urlopen(f"http://{address}:{port}/health", timeout=2)
    assert example.run_local._port_is_free(port)


@pytest.mark.unit
def test_judge_programs_do_not_inherit_the_operators_secrets(example, tmp_path) -> None:
    environment = example.run_local.judge_environment(
        {"PATH": "/usr/bin", "HOME": "/home/me", "WANDB_API_KEY": "secret", "HF_TOKEN": "secret"},
        judge_dir=tmp_path / "environment",
        data_dir=tmp_path / "data",
    )

    assert environment == {
        "PATH": "/usr/bin",
        "HOME": "/home/me",
        "JUDGE_DIR": str(tmp_path / "environment"),
        "DATA_DIR": str(tmp_path / "data"),
    }


@pytest.mark.unit
def test_run_steps_override_stays_within_the_config(example) -> None:
    assert example.run_local.run_steps_from({}, 2) == 2
    assert example.run_local.run_steps_from({"TTTD_RUN_STEPS": "1"}, 2) == 1
    for bad in ("0", "3"):
        with pytest.raises(ValueError, match="TTTD_RUN_STEPS"):
            example.run_local.run_steps_from({"TTTD_RUN_STEPS": bad}, 2)


@pytest.mark.unit
def test_local_judge_refuses_a_busy_port(example, tmp_path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        with (
            pytest.raises(RuntimeError, match="already in use"),
            example.run_local.local_judge(EXAMPLE / "harbor" / TASK, tmp_path, port=port),
        ):
            pass


@pytest.mark.unit
def test_stack_settings_read_the_smoke_grid() -> None:
    from recipes.tttd.examples.tttd.harness.session import StackSettings

    settings = StackSettings.load(EXAMPLE / "serve.ppottt-smoke.yaml")

    assert settings == StackSettings(
        groups_per_step=2, rollouts_per_group=2, steps=2, max_new_tokens=4096, enable_thinking=False
    )
    assert StackSettings.load(EXAMPLE / "serve.yaml").enable_thinking is True


def _harbor_agent_module(monkeypatch):
    modules = {
        "harbor": ModuleType("harbor"),
        "harbor.agents": ModuleType("harbor.agents"),
        "harbor.agents.base": ModuleType("harbor.agents.base"),
        "harbor.environments": ModuleType("harbor.environments"),
        "harbor.environments.base": ModuleType("harbor.environments.base"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.agent": ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.base"].BaseAgent = type("BaseAgent", (), {})
    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.models.agent.context"].AgentContext = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "recipes.tttd.examples.tttd.harness.harbor_agent", raising=False)
    return importlib.import_module("recipes.tttd.examples.tttd.harness.harbor_agent")


@pytest.mark.unit
def test_harbor_agent_builds_the_shared_run_and_writes_the_best_program(monkeypatch, tmp_path) -> None:
    from recipes.tttd.examples.tttd.harness.methods import method_named
    from recipes.tttd.examples.tttd.harness.search import ScoredSolution
    from recipes.tttd.examples.tttd.harness.session import StackSettings

    agent_module = _harbor_agent_module(monkeypatch)
    reef = _FakeReef(tmp_path, "ppottt-smoke-circle-packing-26", 4, session="harbor")
    monkeypatch.setattr(agent_module, "METHOD", method_named("ppottt-smoke"))
    monkeypatch.setattr(agent_module, "TASK", TASK)
    monkeypatch.setattr(agent_module, "SCENARIO", reef.scenario)
    monkeypatch.setattr(agent_module, "SETTINGS", StackSettings(2, 2, 1, 64, False))
    monkeypatch.setattr(agent_module, "SEARCH_STATE_PATH", tmp_path / "search-state.json")
    monkeypatch.setattr(agent_module, "ReefTrainingStatusClient", lambda *args, **kwargs: reef)
    monkeypatch.setattr(
        agent_module, "JudgeScorer", lambda url: lambda solution: ScoredSolution(solution, reward=1.3, value=1.3)
    )

    class _Environment:
        commands: list[str] = []

        async def exec(self, command):
            self.commands.append(command)
            return SimpleNamespace(return_code=0, stderr="")

    agent = object.__new__(agent_module.HarborAgent)
    agent._client = reef
    agent.model_name = "Qwen/Qwen3-8B"
    agent.logger = logging.getLogger("tttd-harbor-test")
    environment = _Environment()
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.run("Pack 26 circles.", environment, context))

    assert "def run_packing" in environment.commands[0]
    assert context.metadata["reef"]["start_step"] == 0
    assert context.metadata["reef"]["next_step"] == 1
    assert len(context.metadata["reef"]["agent_record_ids"]) == 4


@pytest.mark.unit
def test_checkout_metadata_lets_sglang_find_reefs_plugin_without_an_install(tmp_path) -> None:
    import subprocess

    spec = importlib.util.spec_from_file_location("checkout_metadata", EXAMPLE / "apptainer" / "checkout_metadata.py")
    checkout_metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkout_metadata)

    checkout_metadata.write_metadata(ROOT / "pyproject.toml", tmp_path)
    # -S keeps this environment's own Reef install out of the lookup.
    found = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import importlib.metadata as m; print({e.name: e.value for e in m.entry_points(group='sglang.srt.plugins')})",
        ],
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "'reef': 'reef.train.slime_backend.reef_adapters.sglang.plugin:install_sglang_plugin'" in found
