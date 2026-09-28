"""A dry run of apptainer/smoke.sbatch with a stand-in for apptainer and the GPU stack.

The stand-in runs every command the job sends into the container on the
host. ``bash run.sh`` becomes the real ``run_local.py`` against the task's
real judge and a fake Reef that keeps its commits in Reef's own commit log,
so the committed step survives the job's SIGKILL the way a checkpoint does.
What this checks is the job script itself: phase 1 stopping at a step
boundary, the kill, the restart, the checker and the collected results.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "recipes" / "tttd" / "examples" / "tttd"

_FAKE_APPTAINER = textwrap.dedent(
    """\
    #!{python}
    import os, subprocess, sys
    environment = {{key: value for key, value in os.environ.items() if not key.startswith("APPTAINERENV_")}}
    environment.update(
        {{key[len("APPTAINERENV_"):]: value for key, value in os.environ.items() if key.startswith("APPTAINERENV_")}}
    )
    environment["PYTHONPATH"] = {real_repo!r}
    command = sys.argv[-1]
    if command.rstrip().endswith("bash run.sh"):
        example = command.split("cd '", 1)[1].split("'", 1)[0]
        state = os.path.join(example, "work", environment["TTTD_METHOD"], environment["TTTD_TASK"])
        os.makedirs(state, exist_ok=True)
        environment["TTTD_STATE_DIR"] = state
        os.execve(sys.executable, [sys.executable, {stack!r}, example], environment)
    os.execve("/bin/bash", ["bash", "-c", command], environment)
    """
)

_FAKE_STACK = textwrap.dedent(
    """\
    import os, socket, sys
    from pathlib import Path

    example = sys.argv[1]
    sys.path.insert(0, example)
    import run_local
    from harness.run_controller import ScenarioTrainingStatus
    from reef.core import ArtifactRef
    from reef.scenario.state import CommitRecord
    from reef.storage.commit_log import CommitLog

    PROGRAM = (
        "```python\\nimport numpy as np\\n\\ndef run_packing():\\n"
        "    centers = [(0.1 + 0.16 * (i % 6), 0.1 + 0.16 * (i // 6)) for i in range(26)]\\n"
        "    radii = [0.05] * 26\\n    return np.array(centers), np.array(radii), sum(radii)\\n```"
    )
    state = Path(os.environ["TTTD_STATE_DIR"])
    (state / "agent-record").mkdir(parents=True, exist_ok=True)
    log_path = state / "agent-record" / "scenario.commits.jsonl"


    class Reef:
        def __init__(self):
            self.log = CommitLog(log_path)
            self.committed = len(log_path.read_text().splitlines()) if log_path.exists() else 0
            self.pending = 0
            self.session = str(os.getpid())

        def inference_with_record(self, scenario, path, payload, **_):
            return {"choices": [{"message": {"content": PROGRAM}}]}, f"{self.session}-{self.pending}"

        def report(self, scenario, payload, **_):
            self.pending += 1
            if self.pending == 4:
                self.pending = 0
                self.committed += 1
                step = self.committed
                self.log.append(CommitRecord(
                    scenario=scenario, step=step, artifact_ref=ArtifactRef(f"a{step}", f"c{step}", None),
                    checkpoint=True, algorithm_state={"steps": step, "spottt": {"task": [1.3, 4.0, step - 1]}},
                    high_water_sequence=4 * step, high_water_offset=4 * step,
                    metrics={"train/ppottt/critic_updates": 2, "train/ppottt/actor_trained": 1},
                ))
                megatron = state / "checkpoints" / "megatron"
                megatron.mkdir(parents=True, exist_ok=True)
                (megatron / "latest_checkpointed_iteration.txt").write_text(str(step))
                (state / "checkpoints" / "critic" / f"iter_{step}").mkdir(parents=True, exist_ok=True)
            return {}

        def scenario_status(self, scenario):
            return ScenarioTrainingStatus(self.committed, f"{self.session}:{self.committed}", False)


    def free_port():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]


    reef = Reef()
    real_judge = run_local.local_judge
    run_local.local_judge = lambda task_dir, data_dir: real_judge(task_dir, data_dir, port=free_port())
    run_local.ReefClient = lambda *args, **kwargs: reef
    run_local.ReefTrainingStatusClient = lambda *args, **kwargs: reef
    run_local.scenario_status = lambda scenario: {"scenario_step": reef.committed}
    run_local.main()
    """
)


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("pgrep") is None, reason="the job script needs pgrep")
def test_smoke_job_kills_the_stack_resumes_it_and_collects_the_results(tmp_path) -> None:
    repo = tmp_path / "repo"
    example = repo / "recipes" / "tttd" / "examples" / "tttd"
    repo.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", repo / "pyproject.toml")
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("results", "work", "__pycache__"))
    (example / "work" / "model").mkdir(parents=True)
    (example / "work" / "model" / "config.json").write_text("{}")
    work = tmp_path / "work"
    work.mkdir()
    (work / "reef-tttd.sif").write_text("")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stack = tmp_path / "fake_stack.py"
    stack.write_text(_FAKE_STACK)
    apptainer = bin_dir / "apptainer"
    apptainer.write_text(_FAKE_APPTAINER.format(python=sys.executable, real_repo=str(ROOT), stack=str(stack)))
    apptainer.chmod(0o755)
    # A wrapper rather than a symlink: a symlinked venv interpreter loses the venv.
    python = bin_dir / "python3"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)

    environment = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "REEF_REPO": str(repo),
        "REEF_WORK_ROOT": str(work),
        "PHASE_TIMEOUT_S": "300",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        ["bash", str(example / "apptainer" / "smoke.sbatch")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "SMOKE PASSED" in output
    results = next((work / "smoke-results").iterdir())
    for method in ("ppottt-smoke", "spottt-smoke"):
        report = (results / method / "report.md").read_text()
        assert "FAILED" not in report
        assert "phase 2 resumed at step 1 and finished at 2" in report
        for name in ("run-summaries.jsonl", "events.jsonl", "scenario.commits.jsonl", "phase1.out"):
            assert (results / method / name).is_file(), name
        assert "holding_for_kill" in (results / method / "phase1.out").read_text()
        # The job pointed the method's work directory at this job's fresh state.
        assert (example / "work" / method).is_symlink()
