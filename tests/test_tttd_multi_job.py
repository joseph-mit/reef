"""A dry run of apptainer/multi.sbatch with a stand-in for apptainer.

The stand-in answers the GPU ranking with a fixed order and records what
each run's stack would have started with, so the test checks the job
script's own work: fastest GPUs to the runs listed first, separate ports,
Ray directories and step limits per run, and each run's state location.
"""

from __future__ import annotations

import json
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
    import json, os, sys
    environment = {{key[len("APPTAINERENV_"):]: value for key, value in os.environ.items() if key.startswith("APPTAINERENV_")}}
    command = sys.argv[-1]
    if "gpu_check.py' --rank" in command:
        print("GPU 0: 120 TFLOPS sustained")
        print("ranked: 2 0 1")
    elif command.rstrip().endswith("bash run.sh"):
        binds = sys.argv[sys.argv.index("--bind") + 1]
        record = dict(environment, BINDS=binds, RAY=os.environ.get("APPTAINERENV_RAY_TMPDIR", ""))
        with open(os.path.join({records!r}, environment["TTTD_METHOD"] + ".json"), "w") as handle:
            json.dump(record, handle)
    """
)


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("pgrep") is None, reason="the job script needs pgrep")
def test_multi_job_gives_the_fastest_gpus_to_the_first_run(tmp_path) -> None:
    repo = tmp_path / "repo"
    example = repo / "recipes" / "tttd" / "examples" / "tttd"
    repo.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", repo / "pyproject.toml")
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("results", "work", "__pycache__"))
    (example / "work").mkdir()
    work = tmp_path / "work"
    work.mkdir()
    records = tmp_path / "records"
    records.mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    apptainer = bin_dir / "apptainer"
    apptainer.write_text(_FAKE_APPTAINER.format(python=sys.executable, records=str(records)))
    apptainer.chmod(0o755)
    python = bin_dir / "python3"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)

    environment = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "USER": "tester",
        "REEF_REPO": str(repo),
        "REEF_WORK_ROOT": str(work),
        "SLURM_JOB_ID": "4242",
        "CUDA_VISIBLE_DEVICES": "3,5,7",
        "TTTD_RUNS": "spottt:1:12 ppottt:2",
        "TTTD_LOCAL_STATE": "ppottt",
        "STACK_START_GAP_S": "0",
        "GPU_RANK_SECONDS": "1",
        "LANG": "C.UTF-8",
    }
    result = subprocess.run(
        ["bash", str(example / "apptainer" / "multi.sbatch")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    spo = json.loads((records / "spottt.json").read_text())
    ppo = json.loads((records / "ppottt.json").read_text())
    # The ranking put device 2 (CUDA id 7) first, then 0 (3), then 1 (5).
    assert spo["CUDA_VISIBLE_DEVICES"] == "7"
    assert ppo["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert spo["TTTD_RUN_STEPS"] == "12" and "TTTD_RUN_STEPS" not in ppo
    for name in ("TTTD_REEF_PORT", "TTTD_JUDGE_PORT", "TTTD_ROUTER_PORT", "REEF_PORT_BASE", "RAY_TMPDIR"):
        assert spo[name] != ppo[name], name
    assert int(ppo["REEF_PORT_BASE"]) - int(spo["REEF_PORT_BASE"]) == 64
    assert os.readlink(example / "work" / "spottt") == str(work / "runs" / "spottt")
    assert os.readlink(example / "work" / "ppottt") == "/tmp/tester-reef-runs/ppottt"
    assert "/tmp/tester-reef-runs" in ppo["BINDS"]
    assert "spottt: finished" in result.stdout and "ppottt: finished" in result.stdout


@pytest.mark.unit
def test_multi_job_refuses_runs_that_need_more_gpus_than_it_has(tmp_path) -> None:
    repo = tmp_path / "repo"
    example = repo / "recipes" / "tttd" / "examples" / "tttd"
    repo.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", repo / "pyproject.toml")
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("results", "work", "__pycache__"))
    result = subprocess.run(
        ["bash", str(example / "apptainer" / "multi.sbatch")],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path / "home"),
            "REEF_REPO": str(repo),
            "REEF_WORK_ROOT": str(tmp_path / "work"),
            "CUDA_VISIBLE_DEVICES": "0,1",
            "TTTD_RUNS": "spottt:2 ppottt:4",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1
    assert "needs 6 GPUs; this job has 2" in result.stderr
