"""apptainer/status.sh --on-node: one block per recent run, failures filtered."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/apptainer/status.sh"


def _run(root: Path, method: str, events: str, log: str, *, age_s: float = 0) -> Path:
    state = root / method / "circle_packing_26"
    state.mkdir(parents=True)
    (state / "events.jsonl").write_text(events)
    (state / "reef.log").write_text(log)
    stamp = time.time() - age_s
    os.utime(state / "reef.log", (stamp, stamp))
    return state


@pytest.mark.unit
def test_status_reports_steps_activity_and_real_failures_only(tmp_path: Path) -> None:
    now = int(time.time())
    runs, local = tmp_path / "runs", tmp_path / "local"
    _run(
        runs,
        "spottt",
        f'{{"event": "tttd_step_started", "step": 1, "time": {now - 1500}}}\n',
        "starting slime-driver\n  error_injection_rate .... 0\nserver_args=ServerArgs(x)\nTimer train end\n",
    )
    _run(
        local,
        "ppottt",
        f'{{"event": "tttd_step_started", "step": 0, "time": {now - 3600}}}\n',
        "starting slime-driver\n(Ray) Traceback (most recent call last)\nRuntimeError: CUDA error: invalid argument\n",
    )
    _run(runs, "tttd-mean", "{}\n", "starting slime-driver\n", age_s=3 * 86400)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--on-node", str(runs), str(local)],
        env={**os.environ, "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    blocks = result.stdout.split("######## ")[1:]
    assert [block.splitlines()[0] for block in blocks] == [
        "~/runs/spottt/circle_packing_26",
        "~/local/ppottt/circle_packing_26",
    ]
    spo, ppo = blocks
    assert "step 2 running for 25 min" in spo and "no failure lines" in spo
    assert "step 1 running for 60 min" in ppo
    assert "FAILURE LINES" in ppo and "CUDA error: invalid argument" in ppo
