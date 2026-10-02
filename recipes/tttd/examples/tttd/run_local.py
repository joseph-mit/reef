"""The TTT-Discover loop without Harbor, for machines with no Docker daemon.

``run.py`` hands each task to Harbor, which runs the task's judge in its own
container. An HPC node usually cannot do that: Engaging, for one, offers
Apptainer and no Docker. With ``TTTD_DRIVER=local``, ``run.sh`` runs this
script instead:

    judge  - the task's own ``judge_server.py`` and ``score.py``, started as a
             local process that listens on the loopback interface only
    search - the harness and run controller ``harness/harbor_agent.py``
             builds, from the same deployment config (``harness/session.py``)
    result - the best program, a run summary and the step events, written
             under the method's state directory

Rewards are the containerised judge's rewards: the scoring code is the same
file. The programs the judge executes are only as isolated as the process
running this script, so run it inside a container; ``apptainer/`` has the
setup for Engaging.

``TTTD_RUN_STEPS`` stops after that many steps instead of the config's
``training.steps``; with ``TTTD_HOLD_AFTER_RUN=1`` the script then waits to be
killed rather than exiting. The smoke job uses the pair to kill the whole
stack at a step boundary and check that a fresh stack resumes the run.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from reef_client import ReefClient

from harness.methods import DEFAULT_METHOD, method_named, scenario_name, state_dir
from harness.run_controller import ReefTrainingStatusClient
from harness.scorer import JudgeScorer, _codeblock_body, judge_slots
from harness.session import SERVICE_URL, TOKEN, StackSettings, build_run, code_language

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-8B"  # the model run.sh downloaded, as run.py names it
TASKS = ("erdos_min_overlap", "circle_packing_26", "circle_packing_32", "ahc058")
# The tasks' compose files publish the judge on 8082; run.sh sets TTTD_JUDGE_PORT.
JUDGE_PORT = int(os.environ.get("TTTD_JUDGE_PORT", "8082"))
JUDGE_READY_TIMEOUT_S = 120.0

# Serves the task's judge exactly as its container does, but bound to the
# loopback interface: on a shared node, a judge that executes whatever it is
# sent must not be reachable from other machines.
_JUDGE_ENTRY = """
import importlib.util, sys
from http.server import ThreadingHTTPServer
spec = importlib.util.spec_from_file_location("judge_server", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), module.Handler).serve_forever()
"""

# The judge runs model-written programs; they inherit only these variables,
# not the operator's tokens or keys.
_JUDGE_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "PYTHONUNBUFFERED")


def judge_environment(environ: Mapping[str, str], *, judge_dir: Path, data_dir: Path) -> dict[str, str]:
    environment = {name: environ[name] for name in _JUDGE_ENVIRONMENT if name in environ}
    environment.update(JUDGE_DIR=str(judge_dir), DATA_DIR=str(data_dir))
    return environment


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _get_json(url: str, *, token: str | None = None, timeout_s: float = 10.0) -> Any:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


@contextmanager
def local_judge(task_dir: Path, data_dir: Path, *, port: int = JUDGE_PORT) -> Iterator[str]:
    """Run the task's judge for the duration of the block; yields its URL."""
    judge_dir = task_dir / "environment"
    if not (judge_dir / "judge_server.py").is_file():
        raise FileNotFoundError(f"{task_dir} has no environment/judge_server.py")
    if not _port_is_free(port):
        raise RuntimeError(f"port {port} is already in use; is another run on this machine?")
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    log = (data_dir / "judge.log").open("ab")
    process = subprocess.Popen(
        [sys.executable, "-c", _JUDGE_ENTRY, str(judge_dir / "judge_server.py"), str(port)],
        env=judge_environment(os.environ, judge_dir=judge_dir, data_dir=data_dir),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + JUDGE_READY_TIMEOUT_S
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"the judge exited with code {process.returncode}; see {data_dir / 'judge.log'}")
            try:
                if _get_json(url + "/health", timeout_s=2.0).get("ok"):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"the judge did not answer on {url} within {JUDGE_READY_TIMEOUT_S:g}s")
            time.sleep(0.5)
        yield url
    finally:
        # The judge's own children are the model's programs; stop them with it.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.close()


def scenario_status(scenario: str) -> Mapping[str, Any] | None:
    """The scenario's entry in Reef's public status, as Reef reports it."""
    payload = _get_json(SERVICE_URL.rstrip("/") + "/reef/status", token=TOKEN, timeout_s=30.0)
    scenarios = payload.get("scenarios") if isinstance(payload, Mapping) else None
    return scenarios.get(scenario) if isinstance(scenarios, Mapping) else None


def run_steps_from(environ: Mapping[str, str], configured: int) -> int:
    value = environ.get("TTTD_RUN_STEPS")
    if not value:
        return configured
    steps = int(value)
    if not 1 <= steps <= configured:
        raise ValueError(f"TTTD_RUN_STEPS={steps} must lie in [1, {configured}] (the config's training.steps)")
    return steps


def _append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    task = os.environ.get("TTTD_TASK", TASKS[0])
    if task not in TASKS:
        raise SystemExit(f"unknown TTTD_TASK {task!r}; choose {', '.join(TASKS)}")
    method = method_named(os.environ.get("TTTD_METHOD", DEFAULT_METHOD))
    scenario = scenario_name(method, task)
    state = Path(os.environ.get("TTTD_STATE_DIR") or state_dir(HERE, method, task))
    state.mkdir(parents=True, exist_ok=True)
    settings = StackSettings.load(HERE / method.config)
    run_steps = run_steps_from(os.environ, settings.steps)
    task_dir = HERE / "harbor" / task
    instruction = (task_dir / "instruction.md").read_text()
    events_path = state / "events.jsonl"

    def emit(event: Mapping[str, Any]) -> None:
        record = {"time": time.time(), **event}
        _append_json_line(events_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)

    judge_data = state / "judge" / time.strftime("session-%Y%m%d-%H%M%S")
    with local_judge(task_dir, judge_data) as judge_url:
        harness, controller = build_run(
            ReefClient(SERVICE_URL, token=TOKEN, timeout_s=7200),
            ReefTrainingStatusClient(SERVICE_URL, token=TOKEN),
            JudgeScorer(judge_url, max_in_flight=judge_slots(task_dir)),
            instruction,
            method=method,
            scenario=scenario,
            task=task,
            model=MODEL,
            settings=settings,
            state_path=state / f"{method.name}-search-state.json",
            emit=emit,
        )
        outcome = controller.run(run_steps)
        judge = _get_json(judge_url + "/status")

    best = harness.archive.best()
    if best.solution:
        suffix = ".cpp" if code_language(task) == "cpp" else ".py"
        (state / f"best_solution{suffix}").write_text(_codeblock_body(best.solution) + "\n")
    summary = {
        "time": time.time(),
        "method": method.name,
        "task": task,
        "scenario": scenario,
        "start_step": outcome.start_step,
        "next_step": outcome.next_step,
        "configured_steps": settings.steps,
        "runtime_load_id": outcome.runtime_load_id,
        "rollouts": len(outcome.results),
        "scored_rollouts": sum(1 for result in outcome.results if result.solution),
        "judge_submissions": judge.get("used"),
        "archive_size": len(harness.archive.candidates),
        "best_reward": best.reward,
        "reef_status": scenario_status(scenario) if method.train else None,
    }
    _append_json_line(state / "run-summaries.jsonl", summary)
    print(json.dumps({"event": "run_summary", **summary}, sort_keys=True), flush=True)

    if os.environ.get("TTTD_HOLD_AFTER_RUN") == "1":
        # Keep run.sh from stopping the stack gracefully; the caller kills it.
        print(json.dumps({"event": "holding_for_kill", "next_step": outcome.next_step}), flush=True)
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
