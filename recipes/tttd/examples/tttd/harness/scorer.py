"""Scoring abstractions for TTT-Discover.

A ``Scorer`` is a callable that takes a solution string and returns a
``ScoredSolution``. Two implementations are provided:

- :class:`JudgeScorer` — POSTs the solution to a judge HTTP endpoint
  (the standard Harbor task infrastructure). No task-specific Python imports.
- :class:`ProgramScorer` — runs the solution in a subprocess sandbox on the
  host and applies a reward function. Used by the direct experiment runner
  where no judge container is running.
"""

from __future__ import annotations

import json
import math
import re
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .sandbox import ProgramExecutionError, execute_program
from .search import ScoredSolution

Scorer = Callable[[str], ScoredSolution]

_CODEBLOCK_RE = re.compile(r"```python\s+([\s\S]*?)\s*```")


def _codeblock_body(codeblock: str) -> str:
    match = _CODEBLOCK_RE.search(codeblock)
    if match is None:
        raise ValueError("cannot extract Python code")
    return match.group(1).strip()


class JudgeScorer:
    """Score solutions by POSTing to a judge HTTP endpoint.

    The judge runs the program, verifies the result, and returns
    ``{"score": float, "reason": str}``. This is the standard Harbor
    pattern — the task is fully declarative (instruction.md + score.py +
    judge_server.py) and no task-specific Python code is imported.

    The judge runs at most ``max_concurrent_submissions`` programs at once
    and queues the rest, while ``timeout_s`` counts from the moment a request
    is sent. With ``max_in_flight`` set to the judge's slot count, a request
    is sent only when a slot is free, so the timeout covers the program's run
    and not its wait in the judge's queue; the wait happens here, untimed.
    ``None`` sends every request at once, the previous behaviour.
    """

    def __init__(self, judge_url: str, *, timeout_s: float = 7200, max_in_flight: int | None = None) -> None:
        if max_in_flight is not None and max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        self._submit_url = judge_url.rstrip("/") + "/submit"
        self._timeout_s = timeout_s
        self._slots = None if max_in_flight is None else threading.BoundedSemaphore(max_in_flight)

    def __call__(self, solution: str) -> ScoredSolution:
        data = _codeblock_body(solution).encode("utf-8")
        req = urllib.request.Request(
            self._submit_url,
            data=data,
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        if self._slots is None:
            payload = self._submit(req)
        else:
            with self._slots:
                payload = self._submit(req)
        reward = float(payload["score"])
        reason = payload.get("reason", "")
        return ScoredSolution(
            solution=solution,
            reward=reward,
            value=reward,
            metrics={},
            output=reason,
        )

    def _submit(self, req: urllib.request.Request) -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.loads(resp.read())


def judge_slots(task_dir: Path) -> int | None:
    """The number of programs the task's judge runs at once, or None when it sets no limit."""
    config_path = task_dir / "environment" / "judge_config.json"
    if not config_path.is_file():
        return None
    slots = json.loads(config_path.read_text()).get("max_concurrent_submissions")
    return None if slots is None else int(slots)


class ProgramScorer:
    """Score solutions by running them in a subprocess sandbox.

    Extracts the Python code block, runs it via :func:`execute_program`,
    and applies ``reward_fn`` to the return value. The reward function
    maps the program's result to ``(reward, value, metrics)``.

    Used by the direct experiment runner (``runner.py``) where no judge
    container is running and lower-latency host-side execution is needed.
    """

    def __init__(
        self,
        reward_fn: Callable[[Any], tuple[float, float, dict[str, Any]]],
        *,
        eval_timeout_s: float = 1_100,
        num_cpus: int = 1,
        work_dir: str | Path | None = None,
        program_budget_s: float | None = None,
        executor: Callable[..., Any] | None = None,
        prelude: str = "",
    ) -> None:
        if eval_timeout_s <= 0:
            raise ValueError("eval_timeout_s must be positive")
        if num_cpus < 1:
            raise ValueError("num_cpus must be positive")
        self._reward_fn = reward_fn
        self._eval_timeout_s = float(eval_timeout_s)
        self._num_cpus = num_cpus
        self._work_dir = None if work_dir is None else Path(work_dir)
        self._program_budget_s = program_budget_s
        self._executor = executor or execute_program
        self._prelude = prelude

    def __call__(self, solution: str) -> ScoredSolution:
        code = _codeblock_body(solution)
        source = f"{self._prelude}\n\n{code}\n" if self._prelude else f"{code}\n"
        try:
            execution = self._executor(
                source,
                entrypoint="run",
                timeout_s=self._eval_timeout_s,
                max_cpus=self._num_cpus,
                work_dir=self._work_dir,
                entrypoint_kwargs=(
                    {} if self._program_budget_s is None else {"seed": 42, "budget_s": self._program_budget_s}
                ),
            )
        except ProgramExecutionError as exc:
            raise ValueError(str(exc)) from exc

        reward, value, metrics = self._reward_fn(execution.result)
        reward = float(reward)
        value = float(value)
        if not math.isfinite(reward) or not math.isfinite(value):
            raise ValueError("reward function returned a non-finite value")
        return ScoredSolution(
            solution=solution,
            reward=reward,
            value=value,
            metrics=metrics,
            output=execution.stdout,
        )
