"""Harbor ``BaseAgent`` subclass for TTT-Discover.

This module depends on the external ``harbor`` package and is only imported
when Harbor/reef-eval loads the agent at runtime. The remaining harness modules
have no Harbor dependency and can be imported standalone (e.g. in tests).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

from .methods import DEFAULT_METHOD, method_named, scenario_name, state_dir
from .run_controller import ReefTrainingStatusClient
from .scorer import JudgeScorer, _codeblock_body, judge_slots
from .session import SERVICE_URL, TOKEN, StackSettings, build_run

JUDGE_URL = "http://127.0.0.1:8082"  # the task's judge, published by its compose file
SOLUTION_PATH = "/workspace/solution.py"
EXAMPLE_DIR = Path(__file__).resolve().parents[1]

# Everything that follows the chosen task and training method, derived from
# run.sh's two variables.
TASK = os.environ.get("TTTD_TASK", "erdos_min_overlap")
METHOD = method_named(os.environ.get("TTTD_METHOD", DEFAULT_METHOD))
SCENARIO = scenario_name(METHOD, TASK)
SEARCH_STATE_PATH = state_dir(EXAMPLE_DIR, METHOD, TASK) / f"{METHOD.name}-search-state.json"

# The step grid and completion settings, read from the stack config.
SETTINGS = StackSettings.load(EXAMPLE_DIR / METHOD.config)


class HarborAgent(BaseAgent):
    """A Harbor agent that runs TTT-Discover PUCT search through Reef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = ReefClient(SERVICE_URL, token=TOKEN, timeout_s=7200)

    @staticmethod
    def name() -> str:
        return "reef-tttd"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the search runs on the host."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        harness, controller = build_run(
            self._client,
            ReefTrainingStatusClient(SERVICE_URL, token=TOKEN),
            JudgeScorer(JUDGE_URL, max_in_flight=judge_slots(EXAMPLE_DIR / "harbor" / TASK)),
            instruction,
            method=METHOD,
            scenario=SCENARIO,
            task=TASK,
            model=self.model_name,
            settings=SETTINGS,
            state_path=SEARCH_STATE_PATH,
            emit=lambda event: self.logger.info("TTTD event: %s", event),
        )
        outcome = await asyncio.to_thread(controller.run, SETTINGS.steps)

        best = harness.archive.best()
        program_code = _codeblock_body(best.solution)

        result = await environment.exec(
            f"cat > {SOLUTION_PATH} <<'REEF_TTTD_EOF'\n{program_code}\nREEF_TTTD_EOF",
        )
        if result.return_code != 0:
            raise RuntimeError(f"writing {SOLUTION_PATH} failed: {result.stderr}")

        valid_receipts = [r.agent_record_id for r in outcome.results if r.agent_record_id and r.solution]
        metadata = dict(context.metadata or {})
        metadata["reef"] = {
            "agent_record_ids": valid_receipts,
            "start_step": outcome.start_step,
            "next_step": outcome.next_step,
            "runtime_load_id": outcome.runtime_load_id,
        }
        context.metadata = metadata
