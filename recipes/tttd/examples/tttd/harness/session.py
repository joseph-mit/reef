"""One search run against a Reef deployment, as both entry points build it.

The Harbor agent (``harbor_agent.py``) and the Harbor-free runner
(``run_local.py``) differ only in where the task comes from and how a
solution is scored. Everything else, from the sampling settings to the run
identity the resume check compares, is fixed here or read from the
deployment config, so the two build the same run and either one can resume a
state file the other wrote.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from reef_client import ReefClient

from .agent import ReefTTTDiscoverHarness
from .methods import Method
from .run_controller import ReefTrainingStatusClient, TTTDRunController, TTTDRunIdentity, TTTDRunStateStore
from .search import Scorer, TTTDChatRequestBuilder
from .siblings import SiblingSettings
from .step_records import StepRecorder

# The Reef run.sh started; TTTD_REEF_PORT moves it when two stacks share a machine.
SERVICE_URL = f"http://127.0.0.1:{os.environ.get('TTTD_REEF_PORT', '8900')}"
TOKEN = "reef-local"  # matches every serve*.yaml
INFERENCE_PATH = "/v1/chat/completions"

# TTT-Discover's sampling and search settings, written out rather than left to
# SGLang's model-specific defaults. They are part of the run identity.
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
EXPLORATION = 1.0
INVALID_REWARD = 0.0


@dataclass(frozen=True)
class StackSettings:
    """What the harness reads from a deployment config.

    Reef trains only after exactly ``groups_per_step x rollouts_per_group``
    reports arrive, so the harness takes the grid from the same file the
    stack runs rather than repeating it.
    """

    groups_per_step: int
    rollouts_per_group: int
    steps: int
    max_new_tokens: int
    enable_thinking: bool
    # How a step's attempts are spread over parents (the config's search.siblings).
    siblings: SiblingSettings = field(default_factory=SiblingSettings)

    @classmethod
    def load(cls, path: Path) -> StackSettings:
        stack = yaml.safe_load(Path(path).read_text())
        reef, training = stack["reef"], stack["training"]
        return cls(
            groups_per_step=int(reef["groups_per_step"]),
            rollouts_per_group=int(reef["rollouts_per_group"]),
            steps=int(training["steps"]),
            max_new_tokens=int(training["max_new_tokens"]),
            # A reduced smoke turns thinking off so a short completion still reaches code.
            enable_thinking=bool(training.get("enable_thinking", True)),
            siblings=SiblingSettings.from_mapping((stack.get("search") or {}).get("siblings")),
        )


def max_workers(task: str) -> int:
    """Concurrent rollouts and evaluations; the packing tasks need the memory headroom."""
    return 256 if task.startswith("circle_packing") else 512


def build_run(
    client: ReefClient,
    status_reader: ReefTrainingStatusClient,
    scorer: Scorer,
    instruction: str,
    *,
    method: Method,
    scenario: str,
    task: str,
    model: str,
    settings: StackSettings,
    state_path: Path,
    emit: Callable[[Mapping[str, Any]], None],
) -> tuple[ReefTTTDiscoverHarness, TTTDRunController]:
    """The Reef harness and the run controller that pairs its archive with training commits."""
    harness = ReefTTTDiscoverHarness(
        client,
        scorer,
        instruction,
        scenario=scenario,
        model=model,
        inference_path=INFERENCE_PATH,
        groups_per_step=settings.groups_per_step,
        rollouts_per_group=settings.rollouts_per_group,
        exploration=EXPLORATION,
        invalid_reward=INVALID_REWARD,
        max_workers=max_workers(task),
        algorithm=method.algorithm,
        train=method.train,
        request_builder=TTTDChatRequestBuilder(
            max_new_tokens=settings.max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            enable_thinking=settings.enable_thinking,
        ),
        siblings=settings.siblings,
    )
    identity = TTTDRunIdentity(
        scenario=scenario,
        model=model,
        recipe=method.name,
        inference_path=INFERENCE_PATH,
        instruction_sha256=instruction_sha256(instruction),
        groups_per_step=settings.groups_per_step,
        rollouts_per_group=settings.rollouts_per_group,
        max_new_tokens=settings.max_new_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        enable_thinking=settings.enable_thinking,
        exploration=EXPLORATION,
        invalid_reward=INVALID_REWARD,
        siblings=settings.siblings.identity(),
    )
    controller = TTTDRunController(
        harness,
        status_reader,
        TTTDRunStateStore(state_path, identity),
        emit=emit,
        wait_for_training=method.train,
        # Every attempt of every step, next to the search state.
        step_records=StepRecorder(Path(state_path).parent / "attempts", invalid_reward=INVALID_REWARD),
    )
    return harness, controller


def instruction_sha256(instruction: str) -> str:
    return hashlib.sha256(instruction.encode("utf-8")).hexdigest()
