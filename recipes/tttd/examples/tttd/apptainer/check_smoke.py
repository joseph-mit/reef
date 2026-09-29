"""Check a two-phase smoke run stage by stage, from what the run left on disk.

It reads what the harness and Reef themselves recorded, not log text:

- ``run-summaries.jsonl``: one line per phase from ``run_local.py``;
- ``events.jsonl``: the run controller's step events;
- ``<method>-search-state.json``: the search archive paired with a step;
- ``agent-record/*.commits.jsonl``: Reef's durable commit per training step,
  with its checkpoint flag, algorithm state and training metrics.

A stage fails only on a record that contradicts it. A metric the backend does not
report under the expected name is listed as not reported, not as a failure,
since a backend may publish it under another key.

    python3 apptainer/check_smoke.py --method ppottt-smoke --task circle_packing_26 \\
        --state work/ppottt-smoke/circle_packing_26 --report report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from harness.methods import method_named, scenario_name
from harness.session import StackSettings

PASSED, FAILED, NOT_REPORTED = "passed", "FAILED", "not reported"


@dataclass
class Stage:
    name: str
    result: str = PASSED
    details: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        self.details.append(text)

    def require(self, condition: bool, text: str) -> None:
        self.details.append(text if condition else f"expected: {text}")
        if not condition:
            self.result = FAILED


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _commits(state: Path, scenario: str) -> dict[int, dict[str, Any]]:
    commits: dict[int, dict[str, Any]] = {}
    for path in sorted((state / "agent-record").rglob("*.commits.jsonl")):
        for record in _json_lines(path):
            if record.get("scenario") == scenario and record.get("operation", "training") == "training":
                commits[int(record["step"])] = record
    return commits


def _metric(metrics: Mapping[str, Any] | None, suffix: str) -> Any:
    """A metric by the end of its key, whatever namespace the backend put it in."""
    for key, value in (metrics or {}).items():
        if key == suffix or key.endswith("/" + suffix):
            return value
    return None


def check(method_name: str, task: str, state: Path) -> list[Stage]:
    method = method_named(method_name)
    scenario = scenario_name(method, task)
    settings = StackSettings.load(EXAMPLE_DIR / method.config)
    step_size = settings.groups_per_step * settings.rollouts_per_group
    summaries = _json_lines(state / "run-summaries.jsonl")
    events = _json_lines(state / "events.jsonl")
    commits = _commits(state, scenario)
    first = summaries[0] if summaries else {}
    second = summaries[1] if len(summaries) > 1 else {}

    rollout = Stage("rollout")
    rollout.require(len(summaries) == 2, f"two phase summaries ({len(summaries)} found)")
    for label, summary in (("phase 1", first), ("phase 2", second)):
        rollout.require(
            summary.get("rollouts") == step_size,
            f"{label}: {summary.get('rollouts')} of {step_size} rollouts reported against inference receipts",
        )

    evaluation = Stage("evaluation")
    for label, summary in (("phase 1", first), ("phase 2", second)):
        evaluation.require(
            summary.get("judge_submissions") == step_size,
            f"{label}: judge scored {summary.get('judge_submissions')} of {step_size} attempts",
        )
        scored = summary.get("scored_rollouts")
        evaluation.note(f"{label}: {scored} attempts contained a program")
    best = max((summary.get("best_reward") or 0.0) for summary in summaries) if summaries else None
    evaluation.note(f"best reward in the archive: {best}")

    update = Stage("training update")
    for step in (1, 2):
        update.require(step in commits, f"Reef committed training step {step}")
    if method.name.startswith("ppottt"):
        for step in (1, 2):
            metrics = commits.get(step, {}).get("metrics")
            critic = _metric(metrics, "ppottt/critic_updates")
            actor = _metric(metrics, "ppottt/actor_trained")
            if critic is None or actor is None:
                update.note(f"step {step}: critic and actor update counts {NOT_REPORTED} in the commit metrics")
            else:
                update.require(
                    critic >= 1 and actor == 1, f"step {step}: {critic} critic update(s), actor trained = {actor}"
                )
    if method.name.startswith("spottt"):
        tracker = (commits.get(2, {}).get("algorithm_state") or {}).get("spottt")
        update.require(bool(tracker), f"step 2 committed the SPO tracker ({len(tracker or {})} keys)")
    for step, record in sorted(commits.items()):
        metrics = record.get("metrics") or {}
        shown = {key: metrics[key] for key in sorted(metrics) if any(part in key for part in ("clip", "kl", "loss"))}
        if shown:
            update.note(f"step {step} metrics: {json.dumps(shown, sort_keys=True)}")
        # Before the step's first update the trainer and the engine hold the
        # same weights, so this measures serving drift, not training.
        gap = (metrics.get("train_steps") or [{}])[0].get("train/ppo_kl")
        if isinstance(gap, (int, float)):
            update.note(f"step {step}: trainer vs sampler log-prob gap before any update (ppo_kl) {gap:.6f}")

    checkpoint = Stage("checkpoint save")
    for step in (1, 2):
        checkpoint.require(
            commits.get(step, {}).get("checkpoint") is True, f"Reef recorded step {step} as a durable checkpoint"
        )
    latest = state / "checkpoints" / "megatron" / "latest_checkpointed_iteration.txt"
    checkpoint.require(latest.is_file(), f"Megatron checkpoint marker at {latest.relative_to(state)}")
    if latest.is_file():
        checkpoint.note(f"latest Megatron iteration: {latest.read_text().strip()}")
    if method.name.startswith("ppottt"):
        critic_root = state / "checkpoints" / "critic"
        checkpoint.require(
            critic_root.is_dir() and any(critic_root.iterdir()), "critic checkpoint written under checkpoints/critic"
        )

    resume = Stage("resume after SIGKILL")
    resume.require(first.get("start_step") == 0 and first.get("next_step") == 1, "phase 1 ran step 1 from scratch")
    resume.require(
        second.get("start_step") == 1 and second.get("next_step") == 2,
        f"phase 2 resumed at step {second.get('start_step')} and finished at {second.get('next_step')}",
    )
    reef_step = (second.get("reef_status") or {}).get("scenario_step")
    resume.require(reef_step == 2, f"Reef reports scenario step {reef_step} after phase 2")
    search_state_path = state / f"{method.name}-search-state.json"
    search_state = json.loads(search_state_path.read_text()) if search_state_path.is_file() else {}
    resume.require(
        search_state.get("phase") == "committed" and search_state.get("next_step") == 2,
        f"search archive committed at step {search_state.get('next_step')}",
    )
    if any(event.get("event") == "tttd_runtime_load_id_rebound" for event in events):
        resume.note(
            "the restored runtime republished the checkpoint under a new load ID and the archive rebound to it"
        )
    if 1 in commits and 2 in commits:
        before = (commits[1].get("algorithm_state") or {}).get("steps")
        after = (commits[2].get("algorithm_state") or {}).get("steps")
        resume.require(
            (before, after) == (1, 2), f"algorithm state continued across the restart (steps {before} -> {after})"
        )

    return [rollout, evaluation, update, checkpoint, resume]


def render(method: str, task: str, stages: Iterable[Stage]) -> str:
    lines = [f"### {method} on {task}", "", "| Stage | Result | Details |", "| --- | --- | --- |"]
    for stage in stages:
        details = "<br>".join(text.replace("|", "\\|") for text in stage.details)
        lines.append(f"| {stage.name} | {stage.result} | {details} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state", required=True, type=Path, help="the method's state directory for the task")
    parser.add_argument("--report", type=Path, help="also write the table here")
    options = parser.parse_args(argv)
    stages = check(options.method, options.task, options.state)
    report = render(options.method, options.task, stages)
    print(report)
    if options.report is not None:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(report)
    return 0 if all(stage.result != FAILED for stage in stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
