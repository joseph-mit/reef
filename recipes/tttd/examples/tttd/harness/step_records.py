"""Every attempt of a search step, written by the harness, and the step's summary.

Reef keeps its own record of each inference and report, but deletes them
after a retention window, and they do not hold what the search knew: the
parent an attempt started from, the judge's reason for a rejection, or how
long generation and scoring took. After each step the harness writes one
gzip JSON-lines file, ``attempts/step-0003.jsonl.gz``: one line for the step
(selected parents, archive after the expansion), then one line per attempt.
The same pass returns summary numbers that go into the step's
``tttd_step_committed`` event.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .search import PUCTArchive, RolloutResult, SelectedParent, solution_key

OUTPUT_CHARS = 1_000
_NUMBER = re.compile(r"\d+(\.\d+)?")


def failure_kind(result: RolloutResult, invalid_reward: float) -> str:
    """A short, countable name for why an attempt scored nothing; "" when it scored."""
    if result.error:
        if (
            "does not contain a Python code block" in result.error
            or "does not contain a C++ code block" in result.error
        ):
            return "no code block"
        return result.error.split(":", 1)[0]
    if result.reward != invalid_reward:
        return ""
    reason = (result.output or "no reason").split(":", 1)[0].strip()
    return _NUMBER.sub("N", reason)[:60]


def _think_split(action: str) -> tuple[int, int]:
    """Characters of reasoning and of answer in a response with a closing think tag."""
    head, tag, tail = action.partition("</think>")
    return (len(head), len(tail)) if tag else (0, len(action))


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def attempt_record(
    result: RolloutResult,
    parent: SelectedParent | None,
    *,
    invalid_reward: float,
    best_before: float,
) -> dict[str, Any]:
    think_chars, answer_chars = _think_split(result.action)
    improvement = None if parent is None else result.reward - parent.reward
    return {
        "group": result.group,
        "rollout": result.rollout,
        "record_id": result.agent_record_id,
        "parent_id": result.parent_id,
        "parent_reward": None if parent is None else parent.reward,
        "reward": result.reward,
        "search_value": result.search_value,
        "valid": result.reward != invalid_reward,
        "failure": failure_kind(result, invalid_reward),
        "improvement": improvement,
        "beat_parent": improvement is not None and improvement > 0,
        "beat_best": result.reward > best_before,
        "error": result.error,
        "output": (result.output or "")[-OUTPUT_CHARS:],
        "solution_sha1": solution_key(result.solution),
        "solution": result.solution,
        "response_chars": len(result.action),
        "think_chars": think_chars,
        "answer_chars": answer_chars,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "generation_seconds": result.generation_seconds,
        "evaluation_seconds": result.evaluation_seconds,
    }


def archive_summary(archive: PUCTArchive) -> dict[str, Any]:
    candidates = archive.candidates
    scored = [candidate.reward for candidate in candidates if not candidate.seed]
    depths = [archive.depth(candidate.candidate_id) for candidate in candidates]
    return {
        "archive_size": len(candidates),
        "archive_scored": len(scored),
        "archive_best_reward": max(candidate.reward for candidate in candidates),
        "archive_mean_reward": _mean(scored),
        "archive_max_depth": max(depths) if depths else 0,
        "archive_total_expansions": archive.total_expansions,
    }


def step_summary(
    records: Sequence[Mapping[str, Any]],
    selection: Sequence[SelectedParent],
) -> dict[str, Any]:
    """Numbers for the step's event: how good, how varied and how costly its attempts were."""
    rewards = [float(record["reward"]) for record in records]
    valid = [float(record["reward"]) for record in records if record["valid"]]
    top = sorted(rewards)[-max(1, len(rewards) // 10) :] if rewards else []
    solutions = [record["solution_sha1"] for record in records if record["solution_sha1"]]
    tokens = [record["completion_tokens"] for record in records if record["completion_tokens"] is not None]
    finishes = [record["finish_reason"] for record in records if record["finish_reason"] is not None]
    generation = [record["generation_seconds"] for record in records if record["generation_seconds"] is not None]
    evaluation = [record["evaluation_seconds"] for record in records if record["evaluation_seconds"] is not None]
    improvements = [
        record["improvement"] for record in records if record["valid"] and record["improvement"] is not None
    ]
    failures = Counter(record["failure"] for record in records if record["failure"])
    count = max(len(records), 1)
    summary: dict[str, Any] = {
        "valid_fraction": len(valid) / count,
        "valid_mean": _mean(valid),
        "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else None,
        "reward_median": statistics.median(rewards) if rewards else None,
        "reward_top10_mean": _mean(top),
        "beat_parent_fraction": sum(1 for record in records if record["beat_parent"]) / count,
        "beat_best_count": sum(1 for record in records if record["beat_best"]),
        "valid_improvement_mean": _mean(improvements),
        "unique_solution_fraction": len(set(solutions)) / len(solutions) if solutions else None,
        "completion_tokens_mean": _mean(tokens),
        "completion_tokens_p90": _quantile(tokens, 0.9),
        "truncated_fraction": sum(reason == "length" for reason in finishes) / len(finishes) if finishes else None,
        "generation_seconds_mean": _mean(generation),
        "generation_seconds_p90": _quantile(generation, 0.9),
        "evaluation_seconds_mean": _mean(evaluation),
        "failures": dict(failures.most_common()),
        "parent_reward_mean": _mean([parent.reward for parent in selection]),
        "parent_depth_mean": _mean([float(parent.depth) for parent in selection]),
        "parent_seed_count": sum(1 for parent in selection if parent.seed),
    }
    return {key: value for key, value in summary.items() if not (isinstance(value, float) and math.isnan(value))}


class StepRecorder:
    """Write one step's attempts to ``<directory>/step-NNNN.jsonl.gz`` and summarise them."""

    def __init__(self, directory: str | Path, *, invalid_reward: float = 0.0) -> None:
        self.directory = Path(directory)
        self.invalid_reward = float(invalid_reward)

    def path(self, step: int) -> Path:
        return self.directory / f"step-{step + 1:04d}.jsonl.gz"

    def record(
        self,
        step: int,
        results: Sequence[RolloutResult],
        selection: Sequence[SelectedParent],
        archive: PUCTArchive,
        *,
        best_before: float,
        runtime_load_id: str | None,
    ) -> dict[str, Any]:
        """Write the step's file (replacing a file an interrupted run left) and return its summary."""
        parents = {parent.candidate_id: parent for parent in selection}
        records = [
            attempt_record(
                result,
                parents.get(result.parent_id),
                invalid_reward=self.invalid_reward,
                best_before=best_before,
            )
            for result in results
        ]
        summary = step_summary(records, selection)
        header = {
            "step": step + 1,
            "runtime_load_id": runtime_load_id,
            "best_before": best_before if math.isfinite(best_before) else None,
            "selection": [asdict(parent) for parent in selection],
            **archive_summary(archive),
            "archive": [
                {
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "reward": candidate.reward,
                    "visits": candidate.visits,
                    "seed": candidate.seed,
                    "solution_sha1": solution_key(candidate.solution),
                }
                for candidate in archive.candidates
            ],
            "summary": summary,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(step)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(header, sort_keys=True) + "\n")
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
        return summary


def read_step(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The step line and the attempt lines of one step file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    return lines[0], lines[1:]


__all__ = ["StepRecorder", "archive_summary", "attempt_record", "failure_kind", "read_step", "step_summary"]
