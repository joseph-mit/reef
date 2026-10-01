"""Reef harness subclass for TTT-Discover.

``ReefTTTDiscoverHarness`` extends the generic PUCT harness to route
inference through a Reef scenario and report each rollout's reward against
its exact inference receipt.

The Harbor ``BaseAgent`` subclass lives in ``harbor_agent.py`` so this module
stays importable without the external ``harbor`` package (e.g. in tests).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from reef_client import ReefClient

from .search import Candidate, RolloutResult, Scorer, _TTTDiscoverHarnessBase, solution_key
from .siblings import SiblingSettings

# Report tags of methods whose advantage compares a parent's sibling group.
GROUP_ALGORITHMS = ("ttt-discover", "tttd")


class ReefTTTDiscoverHarness(_TTTDiscoverHarnessBase):
    """TTTD with scenario inference and linked reward reports."""

    def __init__(
        self,
        client: ReefClient,
        scorer: Scorer,
        instruction: str,
        *,
        scenario: str,
        model: str,
        release_id: str | None = None,
        inference_path: str = "/v1/chat/completions",
        groups_per_step: int = 8,
        rollouts_per_group: int = 64,
        exploration: float = 1.0,
        invalid_reward: float = 0.0,
        max_workers: int = 32,
        request_builder: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None,
        algorithm: str = "ttt-discover",
        train: bool = True,
        siblings: SiblingSettings | None = None,
    ) -> None:
        if siblings is not None and siblings.adaptive and algorithm in GROUP_ALGORITHMS:
            raise ValueError(
                "adaptive siblings give parents different numbers of attempts; "
                f"{algorithm!r} compares each parent's fixed sibling group"
            )
        super().__init__(
            scorer,
            instruction,
            model=model,
            groups_per_step=groups_per_step,
            rollouts_per_group=rollouts_per_group,
            exploration=exploration,
            invalid_reward=invalid_reward,
            max_workers=max_workers,
            request_builder=request_builder,
            siblings=siblings,
        )
        self.client = client
        self.scenario = scenario
        self.release_id = release_id
        self.inference_path = inference_path
        # The report tag the scenario's recipe accepts: the same search feeds
        # tttd and the single-stream recipes, which check it on ingest.
        self.algorithm = algorithm
        # A search-only run still reports every score, marked ineligible for
        # training through Reef's framework-neutral opt-out, so the records
        # stay comparable with a training run's while the weights never move.
        self.train = train

    def _rollout(
        self,
        parent: Candidate,
        comparison_set: str,
        step: int,
        group_index: int,
        rollout_index: int,
    ) -> RolloutResult:
        release_headers = {} if self.release_id is None else {"x-reef-release-id": self.release_id}
        started = time.monotonic()
        response, agent_record_id = self.client.inference_with_record(
            self.scenario,
            self.inference_path,
            self._request_payload(parent),
            extra_headers=release_headers,
        )
        generation_seconds = time.monotonic() - started

        result = dataclasses.replace(
            self._evaluate_response(parent, response),
            agent_record_id=agent_record_id,
            group=group_index,
            rollout=rollout_index,
            generation_seconds=generation_seconds,
        )

        metadata: dict[str, Any] = {
            "comparison_set": comparison_set,
            "algorithm": self.algorithm,
            "step": step,
            "group": group_index,
            "rollout": rollout_index,
            "groups_per_step": self.groups_per_step,
            "rollouts_per_group": self.rollouts_per_group,
            "parent_id": parent.candidate_id,
            "grandparent_id": parent.parent_id or "",
            "search_value": result.search_value,
            # The parent as the model saw it, so an attempt can be judged
            # against the state it tried to improve.
            "parent_reward": parent.reward,
            "parent_visits": parent.visits,
            "parent_depth": self.archive.depth(parent.candidate_id),
            # Programs by content, so a later attempt can be traced to the
            # attempt whose program its parent state is (stepping-stone credit).
            "solution_sha1": solution_key(result.solution),
            "parent_solution_sha1": solution_key(parent.solution),
        }
        if not self.train:
            metadata["training"] = {"eligible": False}
        self.client.report(
            self.scenario,
            {
                "score": result.reward,
                # The judge rejects a program with a reason and no exception.
                "feedback": result.error or (result.output if result.reward == self.invalid_reward else None),
                "references": [agent_record_id],
                "metadata": metadata,
            },
            extra_headers=release_headers,
        )
        return result
