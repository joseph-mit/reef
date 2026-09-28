"""The training methods this search can drive, and where each one keeps its state.

The PUCT search, the prompts and the Harbor tasks are the same for every
method; what changes is the Reef deployment that trains on the scored
attempts. ``TTTD_METHOD`` picks one. Each method gets its own scenario and its
own state directory, so runs of different methods never share checkpoints,
artifacts or a search archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_METHOD = "tttd"


@dataclass(frozen=True)
class Method:
    name: str
    config: str
    """Deployment config next to ``run.sh``."""
    algorithm: str
    """The ``metadata.algorithm`` tag the method's report contract accepts."""


METHODS: dict[str, Method] = {
    method.name: method
    for method in (
        Method("tttd", "serve.yaml", "ttt-discover"),
        Method("ppottt", "serve.ppottt.yaml", "ppottt"),
        Method("ppottt-smoke", "serve.ppottt-smoke.yaml", "ppottt"),
    )
}


def method_named(name: str) -> Method:
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f"unknown TTTD_METHOD {name!r}; choose {', '.join(METHODS)}") from None


def scenario_name(method: Method, task: str) -> str:
    # tttd keeps its original scenario name so existing state stays valid.
    return f"{method.name}-{task.replace('_', '-')}"


def state_dir(example_dir: Path, method: Method, task: str) -> Path:
    """Durable state root; tttd keeps its original ``work/<task>`` layout."""
    if method.name == DEFAULT_METHOD:
        return example_dir / "work" / task
    return example_dir / "work" / method.name / task
