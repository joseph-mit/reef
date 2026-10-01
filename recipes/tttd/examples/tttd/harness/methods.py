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
    train: bool = True
    """False runs the search on frozen weights, the floor a comparison reads against."""


METHODS: dict[str, Method] = {
    method.name: method
    for method in (
        Method("tttd", "serve.yaml", "ttt-discover"),
        Method("tttd-mean", "serve.tttd-mean.yaml", "ttt-discover"),
        Method("ppottt", "serve.ppottt.yaml", "ppottt"),
        Method("ppottt-smoke", "serve.ppottt-smoke.yaml", "ppottt"),
        Method("spottt", "serve.spottt.yaml", "spottt"),
        Method("spottt-smoke", "serve.spottt-smoke.yaml", "spottt"),
        Method("spottt-adaptive", "serve.spottt-adaptive.yaml", "spottt"),
        Method("espottt", "serve.espottt.yaml", "espottt"),
        Method("espottt-smoke", "serve.espottt-smoke.yaml", "espottt"),
        Method("espottt-adaptive", "serve.espottt-adaptive.yaml", "espottt"),
        Method("espottt-adaptive-smoke", "serve.espottt-adaptive-smoke.yaml", "espottt"),
        Method("hspottt", "serve.hspottt.yaml", "hspottt"),
        Method("hspottt-smoke", "serve.hspottt-smoke.yaml", "hspottt"),
        Method("search-only", "serve.search-only.yaml", "ttt-discover", train=False),
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
