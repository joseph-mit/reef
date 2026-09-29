"""Compare training methods on one task, step by step, from their step events.

Each method's run writes ``events.jsonl`` under its state directory; the
``tttd_step_committed`` events carry the mean reward of the step's attempts
(what a training method changes) and the archive's best reward (what the
search has found). On the circle-packing tasks the published 8x64 TTT-Discover
run from ``results/`` is added as a reference curve.

    python3 compare_runs.py --task circle_packing_26 --methods search-only spottt ppottt
    python3 compare_runs.py --task circle_packing_26 --plot comparison.png

Rows are one per committed step; a method that has not reached a step shows a
blank. Every method in a comparison should use the same grid, so a step means
the same number of verifier calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ("search-only", "tttd", "tttd-mean", "spottt", "ppottt")
PUBLISHED = HERE / "results" / "formal-8x64-v3-packing"
PUBLISHED_TASKS = {"circle_packing_26": "packing26", "circle_packing_32": "packing32"}
REFERENCE = "TTT-Discover (published)"

Curve = dict[int, dict[str, float]]


def events_path(method: str, task: str, work: Path) -> Path:
    # tttd keeps its original work/<task> layout (harness/methods.py).
    root = work / task if method == "tttd" else work / method / task
    return root / "events.jsonl"


def step_curve(events: Iterable[Mapping]) -> Curve:
    """1-based step -> reward_mean and best, from committed-step events; a restart's repeat wins."""
    curve: Curve = {}
    for event in events:
        if event.get("event") != "tttd_step_committed":
            continue
        point = {"best": float(event["archive_best_reward"])}
        if "reward_mean" in event:
            point["mean"] = float(event["reward_mean"])
        curve[int(event["step"]) + 1] = point
    return curve


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def published_curve(task: str, results: Path = PUBLISHED) -> Curve:
    """The formal 8x64 TTT-Discover packing run: W&B mean reward and the archive's best."""
    name = PUBLISHED_TASKS.get(task)
    if name is None or not results.is_dir():
        return {}
    curve: Curve = {}
    with (results / "best_solution_history.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["task"] == name:
                curve.setdefault(int(row["iteration"]), {})["best"] = float(row["best_score_so_far"])
    with (results / "wandb_history.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["task"] == name and row.get("rollout/rewards"):
                curve.setdefault(int(row["reef/step"]), {})["mean"] = float(row["rollout/rewards"])
    return curve


def table(curves: Mapping[str, Curve]) -> str:
    steps = sorted({step for curve in curves.values() for step in curve})
    # The published run has 50 steps; show only as far as the runs being compared.
    reached = [max(curve) for name, curve in curves.items() if name != REFERENCE and curve]
    if reached:
        steps = [step for step in steps if step <= max(reached)]
    names = list(curves)
    header = "| step | " + " | ".join(f"{name} mean | {name} best" for name in names) + " |"
    lines = [header, "|" + " --- |" * (1 + 2 * len(names))]
    for step in steps:
        cells = []
        for name in names:
            point = curves[name].get(step, {})
            cells.append(f"{point['mean']:.3f}" if "mean" in point else "")
            cells.append(f"{point['best']:.6f}" if "best" in point else "")
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(curves: Mapping[str, Curve], task: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (mean_axis, best_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for name, curve in curves.items():
        steps = sorted(curve)
        style = {"linestyle": "--", "color": "grey"} if name == REFERENCE else {"marker": "o"}
        means = [(step, curve[step]["mean"]) for step in steps if "mean" in curve[step]]
        bests = [(step, curve[step]["best"]) for step in steps if "best" in curve[step]]
        if means:
            mean_axis.plot(*zip(*means), label=name, **style)
        if bests:
            best_axis.plot(*zip(*bests), label=name, **style)
    mean_axis.set(title=f"{task}: mean reward of the step's attempts", xlabel="step", ylabel="mean reward")
    best_axis.set(title=f"{task}: best reward found so far", xlabel="step", ylabel="best reward")
    best_axis.ticklabel_format(axis="y", useOffset=False)
    mean_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="circle_packing_26")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--work", type=Path, default=HERE / "work")
    parser.add_argument("--plot", type=Path, help="also save a figure here (needs matplotlib)")
    parser.add_argument("--no-reference", action="store_true", help="leave out the published TTT-Discover run")
    options = parser.parse_args(argv)

    curves: dict[str, Curve] = {}
    for method in options.methods:
        curve = step_curve(read_events(events_path(method, options.task, options.work)))
        if curve:
            curves[method] = curve
    if not options.no_reference and (reference := published_curve(options.task)):
        curves[REFERENCE] = reference
    if not curves:
        print(f"no committed steps found for {options.task} under {options.work}")
        return 1
    print(table(curves))
    if options.plot is not None:
        plot(curves, options.task, options.plot)
        print(f"figure: {options.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
