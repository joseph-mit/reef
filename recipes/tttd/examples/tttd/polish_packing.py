"""Polish a circle packing: refine its last digits numerically, outside the search.

On the packing tasks the last digits of the best score come from numerical
refinement, not from a new arrangement. This script takes a packing program
(for example a run's ``best_program.py``), runs it once, refines the packing
it returns with SLSQP from that starting point, shrinks the result until it
passes the task judge's own checks, and writes a program that returns the
refined packing. It never changes the arrangement, so a gap that remains
after polishing is a search gap, not a precision gap.

    python polish_packing.py circle_packing_26 work/spottt/circle_packing_26/best_program.py polished.py

The written program is graded with the task's ``score.py``, the same code the
judge runs, and the script prints the score before and after.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
# A refined packing is shrunk by this much so that float rounding cannot
# push it past the judge's 1e-12 tolerance.
MARGIN = 1e-13


def task_scorer(task: str) -> ModuleType:
    """The task's judge code, imported from its score.py."""
    path = HERE / "harbor" / task / "environment" / "score.py"
    spec = importlib.util.spec_from_file_location(f"{task}_score", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refine(centers: np.ndarray, radii: np.ndarray, *, iterations: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Maximise the sum of radii from the given packing, then make it strictly valid."""
    count = len(radii)
    upper, lower = np.triu_indices(count, 1)

    def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[: 2 * count].reshape(count, 2), x[2 * count :]

    def walls(x: np.ndarray) -> np.ndarray:
        c, r = split(x)
        return np.concatenate([c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r])

    def walls_jacobian(x: np.ndarray) -> np.ndarray:
        jac = np.zeros((4 * count, 3 * count))
        rows = np.arange(count)
        for block, (axis, sign) in enumerate(((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))):
            jac[block * count + rows, 2 * rows + axis] = sign
            jac[block * count + rows, 2 * count + rows] = -1.0
        return jac

    def gaps(x: np.ndarray) -> np.ndarray:
        c, r = split(x)
        return np.linalg.norm(c[upper] - c[lower], axis=1) - r[upper] - r[lower]

    def gaps_jacobian(x: np.ndarray) -> np.ndarray:
        c, _ = split(x)
        delta = c[upper] - c[lower]
        unit = delta / np.maximum(np.linalg.norm(delta, axis=1), 1e-300)[:, None]
        jac = np.zeros((len(upper), 3 * count))
        rows = np.arange(len(upper))
        for axis in (0, 1):
            jac[rows, 2 * upper + axis] = unit[:, axis]
            jac[rows, 2 * lower + axis] = -unit[:, axis]
        jac[rows, 2 * count + upper] = -1.0
        jac[rows, 2 * count + lower] = -1.0
        return jac

    start = np.concatenate([np.asarray(centers, float).ravel(), np.asarray(radii, float)])
    gradient = np.concatenate([np.zeros(2 * count), -np.ones(count)])
    result = minimize(
        lambda x: -x[2 * count :].sum(),
        start,
        jac=lambda x: gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * count) + [(0.0, 0.5)] * count,
        constraints=(
            {"type": "ineq", "fun": walls, "jac": walls_jacobian},
            {"type": "ineq", "fun": gaps, "jac": gaps_jacobian},
        ),
        options={"maxiter": iterations, "ftol": 1e-16},
    )
    refined_centers, refined_radii = split(result.x)
    return make_valid(refined_centers, refined_radii)


def make_valid(centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shrink radii just enough that every wall and pair constraint holds with MARGIN to spare."""
    centers = np.clip(np.asarray(centers, float), 0.0, 1.0)
    radii = np.asarray(radii, float).copy()
    room = np.minimum.reduce([centers[:, 0], 1 - centers[:, 0], centers[:, 1], 1 - centers[:, 1]])
    radii = np.clip(np.minimum(radii, room - MARGIN), 0.0, None)
    upper, lower = np.triu_indices(len(radii), 1)
    distance = np.linalg.norm(centers[upper] - centers[lower], axis=1)
    for _ in range(10_000):
        overlap = radii[upper] + radii[lower] - distance + MARGIN
        if not (overlap > 0).any():
            break
        cut = np.zeros_like(radii)
        np.maximum.at(cut, upper, np.where(overlap > 0, overlap / 2, 0.0))
        np.maximum.at(cut, lower, np.where(overlap > 0, overlap / 2, 0.0))
        radii = np.clip(radii - cut, 0.0, None)
    else:
        raise RuntimeError("could not make the refined packing valid")
    return centers, radii


def packing_program(centers: np.ndarray, radii: np.ndarray) -> str:
    """A program that returns the packing exactly, in the task's run_packing contract."""
    return (
        "import numpy as np\n\n\n"
        "def run_packing():\n"
        f"    centers = np.array({json.dumps(centers.tolist())})\n"
        f"    radii = np.array({json.dumps(radii.tolist())})\n"
        "    return centers, radii, float(np.sum(radii))\n"
    )


def run_program(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run a packing program's run_packing() and return its centers and radii."""
    namespace: dict[str, object] = {"__name__": "packing_program"}
    exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    centers, radii, _ = namespace["run_packing"]()  # type: ignore[operator]
    return np.asarray(centers, float), np.asarray(radii, float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=("circle_packing_26", "circle_packing_32"))
    parser.add_argument("program", type=Path, help="a program defining run_packing()")
    parser.add_argument("output", type=Path, help="where to write the polished program")
    args = parser.parse_args(argv)

    scorer = task_scorer(args.task)
    before = scorer.grade(str(args.program))
    centers, radii = run_program(args.program)
    polished = packing_program(*refine(centers, radii))
    args.output.write_text(polished)
    after = scorer.grade(str(args.output))
    print(json.dumps({"before": before, "after": after}, indent=2))
    if float(after["reward"]) < float(before["reward"]):
        print("polishing did not help; keep the original program", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
