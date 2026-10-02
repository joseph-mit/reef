"""Verifier and reward for AtCoder Heuristic Contest 058 ("apple" production planning).

The submission is one C++ program. It is compiled once, then run on the
task's public cases in order, each with the contest's 2-second limit. A case
scores the official ``round(1e5 * log2(S))``, where ``S`` is the number of
apples after ``T`` turns, computed by a port of the official tester (Rust,
``tools/src/lib.rs``) that keeps its 64-bit wrapping arithmetic. As in
TTT-Discover's AHC environment, evaluation stops at the first case that is
not accepted, the remaining cases score 0, and the reward is the mean case
score divided by 1500 * 2000; the reason reports the mean raw score.

Differences from TTT-Discover's ALE-Bench evaluation, which this repository
cannot reproduce exactly:

- the 50 cases are the first 50 of the official tools' ``in/`` directory as
  shipped in TTT-Discover's repository; their own cached cases could not be
  checked against them;
- the compiler is the host's ``g++`` with ALE-Bench's C++20 flags; the AtCoder
  Library, Boost, GMP and Eigen are available only when ``AHC_INCLUDE_DIRS``
  names their headers;
- the early stop is by case order, so the reward does not depend on timing.
"""

from __future__ import annotations

import math
import os
import resource
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent / "cases"
TIME_LIMIT_SEC = 2.0
# ALE-Bench's TIME_LIMIT_TOLERANCE: a case may run this much past the limit.
TIME_LIMIT_TOLERANCE_SEC = 0.5
MEMORY_LIMIT_BYTES = 1 << 30
COMPILE_TIMEOUT_SEC = 120
# Cases run two at a time, as TTT-Discover gives each evaluation two CPUs.
CASE_WORKERS = int(os.environ.get("AHC_CASE_WORKERS", "2"))
REWARD_SCALE = 1500.0 * 2000.0
COMPILE_FLAGS = (
    "-std=gnu++20",
    "-O2",
    "-DONLINE_JUDGE",
    "-DATCODER",
    "-march=native",
    "-fconstexpr-depth=2147483647",
    "-fconstexpr-loop-limit=2147483647",
    "-fconstexpr-ops-limit=2147483647",
)

_I64_MOD = 1 << 64
_I64_HALF = 1 << 63


def _i64(value: int) -> int:
    """Wrap to a signed 64-bit integer, as the official tester's release build does."""
    return (value + _I64_HALF) % _I64_MOD - _I64_HALF


def parse_input(text: str) -> tuple[int, int, int, int, list[int], list[list[int]]]:
    tokens = [int(token) for token in text.split()]
    n, levels, turns, apples = tokens[:4]
    a = tokens[4 : 4 + n]
    costs = [tokens[4 + n + level * n : 4 + n + (level + 1) * n] for level in range(levels)]
    return n, levels, turns, apples, a, costs


def parse_output(text: str, n: int, levels: int, turns: int) -> list[tuple[int, int] | None]:
    """The official parse_output: comments skipped, at most ``turns`` actions read, each validated."""
    actions: list[tuple[int, int] | None] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        first = _read_int(tokens[0], -1, levels - 1)
        if first == -1:
            actions.append(None)
            rest = tokens[1:]
        else:
            if len(tokens) < 2:
                raise ValueError("Unexpected EOF")
            actions.append((first, _read_int(tokens[1], 0, n - 1)))
            rest = tokens[2:]
        if rest:
            raise ValueError(f"Too many tokens: {line}")
        if len(actions) == turns:
            break
    if len(actions) < turns:
        raise ValueError(f"Not enough actions: expected {turns}, got {len(actions)}")
    return actions


def _read_int(token: str, low: int, high: int) -> int:
    # Rust's integer parsing: an optional sign, then ASCII digits only (and no
    # minus sign for the unsigned machine ID).
    digits = token[1:] if token[:1] in "+-" else token
    if not digits or not digits.isascii() or not digits.isdigit() or (low >= 0 and token.startswith("-")):
        raise ValueError(f"Parse error: {token}")
    value = int(token)
    if not low <= value <= high:
        raise ValueError(f"Out of range: {value}")
    return value


def compute_score(input_text: str, output_text: str) -> tuple[int, str]:
    """The official compute_score: ``(score, error)``, with score 0 whenever there is an error."""
    n, levels, turns, apples, a, costs = parse_input(input_text)
    try:
        actions = parse_output(output_text, n, levels, turns)
    except ValueError as exc:
        return 0, str(exc)
    count = [[1] * n for _ in range(levels)]
    power = [[0] * n for _ in range(levels)]
    for turn, action in enumerate(actions):
        if action is not None:
            level, machine = action
            cost = _i64(costs[level][machine] * (power[level][machine] + 1))
            if apples < cost:
                return 0, f"Not enough apples at turn {turn}: have {apples}, need {cost}"
            apples = _i64(apples - cost)
            power[level][machine] += 1
        for level in range(levels):
            for machine in range(n):
                if level == 0:
                    produced = _i64(_i64(a[machine] * count[0][machine]) * power[0][machine])
                    apples = _i64(apples + produced)
                else:
                    added = _i64(count[level][machine] * power[level][machine])
                    count[level - 1][machine] = _i64(count[level - 1][machine] + added)
    if apples <= 0:
        return 0, f"Non-positive apples at the end: {apples}"
    # Rust's f64::round rounds halves away from zero; the value is positive here.
    return math.floor(100000.0 * math.log2(float(apples)) + 0.5), ""


CPU_LIMIT_SEC = math.ceil(TIME_LIMIT_SEC + 0.1) + 1


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SEC, CPU_LIMIT_SEC))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _limited_command(binary: Path) -> tuple[list[str], bool]:
    """The command that runs ``binary`` under the CPU and memory limits.

    ``prlimit`` (util-linux), as ALE-Bench uses, sets them without running
    Python between fork and exec, which is unsafe while other threads run
    cases. Without it the limits are set in the child, and the returned flag
    asks for that.
    """
    prlimit = shutil.which("prlimit")
    if prlimit is None:
        return [str(binary)], True
    return [prlimit, f"--cpu={CPU_LIMIT_SEC}", f"--as={MEMORY_LIMIT_BYTES}", str(binary)], False


def run_case(binary: Path, case: Path) -> tuple[int, str]:
    """One case: ``(score, problem)``, where an empty problem means accepted."""
    input_text = case.read_text()
    command, limit_in_child = _limited_command(binary)
    started = time.monotonic()
    try:
        run = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SEC + TIME_LIMIT_TOLERANCE_SEC,
            preexec_fn=_limit_resources if limit_in_child else None,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
    except subprocess.TimeoutExpired:
        return 0, f"{case.stem}: time limit exceeded"
    elapsed = time.monotonic() - started
    if elapsed > TIME_LIMIT_SEC + TIME_LIMIT_TOLERANCE_SEC:
        return 0, f"{case.stem}: time limit exceeded ({elapsed:.2f}s)"
    if run.returncode != 0:
        return 0, f"{case.stem}: runtime error (exit {run.returncode}) {run.stderr.strip()[-300:]}"
    score, error = compute_score(input_text, run.stdout)
    if error:
        return 0, f"{case.stem}: wrong answer: {error}"
    return score, ""


def compile_program(source: str, directory: Path) -> tuple[Path | None, str]:
    compiler = shutil.which("g++")
    if compiler is None:
        return None, "no g++ on the judge host"
    (directory / "Main.cpp").write_text(source)
    includes = [f"-I{path}" for path in os.environ.get("AHC_INCLUDE_DIRS", "").split(":") if path]
    try:
        build = subprocess.run(
            [compiler, *COMPILE_FLAGS, *includes, "-o", "a.out", "Main.cpp"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None, "compilation timed out"
    if build.returncode != 0:
        return None, f"compilation error: {build.stderr.strip()[-1500:]}"
    return directory / "a.out", ""


def evaluate(source: str, cases: list[Path]) -> dict:
    """Compile once and run the cases in order, stopping at the first one not accepted."""
    with tempfile.TemporaryDirectory() as tmp:
        binary, problem = compile_program(source, Path(tmp))
        if binary is None:
            return {"reward": 0.0, "reason": problem}
        scores: list[int] = []
        first_problem = ""
        with ThreadPoolExecutor(max_workers=max(1, CASE_WORKERS)) as pool:
            # Submitted in windows so a failure stops new cases from starting.
            for start in range(0, len(cases), max(1, CASE_WORKERS)):
                window = cases[start : start + max(1, CASE_WORKERS)]
                for score, issue in pool.map(lambda case: run_case(binary, case), window):
                    if first_problem:
                        continue
                    if issue:
                        first_problem = issue
                    else:
                        scores.append(score)
                if first_problem:
                    break
    raw = sum(scores) / len(cases)
    accepted = len(scores)
    reason = f"Evaluated on {len(cases)} public test cases. Passed: {accepted}/{len(cases)}. Raw_score: {raw:.4f}"
    if first_problem:
        reason += f". Stopped at {first_problem}"
    return {"reward": raw / REWARD_SCALE, "reason": reason, "raw_score": raw}


def grade(artifact):
    """Compile the submitted C++ program and score it on the public cases."""
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution file"}
    source = Path(artifact).read_text()
    if not source.strip():
        return {"reward": 0.0, "reason": "empty solution"}
    return evaluate(source, sorted(CASES_DIR.glob("*.txt")))
