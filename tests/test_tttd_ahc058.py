"""AtCoder Heuristic Contest 058: the scorer port, the judge's C++ path and the task contract.

Expected scores come from the official tester (``tools/src/bin/tester.rs``,
built in release mode); 700 random outputs, including inputs whose apples
overflow 64 bits, matched it exactly when the port was written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from recipes.tttd.examples.tttd.harness.scorer import _codeblock_body, judge_slots
from recipes.tttd.examples.tttd.harness.search import extract_solution
from recipes.tttd.examples.tttd.harness.session import code_language

TASK = Path(__file__).resolve().parents[1] / "recipes" / "tttd" / "examples" / "tttd" / "harbor" / "ahc058"
CASES = TASK / "environment" / "cases"

# Buys the cheapest affordable upgrade every turn; the C++ below does the same.
CHEAPEST_CPP = r"""
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, L, T; long long K;
    cin >> n >> L >> T >> K;
    vector<long long> A(n);
    for (auto& a : A) cin >> a;
    vector<vector<long long>> C(L, vector<long long>(n));
    for (auto& row : C) for (auto& c : row) cin >> c;
    vector<vector<long long>> B(L, vector<long long>(n, 1)), P(L, vector<long long>(n, 0));
    __int128 apples = K;
    for (int t = 0; t < T; ++t) {
        int bl = -1, bj = -1; __int128 best = -1;
        for (int l = 0; l < L; ++l) for (int j = 0; j < n; ++j) {
            __int128 c = (__int128)C[l][j] * (P[l][j] + 1);
            if (c <= apples && (best < 0 || c < best)) { best = c; bl = l; bj = j; }
        }
        if (bl >= 0) { apples -= best; P[bl][bj]++; cout << bl << ' ' << bj << '\n'; }
        else cout << -1 << '\n';
        for (int l = 0; l < L; ++l) for (int j = 0; j < n; ++j) {
            if (l == 0) apples += (__int128)A[j] * B[0][j] * P[0][j];
            else B[l - 1][j] += B[l][j] * P[l][j];
        }
    }
}
"""


@pytest.fixture(scope="module")
def score() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tttd_ahc058_score", TASK / "environment" / "score.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cheapest(score: ModuleType, text: str) -> str:
    n, levels, turns, apples, a, costs = score.parse_input(text)
    count = [[1] * n for _ in range(levels)]
    power = [[0] * n for _ in range(levels)]
    lines = []
    for _ in range(turns):
        options = [
            (costs[level][machine] * (power[level][machine] + 1), level, machine)
            for level in range(levels)
            for machine in range(n)
            if costs[level][machine] * (power[level][machine] + 1) <= apples
        ]
        if options:
            cost, level, machine = min(options)
            apples -= cost
            power[level][machine] += 1
            lines.append(f"{level} {machine}")
        else:
            lines.append("-1")
        for level in range(levels):
            for machine in range(n):
                if level == 0:
                    apples += a[machine] * count[0][machine] * power[0][machine]
                else:
                    count[level - 1][machine] += count[level][machine] * power[level][machine]
    return "\n".join(lines) + "\n"


@pytest.mark.unit
@pytest.mark.parametrize(("case", "expected"), [("0000", 1_722_707), ("0007", 1_480_811), ("0049", 1_691_632)])
def test_scores_match_the_official_tester(score: ModuleType, case: str, expected: int) -> None:
    text = (CASES / f"{case}.txt").read_text()
    assert score.compute_score(text, _cheapest(score, text)) == (expected, "")


@pytest.mark.unit
def test_one_upgrade_then_nothing_matches_the_official_tester(score: ModuleType) -> None:
    text = (CASES / "0000.txt").read_text()
    assert score.compute_score(text, "0 0\n" + "-1\n" * 499) == (896_578, "")


@pytest.mark.unit
def test_the_tester_arithmetic_wraps_at_64_bits(score: ModuleType) -> None:
    assert score._i64(2**63) == -(2**63)
    assert score._i64(-(2**63) - 1) == 2**63 - 1
    # Outside the contest's limits apples overflow; the official tester then
    # reports 5,168,941 for this plan.
    text = "10 4 500 1\n" + " ".join(["100"] * 10) + "\n" + "\n".join(" ".join(["1"] * 10) for _ in range(4)) + "\n"
    assert score.compute_score(text, _cheapest(score, text)) == (5_168_941, "")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("output", "error"),
    [
        ("-1\n" * 499, "Not enough actions"),
        ("4 0\n" + "-1\n" * 499, "Out of range"),
        ("0 10\n" + "-1\n" * 499, "Out of range"),
        ("-1 2\n" + "-1\n" * 499, "Too many tokens"),
        ("0 -0\n" + "-1\n" * 499, "Parse error"),
        ("0 1_0\n" + "-1\n" * 499, "Parse error"),
        ("0 9\n" + "-1\n" * 499, "Not enough apples at turn 0"),
    ],
)
def test_invalid_outputs_score_zero_with_the_tester_message(score: ModuleType, output: str, error: str) -> None:
    text = (CASES / "0000.txt").read_text()
    result, message = score.compute_score(text, output)
    assert result == 0 and error in message


@pytest.mark.unit
def test_comments_are_skipped_and_extra_actions_ignored(score: ModuleType) -> None:
    text = (CASES / "0000.txt").read_text()
    assert score.compute_score(text, "# plan\n0 0\n" + "-1\n" * 600)[0] == 896_578


@pytest.mark.unit
def test_task_contract() -> None:
    assert len(sorted(CASES.glob("*.txt"))) == 50
    assert judge_slots(TASK) == 16
    assert code_language("ahc058") == "cpp" and code_language("circle_packing_26") == "python"
    instruction = (TASK / "instruction.md").read_text()
    # TTT-Discover's AhcEnv.get_question() for ahc058 from its initial state, byte for byte.
    assert len(instruction) == 5_976
    assert hashlib.sha256(instruction.encode()).hexdigest() == (
        "7c112f807eb953fef159bd1307bf09cc36cbe39219bd77ea5eb1500f329e32ed"
    )


@pytest.mark.unit
def test_cpp_blocks_are_extracted_for_the_judge() -> None:
    response = "plan\n```cpp\nint main() {}\n```\nthen\n```cpp\nint main() { return 0; }\n```"
    solution = extract_solution(response, "cpp")
    assert solution == "```cpp\nint main() { return 0; }\n```"
    assert _codeblock_body(solution) == "int main() { return 0; }"
    assert extract_solution(response) == ""  # a Python task ignores C++ blocks


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_judge_compiles_and_runs_a_program_on_every_case(score: ModuleType, tmp_path: Path) -> None:
    program = tmp_path / "submission"
    program.write_text(CHEAPEST_CPP)
    result = score.grade(program)
    assert "Passed: 50/50" in result["reason"]
    expected = sum(
        score.compute_score(case.read_text(), _cheapest(score, case.read_text()))[0] for case in CASES.glob("*.txt")
    )
    assert result["raw_score"] == pytest.approx(expected / 50)
    assert result["reward"] == pytest.approx(expected / 50 / 3e6)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_judge_stops_at_the_first_failing_case(score: ModuleType, tmp_path: Path) -> None:
    program = tmp_path / "submission"
    program.write_text("int main() { return 1; }")
    result = score.grade(program)
    assert result["reward"] == 0.0 and "Stopped at 0000: runtime error" in result["reason"]
    program.write_text("int main() { return x; }")
    assert score.grade(program)["reason"].startswith("compilation error")
