"""Score one known packing through the local judge, as run_local.py starts it.

A valid 26-circle packing of radius-0.05 circles on a grid must come back with
reward 1.3 (the sum of the radii), and a program that overlaps two circles
must score 0. Run from the example directory:

    python3 apptainer/judge_selftest.py
"""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR))

from run_local import local_judge

GRID_PACKING = """
import numpy as np

def run_packing():
    centers = [(0.1 + 0.16 * (i % 6), 0.1 + 0.16 * (i // 6)) for i in range(26)]
    radii = [0.05] * 26
    return np.array(centers), np.array(radii), sum(radii)
"""

OVERLAPPING_PACKING = GRID_PACKING.replace("0.05] * 26", "0.09] * 26")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _submit(url: str, program: str) -> dict:
    request = urllib.request.Request(url + "/submit", data=program.encode(), method="POST")
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def main() -> int:
    task_dir = EXAMPLE_DIR / "harbor" / "circle_packing_26"
    with tempfile.TemporaryDirectory() as data, local_judge(task_dir, Path(data), port=_free_port()) as url:
        valid = _submit(url, GRID_PACKING)
        overlapping = _submit(url, OVERLAPPING_PACKING)
    print(json.dumps({"valid": valid, "overlapping": overlapping}, indent=2))
    if abs(valid["score"] - 1.3) > 1e-9 or overlapping["score"] != 0.0:
        print("judge self-test FAILED", file=sys.stderr)
        return 1
    print("judge self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
