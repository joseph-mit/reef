"""The TTT-Discover loop, in the open.

One episode:

    solve  — reef-eval runs the selected harbor/ task under our agent; every
             rollout is
             an inference through Reef, stamped with its step-grid
             coordinates
    verify — the task's judge scores each attempt
    learn  — the agent reports the scores; the selected method's recipe
             (TTTD by default, see harness/methods.py) waits for one complete
             step grid (8 groups x 64 rollouts), trains, and serves the
             updated weights

``Lab.run`` is reef-eval's one primitive: task in, trusted scored row out.
"""

import asyncio
import os
from pathlib import Path

from reef_eval import Lab

MODEL = "Qwen/Qwen3-8B"  # the model run.sh downloaded

# The four tasks this example ships. run.sh picks one with TTTD_TASK and sizes
# the training stack for it; harness/harbor_agent.py reads the same variable.
TASKS = ("erdos_min_overlap", "circle_packing_26", "circle_packing_32", "ahc058")
TASK = os.environ.get("TTTD_TASK", TASKS[0])
if TASK not in TASKS:
    raise SystemExit(f"unknown TTTD_TASK {TASK!r}; choose {', '.join(TASKS)}")

HERE = Path(__file__).resolve().parent
# The training method run.sh selected. run.sh and harness/methods.py derive the
# same state root; tttd keeps its original layout.
METHOD = os.environ.get("TTTD_METHOD", "tttd")
STATE_DIR = HERE / "work" / TASK if METHOD == "tttd" else HERE / "work" / METHOD / TASK
AGENT = {"name": "harness:HarborAgent", "model_name": MODEL}


async def main() -> None:
    lab = Lab(STATE_DIR / "lab")
    row = await lab.run(str(HERE / "harbor" / TASK), AGENT)
    if error := row.tags.get("error"):
        raise RuntimeError(f"Harbor trial failed: {error}")
    if not row.rewards:
        raise RuntimeError(f"Harbor trial returned no rewards: {row.uri}")
    print(f"episode reward: {row.rewards}")


asyncio.run(main())
