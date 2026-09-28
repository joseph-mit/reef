# Running the example without Docker

The example normally runs each task through Harbor, which starts the task's judge in a Docker container. HPC clusters rarely offer Docker; MIT's Engaging cluster offers Apptainer instead. This directory runs the same stack there:

- the training stack runs in an Apptainer image built from `reef-tttd.def`, which mirrors `docker/Dockerfile.reef`'s `tttd` target on the same pinned slime digest, with the Reef checkout bound in rather than baked in;
- `TTTD_DRIVER=local` makes `run.sh` start [`run_local.py`](../run_local.py) instead of `run.py`: the task's own judge runs as a local process inside the container, on the loopback interface only, and the search is built by the same code as the Harbor agent (`harness/session.py`), so rewards, run identity and resume state match a Harbor run.

The job scripts assume Engaging (partition `pi_ppliang`, the `apptainer/1.4.2` module for `--fakeroot`, scratch at `~/orcd/scratch`). Elsewhere, override the `#SBATCH` lines and `REEF_WORK_ROOT`.

## Steps

Clone the fork on the cluster, somewhere with space, and submit everything from the repository root:

```bash
cd ~/orcd/scratch
git clone https://github.com/joseph-mit/reef.git && cd reef
git submodule update --init third_party/reef-client

sbatch recipes/tttd/examples/tttd/apptainer/setup.sbatch   # once: image, model, self-tests
sbatch recipes/tttd/examples/tttd/apptainer/smoke.sbatch   # after setup prints SETUP OK
```

`setup.sbatch` checks network access, builds the image (how long depends on the node and the registries), downloads `Qwen/Qwen3-8B` with its revision recorded, links it to `work/model`, then checks inside the image that the GPU works, that Reef and the recipes import, and that the packing judge scores a known packing correctly.

## What the smoke checks

For each of `ppottt-smoke` and `spottt-smoke` on `circle_packing_26` (2 parents x 2 attempts per step, thinking off):

1. **phase 1**: a fresh stack runs step 1: rollouts through Reef, scoring by the task's judge, one training update, a durable checkpoint;
2. **crash**: every process of the stack is killed with SIGKILL at the step boundary, with no shutdown;
3. **phase 2**: a new stack restores the checkpoint and the search archive and runs step 2;
4. **check**: [`check_smoke.py`](check_smoke.py) reads what Reef committed (`agent-record/*.commits.jsonl`), the run summaries and the search state, and reports each stage (rollout, evaluation, training update, checkpoint save, resume) as passed or failed, with the records behind each result.

The job ends with `SMOKE PASSED` or `SMOKE FAILED`. The reports, logs, commit records and the best program are copied to `$REEF_WORK_ROOT/smoke-results/<job id>/`. Checkpoints go to the node's local disk when it has 600 GB free, otherwise under `REEF_WORK_ROOT`.

Variables: `SMOKE_METHODS` (default `"ppottt-smoke spottt-smoke"`), `TTTD_TASK` (default `circle_packing_26`), `REEF_WORK_ROOT` (default `~/orcd/scratch/reef-work`), `PHASE_TIMEOUT_S` (default 7200).

## Full runs

`run.sbatch` runs one method. `TTTD_RUN_STEPS` ends a job cleanly after that many steps in total, so a 50-step run can be split across jobs and resumed by resubmitting with a larger value:

```bash
TTTD_METHOD=spottt TTTD_TASK=circle_packing_26 TTTD_RUN_STEPS=10 \
    sbatch --gres=gpu:2 recipes/tttd/examples/tttd/apptainer/run.sbatch
```

## Known limits

- None of this has run on a GPU node yet; the scripts are tested on CPU with a stand-in for Apptainer and the GPU stack (`tests/test_tttd_smoke_job.py`, `tests/test_tttd_local_driver.py`).
- Reef's checkpoint storage check sees a shared filesystem's total size, not your quota. A PPO-TTT checkpoint carries a full 8B critic with its optimiser state (from the arithmetic, over 100 GB; not yet measured), and Reef keeps the latest checkpoint while writing the next, so keep several hundred GB free wherever its state lives.
- Two runs cannot share a node: the Reef port (8900), the judge port (8082) and the SGLang router port (30000) are fixed.
