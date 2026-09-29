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

## Checking a run

`run_status.py <state directory>` prints one row per committed step: mean and best reward, step time, training throughput, allocator retries, the trainer-sampler log-prob gap before any update, and the sampled policy's KL to the base. It reads only files, so it runs on the login node while the job runs.

## Known limits

- The SPO-TTT smoke has passed on an Engaging H200 node (all five stages, about 13 minutes). With thinking off, the smoke's attempts all scored 0, so its update had zero advantages: it checks the pipeline, not learning. The scripts are also tested on CPU with a stand-in for Apptainer and the GPU stack (`tests/test_tttd_smoke_job.py`, `tests/test_tttd_local_driver.py`).
- Speed: on two H200s the first SPO-TTT step on circle_packing_26 took about 3.3 hours: 59 minutes writing and scoring the 512 attempts (about 890 generated tokens/s per GPU), 18 minutes of frozen-base log-probabilities and 2 hours of training at about 900 tokens/s, against about 8,300 tokens/s in the published run on two B200s. Two settings came from that B200 config. The trainer's `max_split_size_mb:512` stops the CUDA allocator splitting its large cached blocks, so attempts of different lengths keep asking the driver for new memory, which here goes through torch_memory_saver; the trainer held about 135 of the 140 GB on each GPU (not yet confirmed as the cause). The configs now give the trainer `expandable_segments:False` and nothing else, i.e. the allocator's defaults: the Slime image sets `expandable_segments:True` for every process, and torch_memory_saver, which offloads the colocated trainer, refuses to run under it, so the trainer must switch it off explicitly. SGLang's `--sglang-mem-fraction-static=0.5` left an H200 room for the KV cache of about 25 attempts of this length at once against 128 requested; it is now 0.7, Slime's colocated default, which gives an H200 more KV cache than 0.5 gave a B200. `run_status.py` shows each step's throughput and allocator retries.
- Host memory: the SPO-TTT smoke peaked at about 168 GiB. The first PPO-TTT smoke was killed at a 256 GB limit after loading the actor and the critic, whose full parameters and optimiser state are offloaded to host memory while idle, so `smoke.sbatch` asks for 768 GB and a PPO-TTT `run.sbatch` job needs `--mem=768G`.
- Reef's checkpoint storage check sees a shared filesystem's total size, not your quota. A PPO-TTT checkpoint carries a full 8B critic with its optimiser state (from the arithmetic, over 100 GB; not yet measured), and Reef keeps the latest checkpoint while writing the next, so keep several hundred GB free wherever its state lives.
- Several jobs can share a node: each job takes free ports for Reef, the judge and the SGLang router (`TTTD_REEF_PORT`, `TTTD_JUDGE_PORT`, `TTTD_ROUTER_PORT`; run.sh alone uses 8900, 8082 and 30000). Ray and Slime choose their own free ports, apart from Ray's dashboard agent (52365), whose clash Ray logs and carries on without it.
