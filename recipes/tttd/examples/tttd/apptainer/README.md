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

The job ends with `SMOKE PASSED` or `SMOKE FAILED`. The reports, logs, commit records and the best program are copied to `$REEF_WORK_ROOT/smoke-results/<job id>/`. Checkpoints go to the node's local disk when it has 600 GB free, and are deleted when the job ends; otherwise they go under `REEF_WORK_ROOT` and stay.

Variables: `SMOKE_METHODS` (default `"ppottt-smoke spottt-smoke"`), `TTTD_TASK` (default `circle_packing_26`), `REEF_WORK_ROOT` (default `~/orcd/scratch/reef-work`), `PHASE_TIMEOUT_S` (default 7200).

## Full runs

`run.sbatch` runs one method. `TTTD_RUN_STEPS` ends a job cleanly after that many steps in total, so a 50-step run can be split across jobs and resumed by resubmitting with a larger value:

```bash
TTTD_METHOD=spottt TTTD_TASK=circle_packing_26 TTTD_RUN_STEPS=10 \
    sbatch --gres=gpu:2 recipes/tttd/examples/tttd/apptainer/run.sbatch
```

## Several runs on the best GPUs

A run goes at the pace of its slowest GPU, and Slurm hands a job whichever GPUs are free. On node2500 three GPUs overheat under load: at about 93 C the hardware cuts their clock to 345 MHz and a sustained bf16 matmul reaches about 120 TFLOPS, against 600-660 on the others. `multi.sbatch` takes all the GPUs its runs need, ranks them under a minute of load (`gpu_check.py --rank`) and gives the fastest to the runs listed first:

```bash
TTTD_RUNS="spottt:2 ppottt:4" TTTD_LOCAL_STATE=ppottt \
    sbatch --nodelist=node2500 --gres=gpu:h200:6 --mem=1024G recipes/tttd/examples/tttd/apptainer/multi.sbatch
```

`TTTD_RUNS` lists `method:gpus` or `method:gpus:steps`; a run without a step count goes on until the job's time limit, and the next job resumes it. The methods in `TTTD_LOCAL_STATE` keep their state on the node's local disk, outside your quota. Each run writes `reef-multi-<job>-<method>.out`.

## Checking a run

Each start of a stack moves the previous run's `reef.log` and every service log, including each service's own `stack/<service>/<service>.log`, into `logs/<time>/` under the state directory: a service appends to its own log across restarts, and a new stack reads it from the start into `reef.log`, so otherwise an earlier run's errors reappear in the next run's log.

`run.sbatch` starts with [`gpu_check.py`](gpu_check.py), which prints each GPU's bf16 matrix-multiply rate, which GPUs reach each other directly, the NVIDIA topology and the NCCL all-reduce bandwidth, the traffic tensor parallelism adds to every training layer. It also runs next to a live job (`srun --jobid=<job> --overlap`), where the rates are lower bounds.

Both batch scripts also write every GPU's temperature, clock, power, use and slowdown reasons once a minute to `reef-run-<job>-gpus.csv` or `reef-multi-<job>-gpus.csv` next to the job's output, and `multi.sbatch` prints the PCI bus of each run's GPUs, so a slow step can be matched to a hot GPU afterwards.

`bash status.sh` shows everything about your running job from the login node: for each of its runs, the committed steps, the current step and how long it has run, how long since its log last moved (over half an hour of silence means stuck, not slow) and any failure lines. It reads runs kept on the node's local disk too.

After each step the harness writes every attempt to `attempts/step-NNNN.jsonl.gz` under the state directory: the parent it started from and that parent's score, its score and program, the judge's reason for a rejection, response and reasoning length, whether the token limit cut it off, and how long generation and scoring took. The first line of each file holds the parents PUCT chose (score, visits, depth, PUCT score) and the archive after the step. The step's `tttd_step_committed` event carries a summary: share of attempts beating their parent, attempts beating the best so far, share of distinct programs, failures by reason, token and time costs, and the time spent searching and waiting for training.

`run_report.py <state directory>` prints three tables per run from these files and Reef's commits. Scores: failure rate, mean of valid attempts, median, top-10% mean, the step's best, and how often attempts beat their parent or the best so far. Cost: step time split into search and training wait, tokens, truncation and time per attempt. Training: entropy, KL to the base, clip fraction and gradient norm, and how well each method's baseline predicted the step's scores (explained variance, 1 perfect and 0 no better than a constant): SPO-TTT's tracker, and PPO-TTT's critic before it trained on the step (`value EV`) and after (`critic EV after fit`), with the critic's loss at its first and last update. A fourth table counts failures by reason. Every column also goes to `report/steps.csv` for plotting. Runs started before the step files existed fall back to Reef's records, which each run of the script copies to `report/attempts.csv`, since Reef deletes consumed records after its retention period (the configs keep 90 days).

`run_status.py <state directory>` prints one row per committed step: mean and best reward, step time, training throughput, allocator retries, the trainer-sampler log-prob gap before any update, and the sampled policy's KL to the base. It reads only files, so it runs on the login node while the job runs.

## Known limits

- The SPO-TTT smoke has passed on an Engaging H200 node (all five stages, about 13 minutes). With thinking off, the smoke's attempts all scored 0, so its update had zero advantages: it checks the pipeline, not learning. The scripts are also tested on CPU with a stand-in for Apptainer and the GPU stack (`tests/test_tttd_smoke_job.py`, `tests/test_tttd_local_driver.py`).
- Speed: on two H200s the first SPO-TTT step on circle_packing_26 took about 3.3 hours: 59 minutes writing and scoring the 512 attempts (about 890 generated tokens/s per GPU), 18 minutes of frozen-base log-probabilities and 2 hours of training at about 900 tokens/s, against about 8,300 tokens/s in the published run on two B200s. The cause of the slow training is not yet known; each committed step now records the trainer's allocator retries and throughput (`run_status.py`), which the next full step will show. The trainer's `max_split_size_mb:512` stays: torch_memory_saver refuses expandable segments, and with the allocator's defaults the first adapter publication to SGLang failed (`CUDA error: invalid argument` opening the shared tensors), consistent with the buffers being carved out of large blocks the memory saver owns, which CUDA IPC cannot share. SGLang's `--sglang-mem-fraction-static` is 0.7, Slime's colocated default, instead of the B200 config's 0.5, which left an H200 room for the KV cache of about 25 full-length packing attempts at once against 128 requested.
- Host memory: the SPO-TTT smoke peaked at about 168 GiB. The first PPO-TTT smoke was killed at a 256 GB limit after loading the actor and the critic, whose full parameters and optimiser state are offloaded to host memory while idle, so `smoke.sbatch` asks for 768 GB and a PPO-TTT `run.sbatch` job needs `--mem=768G`.
- Checkpoint space: Reef's size limits read a shared filesystem's total size, not your quota, so on Engaging they never delete anything. Every config therefore keeps only the latest checkpoint (`max_count: 1`), and the disk holds at most two while the next is written. A PPO-TTT checkpoint carries a full 8B critic with its optimiser state (from the arithmetic, over 100 GB; not yet measured), so a PPO-TTT run needs room for two of those in your quota.
- Several jobs can share a node. Each job takes its own ports for Reef, the judge and the SGLang router (`TTTD_REEF_PORT`, `TTTD_JUDGE_PORT`, `TTTD_ROUTER_PORT`; run.sh alone uses 8900, 8082 and 30000) and its own 256-port block where Slime starts probing for the SGLang engines' and Megatron's ports (`REEF_PORT_BASE`). Slime probes before it binds, so without the block two stacks starting together took the same engine port and one failed. The blocks are derived from the job id (see `common.sh`); Ray's dashboard agent keeps its fixed port 52365, whose clash Ray logs and carries on without it.
