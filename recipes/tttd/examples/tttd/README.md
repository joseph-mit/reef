# TTT-Discover on Reef

This example implements the harness side of
[Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175). Its
code is intentionally split so the Reef integration is visible and removable:

The [TTT-Discover guide](../../../../docs/user-guide/recipes/tttd.rst) explains the runtime
sequence, a reduced smoke, trajectory configuration, and recovery behavior. This
README records the example's implementation details, paper fidelity, and completed
reproduction results.

```text
harbor/               self-contained reef-eval/Harbor task definitions
  erdos_min_overlap/
  circle_packing_26/
  circle_packing_32/
    task.toml            metadata, timeouts, resource limits
    instruction.md       exact task prompt shown to the model
    environment/         container, judge service, canonical verifier
    tests/test.sh        asks the judge for the final result, writes reward
harness/              agent harness (PUCT search + Reef adapter)
  search.py             Scorer callable + PUCT archive + search algorithm
  scorer.py             JudgeScorer (HTTP) + ProgramScorer (direct sandbox)
  sandbox.py             isolated subprocess execution for generated programs
  agent.py              ReefTTTDiscoverHarness rollout/report adapter
  run_controller.py     training barrier + paired PUCT resume state
  harbor_agent.py       Harbor BaseAgent (imports harbor package)
  methods.py            the training methods TTTD_METHOD selects, and their state paths
  siblings.py           how many attempts each parent gets (fixed grid or adaptive)
  step_records.py       every attempt of a step, written to attempts/step-NNNN.jsonl.gz
  session.py            the harness and run controller both entry points build
serve.yaml            Reef + Ray + Slime/Megatron + SGLang stack config
serve.ppottt*.yaml    the same stack trained with PPO and a critic (recipes/ppottt)
serve.spottt*.yaml    ... with a baseline from past outcomes (recipes/spottt)
serve.espottt*.yaml   ... with TTT-Discover's entropic advantage over past outcomes (recipes/espottt)
serve.hspottt*.yaml   ... SPO-TTT plus credit for attempts that later led to a success (recipes/hspottt)
serve.tttd-mean.yaml  ... with a group-mean baseline, the control for both
serve.search-only.yaml  the search on frozen weights, the floor of a comparison
run.py                one reef-eval episode owning the complete TTT trajectory
run_local.py          the same trajectory without Harbor or Docker (TTTD_DRIVER=local)
apptainer/            Apptainer image and SLURM jobs for clusters without Docker
run.sh                starts the reef training stack, then runs run.py
pyproject.toml        makes the harness importable
results/              formal result data, generated programs, and plots
```

## The ordinary harness

`TTTDiscoverHarness` accepts the model call that an ordinary harness already
has. Evaluation stays local:

```python
from recipes.tttd.examples.tttd import TTTDiscoverHarness

harness = TTTDiscoverHarness(call_model, MyProblem(), "task instruction", model="my-model")
best = harness.run(steps=50)
```

Its single-rollout path is:

```python
response = self.generate(self._request_payload(parent))
return self._evaluate_response(parent, response)
```

There is no report hook, transport protocol, receipt wrapper, or other
Reef-shaped abstraction in the ordinary harness.

## The changes needed for Reef

Use the concrete Reef harness instead:

```python
from reef_client import ReefClient
from recipes.tttd.examples.tttd import ReefTTTDiscoverHarness

harness = ReefTTTDiscoverHarness(
    ReefClient("http://reef-host:8900", token="secret"),
    MyProblem(),
    "task instruction",
    scenario="my-single-discovery-problem",
    recipe="tttd",
    release_id="checkpoint-v42",  # optional starting checkpoint
    model="reef",
)
best = harness.run(steps=50)
```

`ReefTTTDiscoverHarness` reuses the request construction, evaluation, archive,
and PUCT mechanics. Its overridden `_rollout` method contains the four actual
integration changes in one place:

1. Send inference through a scenario-scoped Reef endpoint.
2. Retain the returned `agent_record_id` as the generation receipt.
3. After the existing evaluator computes a reward, send a Reef report that
   references that exact inference receipt.
4. Include the rollout group's `comparison_set` so the scenario processor can
   construct grouped training data.

The ordinary harness and problem do not import `ReefClient`, know the scenario
name, retain inference IDs, or construct Reef report payloads.

The runtime flow is:

```text
PUCT chooses G parents
  -> harness submits G × R OpenAI-compatible chat requests
  -> the TTTD inference backend renders each prompt once and calls SGLang /generate
  -> Reef stores exact tokens, response loss mask, and rollout log-probabilities
  -> sandbox executes each generated program; the official verifier gives rewards
  -> harness reports every reward against its exact inference receipt
  -> TTTDProcessor waits for the complete G × R step and removes constant groups
  -> Slime prepares adaptive-beta entropic leave-one-out advantages from the reserved batch
  -> Slime applies frozen-base token KL and un-clipped importance sampling
  -> Megatron performs one optimizer step and synchronizes weights to SGLang
  -> valid children update the local PUCT archive
  -> the Harbor controller waits for the durable scenario commit and new runtime load ID
  -> the post-step PUCT archive is atomically paired with that committed version
```

## Included paper problems

`harbor/erdos_min_overlap/environment/score.py` adapts the
[official Erdős minimum-overlap environment](https://github.com/test-time-training/discover/tree/main/examples/erdos_min_overlap),
one of the paper's mathematics tasks. It keeps the same:

- step-function representation using samples over `[0, 2]`;
- constraints `0 ≤ h[i] ≤ 1` and `sum(h) = n/2`;
- full-correlation calculation of the upper bound `C₅`;
- continuous reward `1 / (1e-8 + C₅)`.

As in the reference implementation, the model writes a Python search program
whose `run()` function returns `(h_values, c5_bound, n_points)`. The program is
run in a subprocess with a wall-clock timeout, CPU/thread limits, network
access governed by the deployment, and an isolated temporary working
directory. The task prompt, seed construction, program contract, normalization
semantics, verifier, continuous reward, and PUCT search value are ported from
[`test-time-training/discover@6c40e82`](https://github.com/test-time-training/discover/tree/6c40e82dab9d5de7416ac873ad5cd3106084aaed).

Acknowledgement: this task is adapted from the MIT-licensed TTT-Discover
implementation by Mert Yuksekgonul and collaborators. Full attribution and
license terms are included below.

`harbor/circle_packing_26/` and `harbor/circle_packing_32/` adapt the two
circle-packing tasks from the same paper. In each task, the model writes
`run_packing()`, which returns circle centers, radii, and a reported sum. The
judge independently requires the selected number of circles, checks square
boundaries and pairwise non-overlap, rejects non-finite values, and computes
the reward directly as `sum(radii)`. The reported sum is never trusted. Both
task prompts preserve their corresponding initial TTT-Discover prompt byte for
byte.

## Judge slots and timeouts

Each task's judge runs at most `max_concurrent_submissions` programs at once
(`environment/judge_config.json`: 32 for Erdős, 64 for packing) and queues the
rest, while the harness runs up to 512 attempts at once (256 for packing) and
gives each judge request two hours. Both entry points now send a request only
when a judge slot is free, so the two hours count from when the judge starts
the program, not from when the request joined its queue. Before this, a slow
Erdős step (512 attempts through 32 slots at up to 1,100 s each, about 4.9 h)
could score late attempts 0 while the judge was still running them. Packing
runs were not affected: their worst-case queue is about 35 minutes.

## Choosing a task that shows learning

A task shows what learning adds only when search without training falls well
short of search with training at the same number of attempts. The
`search-only` method is that comparison: run it next to a training method.

26-circle packing is close to saturated for Qwen3-8B. In our SPO-TTT and
PPO-TTT runs, step 1 (512 attempts from the untrained model) already reached
2.624 and 2.627 against the best known 2.635983, and most of what training
improved afterwards was the failure rate (83% to 13% for SPO-TTT). ThetaEvolve
reports the same for an 8B model (DeepSeek-R1-0528-Qwen3-8B): search alone
reached 2.6359831 and search with RL 2.6359857; its second and third
autocorrelation tasks moved by under 1%
([ThetaEvolve, Table 2](https://arxiv.org/abs/2511.23473)).

Candidates with more room, all from
[TTT-Discover](https://arxiv.org/abs/2601.16175) with gpt-oss-120b; none has
a published Qwen3-8B result, and neither is ported here yet (both environments
are in the [TTT-Discover repository](https://github.com/test-time-training/discover)):

| Task | Without training | With TTT-Discover | Best human | Judge |
| --- | ---: | ---: | ---: | --- |
| AtCoder AHC058 (higher is better) | 772,429,752 (best of 25,600 samples) | 848,414,228 | 847,674,723 | CPU, 2 s per test case |
| TriMul kernel on H100 (µs, lower is better) | 2,060.70 (same search, no training) | 1,203.10 | 1,371.1 | GPU, timed |

TriMul's row is the cleaner test, since only the training differs; its judge
needs a GPU, and the leaderboard has no H200 entry. AHC058's judge needs only
CPUs, but many test cases per attempt, so size the judge slots to the CPUs the
training stack leaves free. For quick ablations, packing with a smaller model
still shows a clear gap: ThetaEvolve's ProRL-1.5B-v2 reached 2.1343 with
search alone and 2.5225 with RL after 200 steps.

## Ideas for beating the best known results

These are not implemented; each names the change it would need.

- **Start from the best known solution.** The archive starts from empty
  seeds, so a run can only reach the best known result by finding it again.
  Seeding it with the published best program puts the whole budget on
  improving it. TTT-Discover's Qwen3-8B comparison lists ThetaEvolve with
  this reuse at 1.50314 on the first autocorrelation inequality, against
  1.50681 without.
  Needs: a seed program option scored once at start-up and kept in the
  run's saved search state.
- **Stop rewarding repeats.** From step 16 of our SPO-TTT run, 10 to 14% of
  attempts per step land within 1e-4 of the best score, most likely the
  same packing found again. Training keeps rewarding them, which narrows the
  search just when new ideas are needed. The harness already
  reports each program's key and its parent's; an attempt whose result
  matches one already in the archive could get its parent's reward instead
  of its own.
- **Polish the top of the archive.** On packing and the autocorrelation
  tasks the last digits come from numerical refinement, not new ideas.
  Running the best few archive programs with a longer time budget between
  steps, outside the 512 attempts, would test whether the remaining gap is
  search or precision.
- **Shrink steps once the archive stops improving.** Adaptive siblings
  (below) already decide how many attempts each parent needs. A step could
  likewise get smaller once the best score stops moving, trading attempts per
  step for more training steps, which is the batch-size question for PPO-TTT.

## Setup (once)

```bash
git submodule update --init third_party/reef-client
pip install -e ./third_party/reef-client
pip install -e .
```

```bash
./run.sh
```

That runs `harbor/erdos_min_overlap` on the paper grid: 8 groups of 64
rollouts, one optimizer step, thinking enabled, two GPUs. Nothing has to be
exported first except `TTTD_TASK` — `run.py`, `harness/harbor_agent.py`, and
`serve.yaml` each write out the values they use.

Reef starts and stops the shared Ray runtime automatically; no `ray start`
or fixed Ray port is needed. `run.sh` defaults the local cluster's GPU pool to
`CUDA_VISIBLE_DEVICES=0,1`; override it at launch to choose different GPUs.
`training.num_gpus` still sets Slime's model topology. To use an existing
cluster, set `RAY_ADDRESS`; its nodes control GPU visibility and Reef leaves
it running on exit. The local Slime driver does not reserve model GPUs itself.

We recommend allocating at least 256 GiB of host memory to the reference
8 × 64 setup. With less memory, reduce evaluator concurrency or request a
larger allocation to avoid stalls or termination.

### Another problem, or a smaller grid

`harbor/circle_packing_26` and `harbor/circle_packing_32` are bundled as well.
`TTTD_TASK` selects one:

```bash
TTTD_TASK=circle_packing_26 ./run.sh
```

`run.sh` gives each task its own scenario and `work/<task>/` state directory,
and sizes the memory limits for it — the packing tasks need a longer context
(`32768`) on the same GPUs, so they get a smaller per-GPU token budget.

The step grid has one source: the harness reads `groups_per_step`,
`rollouts_per_group`, `steps`, `max_new_tokens` and `enable_thinking` from the
same config the stack runs (`harness/session.py`), and Slime's
`--global-batch-size` in that config must equal the grid. A plumbing smoke sets
a 2 x 2 grid and `enable_thinking: false`, so a short completion is not spent
entirely in the reasoning channel before it emits a program.

`work/erdos_min_overlap/` holds this problem's checkpoints, artifacts,
scenario records, and PUCT state; a second problem needs its own directory so
the two cannot mix. `work/erdos_min_overlap/tttd-search-state.json` records a
pending archive immediately after a search step and marks it committed only
after Reef's durable training transaction advances the scenario and publishes
a new serving runtime load ID.
Restarting with the same scenario, task instruction, and sampling settings
resumes that paired state; a missing or mismatched step fails closed instead
of silently restarting PUCT against a later model checkpoint. Serving weight
versions are session scoped, so a correctly restored step may rebind the saved
archive to the new engine version.

Use a fresh scenario for each discovery problem: TTT-Discover fine-tunes on one
test problem rather than learning a general task policy.

To add another problem, create a sibling under `harbor/`, write a `score.py`
with a `grade(artifact)` function, and point the values in the table above at
it. The harness scores every generated solution by POSTing it to the task's
judge, so no task-specific imports enter the shared harness.

## Other training methods

The search does not depend on how its attempts are trained, so the same
harness, tasks and verifiers drive other recipes. `TTTD_METHOD` picks one:

| `TTTD_METHOD` | Config | Recipe | GPUs |
| --- | --- | --- | --- |
| `tttd` (default) | `serve.yaml` | TTT-Discover, grouped entropic advantages | 2 |
| `tttd-mean` | `serve.tttd-mean.yaml` | TTT-Discover's grid and loss with a group-mean baseline | 2 |
| `ppottt` | `serve.ppottt.yaml` | [PPO-TTT](../../../ppottt/README.md): clipped PPO with a scalar critic | 4 |
| `ppottt-smoke` | `serve.ppottt-smoke.yaml` | two 2x2 PPO-TTT steps with thinking off | 4 |
| `spottt` | `serve.spottt.yaml` | [SPO-TTT](../../../spottt/README.md): baseline from past outcomes, no critic | 2 |
| `spottt-smoke` | `serve.spottt-smoke.yaml` | two 2x2 SPO-TTT steps with thinking off | 2 |
| `spottt-adaptive` | `serve.spottt-adaptive.yaml` | SPO-TTT with adaptive siblings (below) | 2 |
| `espottt` | `serve.espottt.yaml` | [Entropic SPO-TTT](../../../espottt/README.md): TTT-Discover's entropic advantage against past outcomes | 2 |
| `espottt-smoke` | `serve.espottt-smoke.yaml` | two 2x2 entropic SPO-TTT steps with thinking off | 2 |
| `espottt-adaptive` | `serve.espottt-adaptive.yaml` | entropic SPO-TTT with adaptive siblings | 2 |
| `espottt-adaptive-smoke` | `serve.espottt-adaptive-smoke.yaml` | two 2x2 steps of the above with thinking off | 2 |
| `hspottt` | `serve.hspottt.yaml` | [Stepping-stone SPO-TTT](../../../hspottt/README.md): SPO-TTT plus credit for attempts whose descendants succeed | 2 |
| `hspottt-smoke` | `serve.hspottt-smoke.yaml` | two 2x2 steps of the above with a one-step window | 2 |
| `search-only` | `serve.search-only.yaml` | the same search on frozen weights: reports are not trained on | 2 |

```bash
TTTD_METHOD=ppottt-smoke ./run.sh   # integration check of the whole path
TTTD_METHOD=ppottt ./run.sh         # 50 steps of 8x64 on Erdős
```

Read against each other, the methods separate what each part contributes at
one verifier budget: `search-only` is the floor PUCT reaches without learning,
`tttd` and `tttd-mean` differ only in the objective, and `ppottt` and `spottt`
replace `tttd-mean`'s sibling group with a critic or with past outcomes.
`serve.yaml` ships TTT-Discover's one-step default; set `training.steps: 50`
there to match the other configs, which ship the paper's 50 steps.
`espottt` keeps TTT-Discover's advantage and loss but compares each attempt
with past outcomes from the same state instead of its siblings; its
`baseline: siblings` setting is TTT-Discover's advantage exactly, the control.
`hspottt` trains each step as `spottt` does and also replays, a few steps
later, the attempts that a later success grew from; to trace that, the harness
reports every attempt's program key and its parent state's.

The three changes are separate switches, so each can be measured on its own
against its base: `espottt` against `tttd` (where the comparison set comes
from), `hspottt` against `spottt` (stepping-stone credit), and adaptive
siblings against the same method on the fixed grid (`spottt-adaptive` against
`spottt`, `espottt-adaptive` against `espottt`). Adaptive siblings combine
with any method that does not compare a fixed sibling group; add a
`search.siblings` block to a config to combine them with `hspottt`.

### Adaptive siblings

TTT-Discover's 64 attempts per parent both try to improve on it and measure
what is normal from it. A method that learns what is normal from history needs
the second job only where history is thin. A config's `search.siblings` block
(read by the harness, not by Reef) with `policy: adaptive` spends the same
512 attempts accordingly: each parent, in PUCT's order, gets what brings the
outcomes known from it up to `target`, between `min_siblings` and
`max_siblings`, until the budget is spent, so more parents are tried per step.
A parent never expanded starts from `inherit_fraction` of what is known about
its own parent, as the Reef-side histories do; known outcomes fade with
`half_life` committed steps; and a parent whose new scores sit `surprise_z`
standard errors from what was known is measured again. Each attempt keeps an
address in the 8x64 grid, which then no longer means one parent, so
TTT-Discover's grouped methods refuse it. Absent the block, or with `policy:
fixed`, the search is TTT-Discover's grid.

Every method other than `tttd` keeps its state under `work/<method>/<task>/`
and its scenario as `<method>-<task>`, so runs of different methods never share
checkpoints, artifacts or a search archive; `tttd` keeps its original
`work/<task>/` layout. `harness/methods.py` and `run.sh` must agree on a new
method's config name, and a test checks that they do, and that each config's
grid equals its driver's `--global-batch-size`.

On a SLURM cluster, run the script inside one allocation that holds the whole
stack, for example:

```bash
srun --gres=gpu:4 --cpus-per-task=64 --mem=256G --time=48:00:00 --pty bash
cd recipes/tttd/examples/tttd && TTTD_METHOD=ppottt ./run.sh
```

A run survives a restart of the job: the harness pairs its PUCT archive with
Reef's durable training commit, so launching the same method and task again
resumes at the last committed step. The single-stream configs checkpoint every
step for that reason; `serve.yaml` keeps TTT-Discover's interval of two.

Where there is no Docker, as on most HPC clusters, `TTTD_DRIVER=local ./run.sh`
runs the task's judge as a local process instead of through Harbor
(`run_local.py`), and [`apptainer/`](apptainer/README.md) has the image
recipe and SLURM jobs for Engaging: setup, a two-step smoke with a SIGKILL and
resume between the steps, and full runs.

## Paper fidelity and training ownership

The integration reproduces:

- 8 comparison groups per training step and 64 rollouts per group by default;
  both cardinalities are configurable and the reference setting uses 8 × 64;
- one PUCT-selected initial state shared by each group;
- rank-prior PUCT with maximum-child `Q`, ancestor visit backpropagation, and
  full-lineage diversity blocking;
- the top 2 children per expansion and a top-1000 archive that retains seeds;
- invalid-action reward 0 by default;
- the complete-step barrier and post-barrier removal of constant-reward groups;
- adaptive-beta entropic leave-one-out advantages, using the reference Torch
  float32 bisection procedure;
- frozen-base centered token KL in Slime;
- Tinker's full-batch, token-sum, un-clipped importance-sampling policy loss
  rather than PPO clipping or per-trajectory token averaging;
- Tinker's sampling defaults (`temperature=1`, `top_p=1`, `top_k=-1`)
  explicitly, rather than SGLang's model-specific defaults;
- the reference Adam settings (`lr=4e-5`, betas `0.9/0.95`, `eps=1e-8`,
  zero weight decay, and no gradient clipping);
- raw continuous rewards linked to exact inference tensors rather than
  reconstructed from ordinary HTTP chat JSON;
- Megatron Bridge LoRA with the paper configuration (`rank=32`, `alpha=32`)
  on QKV, attention output, and both MLP projections. The base parameters remain
  frozen in both runtimes. Reef synchronizes the serving-native `lora_A` and
  `lora_B` adapter tensors to SGLang.

The default HTTP backend cannot manufacture training tensors from response
text. The deployment therefore selects Reef's token-native SGLang chat
backend. The public harness route remains `/v1/chat/completions`; inside the
inference backend, Reef renders the prompt once, calls SGLang `/generate`, and
records the engine's sampled token IDs and rollout log-probabilities. Reef's
weight surface names the scenario's own adapter revision on every request;
the harness never selects one and the backend never injects a fallback.

New Slime configs should use the
`recipes.tttd.slime.objective.*` custom-function paths.
The former `reef_adapters.tttd.*` paths
were removed; existing deployment configs must migrate to the backend-owned
module path.

## Earlier Erdős 50-step experiment

We ran a result-level reproduction of the Qwen3-8B Erdős experiment from
[Learning to Discover at Test Time](https://test-time-training.github.io/discover.pdf).
The run used 50 optimizer steps, Qwen3-8B thinking, the two-phase completion
policy, a 26,000-token phase-one budget, the grouped entropic objective,
rank-32 LoRA, and the paper's Adam learning rate. It used eight groups of 32
rollouts on one four-B200 node. The paper's experiment used 64 rollouts per
group, so this run used half of its rollout budget.

### Setup

| Setting | Value |
| --- | --- |
| Task | Erdős minimum-overlap program discovery |
| Model | `Qwen/Qwen3-8B`, thinking enabled |
| Runtime | Reef, Slime, Megatron, and SGLang LoRA serving |
| Hardware | `4 × NVIDIA B200`, using TP2 × DP2 |
| Search grid | 8 groups × 32 rollouts |
| Training | 50 steps, LoRA rank and alpha 32, Adam learning rate `4e-5` |
| Sequence policy | 30,000-token window, 26,000-token phase-one budget, and two-phase completion |
| Evaluation | 1,000-second program budget and 1,100-second timeout |
| Checkpointing | One checkpoint per step, retaining the latest checkpoint |

### Results

| Completed steps | Runtime | Best certified C₅ upper bound | Best reward |
| ---: | ---: | ---: | ---: |
| 50/50 | approximately 22.6 hours | `0.380916` | `2.625249` |

Table 2 of the paper reports `0.380932` for TTT-Discover with Qwen3-8B on the
same Erdős task. The verifier minimizes this bound. Search outcomes depend on
sampling, so the two numbers describe separate trajectories with different
rollout counts.

## Formal 8x64 results

The three trajectories used `Qwen/Qwen3-8B` with thinking enabled on two NVIDIA
B200 GPUs. Each search step contained eight groups of 64 rollouts, followed by
one rank-32 LoRA update. Each task has one trajectory, so these results do not
estimate variance across seeds.

### Erdős minimum overlap

| Steps shown | Certified result | TTT-Discover | Target |
| ---: | ---: | ---: | ---: |
| 25 | `0.38094` | `0.38093` | `0.38080` |

The certified result is the `C₅` upper bound, so lower values are better. The
curve uses the first 25 committed archive states.

![Best certified Erdős solution found by iteration](results/formal-8x64-v3-erdos/best_solution_history.png)

[Erdős result details](results/formal-8x64-v3-erdos/README.md)

### Packing 26

| Steps shown | Certified result | TTT-Discover | Target |
| ---: | ---: | ---: | ---: |
| 50 | `2.635983` | `2.635983` | `2.636` |

The certified result is the verified sum of radii, so higher values are better.

![Best certified Packing 26 solution found by iteration](results/formal-8x64-v3-packing/packing26/best_solution_history.png)

[Packing 26 result details](results/formal-8x64-v3-packing/packing26/README.md)

### Packing 32

| Steps shown | Certified result | TTT-Discover | Target |
| ---: | ---: | ---: | ---: |
| 50 | `2.939573` | `2.939572` | `2.940` |

The certified result is the verified sum of radii. It differs from the value
reported by TTT-Discover by less than `8e-7`.

![Best certified Packing 32 solution found by iteration](results/formal-8x64-v3-packing/packing32/best_solution_history.png)

[Packing 32 result details](results/formal-8x64-v3-packing/packing32/README.md)

### Circle-packing configurations and training metrics

The final programs were executed again before their configurations were
plotted. The replay checked the circle count, finite values, square boundaries,
and pairwise non-overlap.

![Verified circle-packing configurations](results/formal-8x64-v3-packing/packing_configurations.png)

W&B recorded 50 committed rows for each packing task. The mean rollout reward
reached its maximum at step 18 for Packing 26 and step 17 for Packing 32. The
best archived Packing 26 result was found at iteration 13. Packing 32 reached
`2.9395727712072386` at iteration 18 and its final value at iteration 24. The
last change was about `2.5e-13`. Sampled-policy KL increased from about `0.0006`
at step 20 to `0.0491` for Packing 26 and `0.0430` for Packing 32 at step 50.

![W&B metrics from the two packing runs](results/formal-8x64-v3-packing/wandb_training_metrics.png)

New runs also record the grid reward distribution and constant-group filtering
counts under the `tttd` W&B namespace. Each `tttd_step_committed` event includes
the archive size and best reward. These fields were added after the two formal
packing runs and are not present in their stored W&B history.

The [circle-packing overview](results/formal-8x64-v3-packing/README.md) contains
the combined W&B history, verified configurations, generated programs,
milestone summaries, and records of how the results were produced.

## Attribution and license

Parts of this example are adapted from
[`test-time-training/discover@6c40e82`](https://github.com/test-time-training/discover/tree/6c40e82dab9d5de7416ac873ad5cd3106084aaed),
including the TTT-Discover search procedure, adaptive-entropic objective,
two-phase completion behavior, Erdős minimum-overlap task, and circle-packing
tasks for `n=26` and `n=32`.

<details>
<summary>Upstream MIT license</summary>

> MIT License
>
> Copyright (c) 2025 Mert Yuksekgonul
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

</details>
