# ppottt

PPO on TTT-Discover's test-time search, as a Reef weight-training recipe. The search, tasks, verifiers and sampling budget are those of [tttd](../tttd/README.md); what changes is how a scored attempt becomes a gradient. TTT-Discover forms each advantage from a group of sibling attempts at the same parent. PPO-TTT learns a scalar critic instead and trains every attempt against it with a clipped policy objective, so a search step no longer needs siblings to produce a learning signal.

- Methods: [PPO](https://arxiv.org/abs/1707.06347) with [GAE](https://arxiv.org/abs/1506.02438), applied to [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- Pins: the `slime` pin shared by every recipe (`pyproject.toml` `dependency-groups.runtime`); the shipped config uses `Qwen/Qwen3-8B` with thinking, a rank-32 LoRA actor and a full-parameter critic
- Claim scope: an implementation with CPU-verified numerics and configs. No GPU run has been recorded yet; see [the example](../tttd/examples/tttd/README.md) for how to run one.

## Layout

```text
ppottt/
  recipe.py        PPOTTTRecipe: training spec, loss family "ppottt", grid and update schedule
  processor.py     the step barrier, emitting one flat batch per search step
  report.py        PPOTTTRolloutReport, the step-grid report contract
  batches.py       ScheduledPolicyBatch, a policy batch that carries its PPO schedule
  preparer.py      schedules the step; ships no advantages
  slime/           loss family, critic cadence, and the terminal-reward GAE hook
```

The example lives with TTT-Discover's, since the search is shared: `TTTD_METHOD=ppottt` selects `serve.ppottt.yaml` in [`../tttd/examples/tttd`](../tttd/examples/tttd/README.md), and `TTTD_METHOD=ppottt-smoke` a one-step 2x2 check of the integration path.

## What differs from tttd

| | tttd | ppottt |
| --- | --- | --- |
| Baseline | leave-one-out mean of the sibling group's entropic weights | learned value of each response prefix |
| Objective | adaptive-beta entropic utility | expected reward |
| Policy loss | un-clipped importance sampling, one step per batch | Slime's clipped `policy_loss`, `minibatch_size` rollouts per step |
| Constant-reward groups | dropped | kept: they still carry signal against the critic, and train it |
| Attempts per parent | at least 2 | at least 1 |
| KL to the frozen base | centred on the advantages | `-kl_coef * KL` in the per-token reward, actor only |
| GPUs in the shipped config | 2 | 4 (the critic is a second full model) |

Two of those rows change more than the baseline: the objective and the loss. A comparison that wants to isolate the critic should read ppottt against `tttd-mean`, TTT-Discover's grid with a group-mean baseline, not against `tttd` alone; `TTTD_METHOD=tttd-mean` runs it.

## Design notes

- **Episode.** One rollout is one attempt: the prompt holds the parent solution, the response is thinking plus a program, the verifier scores it once. The score goes on the last response token and GAE runs over the response; with `--gamma 1 --lambd 1` each token's advantage is `score - V(token prefix)`. The two parameters stay configurable so a variant that treats a chain of archive states as a multi-step trajectory can reuse the hook.
- **Critic.** Reef trains LoRA on the actor only, so the critic is a full copy of the base with a zero-initialised value head, colocated on the actor GPUs and offloaded while idle. It trains at `--critic-lr 5e-6`, twice per actor step, and alone for the first `--num-critic-only-steps 2` steps, when the search still runs and every attempt still enters the archive. Its regression target is the verifier score: the family zeroes the critic role's KL coefficient. `--critic-save` gives it its own checkpoint root, so a restart does not cold-start the value head.
- **Scale.** Late in an Erdős run, valid programs differ in reward by 1e-4 or less while an invalid one scores 0. `--normalize-advantages` whitens across the step; without it the update is dominated by "do not crash". TTT-Discover's entropic weights are invariant to that scale; PPO is not.
- **Schedule.** The processor stamps `ppo_epochs`, `minibatch_size` and `shuffle_minibatches` on the batch and the preparer turns them into Reef's `StepScheduling`. Old-policy log-probabilities are the sampler's, recorded at generation, so every minibatch after the first is mildly off-policy and the clip bounds it.
- **Verifier budget.** Critic training consumes only rewards the verifier has already produced, so at a fixed number of verifier calls the critic costs compute, not samples.

## Where the rest is documented

[The ppottt recipe page](../../docs/user-guide/recipes/ppottt.rst) covers configuration and metrics; [the tttd page](../../docs/user-guide/recipes/tttd.rst) covers the search, tasks and run controller this recipe reuses.
