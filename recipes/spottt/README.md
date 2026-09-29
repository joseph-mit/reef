# spottt

[Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232) on TTT-Discover's test-time search, as a Reef weight-training recipe. The search, tasks, verifiers and sampling budget are those of [tttd](../tttd/README.md). Each attempt's baseline is a forgetting estimate of what earlier attempts from the same archive state scored, so a step needs neither the sibling group TTT-Discover compares within nor the critic [ppottt](../ppottt/README.md) learns.

- Methods: SPO's value tracker with archive states in place of prompts, applied to [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)
- Pins: the `slime` pin shared by every recipe; the shipped config is tttd's, Qwen3-8B with a rank-32 LoRA on two GPUs
- Claim scope: an implementation with CPU-verified numerics and configs. No GPU run has been recorded yet.

## Layout

```text
spottt/
  tracker.py       the forgetting tracker and advantage assignment, torch-free
  recipe.py        SPOTTTRecipe: training spec, loss family "spottt", tracker and schedule settings
  processor.py     PPO-TTT's step barrier, attaching each attempt's parent and grandparent
  report.py        SPOTTTRolloutReport: PPO-TTT's grid coordinates plus parent ids
  batches.py       SpotttBatch: the scheduled batch plus tracker settings
  preparer.py      advantages from the tracker; the tracker rides the committed algorithm state
  slime/           loss family on Slime's clipped policy_loss, with tttd's frozen-base KL hook
```

Run it from the tttd example with `TTTD_METHOD=spottt`, or `TTTD_METHOD=spottt-smoke` for a two-step 2x2 check of the integration path.

## The tracker

SPO keeps a Beta posterior over a prompt's success rate and discounts it by `rho = 2^(-D / D_half)` before each observation, where `D` is how far the policy has moved; the posterior mean is an adaptive moving average with step `1 / (rho * N + 1)`. SPO-TTT keeps that rule and changes three things.

- **Keys.** An exact state almost never repeats, so a prompt-indexed table would always be cold. `baseline: node` keys the estimate by the archive state an attempt was expanded from; a state seen for the first time starts from its parent's estimate at the start of the step, with `inherit_fraction` of its weight, and falls back to the task-level estimate. `baseline: task` keeps one estimate for the test instance. `baseline: none` subtracts nothing.
- **Rewards.** Continuous: the update `m' = m + (k / n') (mean - m)` with `n' = rho n + k` is exactly the Beta posterior mean when rewards are 0 or 1, and the same moving average otherwise.
- **Forgetting.** Once per policy version, not once per observation: siblings sampled in one step come from one policy, and discounting between them would forget observations no policy change touched. The drift is the number of committed steps since the key was last updated, so `rho = 2^(-steps / half_life)`, clipped to SPO's `[0.875, 0.96]`.

Every baseline is read before the step updates any key, so it never depends on the reward it is subtracted from. The first step, when every key is cold, centres on the step's mean. Advantages are then normalised across the step, as in SPO, with each attempt weighted equally; PPO-TTT uses Slime's per-token whitening, which weights long attempts more, so the two recipes differ there as well as in the baseline. When a step leaves the tracker over its 4,096-key cap, the keys updated longest ago are dropped once the whole step is folded in.

## What it costs

No critic and no extra model: the stack is tttd's, on two GPUs. The tracker holds three numbers per archive state it has seen, at most 4096 keys, and travels with the algorithm state Reef commits at every step.

## Where the rest is documented

[The spottt recipe page](../../docs/user-guide/recipes/spottt.rst) covers configuration and metrics.
