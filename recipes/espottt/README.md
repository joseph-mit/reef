# espottt

Entropic SPO-TTT: [TTT-Discover](../tttd/README.md)'s entropic advantage, with its comparison set taken from a forgetting history of past attempts from the same archive state ([SPO-TTT](../spottt/README.md)'s idea) instead of the step's sibling group. The search, tasks, verifiers, sampling budget and loss are TTT-Discover's.

- Methods: TTT-Discover's adaptive-beta entropic advantage from [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175), with the history rule of [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232)
- Pins: the `slime` pin shared by every recipe; the shipped config is tttd's, Qwen3-8B with a rank-32 LoRA on two GPUs
- Claim scope: an implementation with CPU-verified numerics and configs. No GPU run has been recorded yet.

## Layout

```text
espottt/
  history.py       the weighted reward history and the entropic advantage, torch-free
  recipe.py        ESPOTTTRecipe: training spec, loss family "tttd", history and schedule settings
  processor.py     PPO-TTT's step barrier, attaching each attempt's parent and grandparent
  report.py        SPO-TTT's report under the "espottt" tag
  batches.py       EspotttBatch: the scheduled batch plus history settings
  preparer.py      advantages from the history; the history rides the committed algorithm state
```

Run it from the tttd example with `TTTD_METHOD=espottt`, `TTTD_METHOD=espottt-smoke` for a two-step 2x2 check of the integration path, or `TTTD_METHOD=espottt-adaptive` with adaptive siblings.

## The advantage

TTT-Discover chooses `beta` so that `KL(softmax(beta * r) || uniform) = log 2` over a parent's 64 siblings and gives attempt `i` the advantage `exp(beta * r_i) / mean_{j != i} exp(beta * r_j) - 1`. That favours the best attempts, where SPO-TTT and PPO-TTT compare each attempt with an average, but it needs the siblings at every step.

Here the comparison set is a weighted set of past outcomes. `beta` is solved over the attempt (weight 1) and the comparison set together, with the KL measured against the weights rather than the uniform distribution, and the normaliser is the weighted mean of `exp(beta * r)` over the comparison set alone. With the other siblings of a group as the comparison set, each with weight 1, this is TTT-Discover's advantage exactly, which a test checks against the `tttd` preparer; `baseline: siblings` runs that control.

The comparison set's total weight sets the advantage's scale: against `n` equal outcomes, one better attempt gets about `n / 2`. A history can hold far more effective outcomes than a group of 63 siblings (the task's history gains 512 every step), which would inflate the advantages, and the un-clipped loss with them. So under `baseline: node` and `task` the set's weights are scaled to total `comparison_size` (default `rollouts_per_group - 1`, TTT-Discover's 63, and at least 2): the history decides the shape of the comparison, the group size its scale. Below 2 the KL target `log 2` is reached only as `beta` grows without bound and the best attempt's advantage reaches about `1e12`; TTT-Discover's own advantage does the same for a group of two siblings, so `baseline: siblings` needs groups of three or more. A test checks that one breakthrough gets the same advantage against 63 siblings as against a history of 6,000 equal outcomes.

## The history

Because `beta` and the normaliser depend on the whole distribution of past rewards, not only its mean, each key keeps the rewards themselves with weights. Equal rewards share one entry with their summed weight, which loses nothing (everything computed from a history depends on it only through the weight at each reward) and is where most of the size goes: failures all score 0, and many attempts repeat their parent's score. Past `history_size` distinct rewards the best one is kept exactly and the rest are reduced to weighted quantiles, which keeps the top that the entropic weights depend on most and the shape below it. Forgetting is SPO-TTT's tracker applied to the outcomes: when a key is updated `D` policy versions after its last update, its weights are multiplied by `rho = 2^(-D / half_life)`, clipped to `[rho_min, rho_max]`, and the new rewards join with weight 1. A key's total weight is therefore SPO-TTT's effective count and, until compression, its weighted mean is SPO-TTT's estimate exactly; a test checks both against the tracker.

`baseline: node` keys the history by the archive state an attempt was expanded from. A comparison set thinner than `min_history` effective outcomes is topped up in order: the parent state's history with `inherit_fraction` of its weight, the step's other attempts from the same state, the task's history, then the rest of the step. `pool_siblings: true` always adds the step's other attempts from the same state. Every comparison set is read before the step updates any key, so an attempt is never compared with itself.

With [adaptive siblings](../tttd/examples/tttd/README.md#adaptive-siblings) (`TTTD_METHOD=espottt-adaptive`) a well-known state gets as few as 8 attempts a step and the history supplies the rest of its comparison, which is what lets the same 512 attempts reach more parents.

## What it costs

No critic and no extra model: the stack is tttd's, on two GPUs. The history holds at most `history_size` pairs of numbers per archive state, at most 4,096 states, and travels with the algorithm state Reef commits at every step: about 3 KB per state, 0.2 MB after 15 simulated steps on the 8x64 grid. The preparer computes one comparison per distinct (state, reward) pair, so repeated scores cost nothing, and takes about 0.8 s per 8x64 step on one CPU core, against roughly 50 minutes for the step itself.

## Where the rest is documented

[The espottt recipe page](../../docs/user-guide/recipes/espottt.rst) covers configuration and metrics.
