# hspottt

Stepping-stone SPO-TTT: [SPO-TTT](../spottt/README.md), plus hindsight credit for attempts that later led to a success. An attempt that scores modestly can still be the program a top solution later grows from; this recipe finds those attempts once their descendants succeed and trains them once more, with that credit as their advantage. Every step's own attempts get exactly SPO-TTT's advantages.

- Methods: [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232) through SPO-TTT, and the positive-only replay of [Self-Imitation Learning](https://arxiv.org/abs/1806.05635) with descendants' success as the hindsight return
- Pins: the `slime` pin shared by every recipe; the shipped config is SPO-TTT's, Qwen3-8B with a rank-32 LoRA on two GPUs
- Claim scope: an implementation with CPU-verified numerics and configs. No GPU run has been recorded yet.

## Layout

```text
hspottt/
  stones.py        the store of recent attempts: success, credit, replay, torch-free
  recipe.py        HSPOTTTRecipe: SPO-TTT's fields, the stone settings, loss family "spottt"
  processor.py     SPO-TTT's step barrier, the store, and the replays appended to each batch
  report.py        SPO-TTT's report plus the two program keys that link attempts across steps
  batches.py       HspotttBatch: SPO-TTT's batch with the replays and their credits at its end
  preparer.py      SPO-TTT's advantages for the step's attempts, the credits for the replays
```

Run it from the tttd example with `TTTD_METHOD=hspottt`, or `TTTD_METHOD=hspottt-smoke` for a two-step 2x2 check of the integration path.

## Finding stepping stones

The harness reports two program keys with every attempt (the first 16 hex digits of the program's SHA-1): its own, and that of the archive state it was expanded from. The archive keeps one state per distinct program, so a later attempt whose parent's key equals an earlier attempt's key grew from that attempt's program.

The processor keeps, for `stone_window` steps, the attempts that could have become archive states: the best `stone_keep_per_parent` attempts with a new program from each parent in a step (an attempt that returned its parent's program unchanged is no new state). The archive keeps at most two children per state, so with the default of 2 an archived attempt is missed only when one of a parent's two best programs was already in the archive from an earlier step and the archive took the third instead.

An attempt succeeds when it is among the best `stone_top_fraction` of its step's attempts with a program (`stone_success: top`), or when it beats the best score seen before its step (`stone_success: best`, sparser: on 26-circle packing the best stops moving after about a dozen steps). From each success the processor walks up the parents' keys: the kept attempt `k` generations up is offered `stone_credit * stone_discount^(k - 1)`, up to `stone_generations` up, and keeps the largest credit it is offered. A kept attempt that succeeded itself earns no credit, since its own step already rewarded it, but the walk passes through it to its ancestors.

## The replay

When a kept attempt has waited `stone_window` steps it leaves the store. If it earned credit, its sample is appended to that step's batch and trained once more with the credit as its advantage, in units of the fresh advantages (their standard deviation, as SPO-TTT normalises them). Only positive credit is replayed, so the update raises the probability of a stepping stone, which is self-imitation learning's rule; and each attempt is replayed at most once. The fresh attempts, and only they, update SPO-TTT's tracker: a replay's reward was observed when it was sampled. Replays do share the step with the fresh attempts: they join its optimizer steps, and they enter the batch-wide mean that SPO-TTT's frozen-base KL term is centred on, which moves each fresh token's KL adjustment by a negligible amount while replays are a few percent of the batch.

A replay was sampled by a policy `stone_window` versions old. The `spottt` loss clips its importance ratio, which bounds how far a stale sample can move the policy, and the trainer's staleness check admits samples up to `max_staleness` versions old; the recipe requires `max_staleness >= stone_window`. The trainer refuses a whole batch if any sample is older or comes from another incarnation of the serving runtime, so the processor leaves out any replay that check would refuse instead of risking the step.

The optimizer steps widen evenly to take the replays: SPO-TTT's number of steps is kept (`ceil(fresh / minibatch_size)`), so with `minibatch_size: 128`, 512 fresh attempts and 18 replays train as four steps of about 133, not four of 128 and a fifth of 18.

## Restarts and retries

A step's batch is a pure function of its attempts and of the store as it stood before the step (its base). The base is fixed the first time the step is built, and with `stone_dir` set (the shipped configs use `${TTTD_STATE_DIR}/stones`) it is written there before the batch leaves. So a batch built again, after a failed training attempt, a scenario reload or a restart of the Reef service, is identical, and the trainer recognises a training job it has already run instead of refusing a different one under the same step; tests check both. The store advances to the step's result when the trainer acknowledges the batch, and that result is written too, so a restart after a step's commit plans the next step from where the step left the store. Only the newest step's two files are kept. One case loses credit, never a step: a step the trainer drops as stale still advances the store, so its replays are gone.

## What it costs

Nothing on the GPUs beyond SPO-TTT's stack, plus the replayed samples themselves, a few percent of a step. The store holds at most `stone_keep_per_parent` attempts per parent per step for `stone_window` steps (48 on the 8x64 grid with the defaults, up to 192 with adaptive siblings), with tokens in 4-byte arrays and log-probabilities in 8-byte ones; a 30,000-token attempt takes about 0.4 MB, so the store and its one file on disk stay around 20 MB on the 8x64 grid.

## Where the rest is documented

[The hspottt recipe page](../../docs/user-guide/recipes/hspottt.rst) covers configuration and metrics.
