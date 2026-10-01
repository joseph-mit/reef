Stepping-stone SPO-TTT: credit for attempts that led to a success
=================================================================

Stepping-stone SPO-TTT trains every step exactly as SPO-TTT and, in addition,
gives hindsight credit to attempts that a later successful attempt descends
from. An attempt that scores modestly can still be the program a top solution
grows from; once its descendants succeed, it is trained once more with that
credit as its advantage, the positive-only replay of `Self-Imitation Learning
<https://arxiv.org/abs/1806.05635>`__ with descendants' success as the return.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | a finite ``score`` per rollout, with its parent state ids  |
|             | and the program keys of the attempt and its parent         |
+-------------+------------------------------------------------------------+
| Loss family | ``spottt``                                                 |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/hspottt/``                                       |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, one step per batch, plus replays        |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, a backend that captures tokens and log-probs, and    |
|             | ``max_staleness`` of at least ``stone_window``             |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/tttd/examples/tttd/``, run with                  |
|             | ``TTTD_METHOD=hspottt``                                    |
+-------------+------------------------------------------------------------+

What it does
------------

The harness reports each attempt's program key and its parent state's. The
processor keeps, for ``stone_window`` steps, the best ``stone_keep_per_parent``
attempts of each parent in a step, which covers every attempt the archive can
keep. When an attempt succeeds (among its step's best ``stone_top_fraction``,
or a new best score), each kept attempt ``k`` generations above it that did
not succeed itself is offered ``stone_credit * stone_discount^(k - 1)``. When
its wait ends, a kept attempt that earned credit is appended to that step's
batch with the credit as its advantage.

.. flow::
   :loop: the next search step samples from the updated adapter

   Search :: PUCT picks the step's parents; the model writes attempts
   Feedback :: each score is reported with its parent ids and program keys
   Step* :: SPO-TTT's advantages; credit up each success's parents; replay stones whose wait ended
   Version :: publish the updated LoRA adapter to the engine

How Reef implements it
----------------------

The fresh attempts get exactly SPO-TTT's advantages and only they update its
tracker; replays join the step's optimizer steps. Replays are
``stone_window`` policy versions old: the ``spottt`` loss clips their
importance ratio, the trainer admits them under ``max_staleness``, and the
processor leaves out any replay the trainer's staleness check would refuse,
since that check refuses the whole batch. A step's batch is a pure function of
its attempts and of the store before the step, which is written to
``stone_dir`` the first time the step is built (the store after the step is
written when the trainer acknowledges it), so a batch built again after a
failed training attempt, a scenario reload or a restart is identical, and the
next step starts from where the last one left the store. Tokens and
log-probabilities are kept in compact arrays.

Configuration
-------------

.. config::

   stone_window | 3 | steps a kept attempt waits for descendants before its replay.
   stone_generations | 3 | how far up a success's parents credit reaches.
   stone_discount | 0.5 | share of the credit kept per generation further up.
   stone_credit | 1.0 | credit for a success's parent, in units of the fresh advantages.
   stone_success | top | ``top`` (the step's best ``stone_top_fraction``) or ``best`` (a new best score).
   stone_top_fraction | 0.05 | share of a step's attempts with a program that count as successes.
   stone_keep_per_parent | 2 | attempts kept per parent and step; the archive keeps two children per state.
   stone_tolerance | 1e-9 | relative margin a new best must clear under ``stone_success: best``.
   stone_dir | (empty) | where the store each step is planned from is written; empty keeps it in memory only.
   max_staleness | 3 | versions old a sample may be; at least ``stone_window``.

SPO-TTT's settings (``baseline``, ``normalize_advantages``, ``half_life``,
``rho_min``, ``rho_max``, ``inherit_fraction``, ``groups_per_step``,
``rollouts_per_group``, ``ppo_epochs``, ``minibatch_size``,
``shuffle_minibatches``) apply unchanged.

Run the example
---------------

.. code:: bash

   cd recipes/tttd/examples/tttd
   TTTD_METHOD=hspottt-smoke ./run.sh   # two 2x2 steps with a one-step window
   TTTD_METHOD=hspottt ./run.sh         # 50 steps of 8x64, two GPUs

The step metrics include ``replays`` and ``replay_credit_mean``; the processor
logs ``successes``, ``newly_credited``, ``kept``, ``replays_admitted`` and
``replays_refused`` under ``hspottt``.

Related guides
--------------

- `SPO-TTT <spottt.rst>`__: everything a step trains on besides the replays.
- `Entropic SPO-TTT <espottt.rst>`__: TTT-Discover's advantage against past
  outcomes.
