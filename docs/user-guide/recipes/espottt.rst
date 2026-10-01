Entropic SPO-TTT: TTT-Discover's advantage against past outcomes
================================================================

Entropic SPO-TTT runs the TTT-Discover search and loss unchanged and keeps
TTT-Discover's entropic advantage, which favours the best attempts, but
compares each attempt with a forgetting history of past outcomes from the same
archive state instead of with the step's other siblings. SPO-TTT's idea gives
the history; TTT-Discover's objective is kept, so the two differ in where the
comparison set comes from.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | a finite ``score`` per rollout, with its parent state ids  |
+-------------+------------------------------------------------------------+
| Loss family | ``tttd``                                                   |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/espottt/``                                       |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, one step per batch, no groups           |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/tttd/examples/tttd/``, run with                  |
|             | ``TTTD_METHOD=espottt`` or ``espottt-adaptive``            |
+-------------+------------------------------------------------------------+

What it does
------------

TTT-Discover gives attempt ``i`` the advantage
``exp(beta * r_i) / mean_{j != i} exp(beta * r_j) - 1`` over its 63 siblings,
with ``beta`` chosen so that ``KL(softmax(beta * r) || uniform) = log 2``.
Here the comparison set is a weighted set of past outcomes from the attempt's
archive state. ``beta`` is solved over the attempt and the set together, with
the KL taken against the weights, and the normaliser is the weighted mean of
``exp(beta * r)`` over the set. With the siblings as the set this is
TTT-Discover's advantage exactly (``baseline: siblings``, the control). The
set's weights are scaled to total ``comparison_size``, so the advantages keep
TTT-Discover's scale however much history a state holds.

.. flow::
   :loop: the next search step samples from the updated adapter

   Search :: PUCT picks the step's parents; the model writes attempts
   Feedback :: each score is reported with the attempt's parent and grandparent
   Step* :: entropic advantage against the state's history, then fold the step in
   Version :: publish the updated LoRA adapter to the engine

How Reef implements it
----------------------

The processor is PPO-TTT's step barrier; it attaches each attempt's
``parent_id`` and ``grandparent_id``. The preparer reads the histories from the
algorithm state Reef commits with every training step, so a restart resumes
them exactly. A history keeps the rewards with weights, equal rewards sharing
one entry; forgetting is SPO-TTT's tracker applied to the weights, so a
history's total weight is SPO-TTT's effective count and its weighted mean
SPO-TTT's estimate. A thin history is topped up from the parent state's, the
step's other attempts from the same state, the task's, then the rest of the
step. Every comparison set is read before the step updates any history.

Configuration
-------------

.. config::

   baseline | node | ``node`` (the state's history), ``task`` (one history) or ``siblings`` (TTT-Discover's advantage).
   pool_siblings | false | add the step's other attempts from the same state even to a thick history.
   min_history | 8.0 | effective outcomes a comparison set needs before it is used on its own.
   history_size | 128 | distinct rewards kept per state; past it, the best is kept and the rest reduced to quantiles.
   comparison_size | 0 | total weight a comparison set is scaled to, at least 2; ``0`` means ``rollouts_per_group - 1`` (at least 2).
   half_life | 8.0 | committed steps after which a history keeps half its weight.
   rho_min | 0.875 | lower bound on the per-update forgetting factor.
   rho_max | 0.96 | upper bound on the per-update forgetting factor.
   inherit_fraction | 0.5 | share of a parent's history a new state starts from.
   groups_per_step | 8 | parents PUCT selects per search step.
   rollouts_per_group | 64 | attempts per parent.
   ppo_epochs | 1 | passes over each step's rollouts.
   minibatch_size | 0 | rollouts per optimizer step; ``0``, TTT-Discover's, for the whole step.

Run the example
---------------

.. code:: bash

   cd recipes/tttd/examples/tttd
   TTTD_METHOD=espottt-smoke ./run.sh      # two 2x2 steps: checks the integration path
   TTTD_METHOD=espottt ./run.sh            # 50 steps of 8x64, two GPUs
   TTTD_METHOD=espottt-adaptive ./run.sh   # the same with adaptive siblings

The step metrics include ``comparison_from_<source>_fraction`` (where each
comparison set came from), ``comparison_weight_mean``, ``beta_median`` and the
advantages' range.

Related guides
--------------

- `SPO-TTT <spottt.rst>`__: the history's forgetting rule, with a mean baseline.
- `TTT-Discover <tttd.rst>`__: the search, loss and advantage this recipe keeps,
  and adaptive siblings.
- `Stepping-stone SPO-TTT <hspottt.rst>`__: credit for attempts that later led
  to a success.
