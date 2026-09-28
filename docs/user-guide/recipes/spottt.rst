SPO-TTT: test-time training without siblings or a critic
========================================================

SPO-TTT runs the TTT-Discover search unchanged and gives each attempt a
baseline built from past outcomes: a forgetting estimate of what attempts
from the same archive state have scored. It is the rule of `Single-stream
Policy Optimization <https://arxiv.org/abs/2509.13232>`__ with archive states
in place of prompts, so a search step needs neither a group of siblings nor a
learned critic.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | a finite ``score`` per rollout, with its parent state ids  |
+-------------+------------------------------------------------------------+
| Loss family | ``spottt``                                                 |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/spottt/``                                        |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, one step per batch, no groups           |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/tttd/examples/tttd/``, run with                  |
|             | ``TTTD_METHOD=spottt``                                     |
+-------------+------------------------------------------------------------+

What it does
------------

SPO keeps one running estimate of expected reward per prompt and forgets old
observations as the policy moves. In test-time discovery a state almost never
repeats, so a prompt-indexed table would be empty when it is needed. SPO-TTT
keys the estimate by the archive state an attempt was expanded from
(``baseline: node``) or keeps one for the whole task (``baseline: task``). A
state seen for the first time starts from its own parent's estimate, then the
task's. ``baseline: none`` subtracts nothing, SPO's no-baseline ablation.

.. flow::
   :loop: the next search step samples from the updated adapter

   Search :: PUCT picks the step's parents; the model writes attempts
   Feedback :: each score is reported with the attempt's parent and grandparent
   Step* :: advantage = score minus the tracker's estimate, then update it
   Version :: publish the updated LoRA adapter to the engine

How Reef implements it
----------------------

The processor is PPO-TTT's step barrier; it attaches each attempt's
``parent_id`` and ``grandparent_id`` to its sample. The preparer reads the
tracker from the algorithm state Reef commits with every training step, so a
restart resumes it exactly. Every baseline is read before the step updates any
key, which keeps it independent of the reward it is subtracted from. The
tracker's update is SPO's discounted Beta posterior written for real-valued
rewards, discounting once per policy version rather than once per
observation. Advantages are normalised across the step, as in SPO.

The ``spottt`` loss family trains the shipped advantages with Slime's clipped
``policy_loss`` and applies TTT-Discover's centred KL to the frozen base
unchanged, through the ``tttd`` hook.

Configuration
-------------

.. config::

   baseline | node | ``node``, ``task`` or ``none``: where past outcomes are shared.
   normalize_advantages | true | centre and scale advantages across the step.
   half_life | 8.0 | committed steps after which a key keeps half the weight of its past observations.
   rho_min | 0.875 | lower bound on the per-update forgetting factor.
   rho_max | 0.96 | upper bound on the per-update forgetting factor.
   inherit_fraction | 0.5 | share of a parent's effective count a new state starts from.
   groups_per_step | 8 | parents PUCT selects per search step.
   rollouts_per_group | 64 | attempts per parent; 1 is valid.
   ppo_epochs | 1 | passes over each step's rollouts.
   minibatch_size | 128 | rollouts per optimizer step; ``0`` for the whole step.
   shuffle_minibatches | true | reorder rollouts in every pass.

Run the example
---------------

.. code:: bash

   cd recipes/tttd/examples/tttd
   TTTD_METHOD=spottt-smoke ./run.sh   # one 2x2 step: checks the integration path
   TTTD_METHOD=spottt ./run.sh         # 50 steps of 8x64 on Erdős, two GPUs

The step metrics include ``tracked_fraction``, the share of attempts whose
baseline came from a warm estimate, and ``tracker_keys``.

Related guides
--------------

- `PPO-TTT <ppottt.rst>`__: the critic-based alternative on the same search.
- `TTT-Discover <tttd.rst>`__: the search, tasks and run controller this
  recipe reuses, and the ``tttd-mean`` and ``search-only`` controls.
