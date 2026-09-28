PPO-TTT: test-time training with a learned baseline
===================================================

PPO-TTT runs the TTT-Discover search unchanged and trains on its attempts with
standard PPO: a clipped policy objective and a scalar critic that supplies the
baseline. A TTT-Discover step needs a group of sibling attempts from each
parent to form an advantage; with a critic, every attempt is scored against a
learned estimate of what attempts from that state usually achieve, so a group
of one is enough to train.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | a finite ``score`` per rollout, addressed into a step grid |
+-------------+------------------------------------------------------------+
| Loss family | ``ppottt``                                                 |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/ppottt/``                                        |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, one step per batch, no groups           |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/tttd/examples/tttd/``, run with                  |
|             | ``TTTD_METHOD=ppottt``                                     |
+-------------+------------------------------------------------------------+

What it does
------------

The question PPO-TTT answers is whether a scalar critic can replace the
sibling group as the source of the baseline. The search, the tasks, the
verifiers and the per-step sampling budget stay those of `TTT-Discover
<tttd.rst>`__, so the two recipes differ only in how a scored attempt becomes
a gradient.

.. flow::
   :loop: the next search step samples from the updated adapter

   Search :: PUCT picks the step's parents; the model writes attempts
   Feedback :: the judge scores each attempt, reported against its receipt
   Step* :: fit the critic, then take clipped PPO steps on the actor
   Version :: publish the updated LoRA adapter to the engine

How Reef implements it
----------------------

The processor keeps TTT-Discover's step barrier: it waits for every attempt of
a search step, and discards a step whose reports span two policy releases.
It then hands the step over as one flat batch. Constant-reward groups are not
dropped, because against a critic they still carry signal, and they are the
critic's training data.

The preparer ships no advantages. The ``ppottt`` loss family runs Slime's
stock clipped ``policy_loss`` with the sampler's log-probabilities as the old
policy, and a full-parameter critic with a zero-initialised value head
colocated on the actor GPUs. Its advantage hook places the verifier score on
the last response token, adds ``-kl_coef * KL`` to the frozen base at every
token, and runs GAE. With ``--gamma 1 --lambd 1`` every token of an attempt
carries ``score - V``, the one-step episode of test-time discovery; the
critic regresses on the score alone: the family zeroes the critic role's KL
coefficient, as Slime's own role overrides do.

The step's update schedule travels on the batch: ``minibatch_size`` rollouts
per optimizer step, ``ppo_epochs`` passes, shuffled per pass. The first
``--num-critic-only-steps`` steps fit the critic without moving the policy;
the search still runs during them and every attempt enters the archive.

Configuration
-------------

.. config::

   groups_per_step | 8 | parents PUCT selects per search step.
   rollouts_per_group | 64 | attempts per parent; 1 is valid. The grid must equal the driver's ``--global-batch-size``.
   ppo_epochs | 1 | passes over each step's rollouts.
   minibatch_size | 128 | rollouts per optimizer step; ``0`` for the whole step. Should divide the grid.
   shuffle_minibatches | true | reorder rollouts in every pass.

The objective lives in the Slime flags of ``serve.ppottt.yaml``:
``--eps-clip 0.2``, ``--value-clip 0.2``, ``--gamma 1.0``, ``--lambd 1.0``,
``--normalize-advantages``, ``--kl-coef 0.1``, and for the critic
``--critic-lr 5e-6``, ``--critic-steps-per-actor 2`` and
``--num-critic-only-steps 2``. Advantage whitening matters here: rewards of
valid programs cluster tightly while invalid programs score zero, and
unlike TTT-Discover's entropic weights PPO is not invariant to that scale.

Run the example
---------------

The TTT-Discover example drives every training method from the same search
harness; ``TTTD_METHOD`` picks the deployment config. Each method keeps its own
scenario and state directory, so runs never share checkpoints or an archive.

.. code:: bash

   cd recipes/tttd/examples/tttd
   TTTD_METHOD=ppottt-smoke ./run.sh   # one 2x2 step: checks the integration path
   TTTD_METHOD=ppottt ./run.sh         # 50 steps of 8x64 on Erdős, four GPUs

The critic doubles the training state: the checkpoint preflight expects
roughly sixteen times the model size free on disk. The runtime reports
``pg_clipfrac``, ``critic/explained_variance``, the value loss, and
``ppottt/critic_updates`` and ``ppottt/actor_trained`` per step.

Related guides
--------------

- `TTT-Discover <tttd.rst>`__: the search, tasks and run controller this
  recipe reuses.
- `SAO <sao.rst>`__: the other critic-based recipe, for a stream of tasks.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
