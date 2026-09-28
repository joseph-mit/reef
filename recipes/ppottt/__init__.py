"""PPO-TTT: one method, one package.

Direction I of the single-stream test-time-training study. TTT-Discover's PUCT
search and verifiers stay as they are; the grouped entropic advantage is
replaced by standard PPO, with a clipped policy objective and a learned scalar
critic, so a search step no longer needs sibling attempts to train.

- ``recipe`` — the recipe class, its config fields, and the training spec.
- ``processor`` — the step barrier; emits a flat, scheduled policy batch.
- ``report`` — the step-grid rollout report contract.
- ``batches`` — the policy batch that carries the PPO update schedule.
- ``preparer`` — schedules the step and leaves advantages to the critic.
- ``slime`` — the loss family, critic cadence and GAE hook. Imported by the
  training driver and workers only.
"""

from recipes.ppottt.batches import ScheduledPolicyBatch
from recipes.ppottt.preparer import PpotttPreparer
from recipes.ppottt.processor import PPOTTTProcessor
from recipes.ppottt.recipe import PPOTTTRecipe
from recipes.ppottt.report import PPOTTTRolloutReport
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("ppottt", "recipes.ppottt.slime:PpotttAlgorithm")

__all__ = ["PPOTTTProcessor", "PPOTTTRecipe", "PPOTTTRolloutReport", "PpotttPreparer", "ScheduledPolicyBatch"]
