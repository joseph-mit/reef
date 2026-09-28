"""SPO-TTT: one method, one package.

Direction II of the single-stream test-time-training study. TTT-Discover's
search stays as it is; each attempt's baseline is a forgetting estimate of past
outcomes, SPO's rule with archive states in place of prompts, so a step needs
neither siblings nor a critic.

- ``tracker`` — the forgetting tracker and the advantage assignment, torch-free.
- ``recipe`` — the recipe class, its config fields, and the training spec.
- ``processor`` — PPO-TTT's step barrier, attaching each attempt's parent ids.
- ``report`` — the step-grid report with parent and grandparent ids.
- ``batches`` — the scheduled batch plus tracker settings.
- ``preparer`` — computes advantages and carries the tracker in algorithm state.
- ``slime`` — the loss family and its KL hook. Imported by the training driver
  and workers only.
"""

from recipes.spottt.batches import SpotttBatch
from recipes.spottt.preparer import SpotttPreparer
from recipes.spottt.processor import SPOTTTProcessor
from recipes.spottt.recipe import SPOTTTRecipe
from recipes.spottt.report import SPOTTTRolloutReport
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("spottt", "recipes.spottt.slime:SpotttAlgorithm")

__all__ = ["SPOTTTProcessor", "SPOTTTRecipe", "SPOTTTRolloutReport", "SpotttBatch", "SpotttPreparer"]
