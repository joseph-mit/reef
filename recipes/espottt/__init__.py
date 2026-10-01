"""Entropic SPO-TTT: one method, one package.

TTT-Discover's entropic advantage, which favours the best attempts, with its
comparison set taken from a forgetting history of past attempts from the same
archive state (SPO-TTT's idea) instead of the step's sibling group. The search
is TTT-Discover's; the loss is TTT-Discover's ``tttd`` family.

- ``history`` — the weighted reward history and the entropic advantage, torch-free.
- ``recipe`` — the recipe class, its config fields, and the training spec.
- ``processor`` — PPO-TTT's step barrier, attaching each attempt's parent ids.
- ``report`` — SPO-TTT's report under the ``espottt`` tag.
- ``batches`` — the scheduled batch plus history settings.
- ``preparer`` — computes advantages and carries the history in algorithm state.
"""

# The recipe trains with TTT-Discover's loss family, which the tttd package registers.
import recipes.tttd  # noqa: F401
from recipes.espottt.batches import EspotttBatch
from recipes.espottt.preparer import EspotttPreparer
from recipes.espottt.processor import ESPOTTTProcessor
from recipes.espottt.recipe import ESPOTTTRecipe
from recipes.espottt.report import ESPOTTTRolloutReport

__all__ = ["ESPOTTTProcessor", "ESPOTTTRecipe", "ESPOTTTRolloutReport", "EspotttBatch", "EspotttPreparer"]
