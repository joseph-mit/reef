"""Stepping-stone SPO-TTT: one method, one package.

SPO-TTT, plus hindsight credit: an attempt that a later successful attempt
descends from, and that did not succeed itself, is trained once more a few
steps later with that credit as its advantage (self-imitation of stepping
stones). The step's own attempts train exactly as in SPO-TTT, with its loss
family.

- ``stones`` — the store of recent attempts, success, credit and replay, torch-free.
- ``recipe`` — the recipe class, its config fields, and the training spec.
- ``processor`` — SPO-TTT's step barrier, the store, and the replays.
- ``report`` — SPO-TTT's report plus the program keys that link attempts.
- ``batches`` — SPO-TTT's batch with the replays and their credits at its end.
- ``preparer`` — SPO-TTT's advantages for the step, the credits for the replays.
"""

# The recipe trains with SPO-TTT's loss family, which the spottt package registers.
import recipes.spottt  # noqa: F401
from recipes.hspottt.batches import HspotttBatch
from recipes.hspottt.preparer import HspotttPreparer
from recipes.hspottt.processor import HSPOTTTProcessor
from recipes.hspottt.recipe import HSPOTTTRecipe
from recipes.hspottt.report import HSPOTTTRolloutReport

__all__ = ["HSPOTTTProcessor", "HSPOTTTRecipe", "HSPOTTTRolloutReport", "HspotttBatch", "HspotttPreparer"]
