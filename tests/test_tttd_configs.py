"""Stack settings every TTT-Discover deployment config must share."""

from __future__ import annotations

from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd"
CONFIGS = sorted(EXAMPLE.glob("serve*.yaml"))


@pytest.mark.unit
@pytest.mark.parametrize("config", CONFIGS, ids=lambda path: path.name)
def test_colocated_trainer_keeps_the_memory_saver_usable(config: Path) -> None:
    text = config.read_text()
    # torch_memory_saver, which offloads the colocated trainer while SGLang
    # generates, refuses expandable segments. max_split_size_mb:512 keeps the
    # allocator from carving the buffers the trainer shares with SGLang over
    # CUDA IPC out of large blocks the memory saver owns, which cannot be
    # shared: the default allocator failed at the first adapter publication.
    assert "--colocate" in text
    assert text.count("--train-env-vars") == 1
    assert '--train-env-vars=\'{"PYTORCH_CUDA_ALLOC_CONF":"max_split_size_mb:512"}\'' in text
    assert "expandable_segments" not in text


@pytest.mark.unit
def test_every_config_shares_the_serving_memory_share() -> None:
    shares = {config.name: config.read_text().count("--sglang-mem-fraction-static=0.7") for config in CONFIGS}
    assert len(CONFIGS) >= 7 and set(shares.values()) == {1}, shares
