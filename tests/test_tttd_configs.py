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
    # generates, refuses to run under expandable segments, and the Slime
    # image turns them on for every process: the trainer must switch them off.
    assert "--colocate" in text
    assert text.count("--train-env-vars") == 1
    assert '"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:False"' in text
    assert "expandable_segments:True" not in text and "max_split_size_mb" not in text


@pytest.mark.unit
def test_every_config_shares_the_serving_memory_share() -> None:
    shares = {config.name: config.read_text().count("--sglang-mem-fraction-static=0.7") for config in CONFIGS}
    assert len(CONFIGS) >= 7 and set(shares.values()) == {1}, shares
