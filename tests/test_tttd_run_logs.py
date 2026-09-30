"""run.sh sets every earlier log aside before the stack starts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/run.sh"


def _rotation_block() -> str:
    text = RUN_SH.read_text()
    start = text.index("# Every start writes fresh logs.")
    end = text.index("# Start the Reef training stack.")
    return text[start:end]


@pytest.mark.unit
def test_every_service_log_and_stale_marker_are_cleared(tmp_path: Path) -> None:
    state = tmp_path / "state"
    for relative, text in {
        "reef.log": "old reef",
        "stack/reef.log": "old reef copy",
        "stack/slime-driver.log": "old driver copy",
        # Each service's own log: a new orchestrator reads it from the start.
        "stack/reef/reef.log": "old traceback",
        "stack/slime-driver/slime-driver.log": "old engine lines",
        "stack/slime-driver/bridge.ready": "ready",
        "stack/slime-driver/slime-driver.pid": "123",
    }.items():
        path = state / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    subprocess.run(["bash", "-c", _rotation_block()], env={"TTTD_STATE_DIR": str(state)}, check=True)

    remaining = sorted(str(path.relative_to(state)) for path in state.rglob("*") if path.is_file())
    archived = [name for name in remaining if name.startswith("logs/")]
    assert [name for name in remaining if not name.startswith("logs/")] == ["stack/slime-driver/slime-driver.pid"]
    assert sorted(name.split("/", 2)[2] for name in archived) == [
        "reef.log",
        "stack/reef.log",
        "stack/reef/reef.log",
        "stack/slime-driver.log",
        "stack/slime-driver/slime-driver.log",
    ]
    assert (state / next(name for name in archived if name.endswith("stack/reef/reef.log"))).read_text() == (
        "old traceback"
    )
