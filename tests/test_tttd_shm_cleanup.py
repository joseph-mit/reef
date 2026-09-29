"""shm_cleanup.py: remove only your unheld segments made since a marker."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "recipes/tttd/examples/tttd/apptainer/shm_cleanup.py"
_spec = importlib.util.spec_from_file_location("shm_cleanup", SCRIPT)
shm_cleanup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shm_cleanup)


def _process(proc: Path, pid: int, *, open_paths=(), mapped=()) -> None:
    fd_dir = proc / str(pid) / "fd"
    fd_dir.mkdir(parents=True)
    for number, path in enumerate(open_paths):
        (fd_dir / str(number)).symlink_to(path)
    lines = [f"7f00-7f10 rw-s 00000000 00:1a 42    {path}\n" for path in mapped]
    (proc / str(pid) / "maps").write_text("".join(lines) + "7f20-7f30 r-xp 00000000 08:01 7    /usr/lib/libc.so\n")


@pytest.mark.unit
def test_removes_only_unheld_segments_newer_than_the_marker(tmp_path: Path) -> None:
    shm, proc = tmp_path / "shm", tmp_path / "proc"
    shm.mkdir()
    proc.mkdir()
    (proc / "self").mkdir()  # non-numeric entries are skipped
    old = shm / "old-orphan"
    old.write_text("x")
    os.utime(old, (1000, 1000))
    for name in ("open", "mapped", "orphan", "deleted-mapping", "with space", "with"):
        (shm / name).write_text("x")
    (shm / "ray-dir").mkdir()
    (shm / "ray-dir" / "plasma").write_text("x")
    (shm / "orphan-dir").mkdir()
    (shm / "orphan-dir" / "a").write_text("x")
    _process(proc, 11, open_paths=[shm / "open"], mapped=[shm / "mapped", shm / "ray-dir" / "plasma"])
    _process(proc, 12, mapped=[f"{shm / 'deleted-mapping'} (deleted)", shm / "with space"])
    unreadable = proc / "13"
    unreadable.mkdir()  # a process whose fd and maps cannot be read

    removed, kept = shm_cleanup.remove_unheld(str(shm), str(proc), newer_than=2000, uid=os.getuid())

    assert sorted(path.name for path in shm.iterdir()) == [
        "deleted-mapping",
        "mapped",
        "old-orphan",
        "open",
        "ray-dir",
        "with space",
    ]
    assert (removed, kept) == (3, 5)


@pytest.mark.unit
def test_other_users_segments_are_never_candidates(tmp_path: Path) -> None:
    (tmp_path / "segment").write_text("x")
    assert shm_cleanup.candidates(str(tmp_path), 0, os.getuid() + 1) == []
