"""Register a Reef checkout's entry points without installing it.

The Apptainer image does not install Reef: the job binds the checkout and
puts it on PYTHONPATH, so code changes need no rebuild. Entry points, though,
are package metadata, and SGLang finds Reef's scheduler plugin only through
one (``[project.entry-points."sglang.srt.plugins"]``). Without it the SGLang
scheduler rejects Reef's runtime-load-ID synchronisation and the stack stops
at start-up. This writes the checkout's entry points, read from its
``pyproject.toml``, as a ``.dist-info`` directory that ``importlib.metadata``
finds on PYTHONPATH, which is what ``pip install -e .`` would have recorded.

    python3 checkout_metadata.py <repo>/pyproject.toml <directory on PYTHONPATH>
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; Reef depends on tomli there
    import tomli as tomllib

DISTRIBUTION = "reef_checkout"


def entry_points_text(pyproject: Path) -> str:
    groups = tomllib.loads(pyproject.read_text())["project"].get("entry-points", {})
    lines: list[str] = []
    for group in sorted(groups):
        lines.append(f"[{group}]")
        lines.extend(f"{name} = {target}" for name, target in sorted(groups[group].items()))
        lines.append("")
    return "\n".join(lines)


def _write_atomically(path: Path, text: str) -> None:
    # Concurrent jobs write the same content; a rename never exposes a half-written file.
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(handle, "w") as stream:
        stream.write(text)
    os.replace(temporary, path)


def write_metadata(pyproject: Path, directory: Path) -> Path:
    dist_info = directory / f"{DISTRIBUTION}-0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    _write_atomically(dist_info / "METADATA", f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0\n")
    _write_atomically(dist_info / "entry_points.txt", entry_points_text(pyproject))
    return dist_info


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: checkout_metadata.py <pyproject.toml> <directory>")
    write_metadata(Path(sys.argv[1]), Path(sys.argv[2]))
