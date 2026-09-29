"""Remove your shared-memory segments that no process holds any more.

A stack killed with SIGKILL leaves its segments in /dev/shm (Ray, NCCL,
PyTorch), where they keep occupying memory. Only your segments newer than a
marker file are considered, and a segment stays if any process still has it
open or mapped, so segments of your other jobs on the node are left alone.

    python3 shm_cleanup.py <marker>

The segments are listed before the processes are read, so a segment made
after the listing is never considered. One pass over /proc serves every
segment: checking each segment with fuser rescans every process per segment,
which took over half an hour with a few thousand segments on a busy node.

It runs on the host's Python, which may be as old as 3.6.
"""

import argparse
import contextlib
import os
import shutil
import sys

# How /proc/<pid>/maps marks a mapped file that has since been unlinked.
DELETED = " (deleted)"


def candidates(shm_root, newer_than, uid):
    """Your entries directly under ``shm_root`` modified after ``newer_than``."""
    found = []
    for name in os.listdir(shm_root):
        path = os.path.join(shm_root, name)
        try:
            status = os.lstat(path)
        except OSError:
            continue
        if status.st_uid == uid and status.st_mtime > newer_than:
            found.append(name)
    return found


def held_names(proc_root, shm_root):
    """Names of the entries under ``shm_root`` some process has open or mapped."""
    prefix = shm_root.rstrip("/") + "/"
    held = set()

    def note(path):
        if path.endswith(DELETED):
            path = path[: -len(DELETED)]
        if path.startswith(prefix):
            held.add(path[len(prefix) :].split("/", 1)[0])

    for pid in os.listdir(proc_root):
        if not pid.isdigit():
            continue
        fd_dir = os.path.join(proc_root, pid, "fd")
        try:
            descriptors = os.listdir(fd_dir)
        except OSError:
            descriptors = []
        for descriptor in descriptors:
            with contextlib.suppress(OSError):
                note(os.readlink(os.path.join(fd_dir, descriptor)))
        try:
            with open(os.path.join(proc_root, pid, "maps")) as maps:
                for line in maps:
                    fields = line.rstrip("\n").split(None, 5)
                    if len(fields) == 6:
                        note(fields[5])
        except OSError:
            pass
    return held


def remove_unheld(shm_root, proc_root, newer_than, uid):
    """Remove the unheld candidates; return how many were removed and kept."""
    names = candidates(shm_root, newer_than, uid)
    held = held_names(proc_root, shm_root)
    removed = 0
    for name in names:
        if name in held:
            continue
        path = os.path.join(shm_root, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed, len(names) - removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("marker", help="only segments modified after this file are considered")
    parser.add_argument("--shm", default="/dev/shm")
    parser.add_argument("--proc", default="/proc")
    options = parser.parse_args(argv)
    removed, kept = remove_unheld(options.shm, options.proc, os.stat(options.marker).st_mtime, os.getuid())
    print(f"shared memory: removed {removed} unheld segment(s), kept {kept} held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
