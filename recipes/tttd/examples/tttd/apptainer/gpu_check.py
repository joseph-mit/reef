"""How fast a job's GPUs compute and how fast they reach each other.

Run in the image on the job's GPUs:

    python3 apptainer/gpu_check.py
    python3 apptainer/gpu_check.py --burn 90   # load every GPU, then read temperatures

It prints each GPU's temperature, clock and slowdown reasons, its bf16
matrix-multiply rate, which GPUs can read each other's memory directly (peer
access), the NVIDIA topology, and the NCCL
all-reduce bus bandwidth across all visible GPUs: the traffic tensor
parallelism adds to every layer of every training step. Next to a running job
the GPUs are shared, so the rates are lower bounds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Rough expectations on H200 SXM, for reading the output.
EXPECTED = (
    "expected on H200 SXM: about 600-750 TFLOPS per GPU; all-reduce bus bandwidth "
    "above 300 GB/s over NVLink, about 40 GB/s over PCIe, 10-20 GB/s through host memory"
)


def matmul_tflops(device: int, size: int = 8192, repeats: int = 30) -> float:
    left = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    right = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    for _ in range(3):
        left @ right
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        left @ right
    torch.cuda.synchronize(device)
    return 2 * size**3 * repeats / (time.perf_counter() - start) / 1e12


HEALTH_FIELDS = (
    "index,pci.bus_id,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,"
    "utilization.gpu,clocks_event_reasons.active"
)


def gpu_health() -> str:
    """Temperature, clock and slowdown reasons per GPU, from nvidia-smi.

    In the reasons, 0x20 and 0x40 are thermal slowdown and 0x08 a hardware
    slowdown; 0x01 means idle.
    """
    if not shutil.which("nvidia-smi"):
        return "nvidia-smi not found"
    health = subprocess.run(
        ["nvidia-smi", f"--query-gpu={HEALTH_FIELDS}", "--format=csv"], capture_output=True, text=True, check=False
    )
    return health.stdout.strip() or health.stderr.strip()


def burn(seconds: float, size: int = 8192, batch: int = 20) -> None:
    """Keep every visible GPU busy with bf16 matmuls and report how each copes.

    A GPU whose cooling falls short heats up and the hardware lowers its
    clock, so its rate falls well below the others'. Each GPU always has two
    batches queued, so a fast GPU never waits for a slow one, and the
    readings are taken while all of them are busy.
    """
    count = torch.cuda.device_count()
    operands = [
        (
            torch.randn(size, size, device=device, dtype=torch.bfloat16),
            torch.randn(size, size, device=device, dtype=torch.bfloat16),
        )
        for device in range(count)
    ]
    queued: list[list[tuple[torch.cuda.Event, torch.cuda.Event]]] = [[] for _ in range(count)]
    busy_ms = [0.0] * count
    done = [0] * count

    def enqueue(device: int) -> None:
        left, right = operands[device]
        with torch.cuda.device(device):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(batch):
                left @ right
            end.record()
        queued[device].append((begin, end))

    def collect(device: int, *, wait: bool) -> None:
        while queued[device] and (wait or queued[device][0][1].query()):
            begin, end = queued[device].pop(0)
            end.synchronize()
            busy_ms[device] += begin.elapsed_time(end)
            done[device] += batch

    start = time.perf_counter()
    next_report = start + min(15.0, seconds / 2)
    while time.perf_counter() - start < seconds:
        for device in range(count):
            collect(device, wait=False)
            while len(queued[device]) < 2:
                enqueue(device)
        if time.perf_counter() >= next_report:
            print(f"after {time.perf_counter() - start:.0f}s under load:\n{gpu_health()}", flush=True)
            next_report += 30
        time.sleep(0.01)
    print(f"at the end, {time.perf_counter() - start:.0f}s under load:\n{gpu_health()}", flush=True)
    for device in range(count):
        collect(device, wait=True)
        rate = 2 * size**3 * done[device] / (busy_ms[device] / 1e3) / 1e12
        print(f"GPU {device}: {rate:.0f} TFLOPS sustained")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _allreduce_worker(rank: int, world: int, port: int, megabytes: int, repeats: int, results) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    tensor = torch.ones(megabytes * 2**20 // 2, device=rank, dtype=torch.bfloat16)
    for _ in range(3):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    if rank == 0:
        # NCCL's bus bandwidth: the per-link rate, comparable across GPU counts.
        results.put(tensor.numel() * 2 / elapsed * 2 * (world - 1) / world / 1e9)
    dist.destroy_process_group()


def allreduce_bus_bandwidth(world: int, megabytes: int = 256, repeats: int = 20) -> float:
    context = mp.get_context("spawn")
    results = context.Queue()
    mp.start_processes(
        _allreduce_worker,
        args=(world, _free_port(), megabytes, repeats, results),
        nprocs=world,
        start_method="spawn",
    )
    return results.get(timeout=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--burn", type=float, metavar="SECONDS", help="only load every GPU this long and report")
    options = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 1
    if options.burn:
        burn(options.burn)
        return 0
    count = torch.cuda.device_count()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}, {count} GPU(s)")
    print(EXPECTED)
    for device in range(count):
        properties = torch.cuda.get_device_properties(device)
        print(
            f"GPU {device}: {properties.name}, {properties.total_memory / 2**30:.0f} GiB, "
            f"bf16 matmul {matmul_tflops(device):.0f} TFLOPS"
        )
    for device in range(count):
        peers = [
            str(other)
            for other in range(count)
            if other != device and torch.cuda.can_device_access_peer(device, other)
        ]
        print(f"GPU {device} peer access to: {', '.join(peers) or 'none'}")
    print(gpu_health())
    if shutil.which("nvidia-smi"):
        topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, check=False)
        print(topology.stdout.strip() or topology.stderr.strip())
    if count > 1:
        print(f"NCCL all-reduce bus bandwidth across {count} GPUs: {allreduce_bus_bandwidth(count):.0f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
