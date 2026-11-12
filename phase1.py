#!/usr/bin/env python3
"""
Phase 1 — False Sharing experiment for the supplied lab.
Windows/macOS/Linux; Python standard library only.

Run:
    python phase1.py
Optional shorter validation run:
    python phase1.py --quick

The full run follows the lab's M=24,000,000 and 2 warmups + 7 timed runs.
It can take a long time because it launches many worker processes.
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from multiprocessing import shared_memory

STEPS_FULL = 24_000_000
STRIDES = (1, 2, 4, 8, 16, 32)
MASK = 0xFFFFFFFFFFFFFFFF
MULT = 6364136223846793005
OUT_DIR = Path("parallel_lab")
OUT_DIR.mkdir(exist_ok=True)


def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & MASK
    return h


def worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * MULT + (i | 1)) & MASK
        # Store each iteration so workers truly contend on shared cache lines.
        arr[slot_idx] = acc
    del arr
    shm.close()


def run_once(p: int, stride: int, total_steps: int) -> float:
    # 512 uint64 slots = 4096 bytes; enough for the largest tested layout.
    shm = shared_memory.SharedMemory(create=True, size=4096)
    processes = []
    try:
        view = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for i in range(512):
            view[i] = 1
        del view

        base, remainder = divmod(total_steps, p)
        start = time.perf_counter_ns()
        for worker_id in range(p):
            iters = base + (1 if worker_id < remainder else 0)
            proc = mp.Process(
                target=worker,
                args=(shm.name, worker_id * stride, iters),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Worker process {proc.pid} exited with code {proc.exitcode}"
                )
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        return elapsed_ms
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join()
        shm.close()
        shm.unlink()


def benchmark(p: int, stride: int, steps: int, warmups: int, trials: int):
    for _ in range(warmups):
        run_once(p, stride, steps)
    samples = [run_once(p, stride, steps) for _ in range(trials)]
    samples.sort()
    med = statistics.median(samples)
    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]
    iqr = q3 - q1
    throughput = steps / (med / 1000.0) / 1e6
    return med, iqr, throughput, samples


def scaling_counts(logical: int, physical: int):
    # Include powers of two, P_log, and P_phys because the Phase 1 hash
    # is explicitly defined using the measurements at P_phys.
    counts = [1, 2, 4, 8, 16, 32]
    return sorted(set(x for x in counts if x <= logical) | {logical, physical})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="short validation: 300,000 steps, 1 warmup, 3 trials")
    args = parser.parse_args()

    student_id = input("Enter your Student ID: ").strip()
    if not student_id:
        raise SystemExit("Student ID cannot be empty.")

    logical = mp.cpu_count()
    physical = 6  # Measured in Phase 0 from Windows WMI: Ryzen 5 7430U.
    steps = 300_000 if args.quick else STEPS_FULL
    warmups, trials = (1, 3) if args.quick else (2, 7)

    print(f"\nPhase 1 | P_phys={physical}, P_log={logical}, M={steps:,}")
    if args.quick:
        print("QUICK MODE: these results are for code validation, not final lab submission.")

    stride_rows = []
    print("\n=== Table 1A: stride sweep at P_phys ===")
    for stride in STRIDES:
        med, iqr, mops, samples = benchmark(
            physical, stride, steps, warmups, trials
        )
        stride_rows.append((stride, med, iqr, mops, samples))
        print(f"stride={stride:>2} ({stride*8:>3} bytes): "
              f"median={med:.3f} ms, IQR={iqr:.3f} ms, "
              f"throughput={mops:.3f} Mops/s")

    print("\n=== Table 1B: process scaling ===")
    counts = scaling_counts(logical, physical)
    scaling_rows = []
    medians = {}
    for stride in (1, 16):
        for p in counts:
            med, iqr, mops, samples = benchmark(
                p, stride, steps, warmups, trials
            )
            medians[(stride, p)] = med
            scaling_rows.append((stride, p, med, iqr, mops, samples))

    for stride in (1, 16):
        t1 = medians[(stride, 1)]
        print(f"\nstride={stride}")
        print("p, median_ms, speedup, efficiency, Karp-Flatt e(p)")
        for p in counts:
            tp = medians[(stride, p)]
            speedup = t1 / tp
            efficiency = speedup / p
            if p == 1:
                e = None
                print(f"{p}, {tp:.3f}, {speedup:.4f}, {efficiency:.4f}, N/A")
            else:
                e = (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)
                print(f"{p}, {tp:.3f}, {speedup:.4f}, {efficiency:.4f}, {e:.6f}")

    m1 = medians[(1, physical)]
    m16 = medians[(16, physical)]
    payload = f"PHASE1|{student_id}|{logical}|{m1:.4f}|{m16:.4f}"
    verify = f"VERIFY_HASH[PHASE1]: {fnv1a_hash(payload):016X} (P_log={logical})"

    report = OUT_DIR / "phase1_results.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Phase 1 — False Sharing\n\n")
        f.write(f"Student ID: {student_id}\nP_phys: {physical}\nP_log: {logical}\n")
        f.write(f"Steps per trial (total across workers): {steps}\n")
        f.write(f"Warmups: {warmups}; timed trials: {trials}\n")
        if args.quick:
            f.write("WARNING: QUICK MODE; do not submit these reduced-workload measurements as full results.\n")
        f.write("\n## Table 1A — stride sweep at P_phys\n")
        f.write("| stride | separation_bytes | median_ms | IQR_ms | throughput_Mops_s |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for stride, med, iqr, mops, _ in stride_rows:
            f.write(f"| {stride} | {stride*8} | {med:.3f} | {iqr:.3f} | {mops:.3f} |\n")
        f.write("\n## Table 1B — scaling\n")
        f.write("| stride | p | median_ms | speedup | efficiency | Karp-Flatt_e |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for stride in (1, 16):
            t1 = medians[(stride, 1)]
            for p in counts:
                tp = medians[(stride, p)]
                speedup = t1 / tp
                efficiency = speedup / p
                e = "N/A" if p == 1 else f"{(1/speedup - 1/p)/(1 - 1/p):.6f}"
                f.write(f"| {stride} | {p} | {tp:.3f} | {speedup:.4f} | {efficiency:.4f} | {e} |\n")
        f.write("\n## Verification\n\n")
        f.write(verify + "\n")
        f.write("\n## Interpretation prompts\n")
        f.write("- Compare the stride=1 and stride=16 Karp-Flatt values as p increases.\n")
        f.write("- The tested byte separations are stride × 8; infer a cache-line boundary only from measured results and hardware documentation.\n")

    print("\n" + verify)
    print(f"Results saved to: {report.resolve()}")
    print("Note: process startup is included in these wall-clock measurements.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
