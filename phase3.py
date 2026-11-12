#!/usr/bin/env python3
"""Phase 2: strong/weak scaling experiment for the parallel_lab assignment.

Uses only the Python standard library. Pools are created outside timed regions.
The default run performs 2 warm-ups and 7 timed repetitions per point.
"""

import math
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from pathlib import Path

N_STRONG = 1_600_000
N0_WEAK = 250_000
WARMUPS = 2
TIMED_RUNS = 7
STAGE_A_FRACTION = 0.05
STAGE_A_OPS = 12
STAGE_B_OPS = 35


def stage_a(n: int) -> float:
    """Serial prefix stage; return a checksum to prevent dead-code-like simplification."""
    count = max(1, int(n * STAGE_A_FRACTION))
    acc = 0.123456789
    for i in range(count):
        x = (i % 997) * 0.001 + 0.1
        for j in range(STAGE_A_OPS):
            acc += math.sin(x + j * 0.001)
            acc += math.sqrt(abs(x) + 1.0)
            x = (x * 1.0000001 + 0.0000003) % 10.0
    return acc


def stage_b_worker(bounds):
    """Compute the assigned range using a deterministic 35-step FP/trig kernel."""
    start, end = bounds
    acc = 0.0
    for i in range(start, end):
        x = (i % 997) * 0.001 + 0.1
        for j in range(STAGE_B_OPS):
            x = math.sin(x * 1.0000001 + 0.0000003)
        acc += x
    return acc


def ranges_for(n: int, p: int):
    q, r = divmod(n, p)
    out = []
    start = 0
    for worker in range(p):
        size = q + (1 if worker < r else 0)
        out.append((start, start + size))
        start += size
    return out


def one_trial(pool, p: int, n: int) -> tuple[float, float]:
    """Return total elapsed ms and Stage-A-only elapsed ms for one pipeline trial."""
    t0 = time.perf_counter_ns()
    a0 = time.perf_counter_ns()
    checksum_a = stage_a(n)
    a1 = time.perf_counter_ns()
    chunks = ranges_for(n, p)
    results = pool.map(stage_b_worker, chunks)
    # Consume results deterministically.
    checksum = checksum_a + sum(results)
    if not math.isfinite(checksum):
        raise RuntimeError("Non-finite checksum")
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, (a1 - a0) / 1e6


def median_iqr(values):
    s = sorted(values)
    med = statistics.median(s)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    return med, q3 - q1


def benchmark_point(p: int, n: int):
    # Persistent Pool is deliberately outside all timed regions.
    with mp.Pool(processes=p) as pool:
        for _ in range(WARMUPS):
            one_trial(pool, p, n)
        totals, serials = [], []
        for _ in range(TIMED_RUNS):
            total_ms, serial_ms = one_trial(pool, p, n)
            totals.append(total_ms)
            serials.append(serial_ms)
    total_med, total_iqr = median_iqr(totals)
    serial_med, _ = median_iqr(serials)
    return total_med, total_iqr, serial_med


def fnv1a_hash(text: str) -> int:
    h = 14695981039346656037
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def verify_hash(student_id: str, p_log: int, strong_speedup: float, gustafson_speedup: float):
    payload = f"PHASE2|{student_id}|{p_log}|{strong_speedup:.4f}|{gustafson_speedup:.4f}"
    return f"VERIFY_HASH[PHASE2]: {fnv1a_hash(payload):016X} (P_log={p_log})"


def main():
    print("Phase 2 | Strong and weak scaling")
    student_id = input("Enter your Student ID: ").strip()
    if not student_id.isdigit():
        print("Student ID must contain digits only.")
        return

    p_log = mp.cpu_count()
    p_phys = 6  # Phase 0 Windows CIM result for this laptop (Ryzen 5 7430U).
    # Keep topology grounded in the successful Phase 0 measurement.
    if p_log != 12:
        print(f"WARNING: Phase 0 reported P_log=12, but this run sees P_log={p_log}.")
    print(f"OS={platform.platform()} | Python={sys.version.split()[0]} | P_phys={p_phys} | P_log={p_log}")
    print(f"Strong N={N_STRONG:,}; weak N0={N0_WEAK:,}; warmups={WARMUPS}; timed_runs={TIMED_RUNS}")
    print("NOTE: pool startup is excluded; serial Stage A and pool.map dispatch are included in trial time.")

    report_dir = Path("parallel_lab")
    report_dir.mkdir(exist_ok=True)
    out_path = report_dir / "phase2_results.txt"
    lines = []
    lines.append("PHASE 2 RAW RESULTS")
    lines.append(f"Student ID: {student_id}")
    lines.append(f"OS: {platform.platform()}")
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"P_phys: {p_phys}; P_log: {p_log}")
    lines.append(f"N_strong: {N_STRONG}; N0_weak: {N0_WEAK}; warmups: {WARMUPS}; timed_runs: {TIMED_RUNS}")

    ps = list(range(1, p_log + 1))
    strong = {}
    stage_a_one = None
    print("\n=== Experiment 2A: Strong scaling ===")
    print("p, median_ms, IQR_ms, speedup, efficiency, StageA_median_ms")
    lines.append("\n=== Experiment 2A: Strong scaling ===")
    lines.append("p, median_ms, IQR_ms, speedup, efficiency, StageA_median_ms")
    for p in ps:
        print(f"Running strong-scaling p={p}/{p_log} ...", flush=True)
        med, iqr, a_med = benchmark_point(p, N_STRONG)
        strong[p] = (med, iqr, a_med)
        if p == 1:
            stage_a_one = a_med
        print(f"{p}, {med:.3f}, {iqr:.3f}, pending, pending, {a_med:.3f}")
        lines.append(f"{p}, {med:.3f}, {iqr:.3f}, pending, pending, {a_med:.3f}")

    t1 = strong[1][0]
    strong_rows = {}
    print("\nStrong scaling derived values:")
    lines.append("\nStrong scaling derived values:")
    lines.append("p, median_ms, IQR_ms, speedup, efficiency, StageA_median_ms")
    for p in ps:
        med, iqr, a_med = strong[p]
        speedup = t1 / med
        efficiency = speedup / p
        strong_rows[p] = (med, iqr, speedup, efficiency, a_med)
        row = f"{p}, {med:.3f}, {iqr:.3f}, {speedup:.4f}, {efficiency:.4f}, {a_med:.3f}"
        print(row)
        lines.append(row)

    weak = {}
    print("\n=== Experiment 2B: Weak scaling ===")
    print("p, N(p), median_ms, IQR_ms, alpha, Gustafson_speedup, efficiency")
    lines.append("\n=== Experiment 2B: Weak scaling ===")
    lines.append("p, N(p), median_ms, IQR_ms, alpha, Gustafson_speedup, efficiency")
    for p in ps:
        n = p * N0_WEAK
        print(f"Running weak-scaling p={p}/{p_log}, N={n:,} ...", flush=True)
        med, iqr, a_med = benchmark_point(p, n)
        alpha = a_med / med if med else 0.0
        weak[p] = (med, iqr, alpha, a_med)
        # Gustafson scaled speedup, using alpha(p) measured for this weak-scaling run.
        gust = p - alpha * (p - 1)
        eff = gust / p
        row = f"{p}, {n}, {med:.3f}, {iqr:.3f}, {alpha:.6f}, {gust:.4f}, {eff:.4f}"
        print(row)
        lines.append(row)

    s = stage_a_one / strong[1][0] if strong[1][0] else 0.0
    s_inf = (1.0 / s) if s > 0 else float("inf")
    amdahl_log = 1.0 / (s + (1.0 - s) / p_log) if s > 0 else float("inf")
    observed_log = strong_rows[p_log][2]
    gust_log = p_log - weak[p_log][2] * (p_log - 1)

    # Marginal speedup per added worker around the physical/logical boundary.
    before = strong_rows[p_phys][2] - strong_rows[p_phys - 1][2] if p_phys > 1 else float("nan")
    after = strong_rows[p_phys + 1][2] - strong_rows[p_phys][2] if p_phys < p_log else float("nan")
    drop_pct = ((before - after) / before * 100.0) if before > 0 and math.isfinite(after) else float("nan")

    lines.append("\n=== Analytical summary ===")
    lines.append(f"Serial fraction s = StageA_median(1)/Strong_median(1) = {s:.8f}")
    lines.append(f"Amdahl S_infinity = {s_inf:.6f}")
    lines.append(f"Amdahl theoretical speedup at P_log={p_log}: {amdahl_log:.6f}")
    lines.append(f"Observed strong speedup at P_log={p_log}: {observed_log:.6f}")
    lines.append(f"Gustafson scaled speedup at P_log={p_log}: {gust_log:.6f}")
    lines.append(f"Marginal strong-speedup gain p={p_phys-1}->{p_phys}: {before:.6f}")
    lines.append(f"Marginal strong-speedup gain p={p_phys}->{p_phys+1}: {after:.6f}")
    lines.append(f"Marginal gain drop across P_phys boundary: {drop_pct:.3f}%")
    tag = verify_hash(student_id, p_log, observed_log, gust_log)
    lines.append(tag)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n=== Analytical summary ===")
    print(f"Serial fraction s = {s:.8f}")
    print(f"Amdahl S_infinity = {s_inf:.6f}")
    print(f"Amdahl theoretical speedup at P_log={p_log}: {amdahl_log:.6f}")
    print(f"Observed strong speedup at P_log={p_log}: {observed_log:.6f}")
    print(f"Gustafson scaled speedup at P_log={p_log}: {gust_log:.6f}")
    print(f"Marginal gain p={p_phys-1}->{p_phys}: {before:.6f}")
    print(f"Marginal gain p={p_phys}->{p_phys+1}: {after:.6f}")
    print(f"Marginal gain drop across P_phys boundary: {drop_pct:.3f}%")
    print(tag)
    print(f"Results saved to: {out_path.resolve()}")
    print("Keep the laptop plugged in and avoid other heavy workloads during the run.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
