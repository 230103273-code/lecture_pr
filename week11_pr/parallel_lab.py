#!/usr/bin/env python3
"""
Minimal Python parallel systems lab runner.
Python 3.8+; standard library only.

Run:
    python parallel_lab.py
    python parallel_lab.py --id YOUR_ID
    python parallel_lab.py --quick --id YOUR_ID

The default run uses reduced workloads so you can verify that everything works.
Use --full for the assignment-scale workloads. Measurements are machine-dependent.
"""
import argparse
import ctypes
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import statistics
import sys
import time
from pathlib import Path

MASK64 = (1 << 64) - 1
M_FULL = 16_000_000
N_GEMM = 256
N_STRONG = 1_200_000
N0_WEAK = 200_000
N_SCHED = 8_000

def fnv1a_hash(s):
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & MASK64
    return h

def verify(phase, seed, m1, m2, cores):
    sig = fnv1a_hash(f"{phase}|{seed}|{cores}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH[{phase}]: {sig:016X} (P_log={cores})"

def stats(samples):
    x = sorted(samples)
    n = len(x)
    return {
        "min": x[0],
        "median": statistics.median(x),
        "iqr": x[(3*n)//4] - x[n//4],
    }

def bench(fn, runs=3, warmup=1):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - t0) / 1e6)
    return stats(out)

# ---- Phase 1: false sharing ----
def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & MASK64
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_fs(p, stride, total_steps):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    for i in range(512):
        arr[i] = i + 1
    del arr
    procs = []
    each = total_steps // p
    t0 = time.perf_counter_ns()
    try:
        for w in range(p):
            proc = mp.Process(target=fs_worker, args=(shm.name, w * stride, each))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("False-sharing worker failed")
        return (time.perf_counter_ns() - t0) / 1e6
    finally:
        shm.close()
        shm.unlink()

# ---- Phase 2: scaling ----
def stage_a(n):
    total = 0.0
    for i in range(max(1, int(n * 0.05))):
        x = i * 0.00001 + 0.1
        for _ in range(10):
            total += math.sin(x) + math.sqrt(abs(x))
    return total

def stage_b_range(args):
    start, end = args
    total = 0.0
    for i in range(start, end):
        x = i * 0.000001 + 0.01
        # A repeatable trig-heavy workload (30 operations per element).
        for _ in range(10):
            total += math.sin(x) * math.cos(x) + math.sqrt(abs(x))
            x += 0.000001
    return total

def chunks(n, p):
    q, r = divmod(n, p)
    out, start = [], 0
    for i in range(p):
        end = start + q + (1 if i < r else 0)
        out.append((start, end))
        start = end
    return out

def pipeline(pool, n, p):
    a0 = time.perf_counter_ns()
    stage_a(n)
    a_ms = (time.perf_counter_ns() - a0) / 1e6
    b0 = time.perf_counter_ns()
    pool.map(stage_b_range, chunks(n, p))
    b_ms = (time.perf_counter_ns() - b0) / 1e6
    return a_ms + b_ms, a_ms

# ---- Phase 3: roofline and GEMM ----
def copy_worker(args):
    name_a, name_b, start, end = args
    a = shared_memory.SharedMemory(name=name_a)
    b = shared_memory.SharedMemory(name=name_b)
    av, bv = a.buf, b.buf
    av[start:end] = bv[start:end]
    del av, bv
    a.close()
    b.close()

def bandwidth_test(p, size_bytes):
    a = shared_memory.SharedMemory(create=True, size=size_bytes)
    b = shared_memory.SharedMemory(create=True, size=size_bytes)
    try:
        b.buf[:] = b"x" * size_bytes
        ranges = chunks(size_bytes, p)
        jobs = [(a.name, b.name, s, e) for s, e in ranges]
        with mp.Pool(p) as pool:
            def once():
                pool.map(copy_worker, jobs)
            st = bench(once, runs=3, warmup=1)
        # One read + one write per byte copied.
        gbps = (2 * size_bytes) / (st["median"] * 1e6)
        return st, gbps
    finally:
        a.close(); a.unlink()
        b.close(); b.unlink()

def compute_worker(iters):
    vals = [0.1 + i * 0.001 for i in range(16)]
    for k in range(iters):
        j = k & 15
        vals[j] = vals[j] * 1.0000001 + 0.0000003
    return sum(vals)

def gemm_worker(args):
    n, a_name, b_name, c_name, i0, i1, mode, bs = args
    a_sh = shared_memory.SharedMemory(name=a_name)
    b_sh = shared_memory.SharedMemory(name=b_name)
    c_sh = shared_memory.SharedMemory(name=c_name)
    A = (ctypes.c_double * (n*n)).from_buffer(a_sh.buf)
    B = (ctypes.c_double * (n*n)).from_buffer(b_sh.buf)
    C = (ctypes.c_double * (n*n)).from_buffer(c_sh.buf)
    if mode == "naive":
        for i in range(i0, i1):
            ib = i*n
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[ib+k] * B[k*n+j]
                C[ib+j] = s
    elif mode == "ikj":
        for i in range(i0, i1):
            ib = i*n
            for k in range(n):
                aik = A[ib+k]
                kb = k*n
                for j in range(n):
                    C[ib+j] += aik * B[kb+j]
    else:
        for ii in range(i0, i1, bs):
            iend = min(ii+bs, i1)
            for kk in range(0, n, bs):
                kend = min(kk+bs, n)
                for jj in range(0, n, bs):
                    jend = min(jj+bs, n)
                    for i in range(ii, iend):
                        ib = i*n
                        for k in range(kk, kend):
                            aik = A[ib+k]
                            kb = k*n
                            for j in range(jj, jend):
                                C[ib+j] += aik * B[kb+j]
    del A, B, C
    a_sh.close(); b_sh.close(); c_sh.close()

def run_gemm(n, p, mode, bs=32):
    a = shared_memory.SharedMemory(create=True, size=n*n*8)
    b = shared_memory.SharedMemory(create=True, size=n*n*8)
    c = shared_memory.SharedMemory(create=True, size=n*n*8)
    A = (ctypes.c_double * (n*n)).from_buffer(a.buf)
    B = (ctypes.c_double * (n*n)).from_buffer(b.buf)
    C = (ctypes.c_double * (n*n)).from_buffer(c.buf)
    for i in range(n*n):
        A[i] = ((i % 17) + 1) / 17.0
        B[i] = ((i % 13) + 1) / 13.0
        C[i] = 0.0
    del A, B, C
    jobs = [(n, a.name, b.name, c.name, s, e, mode, bs) for s, e in chunks(n, p)]
    try:
        with mp.Pool(p) as pool:
            def once():
                # Reset C before each sample so accumulation variants are comparable.
                Creset = (ctypes.c_double * (n*n)).from_buffer(c.buf)
                for z in range(n*n):
                    Creset[z] = 0.0
                del Creset
                pool.map(gemm_worker, jobs)
            st = bench(once, runs=3, warmup=0)
        return st
    finally:
        a.close(); a.unlink()
        b.close(); b.unlink()
        c.close(); c.unlink()

# ---- Phase 4: scheduling ----
def item_work(i):
    k = 1 + (i // 40) ** 2
    x = i * 0.0001 + 0.1
    z = 0.0
    for _ in range(k):
        z += math.sin(x) + math.cos(x) + math.sqrt(abs(x))
        x += 0.00001
    return z

def sched_worker_static(worker, p, n, mode, times, result):
    t0 = time.perf_counter_ns()
    if mode == "block":
        s, e = (n*worker)//p, (n*(worker+1))//p
        ids = range(s, e)
    else:
        ids = range(worker, n, p)
    v = 0.0
    for i in ids:
        v += item_work(i)
    times[worker] = (time.perf_counter_ns()-t0)/1e6
    result[worker] = v

def sched_worker_dynamic(worker, p, n, chunk, counter, lock, times, result):
    t0 = time.perf_counter_ns()
    v = 0.0
    while True:
        with lock:
            s = counter.value
            counter.value += chunk
        if s >= n:
            break
        for i in range(s, min(s+chunk, n)):
            v += item_work(i)
    times[worker] = (time.perf_counter_ns()-t0)/1e6
    result[worker] = v

def schedule_run(p, n, mode, chunk=1):
    times = mp.Array('d', p)
    result = mp.Array('d', p)
    procs = []
    t0 = time.perf_counter_ns()
    if mode == "dynamic":
        counter = mp.Value('i', 0)
        lock = mp.Lock()
        for w in range(p):
            q = mp.Process(target=sched_worker_dynamic,
                args=(w,p,n,chunk,counter,lock,times,result))
            q.start(); procs.append(q)
    else:
        for w in range(p):
            q = mp.Process(target=sched_worker_static,
                args=(w,p,n,mode,times,result))
            q.start(); procs.append(q)
    for q in procs:
        q.join()
        if q.exitcode:
            raise RuntimeError("Scheduling worker failed")
    wall = (time.perf_counter_ns()-t0)/1e6
    active = list(times)
    mean = sum(active)/p if p else 0.0
    imbalance = max(active)/mean if mean else 0.0
    return wall, imbalance, active

def main():
    ap = argparse.ArgumentParser(description="Parallel computing laptop lab")
    ap.add_argument("--id", default="YOUR_ID", help="Student ID for report/hash seed")
    ap.add_argument("--quick", action="store_true", help="Short validation run")
    ap.add_argument("--full", action="store_true", help="Use assignment-scale workloads")
    args = ap.parse_args()
    seed = args.id
    logical = mp.cpu_count() or 1
    physical_guess = max(1, (os.cpu_count() or 1)//2)
    physical = int(input(f"Enter physical core count reported by your OS [{physical_guess}]: ") or physical_guess)
    physical = max(1, min(physical, logical))
    total_steps = M_FULL if args.full else 300_000
    strong_n = N_STRONG if args.full else 60_000
    weak_n0 = N0_WEAK if args.full else 10_000
    gemm_n = N_GEMM if args.full else 64
    sched_n = N_SCHED if args.full else 800

    lines = []
    def log(s=""):
        print(s)
        lines.append(str(s))

    log("# Parallel Computing Lab Report")
    log(f"\n- Student ID: `{seed}`")
    log(f"- Date/time: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    log(f"- OS: `{platform.platform()}`")
    log(f"- Python: `{sys.version.split()[0]}`")
    log(f"- Logical CPUs (`mp.cpu_count()`): **{logical}**")
    log(f"- Physical cores used (`P_phys`): **{physical}**")
    log(f"- Mode: **{'FULL' if args.full else 'QUICK validation'}**")
    log("\n> Record exact CPU model, physical/logical cores, cache sizes and cache-line size from the OS commands in the assignment. The standard library does not expose these reliably on every OS.")
    # Phase 0
    samples = []
    for _ in range(10_000 if args.full else 1_000):
        t0 = time.perf_counter_ns(); time.perf_counter_ns()
        samples.append(time.perf_counter_ns()-t0)
    call_ns = statistics.mean(samples)
    tag0 = verify("PHASE0", seed, call_ns, logical, logical)
    log("\n## Phase 0 — Hardware and timing")
    log(f"- Average perf_counter_ns call-pair latency: {call_ns:.2f} ns")
    log(tag0)

    # Phase 1
    log("\n## Phase 1 — False sharing and scaling")
    strides = [1,2,4,8,16,32]
    fs_results = {}
    for stride in strides:
        ms = run_fs(physical, stride, total_steps)
        fs_results[stride] = ms
        log(f"- stride={stride} ({stride*8} B), p={physical}: {ms:.3f} ms, throughput={total_steps/ms/1000:.3f} Mops/s")
    scale_results = {}
    ps = sorted(set([1,2,4,8,logical]))
    ps = [p for p in ps if p <= logical]
    for stride in (1,16):
        t1 = None
        for p in ps:
            ms = run_fs(p, stride, total_steps)
            if p == 1: t1 = ms
            speed = t1/ms
            eff = speed/p
            karp = ((1/speed)-(1/p))/(1-(1/p)) if p > 1 else float("nan")
            scale_results[(stride,p)] = ms
            log(f"- stride={stride}, p={p}: {ms:.3f} ms, S={speed:.3f}, E={eff:.3f}, Karp-Flatt={karp:.5f}")
    tag1 = verify("PHASE1", seed, fs_results[1], fs_results[16], logical)
    log(tag1)

    # Phase 2
    log("\n## Phase 2 — Strong and weak scaling")
    strong = {}
    for p in range(1, logical+1):
        with mp.Pool(processes=p) as pool:
            ms, a_ms = pipeline(pool, strong_n, p)
        strong[p] = (ms, a_ms)
        t1 = strong.get(1, (ms,a_ms))[0]
        log(f"- Strong p={p}: T={ms:.3f} ms, StageA={a_ms:.3f} ms, speedup={t1/ms:.3f}, efficiency={t1/ms/p:.3f}")
    weak = {}
    for p in range(1, logical+1):
        with mp.Pool(processes=p) as pool:
            ms, a_ms = pipeline(pool, weak_n0*p, p)
        weak[p] = (ms,a_ms)
        t1 = weak.get(1, (ms,a_ms))[0]
        alpha = a_ms/ms if ms else 0.0
        gust = p - alpha*(p-1)
        log(f"- Weak p={p}: T={ms:.3f} ms, efficiency={t1/ms:.3f}, alpha={alpha:.4f}, Gustafson={gust:.3f}")
    sstrong = strong[1][0]/strong[logical][0]
    alpha_last = weak[logical][1]/weak[logical][0]
    gust_last = logical-alpha_last*(logical-1)
    tag2 = verify("PHASE2", seed, sstrong, gust_last, logical)
    log(f"- Strong speedup at P_log={logical}: {sstrong:.3f}; Gustafson scaled speedup: {gust_last:.3f}")
    log(tag2)

    # Phase 3
    log("\n## Phase 3 — Empirical roofline and GEMM")
    size = 64*1024*1024 if args.full else 512*1024
    bw_stats, bw = bandwidth_test(physical, size)
    log(f"- Shared-memory copy: median={bw_stats['median']:.3f} ms, sustained bandwidth={bw:.3f} GB/s")
    # Approximate compute ceiling using 16 independent accumulators.
    iters = 30_000_000 if args.full else 100_000
    with mp.Pool(physical) as pool:
        t0 = time.perf_counter_ns()
        pool.map(compute_worker, [iters]*physical)
        compute_ms = (time.perf_counter_ns()-t0)/1e6
    # This loop is roughly two floating operations per iteration; explicitly label estimate.
    p_gflops = (physical*iters*2)/(compute_ms*1e6)
    ridge = p_gflops/bw if bw else 0.0
    log(f"- Scalar-loop compute estimate: {p_gflops:.4f} GFLOP/s; ridge point={ridge:.6f} FLOP/byte")
    gemm = {}
    for mode in ("naive","ikj"):
        st = run_gemm(gemm_n, physical, mode)
        gflops = (2*gemm_n**3)/(st["median"]*1e6)
        gemm[mode] = (st["median"],gflops)
        log(f"- GEMM {mode}: median={st['median']:.3f} ms, {gflops:.5f} GFLOP/s")
    bs_candidates = [8,16,32,64,96,128]
    # Keep tile sizes sensible for the selected quick/full matrix.
    bs_candidates = [b for b in bs_candidates if b <= gemm_n]
    for bs in bs_candidates:
        st = run_gemm(gemm_n, physical, "blocked", bs)
        gflops = (2*gemm_n**3)/(st["median"]*1e6)
        gemm[("blocked",bs)] = (st["median"],gflops)
        log(f"- GEMM blocked BS={bs}: median={st['median']:.3f} ms, {gflops:.5f} GFLOP/s, tile footprint={3*bs*bs*8} B")
    best_bs = max(bs_candidates, key=lambda b: gemm[("blocked",b)][1])
    tag3 = verify("PHASE3", seed, gemm["naive"][1], gemm[("blocked",best_bs)][1], logical)
    log(f"- Best measured blocked tile: BS={best_bs}; OI estimate={best_bs/12:.3f} FLOP/byte")
    log(tag3)

    # Phase 4
    log("\n## Phase 4 — Skewed scheduling")
    block = schedule_run(physical,sched_n,"block")
    cyclic = schedule_run(physical,sched_n,"cyclic")
    log(f"- Block static: wall={block[0]:.3f} ms, lambda={block[1]:.3f}")
    log(f"- Cyclic static: wall={cyclic[0]:.3f} ms, lambda={cyclic[1]:.3f}")
    dyn = {}
    for c in (1,8,64,512):
        dyn[c] = schedule_run(physical,sched_n,"dynamic",c)
        log(f"- Dynamic C={c}: wall={dyn[c][0]:.3f} ms, lambda={dyn[c][1]:.3f}")
    best_c = min(dyn, key=lambda c: dyn[c][0])
    tag4 = verify("PHASE4", seed, block[0], dyn[best_c][0], logical)
    log(f"- Best measured dynamic chunk: C={best_c}")
    log(tag4)

    log("\n## Diagnostic notes")
    log("- False sharing: compare stride 1 with stride 16; a sharp improvement at larger stride is consistent with cache-line contention. The actual cache-line boundary must be checked against your CPU.")
    log("- Karp–Flatt: increasing values with p indicate growing non-parallel overheads; compare both strides rather than assuming all overhead is serial code.")
    log("- Strong scaling: speedup is limited by serial Stage A, process scheduling, and memory/cache effects. Weak scaling is sensitive to how Stage A grows with total N.")
    log("- Roofline: Python interpreter overhead means these are empirical Python-workload ceilings, not hardware peak FLOP/s. GEMM loop order changes locality; blocked tile size should be interpreted alongside actual L1/L2 sizes.")
    log("- Scheduling: C=1 increases lock traffic; very large chunks can leave a long tail. Select C* from the measured minimum wall time, not from a theoretical guess.")
    report = Path(f"lab_report_{seed}.md")
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"\nReport written to: {report.resolve()}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
