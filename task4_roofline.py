import time
import multiprocessing as mp
import numpy as np


def compute_heavy(n):
    x = 1.0001

    for _ in range(n):
        x = (x * 1.000001) + 0.00001

    return x


def memory_heavy(size):
    arr = np.ones(size, dtype=np.float64)
    arr = arr * 2.0 + 1.0
    return arr[0]


def run_suite():

    print("=== Compute-Bound Suite (Register Math) ===")

    compute_times = {}

    for w in [1, 2, 4]:
        t0 = time.perf_counter()

        with mp.Pool(w) as p:
            p.map(compute_heavy, [25_000_000] * w)

        elapsed = time.perf_counter() - t0
        compute_times[w] = elapsed

        print(f"Workers: {w} | ExecutionTime: {elapsed:.4f}s")

    print("\n=== Memory-Bound Suite (DRAM Bandwidth Saturation) ===")

    memory_times = {}

    for w in [1, 2, 4]:
        t0 = time.perf_counter()

        with mp.Pool(w) as p:
            p.map(memory_heavy, [50_000_000] * w)

        elapsed = time.perf_counter() - t0
        memory_times[w] = elapsed

        print(f"Workers: {w} | ExecutionTime: {elapsed:.4f}s")

    # Scaling analysis
    print("\n=== Scaling Analysis ===")

    t_compute_1 = compute_times[1]
    t_memory_1 = memory_times[1]

    print("Workers | Compute Time | Compute Tw/T1 | Memory Time | Memory Tw/T1")
    print("-" * 70)

    for w in [1, 2, 4]:
        compute_ratio = compute_times[w] / t_compute_1
        memory_ratio = memory_times[w] / t_memory_1

        print(
            f"{w:7d} | "
            f"{compute_times[w]:12.4f} | "
            f"{compute_ratio:14.2f} | "
            f"{memory_times[w]:11.4f} | "
            f"{memory_ratio:13.2f}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    run_suite()



# ============================================================
# TASK 4 — QUESTIONS / ANSWERS
# ============================================================

# Q4.1 — Compute-Bound vs Memory-Bound Scaling
#
# Measured results:
#
# Compute-Bound:
#   1 worker = 2.1465 s
#   2 workers = 2.5129 s
#   4 workers = 2.2646 s
#
# Memory-Bound:
#   1 worker = 1.0866 s
#   2 workers = 1.2724 s
#   4 workers = 2.1424 s
#
# Compute scaling for 4 workers:
#
#   T4 / T1 = 2.2646 / 2.1465 = 1.06x
#
# Therefore, four parallel compute tasks complete in
# approximately the same time as one task.
#
# The compute-bound workload is primarily limited by CPU
# computational resources. Multiple processes can execute
# arithmetic operations in parallel on different CPU cores.
#
# Memory scaling for 4 workers:
#
#   T4 / T1 = 2.1424 / 1.0866 = 1.97x
#
# The memory-bound workload almost doubles in execution time.
#
# Each worker processes approximately:
#
#   50,000,000 * 8 bytes = 400,000,000 bytes
#
# Multiple workers generate simultaneous DRAM traffic.
# The memory subsystem has a finite bandwidth shared by all
# CPU cores. Once the DRAM bandwidth is saturated, adding more
# workers does not increase throughput.
#
# Instead, workers compete for the memory bus and memory
# controllers, increasing queueing latency and total execution
# time.
#
# This experiment demonstrates the memory wall:
#
#     More CPU parallelism != more performance
#     when the workload is limited by DRAM bandwidth.


# Q4.2 — Host Memory Architecture
#
# CPU:
#     AMD Ryzen 5 7430U with Radeon Graphics
#
# Physical CPU cores:
#     6
#
# Logical processors:
#     12
#
# Installed memory:
#     16 GB
#
# Memory speed:
#     DDR4-3200
#
# DDR4-3200 transfers:
#     3200 MT/s
#
# A 64-bit memory channel transfers 8 bytes per cycle.
#
# Theoretical bandwidth for one channel:
#
#     3200 MT/s * 8 bytes
#     = 25.6 GB/s
#
# If the system operates in dual-channel mode:
#
#     25.6 GB/s * 2
#     = 51.2 GB/s
#
# This is a theoretical peak bandwidth. Real measured
# bandwidth is lower because of memory controller overhead,
# access patterns, refresh operations and other system activity.
#
# The benchmark demonstrates why memory bandwidth limits
# multi-core scalability. All CPU cores share the same memory
# subsystem. When several workers continuously stream data from
# DRAM, the memory bus can become saturated.
#
# After saturation, adding CPU workers only increases
# contention and queueing instead of increasing useful
# throughput.


# Q4.3 — Architectural Choice for ML Embedding Pipeline
#
# For a memory-bandwidth-saturated embedding pipeline,
# simply adding more CPU OpenMP workers does not remove the
# primary bottleneck.
#
# Multiple CPU cores would compete for the same DRAM bandwidth.
# The Task 4 benchmark demonstrates this behavior:
#
#     Memory T1 = 1.0866 s
#     Memory T4 = 2.1424 s
#
#     T4/T1 = 1.97x
#
# Therefore, increasing CPU parallelism can increase memory
# contention rather than improving throughput.
#
# CUDA GPU kernels can be useful when the GPU provides much
# higher memory bandwidth and the data can remain in GPU memory.
# However, transferring large feature vectors repeatedly between
# CPU DRAM and GPU memory can introduce PCIe transfer overhead.
#
# Therefore, the architectural decision should depend on where
# the data resides and what limits performance.
#
# For a genuinely memory-bandwidth-bound workload, the main
# optimization targets are:
#
#   1. Increase available memory bandwidth.
#   2. Reduce the amount of data transferred.
#   3. Improve memory locality and access patterns.
#   4. Reuse data in cache when possible.
#   5. Keep data on the accelerator when using a GPU.
#
# The key architectural principle is:
#
#     When memory bandwidth is the bottleneck, adding more
#     compute resources alone does not solve the problem.
#
# Task 4 demonstrates the difference between compute saturation
# and memory bandwidth saturation.