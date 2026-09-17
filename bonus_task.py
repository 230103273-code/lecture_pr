import time
from multiprocessing import Pool
import threading


def cpu_work(n):
    # Pure CPU crunch
    count = 0
    for i in range(n):
        count += i * i
    return count


TOTAL_WORK = 50_000_000
WORKERS = [1, 2, 4, 8, 16, 32]


def run_threading(workers):
    work_per_thread = TOTAL_WORK // workers
    threads = []

    start = time.perf_counter()

    for _ in range(workers):
        t = threading.Thread(
            target=cpu_work,
            args=(work_per_thread,)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return time.perf_counter() - start


def run_multiprocessing(workers):
    work_per_process = TOTAL_WORK // workers

    start = time.perf_counter()

    with Pool(workers) as p:
        p.map(cpu_work, [work_per_process] * workers)

    return time.perf_counter() - start


if __name__ == "__main__":
    print("=== BONUS TASK: GIL vs MULTIPROCESSING ===")
    print()

    print("Workers | Threading Time | Multiprocessing Time")
    print("-" * 52)

    threading_results = {}
    multiprocessing_results = {}

    for workers in WORKERS:
        t_thread = run_threading(workers)
        t_process = run_multiprocessing(workers)

        threading_results[workers] = t_thread
        multiprocessing_results[workers] = t_process

        print(
            f"{workers:7d} | "
            f"{t_thread:15.4f} | "
            f"{t_process:21.4f}"
        )

# ============================================================
# BONUS TASK — ARCHITECTURAL AUTOPSY
# ============================================================

# Experimental results:
#
# Workers | Threading Time | Multiprocessing Time
# ------------------------------------------------
#       1 | 5.9061 s        | 4.6087 s
#       2 | 4.3191 s        | 2.5243 s
#       4 | 4.4093 s        | 1.5166 s
#       8 | 4.2840 s        | 0.9856 s
#      16 | 4.2661 s        | 1.0400 s
#      32 | 4.3984 s        | 1.6601 s
#
#
# 1. EMPIRICAL CURVE
#
# The threading execution time remains approximately flat
# after the first worker:
#
#     1 worker  = 5.9061 s
#     2 workers = 4.3191 s
#     4 workers = 4.4093 s
#     8 workers = 4.2840 s
#     16 workers = 4.2661 s
#     32 workers = 4.3984 s
#
# The multiprocessing execution time decreases strongly
# as the number of workers increases up to 8 workers:
#
#     1 worker  = 4.6087 s
#     2 workers = 2.5243 s
#     4 workers = 1.5166 s
#     8 workers = 0.9856 s
#
# After 8 workers, performance starts to degrade:
#
#     16 workers = 1.0400 s
#     32 workers = 1.6601 s
#
#
# 2. INFLECTION CLIFF
#
# The exact observed inflection point in this experiment
# is 8 workers.
#
# The best measured multiprocessing execution time was:
#
#     8 workers = 0.9856 s
#
# Increasing the number of workers from 8 to 16 increased
# execution time:
#
#     0.9856 s -> 1.0400 s
#
# Increasing to 32 workers increased it further:
#
#     0.9856 s -> 1.6601 s
#
# Therefore, 8 workers is the first sampled point after which
# adding more workers makes the program slower.
#
# This is an empirical inflection point. It should not be
# interpreted as the exact physical core count because the
# experiment sampled 1, 2, 4, 8, 16 and 32 workers and did not
# test exactly 6 workers.
#
#
# 3. ARCHITECTURAL AUTOPSY — GIL
#
# The threading curve remains approximately flat because the
# benchmark is CPU-bound Python code:
#
#     count += i * i
#
# CPython uses the Global Interpreter Lock (GIL), which prevents
# multiple threads from simultaneously executing Python bytecode
# in the same process.
#
# Therefore, adding more CPU threads does not provide linear
# CPU parallelism for this workload.
#
# Additional threads introduce scheduling and context-switching
# overhead without providing proportional computational progress.
#
#
# 4. ARCHITECTURAL AUTOPSY — MULTIPROCESSING
#
# multiprocessing.Pool creates separate Python processes.
# Each process has its own Python interpreter and its own GIL.
#
# Therefore, different processes can execute CPU-bound Python
# code simultaneously on different CPU cores.
#
# This explains the observed scaling:
#
#     1 worker  = 4.6087 s
#     2 workers = 2.5243 s
#     4 workers = 1.5166 s
#     8 workers = 0.9856 s
#
#
# 5. CPU TOPOLOGY
#
# Host CPU:
#
#     AMD Ryzen 5 7430U with Radeon Graphics
#
# Physical CPU cores:
#     6
#
# Logical processors:
#     12
#
# Therefore, the processor has 6 physical cores and 12 logical
# processors.
#
# The experiment does not contain a measurement at exactly
# 6 workers, so the exact physical-core saturation point cannot
# be proven from the sampled data alone.
#
# The measured curve reaches its minimum at 8 workers and then
# becomes slower at 16 and 32 workers.
#
#
# 6. WHY PERFORMANCE DROPS AFTER SATURATION
#
# Once available CPU execution resources are saturated, adding
# more processes does not create additional physical CPU cores.
#
# Instead, the operating system must schedule more runnable
# processes onto the available cores.
#
# This increases:
#
#     - context-switch overhead;
#     - scheduler overhead;
#     - CPU-cache disruption;
#     - competition for execution resources.
#
# Consequently, the total execution time can increase even
# though the number of workers is larger.
#
#
# FINAL OBSERVATION
#
# The experiment demonstrates two important properties of CPython:
#
#     Threads:
#         CPU-bound Python workloads do not scale well because
#         of the GIL.
#
#     Processes:
#         CPU-bound workloads can scale across physical cores,
#         but only until the available CPU resources become
#         saturated.
#
# In this experiment the best sampled multiprocessing result
# occurred at 8 workers with an execution time of 0.9856 seconds.
# Performance degraded at 16 and 32 workers.