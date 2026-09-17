import time
import os
import multiprocessing as mp

def cpu_intensive(chunk_size):
    acc = 0
    for i in range(chunk_size):
        acc += (i%7)*(i%11)
    return acc

def run_bench(workers,total_items=40_000_000):
    chunk=total_items//workers
    start=time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        pool.map(cpu_intensive,[chunk]*workers)
    return time.perf_counter()-start

if __name__=="__main__":
    cores=[1,2,4,8,12,16]
    base=run_bench(1)
    print(f"OSReportedLogicalCores:{os.cpu_count()}")
    print(f"SingleProcessBaselineTimeT(1):{base:.4f}s\n")
    print("Cores|Time(s)|ObservedSpeedup|TheoreticalLinear")
    print("-"*55)
    for c in cores:
        t=run_bench(c)
        speedup=base/t
        print(f"{c:5d}|{t:7.4f}|{speedup:16.2f}x|{c:18.2f}x")



# ============================================================
# TASK 1 - ANSWERS
# ============================================================

# Q1.1 - Inflection Point
#
# The observed speedup increases up to 12 workers:
#
# p=1   -> 1.00x
# p=2   -> 2.42x
# p=4   -> 4.09x
# p=8   -> 5.89x
# p=12  -> 6.41x
# p=16  -> 5.41x
#
# Therefore, the inflection point is p* = 12 workers.
# At 12 workers the execution time is the lowest: 0.9324 s.
# Increasing workers from 12 to 16 decreases performance:
# speedup falls from 6.41x to 5.41x.
#
# The machine reports 12 logical CPU cores.
# This suggests that 16 processes exceed the available logical
# CPU capacity and cause additional scheduling/context-switching
# and resource-contention overhead.
#
# Note: os.cpu_count() reports logical CPUs only. It does not by
# itself distinguish physical Performance cores, Efficiency cores,
# and SMT/Hyper-Threading threads. Exact topology requires checking
# the CPU information with a hardware-topology command such as lscpu.


# Q1.2 - Amdahl Parallel Fraction
#
# Measured values:
# T(1) = 5.9742 s
# T(4) = 1.4604 s
#
# First calculate the measured speedup:
#
# S(4) = T(1) / T(4)
#      = 5.9742 / 1.4604
#      = 4.0904
#
# Amdahl's Law:
#
# S(4) = 1 / [(1-P) + P/4]
#
# Therefore:
#
# 4.0904 = 1 / [1 - P + P/4]
#
# 4.0904 = 1 / [1 - 3P/4]
#
# 1 / 4.0904 = 1 - 3P/4
#
# 0.24448 = 1 - 3P/4
#
# 3P/4 = 0.75552
#
# P = (0.75552 * 4) / 3
#
# P = 1.00736
#
# Therefore:
# P ≈ 100.74%
#
# This value is not physically valid because the parallel fraction
# must be between 0% and 100%. The result is caused by measurement
# variation and benchmark overhead. In particular, the initial
# single-process baseline (5.9742 s) is different from the later
# p=1 measurement (4.4085 s).
#
# Therefore, the calculated P should be reported as a mathematical
# result of the provided measurements, not as a physically valid
# parallel fraction.


# Q1.3 - Asymptotic Speedup Limit
#
# Amdahl's Law gives:
#
# S_max = 1 / (1-P)
#
# Using P = 1.00736 would produce:
#
# S_max = 1 / (1 - 1.00736)
#       ≈ -135.9
#
# A negative speedup is physically impossible.
# Therefore, a physically meaningful asymptotic speedup cannot be
# calculated from this measured P because P > 1 violates the
# assumptions of Amdahl's Law.
#
# The practical experiment shows that increasing the number of
# workers eventually stops improving performance:
#
# 12 workers -> 0.9324 s -> 6.41x speedup
# 16 workers -> 1.1052 s -> 5.41x speedup
#
# Therefore, a 128-core server would NOT automatically provide
# a 128x speedup for this workload. The benchmark already shows
# diminishing returns and performance degradation at higher worker
# counts due to overhead, scheduling, and resource contention.