import threading
import time

ITERATIONS = 5_000_000


def worker_adjacent(shared_list, index):
    for _ in range(ITERATIONS):
        shared_list[index] += 1


def worker_padded(shared_list, index):
    # Offset by 16 integers (16 * 8 bytes = 128 bytes > 64-byte cache line)
    padded_idx = index * 16

    for _ in range(ITERATIONS):
        shared_list[padded_idx] += 1


def run_test(target_fn, size):
    arr = [0] * size

    threads = [
        threading.Thread(
            target=target_fn,
            args=(arr, i)
        )
        for i in range(4)
    ]

    start = time.perf_counter()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return time.perf_counter() - start


if __name__ == "__main__":

    t_adjacent = run_test(worker_adjacent, size=4)
    t_padded = run_test(worker_padded, size=64)

    print(f"AdjacentIndices(FalseSharing): {t_adjacent:.4f}s")
    print(f"PaddedIndices(Cache-Aligned):  {t_padded:.4f}s")
    print(f"SlowdownFactor: {t_adjacent / t_padded:.2f}x")


# ============================================================
# TASK 2 - RESULTS AND ANSWERS
# ============================================================

# Benchmark results:
#
#                    Trial 1    Trial 2    Trial 3    Median
# Adjacent:          1.9561 s    1.9659 s   2.0327 s   1.9659 s
# Padded:            1.9501 s    1.9187 s   2.1153 s   1.9501 s
#
# Median slowdown:
#
# 1.9659 / 1.9501 = 1.0081
#
# Therefore:
#
# Median Slowdown = approximately 1.01x
#
# The measured difference is only about 0.81%, so this experiment
# does not show a significant false-sharing performance penalty
# on this Python/CPython environment.


# ============================================================
# Q2.1 - Memory Footprint and Cache-Line Separation
# ============================================================
#
# On a typical 64-bit CPython system, a Python list stores
# 8-byte references (pointers) to Python objects.
#
# Therefore:
#
#     1 reference = 8 bytes
#
# A cache line is 64 bytes:
#
#     64 / 8 = 8 references per cache line
#
# In the padded version, the worker index is multiplied by 16:
#
#     padded_idx = index * 16
#
# The distance between two worker indices is therefore:
#
#     16 * 8 bytes = 128 bytes
#
# Since 128 bytes is greater than one 64-byte cache line, adjacent
# worker locations are separated by at least two cache-line widths.
#
# Therefore, the padded indices are placed on different cache lines
# and avoid sharing the same cache line.
#
# Note: CPython lists contain references to Python integer objects,
# not the integer values directly.


# ============================================================
# Q2.2 - MESI State Transitions and False Sharing
# ============================================================
#
# Assume arr[0] and arr[1] are located in the same 64-byte cache line.
#
# Initially, both cores can have the cache line in the Shared (S)
# state.
#
# Core 0 executes:
#
#     arr[0] += 1
#
# Core 0 needs write ownership, so the cache line becomes Modified
# on Core 0 and the copy on Core 1 is invalidated:
#
#     Core 0: Shared -> Modified
#     Core 1: Shared -> Invalid
#
# Core 1 then executes:
#
#     arr[1] += 1
#
# Core 1 needs the same cache line for writing. The line must be
# obtained from Core 0 and Core 0's copy becomes invalid:
#
#     Core 0: Modified -> Invalid
#     Core 1: Invalid -> Modified
#
# If Core 0 writes again, the process repeats:
#
#     Core 1: Modified -> Invalid
#     Core 0: Invalid -> Modified
#
# Thus, the cache line repeatedly moves between cores.
#
# This is false sharing: the threads modify different logical
# variables, so there is no logical data race, but the variables
# share the same cache line and cause cache-coherence traffic.


# ============================================================
# Q2.3 - Industry-Standard False-Sharing Prevention
# ============================================================
#
# 1. C++11 alignas(64)
#
# C++ provides alignas() for controlling object alignment.
# For example:
#
#     struct alignas(64) Counter {
#         long value;
#     };
#
# This can be used to align frequently accessed data to cache-line
# boundaries and help prevent independent data from sharing a line.
#
#
# 2. Java @Contended
#
# Java provides the @Contended annotation for reducing false sharing
# between frequently modified fields.
#
#     @Contended
#     class Counter {
#         volatile long value;
#     }
#
# The JVM can insert padding around contended fields so that
# independent variables do not occupy the same cache line.
#
# Both mechanisms allow programmers to address false sharing without
# manually calculating large array offsets.