import threading
import time

TOTAL_OPS = 2_000_000
NUM_THREADS = 4


class UnsafeCounter:
    def __init__(self):
        self.val = 0

    def inc(self):
        self.val += 1


class LockedCounter:
    def __init__(self):
        self.val = 0
        self.lock = threading.Lock()

    def inc(self):
        with self.lock:
            self.val += 1


def bench(counter_type):
    c = counter_type()
    ops_per_thread = TOTAL_OPS // NUM_THREADS

    def work():
        for _ in range(ops_per_thread):
            c.inc()

    threads = [
        threading.Thread(target=work)
        for _ in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return c.val, time.perf_counter() - start


def bench_lockless():
    ops_per_thread = TOTAL_OPS // NUM_THREADS

    local_results = [0] * NUM_THREADS

    def work(thread_id):
        local_count = 0

        for _ in range(ops_per_thread):
            local_count += 1

        local_results[thread_id] = local_count

    threads = [
        threading.Thread(
            target=work,
            args=(i,)
        )
        for i in range(NUM_THREADS)
    ]

    start = time.perf_counter()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    final_value = sum(local_results)

    elapsed = time.perf_counter() - start

    return final_value, elapsed


if __name__ == "__main__":
    val_unsafe, t_unsafe = bench(UnsafeCounter)
    val_locked, t_locked = bench(LockedCounter)

    print(
        f"Unsafe: Value={val_unsafe:,}/{TOTAL_OPS:,} | "
        f"Time: {t_unsafe:.4f}s"
    )

    print(
        f"Locked: Value={val_locked:,}/{TOTAL_OPS:,} | "
        f"Time: {t_locked:.4f}s"
    )

    print(
        f"ContentionCostMultiplier: "
        f"{t_locked / t_unsafe:.2f}x"
    )

    val_lockless, t_lockless = bench_lockless()

    print(
        f"Lockless: Value={val_lockless:,}/{TOTAL_OPS:,} | "
        f"Time: {t_lockless:.4f}s"
    )

    print(
        f"Speedup over Locked: "
        f"{t_locked / t_lockless:.2f}x"
    )



# ============================================================
# TASK 3 — QUESTIONS / ANSWERS
# ============================================================

# Q3.1 — UnsafeCounter and LOAD/ADD/STORE
#
# UnsafeCounter returned:
# Value = 2,000,000 / 2,000,000
#
# In this particular run, no lost updates were observed.
# However, the operation:
#
#     self.val += 1
#
# is logically composed of three steps:
#
#     LOAD  -> read current value
#     ADD   -> add 1
#     STORE -> write the new value
#
# A race condition can occur if two threads interleave these
# operations:
#
# Thread A: LOAD 100
# Thread B: LOAD 100
# Thread A: ADD 1 -> 101
# Thread B: ADD 1 -> 101
# Thread A: STORE 101
# Thread B: STORE 101
#
# Two increments then produce only one effective increment.
# This is called a lost update.
#
# In my CPython run the final value was correct, so the race
# condition did not manifest in this particular execution.
# The correct result does not make the shared operation a
# reliable synchronization mechanism.


# Q3.2 — Lockless Thread-Local Accumulator
#
# The lockless implementation uses a thread-local counter.
# Each worker performs all increments on its own local variable:
#
#     local_count += 1
#
# The workers do not modify one shared counter.
# After all threads finish, the local results are combined:
#
#     final_value = sum(local_results)
#
# This is a Map-Reduce style approach:
#
# Map:
#   each thread independently calculates its own partial result.
#
# Reduce:
#   the partial results are summed at the end.
#
# Measured result:
#
# Unsafe:
#     Value = 2,000,000 / 2,000,000
#     Time  = 0.2583 s
#
# Locked:
#     Value = 2,000,000 / 2,000,000
#     Time  = 0.8335 s
#
# Lockless:
#     Value = 2,000,000 / 2,000,000
#     Time  = 0.1432 s
#
# Speedup over Locked:
#     0.8335 / 0.1432 = 5.82x
#
# The required speedup was at least 2.0x.
# The measured speedup was 5.82x, so the requirement is satisfied.
#
# The lockless result is also exactly correct:
#     2,000,000 / 2,000,000


# Q3.3 — Why thread-local state is fundamentally better
# than optimizing the lock
#
# A lock protects shared mutable state, but the lock itself
# remains a serialization point. Every thread must acquire and
# release the same lock before modifying the counter.
#
# This creates synchronization overhead:
# - lock acquisition and release;
# - contention between threads;
# - waiting when another thread owns the lock;
# - additional shared-state and cache-coherence traffic.
#
# The thread-local design removes the shared mutable counter
# from the hot path completely.
#
# Each thread works independently on its own local variable.
# Synchronization is only needed after the workers finish,
# when the small number of partial results is reduced with sum().
#
# Therefore, instead of trying to make the lock faster, the
# better architectural solution is to eliminate the need for
# the lock in the main computation.
#
# This follows the principle:
#
#     avoid shared mutable state instead of protecting it
#     with increasingly expensive synchronization mechanisms.
#
# In this benchmark the result demonstrates the benefit:
#
#     Locked:  0.8335 s
#     Lockless: 0.1432 s
#     Speedup:  5.82x
#
# Thus, the thread-local Map-Reduce design is substantially
# faster than the shared locked counter while preserving
# correctness.