#!/usr/bin/env python3
"""
Phase 0 starter for the Parallel Computing lab.
Run with: python phase0.py
Uses only the Python standard library. It records real measurements from this laptop.
"""
from __future__ import annotations

import os
import platform
import multiprocessing as mp
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("parallel_lab")
OUT.mkdir(exist_ok=True)
REPORT = OUT / "lab_report_phase0.md"


def run_command(args):
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=10, check=False
        )
        return (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def cpu_info():
    system = platform.system()
    details = {
        "OS": platform.platform(),
        "Python": sys.version.replace("\n", " "),
        "Machine": platform.machine(),
        "Logical CPUs (multiprocessing)": str(mp.cpu_count()),
        "Physical cores": "Not reliably available from the Python standard library",
        "CPU model": platform.processor() or "Not reported by platform.processor()",
        "Cache sizes": "Not available portably through the Python standard library",
        "Cache-line size": "Not available portably through the Python standard library",
    }

    if system == "Windows":
        ps = (
            "Get-CimInstance Win32_Processor | "
            "Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,"
            "L2CacheSize,L3CacheSize | Format-List"
        )
        output = run_command(["powershell", "-NoProfile", "-Command", ps])
        if output:
            details["Windows CPU details"] = output

    elif system == "Darwin":
        output = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if output:
            details["CPU model"] = output
        for key in (
                "hw.physicalcpu", "hw.logicalcpu", "hw.cachelinesize",
                "hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize"
        ):
            value = run_command(["sysctl", "-n", key])
            if value:
                details[key] = value

    elif system == "Linux":
        output = run_command(["bash", "-lc", "lscpu"])
        if output:
            wanted = (
                "Model name", "Socket(s)", "Core(s) per socket",
                "Thread(s) per core", "CPU(s)", "L1d", "L2", "L3"
            )
            selected = [
                line.strip() for line in output.splitlines()
                if any(line.strip().startswith(k) for k in wanted)
            ]
            if selected:
                details["Linux lscpu details"] = "\n".join(selected)
        line_size = run_command(
            ["bash", "-lc", "getconf LEVEL1_DCACHE_LINESIZE"]
        )
        if line_size:
            details["Cache-line size (bytes)"] = line_size

    return details


def calibrate_timer(reads=2_000_000, batches=7):
    # Measure batches of perf_counter_ns calls and report average ns per call.
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for _ in range(reads):
            time.perf_counter_ns()
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / reads)
    return samples


def fnv1a_hash(text):
    value = 14695981039346656037
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def main():
    print("=== Parallel Computing Lab: Phase 0 ===")
    student_id = input("Enter your Student ID (used as the hash seed): ").strip()
    if not student_id:
        print("Student ID cannot be empty.")
        raise SystemExit(2)

    details = cpu_info()
    print("\n--- Hardware / software information ---")
    for key, value in details.items():
        print(f"{key}: {value}")

    print("\nCalibrating timer: 7 batches × 2,000,000 clock reads...")
    samples = calibrate_timer()
    median_ns = statistics.median(samples)
    iqr_ns = (
            statistics.quantiles(samples, n=4, method="inclusive")[2]
            - statistics.quantiles(samples, n=4, method="inclusive")[0]
    )
    print("Per-batch average ns/call:", ", ".join(f"{x:.3f}" for x in samples))
    print(f"Median: {median_ns:.3f} ns/call; IQR: {iqr_ns:.3f} ns/call")

    logical = mp.cpu_count()
    payload = f"PHASE0|{student_id}|{logical}|{median_ns:.4f}|{iqr_ns:.4f}"
    verify = f"VERIFY_HASH[PHASE0]: {fnv1a_hash(payload):016X} (P_log={logical})"

    lines = [
        "# Parallel Computing Lab — Phase 0",
        "",
        "> Generated from measurements on this computer. Do not replace measured values with estimates.",
        "",
        "## Hardware and software baseline",
        "",
    ]
    for key, value in details.items():
        lines.append(f"- **{key}:** {value.replace(chr(10), '; ')}")
    lines += [
        "",
        "## Timer calibration",
        "",
        f"- Clock reads per batch: {2_000_000:,}",
        f"- Number of batches: {len(samples)}",
        f"- Samples (ns/call): {', '.join(f'{x:.3f}' for x in samples)}",
        f"- Median (ns/call): {median_ns:.3f}",
        f"- IQR (ns/call): {iqr_ns:.3f}",
        "",
        "## Verification",
        "",
        f"- Student ID seed: `{student_id}`",
        f"- Logical CPU count: `{logical}`",
        f"- {verify}",
        "",
        "## Notes",
        "",
        "Physical-core count, cache capacities, and cache-line size must be verified from the OS/hardware utility where unavailable above.",
        "This file contains Phase 0 only; later phases must use measured values from this machine.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + verify)
    print(f"\nReport saved to: {REPORT.resolve()}")
    print("\nPhase 0 complete. Keep this output; later phases depend on actual hardware measurements.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
