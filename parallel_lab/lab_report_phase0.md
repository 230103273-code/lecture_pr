# Parallel Computing Lab — Phase 0

> Generated from measurements on this computer. Do not replace measured values with estimates.

## Hardware and software baseline

- **OS:** Windows-11-10.0.26200-SP0
- **Python:** 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]
- **Machine:** AMD64
- **Logical CPUs (multiprocessing):** 12
- **Physical cores:** Not reliably available from the Python standard library
- **CPU model:** AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD
- **Cache sizes:** Not available portably through the Python standard library
- **Cache-line size:** Not available portably through the Python standard library
- **Windows CPU details:** Name                      : AMD Ryzen 5 7430U with Radeon Graphics; NumberOfCores             : 6; NumberOfLogicalProcessors : 12; L2CacheSize               : 3072; L3CacheSize               : 8192

## Timer calibration

- Clock reads per batch: 2,000,000
- Number of batches: 7
- Samples (ns/call): 156.917, 141.321, 145.288, 156.888, 156.569, 145.546, 155.507
- Median (ns/call): 155.507
- IQR (ns/call): 11.312

## Verification

- Student ID seed: `230103273`
- Logical CPU count: `12`
- VERIFY_HASH[PHASE0]: 28A605AF18D9893F (P_log=12)

## Notes

Physical-core count, cache capacities, and cache-line size must be verified from the OS/hardware utility where unavailable above.
This file contains Phase 0 only; later phases must use measured values from this machine.
