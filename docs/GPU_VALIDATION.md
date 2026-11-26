# RTX 4060 Laptop validation

## Scope and hardware

The supported backends are NumPy, CuPy, Torch and JAX. GPU reports dated 2026-11-26 identify an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB and driver 621.14. Measurements below use XPOSim 0.34.1, CuPy 14.2.0 and Torch 2.5.1+cu121. A separate NumPy run supplies the CPU reference. JAX requires a subsequent actual-device run.

All 32 diagnostic checks passed in each of the NumPy, CuPy and Torch reports, without reported exceptions. Maximum GPU relative L2 difference from the NumPy reference was approximately `3.11e-12`.

Checks cover complex Gaussian propagation against an analytic reference, integrated power, 1D/2D CRL fields and metrics at batch sizes 1/4/auto, repeated z=0 positions, signed free propagation, local Fourier detectors and high-level Gaussian sources using point quadrature and convolution. This evidence applies to those configurations.

## Synchronization and memory

For a 19-plane scan at batch size 4, both accelerator reports recorded one explicit device-array conversion through the library helpers and zero explicit scalar conversions. Internal framework transfers and device placement require GPU traces for detailed measurement.

Active allocation remained constant throughout eight repeated scans. After releasing the lens, Torch reported zero allocated bytes and CuPy's pool reported 408576 used bytes. Reserved allocator memory remained available for reuse. Long-duration and shape-changing workloads require additional tests.

## Timing measurements

Each time is the median of three warmed calls. A call computes a three-element CRL and scans nine detector planes, including metrics and their host export. Times are milliseconds.

| Grid | NumPy, separate CPU run | CuPy auto | Torch auto |
| --- | --- | --- | --- |
| 128×128 | 5.47 | 10.21 | 6.32 |
| 256×256 | 25.12 | 12.99 | 8.39 |
| 512×512 | 130.07 | 12.17 | 12.20 |

The CPU run and CuPy environment used NumPy 2.4.6 and SciPy 1.17.1. The Torch environment used NumPy 1.26.4 and SciPy 1.13.1. In-environment CPU controls also differ. Short samples show substantial variation in some cases, so a ranking requires more repeated measurements. Clock state, background workload and runtime versions should accompany broader comparisons.

## Extended grid comparison

The notebook benchmark supplied separately measures total plane sizes `nx*ny=2**p`, for p from 8 through 22, with squares on even powers and 2:1 rectangles on odd powers. It retains per-size samples, initialization time, batching metadata, library/source identity and hardware versions. Small CPU cases use bounded vectorized batches, while GPU cases use the library's automatic batch heuristic. Large CPU cases use fewer repetitions to bound measurement time.

Results for the expanded grid range remain pending. A CPU/GPU crossover should be reported as a measured interval for the specified CRL workload, with sample variation and batching policy. Optical discretization requires separate convergence validation.
