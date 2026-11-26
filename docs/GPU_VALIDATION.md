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

The expanded benchmark covers total plane sizes `nx*ny=2**p` for p from 8 through 22. Even powers use square grids, odd powers use 2:1 rectangles. All 15 sizes completed in each report. The reports share fingerprint `527bf6cc7eb2e18b` and XPOSim 0.34.1.

The CPU is an Intel Core i5-12450H. GPU runs use the NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM and driver 621.14. The NumPy CPU baseline and Torch environment use NumPy 1.26.4 and SciPy 1.13.1. The CuPy environment uses NumPy 2.4.6 and SciPy 1.17.1. Accelerator versions are CuPy 14.2.0 and Torch 2.5.1+cu121. JAX has no measurements in this dataset.

Each warmed scan evaluates a three-element CRL and nine detector planes from 0.01 to 0.05 m, exporting metrics with `return_fields=False`. Grid steps are 200 nm on both axes. Parameters are R=6.25 um, aperture=14 um, bridge=2 um, 12 keV, delta=3.35e-6 and beta=3.7e-8. The input amplitude is Gaussian with widths 3/2 um, offsets +0.31/-0.23 um and an x phase gradient of 1e5 rad/m. Fields use complex128.

NumPy uses vectorized batches based on a conservative 64 MiB estimate for six field buffers, with a minimum batch of one. GPU cases use the library's automatic batching. Setup and the first warm-up scan are excluded from the times below. Each point uses the median of one, three or five timed runs, selected from the warm-up duration. CPU points from `2**20` onward have one timed run. Some GPU points also have one run. Figure bands show the sample 25th–75th percentiles, which collapse at single-sample points.

| Points per plane | Grid, nx × ny | NumPy ms | CuPy ms | Torch ms |
| --- | --- | --- | --- | --- |
| 256 | 16 × 16 | 0.654 | 6.112 | 5.297 |
| 512 | 32 × 16 | 0.648 | 8.614 | 5.643 |
| 1,024 | 32 × 32 | 0.840 | 10.317 | 5.439 |
| 2,048 | 64 × 32 | 1.248 | 10.823 | 7.655 |
| 4,096 | 64 × 64 | 2.396 | 11.298 | 6.044 |
| 8,192 | 128 × 64 | 3.007 | 9.089 | 5.536 |
| 16,384 | 128 × 128 | 6.947 | 9.391 | 6.469 |
| 32,768 | 256 × 128 | 15.162 | 10.007 | 6.274 |
| 65,536 | 256 × 256 | 37.362 | 9.902 | 7.344 |
| 131,072 | 512 × 256 | 83.844 | 9.460 | 7.155 |
| 262,144 | 512 × 512 | 158.681 | 11.571 | 11.888 |
| 524,288 | 1024 × 512 | 344.858 | 22.054 | 22.356 |
| 1,048,576 | 1024 × 1024 | 846.499 | 42.410 | 38.234 |
| 2,097,152 | 2048 × 1024 | 2156.120 | 102.942 | 82.572 |
| 4,194,304 | 2048 × 2048 | 4641.217 | 222.335 | 195.206 |

NumPy has the lower median through 8,192 points. At 16,384 points, Torch's median is 6.47 ms against NumPy's 6.95 ms, with overlapping sample interquartile ranges. Both measured GPU backends have lower medians at 32,768 points and all larger measured sizes. This places the observed transition in the approximate 8,192–32,768 range, depending on backend and sample variation.

At 4,194,304 points, the observed times are 4.641 s for NumPy, 0.222 s for CuPy and 0.195 s for Torch. Their ratios correspond to 20.9× and 23.8× GPU speedup for this workload. The CPU endpoint has one timed sample. Batching, environment versions, clock state and background workload affect these ratios and the crossover. Optical discretization requires separate convergence validation.

[Open the full-resolution comparison](assets/backend-benchmark.png).
