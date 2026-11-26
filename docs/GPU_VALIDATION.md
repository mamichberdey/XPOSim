# RTX 4060 Laptop validation and TensorFlow metric optimization

## Hardware evidence from XPOSim 0.34.0

Four user-supplied diagnostic reports dated 2026-11-26 exercised NumPy, CuPy 14.2.0, PyTorch 2.5.1+cu121 and TensorFlow 2.21.0. GPU reports identify an NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB and driver 621.14. Each report completed all 32 checks successfully, without reported exceptions. The largest relative L2 difference across the GPU checks was approximately `3.11e-12`.

Checks covered complex Gaussian propagation against an analytic reference, integrated power, 1D/2D CRL fields and metrics at batch sizes 1/4/auto, repeated z=0 positions, signed free propagation, local Fourier detectors and high-level Gaussian sources using point quadrature and convolution. This evidence applies to those test configurations. JAX was outside this report set.

For a 19-plane scan at batch size 4, each accelerator reported one explicit device-array conversion through the library helpers and zero explicit scalar conversions. These counters do not capture internal framework transfers or kernel placement. An actual device trace would be needed to quantify hidden PCIe traffic.

Active allocation remained constant throughout eight repeated scans. After releasing the lens, Torch reported zero allocated bytes, TensorFlow reported 131072 bytes, and CuPy's pool reported 408576 used bytes. Reserved allocator memory remained available for reuse. This is evidence of stable allocation for this scenario. Long-duration and shape-changing workloads require further checks.

## Timing evidence

Each time is the median of three warmed calls. A call computes a three-element CRL and scans nine detector planes, including metrics and their host export. Times are milliseconds.

| Grid | CuPy auto | Torch auto | TensorFlow auto | TensorFlow batch 4 |
| --- | --- | --- | --- | --- |
| 128×128 | 12.65 | 9.75 | 23.91 | 65.03 |
| 256×256 | 12.08 | 8.40 | 29.18 | 74.96 |
| 512×512 | 11.18 | 11.67 | 175.00 | 109.33 |

The separate NumPy report measured 5.90, 24.85 and 152.73 ms on those grids. Environments differ. Torch and standalone NumPy used NumPy 1.26.4 and SciPy 1.13.1, CuPy used NumPy 2.4.6 and SciPy 1.17.1, and TensorFlow used NumPy 2.0.1 and SciPy 1.17.1. Their in-environment NumPy controls also differ. CuPy and Torch samples show substantial timing variation in some cases. A universal ordering between their close results is unsupported by these short runs.

The conservative TensorFlow auto-batch budget is 64 MiB with six field buffers in its estimate. For complex128 at 512×512 this selects two planes per batch, making five metric-graph calls for nine planes. Batch 4 makes three calls. At the smaller grids auto can fit all nine planes in one batch. This explains the change in call count. It does not independently measure kernel efficiency.

## Profile and implemented change in 0.34.1

In the TensorFlow 256×256 batch-4 Python profile, approximately 51 ms of an approximately 80 ms scan was attributed to `_device_metrics`. Calls can include waiting for prior GPU work. The profile identifies a target for investigation, without isolating individual kernel time or PCIe traffic.

Tracing the previous metric graph for `(4,256,256)` on the development host produced 1261 operations, including 257 casts and 52 GatherV2 operations. Much of this machinery came from advanced indexing through `tf.experimental.numpy`.

Version 0.34.1 uses native TensorFlow batched gathers for peak extraction, transverse profiles and crossing interpolation. It reuses the already determined global peak for profile widths and gathers all four crossing samples together. The traced graph has 363 operations, including 90 casts and nine GatherV2 operations. These counts concern the traced graph before runtime optimization. They are not GPU kernel counts.

The new regression tests compare against NumPy for float32/float64, 1D/2D, single-sample and two-sample profiles, zero/negative/constant profiles, NaN/Inf values, edge peaks, ties, exact half-height plateaus, disabled y-FWHM and integer input-field metrics. Existing full-workflow and physical tests also cover the new entry point.

## Development-host microbenchmark

TensorFlow ran on CPU with Python 3.12.14, TensorFlow 2.21.0, NumPy 2.4.6 and one intra/inter-op thread. Each graph was warmed three times, then measured 25 times with `.numpy()` completion. The before/after graphs used the same seeded uniform float64 inputs and coordinate tensors. The benchmark measures metric reduction only.

| Batch shape | Before median ms | After median ms |
| --- | --- | --- |
| 4×256×256 | 1.650 | 1.059 |
| 2×512×512 | 2.858 | 1.745 |
| 9×128×128 | 1.196 | 0.776 |

The observed CPU improvement is approximately 1.54–1.64 times. GPU acceleration from this change remains unmeasured. Repeat the same TensorFlow diagnostic on the target GPU with version 0.34.1 and compare warmed times, numerical checks and allocator statistics. Keep the earlier reports as the baseline. A subsequent device trace should be considered if the metric cost remains dominant.

The reviewable regression tree completed 33 tests with 28 passes and five hardware-related skips after this change. TensorFlow-specific tests and isolated Ruff checks passed locally. Actual GPU timing of 0.34.1 remains pending.
