"""Benchmark steady-state CRL.focus_scan with a fixed double-precision workload."""

import argparse
from importlib import import_module
import platform
from statistics import median
from time import perf_counter

import numpy as np

import XPOSim as xpos


def device_name(backend, module, field):
    if backend == "cupy":
        name = module.cuda.runtime.getDeviceProperties(field.device.id)["name"]
        return name.decode() if isinstance(name, bytes) else name
    if backend == "torch":
        return module.cuda.get_device_name(field.device)
    if backend == "jax":
        return field.device.device_kind
    if backend == "tensorflow":
        return field.device
    return f"CPU ({platform.processor() or platform.machine()})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "numpy", "cupy", "torch", "tensorflow", "jax"))
    args = parser.parse_args()
    try:
        backend = xpos.select_backend(args.backend)
    except (ImportError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Cannot run {args.backend}: {exc}\n")

    grid = xpos.Grid2D(dx=50e-9, dy=50e-9, nx=512, ny=512, backend=backend)
    source = xpos.PointSource(grid=grid, position=(0., 0.), energy_kev=12.)
    field = source.E(z=15.)
    lens = xpos.CRL(input_field=field, grid=grid, energy_kev=12.,
                    radius=6.25e-6, aperture=20e-6, bridge_thickness=2e-6,
                    n_elements=26, material=xpos.Material(formula="Si", density=2.33))
    image_distance = lens.effective_focus(15.)
    distances = np.linspace(0.9 * image_distance, 1.1 * image_distance, 33)
    batch = 8
    module = import_module(backend)
    print(f"XPOSim {xpos.__version__} | Python {platform.python_version()} | {platform.platform()}")
    print(f"Backend: {backend} {module.__version__} | Device: {device_name(backend, module, field)}")
    print(f"CRL.focus_scan | {grid.nx} x {grid.ny} | {field.dtype} | "
          f"{len(distances)} planes | batch {batch} | metrics only", flush=True)

    def scan():
        # focus_scan reads each batch's metrics back to the host. Every timed
        # call therefore waits for its GPU propagation and reductions to finish.
        return lens.focus_scan(distances, batch_size=batch, return_fields=False,
                               calculate_divergence=False)

    scan()  # Prime FFT plans, compilation and lens caches outside the timer.
    times = []
    for _ in range(5):
        start = perf_counter()
        result = scan()
        times.append(perf_counter() - start)
        peaks = [m.max_intensity for m in result.metrics]
        if len(peaks) != len(distances) or not np.all(np.isfinite(peaks)) or max(peaks) <= 0:
            raise RuntimeError("Incomplete scan or invalid intensity.")

    elapsed = median(times)
    print("Scan times (s):", ", ".join(f"{t:.4f}" for t in times))
    print(f"Median: {elapsed:.4f} s | Throughput: {len(distances) / elapsed:.2f} planes/s")
    print(f"Peak plane: {result.peak_z * 1e3:.4f} mm | Relative peak intensity: {max(peaks):.6g}")


if __name__ == "__main__":
    main()
