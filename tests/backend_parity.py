"""Shared optical reference calculations for optional array backends."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import XPOSim as xpos


def compare_results(test, candidate):
    for grid_type, args in ((xpos.Grid1D, (2e-7, 65)), (xpos.Grid2D, (2e-7, 3e-7, 65, 43))):
        results = []
        for backend in ("numpy", candidate):
            grid = grid_type(*args, backend=backend)
            field = grid.xp.exp(-((grid.x-3e-7)/3e-6)**2 + 1j*grid.x*1e5)
            if grid.dimension == 2:
                field *= grid.xp.exp(-((grid.y+0.3e-6)/2e-6)**2)
            lens = xpos.CRL(field, grid, radius=6.25e-6, aperture=14e-6,
                            bridge_thickness=2e-6, n_elements=3, energy_kev=12,
                            material=xpos.Material(delta=3.35e-6, beta=3.7e-8))
            host_reads = []
            to_numpy = xpos._to_numpy
            def record_read(value):
                if xpos._backend_name(value) == candidate:
                    host_reads.append(tuple(value.shape))
                return to_numpy(value)
            with patch.object(xpos, "_to_numpy", side_effect=record_read):
                scan = lens.focus_scan([0., .01, .02, .03, .04], return_fields=True,
                                       batch_size=2)
            if backend == candidate:
                test.assertEqual(host_reads, [(5, 4 if grid.dimension == 1 else 6)])
            divergence = xpos.divergence(field, grid, lens.k)
            stack = grid.propagate_many(field, [0., .01, -.02], lens.k)
            ax = np.linspace(-2e-6, 2e-6, 5)
            data = np.arange(5**grid.dimension).reshape((5,)*grid.dimension)
            error = xpos.SurfaceError([data, *([ax]*grid.dimension)]).on_grid(grid)
            if backend == candidate:
                for tensor in (field, scan.E, stack, error):
                    test.assertEqual(tensor.device, grid.x.device)
            filtered = (xpos.gaussian_filter_1d(field, grid.x, 4e-7) if grid.dimension == 1
                        else xpos.gaussian_filter_2d(field, grid.x_axis, grid.y_axis, 4e-7, 6e-7))
            results.append((scan.E, [m.max_intensity for m in scan.metrics], stack, error,
                            [m.fwhm for m in scan.metrics], filtered, divergence))
        for measured, expected in zip(results[1], results[0]):
            np.testing.assert_allclose(xpos._to_numpy(measured), xpos._to_numpy(expected),
                                       rtol=2e-10, atol=2e-12)

    for dimension, method in ((1, "points"), (2, "convolution")):
        cfg = xpos.CRLFocusConfig(backend="numpy", grid_dimension=dimension,
            grid_points=128, grid_points_y=128 if dimension==2 else None, scan_points=7,
            adaptive_window=False, source_type="gaussian", source_method=method, source_points=3)
        reference = xpos.estimate_crl_focus(cfg)
        result = xpos.estimate_crl_focus(replace(cfg, backend=candidate))
        for name in ("best_intensity", "peak_z_m", "transmission", "fwhm_m"):
            np.testing.assert_allclose(getattr(result, name), getattr(reference, name),
                                       rtol=1e-8, atol=1e-13)

    # Off-centre detector coordinates exercise the local Fourier transform and
    # ROI selection. Point-source quadrature also checks device spectrum reuse.
    cfg = xpos.CRLFocusConfig(backend="numpy", grid_dimension=1,
        grid_points=128, scan_points=7, adaptive_window=False,
        accuracy="refined", detector_roi_m=(-2e-6, 4e-6),
        source_type="gaussian", source_method="points", source_points=3)
    reference = xpos.estimate_crl_focus(cfg)
    result = xpos.estimate_crl_focus(replace(cfg, backend=candidate))
    for name in ("best_intensity", "x_m", "longitudinal_intensity", "transmission"):
        np.testing.assert_allclose(getattr(result, name), getattr(reference, name),
                                   rtol=1e-8, atol=1e-13)
    if candidate in {"tensorflow", "jax"} or result._session.grid._free_gpu_memory() is not None:
        test.assertTrue(all(xpos._backend_name(batch) == candidate
                            for batch in result._session._mode_batches))
