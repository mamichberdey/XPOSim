"""Run identical TensorFlow kernel checks in CPU CI and on a real GPU."""
from dataclasses import replace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import XPOSim as xpos

try:
    import tensorflow as tf
except (ImportError, OSError):
    tf = None


@skipUnless(tf is not None, "TensorFlow is optional")
class TensorFlowBackendTests(TestCase):
    def compare_results(self):
        for grid_type, args in ((xpos.Grid1D, (2e-7, 65)), (xpos.Grid2D, (2e-7, 3e-7, 65, 43))):
            results = []
            for backend in ("numpy", "tensorflow"):
                grid = grid_type(*args, backend=backend)
                field = grid.xp.exp(-((grid.x-3e-7)/3e-6)**2 + 1j*grid.x*1e5)
                if grid.dimension == 2:
                    field *= grid.xp.exp(-((grid.y+0.3e-6)/2e-6)**2)
                lens = xpos.CRL(field, grid, radius=6.25e-6, aperture=14e-6,
                                bridge_thickness=2e-6, n_elements=3, energy_kev=12,
                                material=xpos.Material(delta=3.35e-6, beta=3.7e-8))
                scan = lens.focus_scan([0., .01, .02, .03, .04], return_fields=True,
                                       calculate_divergence=True, batch_size=2)
                stack = grid.propagate_many(field, [0., .01, -.02], lens.k)
                ax = np.linspace(-2e-6, 2e-6, 5)
                data = np.arange(5**grid.dimension).reshape((5,)*grid.dimension)
                error = xpos.SurfaceError([data, *([ax]*grid.dimension)]).on_grid(grid)
                if backend == "tensorflow":
                    for tensor in (field, scan.E, stack, error):
                        self.assertEqual(tensor.device, grid.x.device)
                results.append((scan.E, [m.max_intensity for m in scan.metrics], stack, error))
            for measured, expected in zip(results[1], results[0]):
                np.testing.assert_allclose(measured, expected, rtol=2e-10, atol=2e-12)

        for dimension, method in ((1, "points"), (2, "convolution")):
            cfg = xpos.CRLFocusConfig(backend="numpy", grid_dimension=dimension,
                grid_points=128, grid_points_y=128 if dimension==2 else None, scan_points=7,
                adaptive_window=False, source_type="gaussian", source_method=method, source_points=3)
            reference = xpos.estimate_crl_focus(cfg)
            result = xpos.estimate_crl_focus(replace(cfg, backend="tensorflow"))
            for name in ("best_intensity", "peak_z_m", "transmission", "fwhm_m"):
                np.testing.assert_allclose(getattr(result, name), getattr(reference, name),
                                           rtol=1e-8, atol=1e-13)

    def test_kernels(self):
        with tf.device("/CPU:0"):
            xp = xpos._tensorflow_arrays(tf.constant(1).device)
            with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
                self.compare_results()

    def test_gpu(self):
        if not tf.config.list_physical_devices("GPU"):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                xpos.select_backend("tensorflow")
            self.skipTest("TensorFlow GPU unavailable")
        self.assertEqual(xpos.select_backend("tensorflow"), "tensorflow")
        self.compare_results()
