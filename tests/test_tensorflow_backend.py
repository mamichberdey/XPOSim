"""Run identical TensorFlow kernel checks in CPU CI and on a real GPU."""
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np

from backend_parity import compare_results
import XPOSim as xpos

try:
    import tensorflow as tf
except (ImportError, OSError):
    tf = None


@skipUnless(tf is not None, "TensorFlow is optional")
class TensorFlowBackendTests(TestCase):
    def test_kernels(self):
        with tf.device("/CPU:0"):
            xp = xpos._tensorflow_arrays(tf.constant(1).device)
            with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
                compare_results(self, "tensorflow")

    def test_metric_graph_uses_current_roi_coordinates(self):
        # Same shape across calls, but different axis values and peak positions.
        xp = xpos._tensorflow_arrays(tf.constant(0).device)
        grid = xpos.Grid2D(1., 1., 5, 3, backend="numpy")
        with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
            tf_grid = xpos.Grid2D(1., 1., 5, 3, backend="tensorflow")
            values = np.array([[[0., 1., 4., 1., 0.]], [[0., 0., 1., 4., 0.]]])
            for offset in (0., 10.):
                axes = (grid.x_axis + offset, grid.y_axis[:1])
                kwargs = dict(measure_y_fwhm=False, coordinates=axes)
                expected = xpos._intensity_metrics_batch(values, grid, (), **kwargs)
                kwargs["coordinates"] = tuple(map(xp.asarray, axes))
                actual = xpos._intensity_metrics_batch(xp.asarray(values), tf_grid, (), **kwargs)
                self.assertEqual(actual, expected)

    def test_gpu(self):
        if not tf.config.list_physical_devices("GPU"):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                xpos.select_backend("tensorflow")
            self.skipTest("TensorFlow GPU unavailable")
        self.assertEqual(xpos.select_backend("tensorflow"), "tensorflow")
        compare_results(self, "tensorflow")
