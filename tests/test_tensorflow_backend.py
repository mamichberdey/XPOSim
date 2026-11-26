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
                kwargs = {"measure_y_fwhm": False, "coordinates": axes}
                expected = xpos._intensity_metrics_batch(values, grid, (), **kwargs)
                kwargs["coordinates"] = tuple(map(xp.asarray, axes))
                actual = xpos._intensity_metrics_batch(xp.asarray(values), tf_grid, (), **kwargs)
                self.assertEqual(actual, expected)

    def test_native_metric_graph_matches_numpy_edge_cases(self):
        xp = xpos._tensorflow_arrays(tf.constant(0).device)
        rng = np.random.default_rng(6207)
        for shape in ((12, 1), (12, 2), (12, 17), (12, 3, 5), (12, 11, 17)):
            for dtype in (np.float32, np.float64):
                values = rng.random(shape).astype(dtype)
                values[0] = 0
                values[1] = 1
                values[2] = -1
                values[3] = np.nan
                values[4] = np.inf
                values[5] = -np.inf
                values[6].flat[0] = np.nan
                values[7].flat[-1] = np.inf
                values[8] = 0
                values[8].flat[0] = 1
                values[9] = 0
                values[9].flat[-1] = 1
                # Multiple equal peaks and an exact half-height plateau.
                values[10] = 0
                values[10].flat[::2] = 1
                values[11] = .5
                values[11].flat[values[11].size//2] = 1
                x = np.linspace(-2., 3., shape[-1])
                y = np.linspace(4., 7., shape[-2]) if len(shape) == 3 else None
                for measure_y in (False, True):
                    with self.subTest(shape=shape, dtype=dtype, measure_y=measure_y):
                        with np.errstate(invalid="ignore"):
                            expected = xpos._metric_table(values, x, y, measure_y, np)
                        graph = xpos._tensorflow_metric_graph(
                            xp, shape, tf.as_dtype(dtype), tf.float64, measure_y)
                        axes = (xp.asarray(x),) if y is None else (xp.asarray(x), xp.asarray(y))
                        actual = graph(xp.asarray(values), *axes).numpy()
                        tolerance = 2e-6 if dtype == np.float32 else 2e-12
                        np.testing.assert_allclose(actual, expected, rtol=tolerance,
                                                   atol=tolerance, equal_nan=True)

    def test_integer_field_metrics_match_numpy(self):
        xp = xpos._tensorflow_arrays(tf.constant(0).device)
        horizontal = np.array([0, 1, 3, 7, 3, 1, 0], dtype=np.int64)
        field = np.array([0, 1, 4, 1, 0])[:, None] * horizontal
        reference = xpos.Grid2D(.1, .2, 7, 5, backend="numpy")
        with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
            grid = xpos.Grid2D(.1, .2, 7, 5, backend="tensorflow")
            self.assertEqual(xpos.beam_metrics(xp.asarray(field), grid),
                             xpos.beam_metrics(field, reference))

    def test_repeated_stack_matches_sequential_transmissions(self):
        # Check phase as well as intensity on an even-by-odd grid, using
        # different fields on successive calls to the same compiled graph.
        xp = xpos._tensorflow_arrays(tf.constant(0).device)
        with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
            grid = xpos.Grid2D(2e-7, 3e-7, 32, 27, backend="tensorflow")
            source = xpos.PointSource(grid, position=(2e-7, -5e-7), energy_kev=12.)
            incident = source.E(15.)
            for count in (1, 8):
                lens = xpos.CRL(incident, grid, radius=6.25e-6, aperture=8e-6,
                    bridge_thickness=2e-6, n_elements=count, energy_kev=12.,
                    material=xpos.Material(delta=3.35e-6, beta=3.7e-8))
                sequential = lens.with_parameters(reuse_element_geometry=False)
                for multiplier in (1., 0.2j):
                    lens.input_field = sequential.input_field = incident * multiplier
                    np.testing.assert_allclose(lens.E(), sequential.E(), rtol=2e-11, atol=1e-12)
                fields = xp.stack((incident, incident * 0.2j))
                np.testing.assert_allclose(lens.apply_to_fields(fields),
                    sequential.apply_to_fields(fields), rtol=2e-11, atol=1e-12)

    def test_gpu(self):
        if not tf.config.list_physical_devices("GPU"):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                xpos.select_backend("tensorflow")
            self.skipTest("TensorFlow GPU unavailable")
        self.assertEqual(xpos.select_backend("tensorflow"), "tensorflow")
        compare_results(self, "tensorflow")
