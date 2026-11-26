"""NumPy-only numerical helpers and optional independent SciPy references."""
from unittest import TestCase, skipUnless

import numpy as np

import XPOSim as xpos

try:
    import scipy
except ImportError:
    scipy = None


class NumpyNumericsTests(TestCase):
    def test_fft_padding_is_minimal_and_smooth(self):
        for radices in ((2, 3, 5, 7), (2, 3, 5, 7, 11)):
            for target in range(1, 300):
                length = xpos._next_fast_len(target, radices)
                for candidate in range(target, length+1):
                    remainder = candidate
                    for prime in radices:
                        while remainder % prime == 0:
                            remainder //= prime
                    self.assertEqual(remainder == 1, candidate == length)

    def test_bilinear_interpolation_and_outside_fill(self):
        grid = xpos.Grid2D(1., 1., 7, 5, backend="numpy")
        x, y = np.array([-2., -.3, 1.5]), np.array([-1.8, .1, 1.5])
        def surface(x, y):
            return 2+3*x-4*y+.7*x*y
        values = surface(x[None, :], y[:, None])
        actual = xpos._interpolate_map(values, x, y, grid, -7.)
        inside = (grid.x >= x[0]) & (grid.x <= x[-1]) & (grid.y >= y[0]) & (grid.y <= y[-1])
        np.testing.assert_allclose(actual, np.where(inside, surface(grid.x, grid.y), -7.),
                                   rtol=0, atol=3e-15)

    def test_gaussian_blur_has_zero_boundaries_and_unit_mass(self):
        impulse = np.zeros(101)
        impulse[50] = 1
        actual = xpos._numpy_gaussian_filter(impulse, 2.)
        self.assertAlmostEqual(actual.sum(), 1., places=14)
        np.testing.assert_allclose(actual, actual[::-1], atol=1e-15)
        np.testing.assert_array_equal(xpos._numpy_gaussian_filter(impulse, 0.), impulse)
        boundary = xpos._numpy_gaussian_filter(np.ones(31), 2.)
        self.assertLess(boundary[0], 1.)
        self.assertAlmostEqual(boundary[15], 1., places=14)

    def test_gaussian_quadrature_moments(self):
        for count in (3, 7, 31, 201):
            for equal_probability in (False, True):
                nodes, weights = xpos._gaussian_source_quadrature(2., count,
                    equal_probability=equal_probability)
                self.assertTrue(np.all(weights >= 0))
                self.assertTrue(np.all(np.diff(nodes) > 0))
                self.assertAlmostEqual(weights.sum(), 1., places=14)
                self.assertAlmostEqual(np.dot(weights, nodes), 0., places=13)
                self.assertAlmostEqual(np.dot(weights, nodes**2), 4., places=12)
                if not equal_probability:
                    self.assertAlmostEqual(np.dot(weights, nodes**4), 48., places=10)

    def test_bounded_minimum_and_evaluation_budget(self):
        for center in (0., .2, .8, 1.):
            result = xpos._bounded_minimum(lambda x, center=center: (x-center)**2, 0., 1., 1e-12, 100)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.x, center, delta=5e-8)
        calls = []
        def objective(x):
            calls.append(x)
            return (x-.3)**2
        self.assertFalse(xpos._bounded_minimum(objective, 0., 1., 1e-12, 2).converged)
        self.assertEqual(len(calls), 2)

    def test_bracketed_roots_and_invalid_bracket(self):
        for function, expected in ((lambda x: x**3-2, 2**(1/3)),
                                   (lambda x: (x-.2)**3, .2),
                                   (lambda x: 1-x, 1.)):
            result = xpos._bracketed_root(function, -1., 2., 1e-12)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.x, expected, delta=2e-12)
        for scale in (1e-250, 1e250):
            root = xpos._bracketed_root(lambda x, scale=scale: scale*(x**3-2), 0., 2., 1e-12)
            self.assertTrue(root.converged)
            self.assertAlmostEqual(root.x, 2**(1/3), delta=2e-12)
        self.assertEqual(xpos._bracketed_root(lambda x: x, 0., 1., 1e-12).x, 0.)
        with self.assertRaises(ValueError):
            xpos._bracketed_root(lambda x: x*x+1, -1., 1., 1e-12)


@skipUnless(scipy is not None, "SciPy reference checks are optional")
class ScipyReferenceTests(TestCase):
    def test_smoothing_and_interpolation(self):
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(491)
        for shape, sigmas in (((3,), (8.,)), ((21,), (1.3,)), ((7, 13), (3.7, .2))):
            values = rng.random(shape)
            np.testing.assert_allclose(xpos._numpy_gaussian_filter(values, sigmas),
                gaussian_filter(values, sigmas, mode="constant", truncate=5.), rtol=2e-14, atol=2e-15)
        grid = xpos.Grid2D(.3, .4, 19, 13, backend="numpy")
        x, y = np.array([-2., -.3, 1.5]), np.array([-1.8, .1, 1.5])
        values = rng.random((3, 3))
        expected = RegularGridInterpolator((y, x), values, bounds_error=False,
                                           fill_value=-7.)(np.stack((grid.y, grid.x), axis=-1))
        np.testing.assert_allclose(xpos._interpolate_map(values, x, y, grid, -7.),
                                   expected, rtol=2e-14, atol=2e-15)

    def test_quadrature_and_fft_lengths(self):
        from scipy.fft import next_fast_len
        from scipy.special import ndtri, roots_hermitenorm
        for target in (*range(1, 500), 1025, 8193, 65537, 1048577):
            self.assertEqual(xpos._next_fast_len(target), next_fast_len(target))
        for count in (3, 7, 16, 51, 201):
            nodes, weights = xpos._gaussian_source_quadrature(1., count)
            expected_nodes, expected_weights = roots_hermitenorm(count)
            np.testing.assert_allclose(nodes, expected_nodes, rtol=2e-14, atol=1e-14)
            np.testing.assert_allclose(weights, expected_weights/expected_weights.sum(), atol=2e-14)
            edges = ndtri(np.linspace(0., 1., count+1))
            density = np.exp(-.5*edges**2)/np.sqrt(2*np.pi)
            expected = count*(density[:-1]-density[1:])
            expected /= np.sqrt(np.mean(expected**2))
            actual, _ = xpos._gaussian_source_quadrature(1., count, equal_probability=True)
            np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-13)
