"""Scale invariance, degeneracies, budgets and quadrature cache ownership."""
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import XPOSim as xpos


class NumericalStabilityTests(TestCase):
    def test_minimizer_scale_invariance_and_flat_objective(self):
        for scale in (1e-250, 1., 1e250):
            result = xpos._bounded_minimum(lambda x, scale=scale: scale*(x-.37)**2,
                                          0., 1., 1e-11, 100)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.x, .37, delta=5e-8)
        calls = []
        def flat(x):
            calls.append(x)
            return 1.
        result = xpos._bounded_minimum(flat, -.1, .2, 1e-10, 100)
        self.assertTrue(result.converged)
        self.assertTrue(all(-.1 <= x <= .2 for x in calls))
        self.assertLessEqual(len(calls), 100)
        for invalid in (np.nan, np.inf):
            with self.assertRaises(ArithmeticError):
                xpos._bounded_minimum(lambda x, invalid=invalid: invalid, 0., 1., 1e-9, 10)

    def test_peak_search_reports_flatness_and_preserves_budget(self):
        result = xpos.find_peak_intensity_distance(lambda z: 1., 1., max_evaluations=8)
        self.assertFalse(result.converged)
        self.assertIn("flat", result.message)
        calls = []
        def narrow(z):
            calls.append(z)
            return np.exp(-((z-1.3)/.001)**2)
        result = xpos.find_peak_intensity_distance(narrow, 1.3, max_evaluations=8,
                                                   tolerance_m=1e-15)
        self.assertLessEqual(len(calls), 8)
        self.assertEqual(result.evaluations, len(calls))

    def test_root_budget_reversed_bracket_and_nonfinite_values(self):
        for bounds in ((0., 2.), (2., 0.)):
            result = xpos._bracketed_root(lambda x: x**3-2, *bounds, 1e-12)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(result.x, 2**(1/3), delta=2e-12)
        result = xpos._bracketed_root(lambda x: x**3-2, 0., 2., 1e-15, maxiter=1)
        self.assertFalse(result.converged)
        with self.assertRaises(ValueError):
            xpos._bracketed_root(lambda x: np.nan, 0., 1., 1e-10)

    def test_propagation_and_width_are_invariant_under_field_scaling(self):
        rng = np.random.default_rng(425)
        for count in (64, 65):
            grid = xpos.Grid1D(1e-7, count, backend="numpy")
            field = rng.normal(size=count)+1j*rng.normal(size=count)
            forward = grid.propagate(field, .04, 6e10)
            np.testing.assert_allclose(grid.propagate(forward, -.04, 6e10), field,
                                       rtol=2e-12, atol=2e-12)
            profile = np.exp(-(grid.x/1e-6)**2)
            reference = xpos.full_width_half_max(profile, grid.x)
            for scale in (1e-240, 1e240):
                self.assertAlmostEqual(xpos.full_width_half_max(profile*scale, grid.x),
                                       reference, delta=1e-19)

    def test_quadrature_cache_is_bounded_and_caller_arrays_are_independent(self):
        xpos._standard_source_quadrature.cache_clear()
        with patch.object(np.linalg, "eigh", wraps=np.linalg.eigh) as eigensolve:
            first, weights = xpos._gaussian_source_quadrature(1., 7)
            expected = first.copy()
            first[:] = 0
            weights[:] = 0
            second, second_weights = xpos._gaussian_source_quadrature(2., 7)
            np.testing.assert_allclose(second, 2*expected)
            self.assertAlmostEqual(second_weights.sum(), 1.)
            self.assertEqual(eigensolve.call_count, 1)
        for count in range(3, 24):
            xpos._gaussian_source_quadrature(1., count)
        self.assertLessEqual(xpos._standard_source_quadrature.cache_info().currsize, 16)
