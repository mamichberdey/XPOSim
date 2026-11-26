"""Independent physical references and discretization checks, see docs/VALIDATION.md."""
import unittest

import numpy as np
from scipy.constants import c, elementary_charge, h
from scipy.special import fresnel

import XPOSim as xpos


class PhysicsValidationTests(unittest.TestCase):
    wavelength = 1e-10
    k = 2 * np.pi / wavelength

    def test_fourier_quadrature_and_parseval(self):
        rng = np.random.default_rng(937)
        for count in (32, 33):
            with self.subTest(count=count):
                grid = xpos.Grid1D(2e-7, count, backend="numpy")
                field = rng.normal(size=count) + 1j * rng.normal(size=count)
                expected = grid.dx * np.exp(-1j * grid.qx[:, None] * grid.x).dot(field)
                spectrum = grid.fft(field)
                np.testing.assert_allclose(spectrum, expected, rtol=1e-11, atol=1e-18)
                np.testing.assert_allclose(grid.ifft(spectrum), field, rtol=0, atol=1e-12)
                self.assertAlmostEqual(
                    np.sum(abs(spectrum)**2) * grid.dqx / (2 * np.pi),
                    np.sum(abs(field)**2) * grid.dx, delta=1e-17)

    def test_gaussian_complex_field(self):
        width, z = 2e-6, 0.08
        for count in (1024, 1025):
            grid = xpos.Grid1D(40e-9, count, backend="numpy")
            field = np.exp(-(grid.x / width)**2)
            spread = 1 + 2j * z / (self.k * width**2)
            expected = np.exp(-grid.x**2 / (width**2 * spread)) / np.sqrt(spread)
            np.testing.assert_allclose(grid.propagate(field, z, self.k), expected,
                                       rtol=1e-8, atol=2e-10)

    def test_two_dimensional_gaussian_and_power(self):
        grid = xpos.Grid2D(1e-7, 1.2e-7, 257, 225, backend="numpy")
        field = np.exp(-(grid.x / 1.4e-6)**2 - (grid.y / 1.8e-6)**2)
        z = 0.04
        bx, by = (1 + 2j * z / (self.k * w**2) for w in (1.4e-6, 1.8e-6))
        expected = np.exp(-grid.x**2 / (1.4e-6**2 * bx)
                          - grid.y**2 / (1.8e-6**2 * by)) / np.sqrt(bx * by)
        propagated = grid.propagate(field, z, self.k)
        np.testing.assert_allclose(propagated, expected, rtol=0, atol=2e-10)
        self.assertAlmostEqual(np.sum(abs(propagated)**2) / np.sum(abs(field)**2),
                               1., delta=1e-12)
        np.testing.assert_allclose(grid.propagate(propagated, -z, self.k), field,
                                   rtol=0, atol=1e-12)

    def test_thin_lens_gaussian_focus(self):
        grid = xpos.Grid1D(40e-9, 2049, backend="numpy")
        width, focus = 3e-6, 0.05
        field = np.exp(-(grid.x / width)**2 - 1j * self.k * grid.x**2 / (2 * focus))
        a = 1 / width**2 + 1j * self.k / (2 * focus)
        spread = 1 + 2j * a * focus / self.k
        expected = np.exp(-a * grid.x**2 / spread) / np.sqrt(spread)
        actual = grid.propagate(field, focus, self.k)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
        expected_width = np.sqrt(2 * np.log(2)) * 2 * focus / (self.k * width)
        measured = xpos.full_width_half_max(abs(actual)**2, grid.x)
        self.assertAlmostEqual(measured / expected_width, 1., delta=0.002)

    def test_slit_fresnel_quadrature_convergence(self):
        # Independent Fresnel integral, evaluated on a local detector to exclude
        # FFT window wraparound. Slit boundaries are cell edges at +/-2 um.
        errors = []
        detector = xpos.Grid1D(80e-9, 61, backend="numpy")
        z, half_width = 0.02, 2e-6
        scale = np.sqrt(2 / (self.wavelength * z))
        lower = scale * (-half_width - detector.x)
        upper = scale * (half_width - detector.x)
        s0, c0 = fresnel(lower)
        s1, c1 = fresnel(upper)
        expected = ((c1 - c0) + 1j * (s1 - s0)) / np.sqrt(2j)
        for step in (40e-9, 20e-9):
            # Use direct Fresnel quadrature as an independent check of sampling.
            source_x = np.arange(-half_width + step / 2, half_width, step)
            kernel = np.exp(1j * self.k * (detector.x[:, None] - source_x)**2 / (2 * z))
            quadrature = step * kernel.sum(axis=1) / np.sqrt(1j * self.wavelength * z)
            errors.append(np.max(abs(quadrature - expected)))
            grid = xpos.Grid1D(step, round(160e-6 / step), backend="numpy")
            field = (abs(grid.x) < half_width).astype(complex)
            actual = grid.propagate_to_grid(field, z, self.k, detector)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0.004)
        self.assertLess(errors[1], errors[0] / 3)

    def test_crl_material_transmission(self):
        grid = xpos.Grid1D(20e-9, 257, backend="numpy")
        delta, beta, radius, bridge = 3e-6, 2e-8, 50e-6, 2e-6
        lens = xpos.CRL(grid=grid, radius=radius, aperture=20e-6,
                        bridge_thickness=bridge, n_elements=1,
                        material=xpos.Material(delta=delta, beta=beta),
                        wavelength_m=self.wavelength)
        thickness = bridge + grid.x**2 / radius
        expected = np.exp(-1j * self.k * delta * thickness - self.k * beta * thickness)
        np.testing.assert_allclose(lens.transmission(0), expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(abs(lens.transmission(0))**2,
                                   np.exp(-2 * self.k * beta * thickness), rtol=0, atol=1e-14)

    def test_sampled_plane_wave_eigenmode(self):
        for count in (64, 65):
            grid = xpos.Grid1D(1e-6, count, backend="numpy")
            q = grid.qx[count // 2 + 2]
            field = np.exp(1j * q * grid.x)
            expected = field * np.exp(-0.5j * 3 * q**2 / self.k)
            np.testing.assert_allclose(grid.propagate(field, 3., self.k), expected, rtol=0, atol=1e-12)

    def test_energy_wavelength_units(self):
        wavelength, energy = xpos.wavelength_energy(energy_kev=12.)
        self.assertEqual(energy, 12.)
        # The current API uses rounded hc. Quantify its accuracy without silently
        # changing a constant that affects all established numerical results.
        codata = h * c / (12e3 * elementary_charge)
        self.assertAlmostEqual(wavelength / codata, 1., delta=2e-6)
