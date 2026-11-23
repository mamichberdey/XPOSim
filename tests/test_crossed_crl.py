"""Crossed CRL alignment and sequential wave propagation on a small CPU grid."""

from dataclasses import replace
import unittest

import numpy as np

from XPOSim import CRL, CRLFocusConfig, CRLStackConfig, Grid2D, analyze_crossed_crl, estimate_crl_focus
from XPOSim import _build_focus_session, _CRLFocusSession, _find_equal_fwhm_plane, _focus_lens_parameters


class CrossedCRLTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CRLFocusConfig(
            geometry="crossed", radius_m=6.25e-6, aperture_m=50e-6,
            bridge_thickness_m=2e-6, n_elements=26, structure_depth_m=35e-6,
            material_delta=3.35e-6, material_beta=3.7e-8,
            second_crl=CRLStackConfig(
                radius_m=4.5e-6, aperture_m=48e-6, structure_depth_m=36e-6,
                material_delta=3.6e-6, material_beta=4.1e-8),
            backend="numpy", grid_points=128, grid_points_y=128,
            scan_points=7, adaptive_window=False,
        )

    def assert_longitudinal_sections(self, result):
        self.assertEqual(result.longitudinal_intensity.shape, (len(result.z_m), len(result.x_m)))
        self.assertEqual(result.longitudinal_intensity_y.shape, (len(result.z_m), len(result.y_m)))
        for index in (0, result.peak_index, len(result.z_m)-1):
            p = result.plane_at(result.z_m[index], full=True)
            xs = np.searchsorted(p.x_m, result.x_m)
            ys = np.searchsorted(p.y_m, result.y_m)
            np.testing.assert_allclose(result.longitudinal_intensity[index],
                                       p.intensity[np.argmin(abs(p.y_m)), xs])
            np.testing.assert_allclose(result.longitudinal_intensity_y[index],
                                       p.intensity[ys, np.argmin(abs(p.x_m))])

    def test_auto_and_manual_gaps_use_both_stack_lengths(self):
        for cfg in (self.cfg, replace(self.cfg, source_type="plane"),
                    replace(self.cfg, source_distance_m=.09)):
            with self.subTest(source=cfg.source_type, distance=cfg.source_distance_m):
                a = analyze_crossed_crl(cfg)
                self.assertTrue(a.aligned)
                gap = a.recommended_gap_m
                self.assertGreater(gap, 0)
                self.assertAlmostEqual(a.x_lens.image_distance_m,
                    gap + a.y_lens.lens_length_m + a.y_lens.image_distance_m, places=12)
                wrong = analyze_crossed_crl(replace(cfg, gap_m=0.))
                self.assertFalse(wrong.aligned)
                self.assertAlmostEqual(wrong.recommended_gap_m, gap, places=12)
                installed = analyze_crossed_crl(replace(cfg, gap_m=gap))
                self.assertTrue(installed.aligned)
                if cfg.source_type == "plane":
                    self.assertAlmostEqual(gap, a.x_lens.collimated_focus_m
                        - a.y_lens.lens_length_m - a.y_lens.collimated_focus_m, places=12)

    def test_crossed_requires_two_dimensional_grid(self):
        with self.assertRaisesRegex(ValueError, "requires a 2D grid"):
            replace(self.cfg, grid_dimension=1)

    def test_no_nonnegative_matching_gap(self):
        cfg = replace(self.cfg, second_crl=replace(self.cfg.second_crl, radius_m=20e-6))
        alignment = analyze_crossed_crl(cfg)
        self.assertIsNone(alignment.gap_m)
        self.assertFalse(alignment.aligned)
        self.assertIn("nonnegative gap", alignment.message)
        with self.assertRaisesRegex(ValueError, "nonnegative gap"):
            estimate_crl_focus(cfg)

    def test_wave_field_matches_two_stacks_and_explicit_free_space(self):
        cfg = replace(self.cfg, source_type="plane")
        result = estimate_crl_focus(cfg)
        analysis = result.crossed_alignment
        self.assertEqual(result.fwhm_axes, ("x", "y"))
        self.assertAlmostEqual(result.lens_length_m,
            analysis.x_lens.lens_length_m + analysis.gap_m + analysis.y_lens.lens_length_m, places=13)
        grid = Grid2D(result.grid_step_m, result.grid_step_y_m, *reversed(result.grid_shape), backend="numpy")
        y_cfg = cfg.second_crl._as_focus_config(cfg.energy_kev, None)
        lenses = []
        for axis, stack in enumerate((cfg, y_cfg)):
            lenses.append(CRL(grid=grid, radius=stack.radius_m, aperture=stack.aperture_m,
                bridge_thickness=stack.bridge_thickness_m, n_elements=stack.n_elements,
                transverse_length=stack.structure_depth_m, rotation=axis * np.pi / 2,
                material=stack.create_material(), energy_kev=cfg.energy_kev))
        first, second = lenses
        first.input_field = first._entrance_pupil().astype(complex)
        second.input_field = first.propagate(first.E(), analysis.gap_m) * second._entrance_pupil()
        direct = second.E()
        exit_field = result._session.exit_field
        np.testing.assert_allclose(exit_field, direct, rtol=2e-12, atol=1e-12)
        self.assertAlmostEqual(result.transmission, np.sum(abs(direct)**2) /
            np.sum(abs(first.input_field)**2), places=12)
        self.assertEqual(result.z_m[result.peak_index], result.peak_z_m)
        summary = result.summary()
        self.assertEqual(summary["distance_reference"], "y CRL exit")
        self.assertTrue(summary["crossed_alignment"]["aligned"])
        self.assert_longitudinal_sections(result)

    def test_point_focus_sections_use_nearest_available_axis_in_manual_roi(self):
        cfg = replace(self.cfg, geometry="two-dimensional", second_crl=None,
                      detector_roi_m=((.5e-6,4e-6),(1e-6,5e-6)))
        result = estimate_crl_focus(cfg)
        self.assertGreater(result.x_m[0], 0.)
        self.assertGreater(result.y_m[0], 0.)
        self.assert_longitudinal_sections(result)

    def test_equal_fwhm_refines_the_crossing_nearest_the_intensity_peak(self):
        z = np.linspace(0.01, 0.05, 6)

        def measure(distance):
            return 1e-6 * (1 + 4000 * (distance - .021) * (distance - .038)), 1e-6

        widths = np.array([measure(distance) for distance in z])
        for reference, expected in ((.020, .021), (.043, .038)):
            with self.subTest(reference=reference):
                plane = _find_equal_fwhm_plane(z, widths, reference, measure)
                self.assertAlmostEqual(plane.z_m, expected, places=9)
                self.assertAlmostEqual(plane.fwhm_x_m, plane.fwhm_y_m, places=12)

    def test_equal_fwhm_handles_tangency_and_rejects_unresolved_widths(self):
        z = np.linspace(.01, .05, 5)

        def tangent(distance):
            return 1e-6 * (1 + 5000 * (distance - .025)**2), 1e-6

        widths = np.array([tangent(distance) for distance in z])
        plane = _find_equal_fwhm_plane(z, widths, .031, tangent)
        self.assertAlmostEqual(plane.z_m, .025, places=8)
        widths[2] = np.nan
        self.assertIsNone(_find_equal_fwhm_plane(z, widths, .031, tangent))

    def test_equal_fwhm_rejects_a_lobe_switch_without_an_actual_match(self):
        z = np.linspace(.01, .05, 5)

        def discontinuity(distance):
            return (1.2e-6 if distance < .035 else .8e-6), 1e-6

        widths = np.array([discontinuity(distance) for distance in z])
        self.assertIsNone(_find_equal_fwhm_plane(z, widths, .03, discontinuity))

    def test_gaussian_projection_and_point_sampling_for_a_wide_y_source(self):
        cfg = replace(self.cfg, radius_m=6.25e-6, n_elements=6, structure_depth_m=30e-6,
            material_delta=None, material_beta=None, material_formula="Si", density_g_cm3=2.33,
            second_crl=CRLStackConfig(radius_m=6.25e-6, n_elements=58, structure_depth_m=30e-6),
            source_type="gaussian", source_fwhm_x_m=0, source_fwhm_y_m=1e-3,
            grid_points=192, grid_points_y=384, batch_size=4, source_points_y=15)
        alignment = analyze_crossed_crl(cfg)
        z = alignment.common_image_m
        downstream = z + 1.05e-3
        grid, lens, convolution, _ = _build_focus_session(
            cfg, "numpy", _focus_lens_parameters(cfg), z, alignment=alignment)
        magnification = lens.magnification(z, cfg.source_distance_m)
        self.assertEqual(convolution.projected_source_fwhm(z)[0], 0.)
        near_width = convolution.projected_source_fwhm(z)[1]
        far_width = convolution.projected_source_fwhm(downstream)[1]
        self.assertLess(abs(near_width / (abs(magnification[1]) * cfg.source_fwhm_y_m) - 1), .1)
        self.assertGreater(far_width, .95 * near_width)
        legacy_far_width = abs(lens.magnification(downstream, cfg.source_distance_m)[1]) * cfg.source_fwhm_y_m
        self.assertGreater(far_width, 1.5 * legacy_far_width)

        points = _CRLFocusSession(replace(cfg, source_method="points"), grid, lens,
                                  convolution.incident, coherent_state=convolution)
        self.assertEqual(tuple(map(len, points.source_nodes)), (1, 15))
        np.testing.assert_allclose(points.source_nodes[1], -points.source_nodes[1][::-1])
        self.assertAlmostEqual(np.sum(points.weights), 1.)
        self.assertAlmostEqual(np.sqrt(np.sum(points.weights * points.source_nodes[1]**2)),points.source_sigma[1])
        self.assertLess(abs(points.transmission / convolution.transmission - 1), .01)
        direct_at_image = points.plane(z).metrics
        blur_at_image = convolution.plane(z).metrics
        for actual, projected in zip(direct_at_image.fwhm, blur_at_image.fwhm):
            self.assertLess(abs(actual / projected - 1), .07)
        direct_ratio = points.plane(downstream).metrics.max_intensity / direct_at_image.max_intensity
        blur_ratio = convolution.plane(downstream).metrics.max_intensity / blur_at_image.max_intensity
        self.assertLess(direct_ratio, 1.)
        self.assertLess(blur_ratio, 1.)
        self.assertAlmostEqual(direct_ratio, blur_ratio, delta=.12)


if __name__ == "__main__":
    unittest.main()
