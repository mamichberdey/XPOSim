"""Crossed CRL alignment and sequential wave propagation on a small CPU grid."""

from dataclasses import replace
import unittest

import numpy as np

from XPOSim import CRL, CRLFocusConfig, CRLStackConfig, Grid2D, analyze_crossed_crl, estimate_crl_focus


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


if __name__ == "__main__":
    unittest.main()
