"""Public convergence diagnostics, storage estimates and retained-result lifetime."""
import json
from dataclasses import asdict, replace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

import XPOSim as xpos


def config(**options):
    values = {"backend": "numpy", "grid_points": 64, "scan_points": 7, "adaptive_window": False,
                  "material_delta": 3e-6, "material_beta": 2e-8, "n_elements": 3,
                  "radius_m": 50e-6, "aperture_m": 40e-6, "source_type": "plane"}
    values.update(options)
    return xpos.CRLFocusConfig(**values)


class FocusReliabilityTests(TestCase):
    def test_memory_estimate_batches_and_retained_storage(self):
        grid = xpos.Grid2D(1e-7, 2e-7, 65, 32, backend="numpy")
        plan = grid.estimate_memory(11, batch_size=3, source_modes=7, return_fields=True)
        field_bytes = 65*32*16
        self.assertEqual(plan["field_bytes"], field_bytes)
        self.assertEqual(plan["batch_size"], 3)
        self.assertEqual(plan["working_bytes_estimate"], 18*field_bytes)
        self.assertEqual(plan["output_stack_bytes"], 11*field_bytes)
        self.assertEqual(plan["source_spectra_bytes"], 7*field_bytes)
        self.assertEqual(grid.estimate_memory(11)["batch_size"], 1)
        self.assertEqual(grid.estimate_memory(11)["output_stack_bytes"], 0)
        with patch.object(grid, "_free_gpu_memory", return_value=6*field_bytes*8*4):
            self.assertEqual(grid.estimate_memory(11)["batch_size"], 8)
        for options in ({"n_planes":0}, {"source_modes":0}, {"batch_size":0}):
            with self.assertRaises(ValueError):
                grid.estimate_memory(**options)

    def test_numerical_report_and_preflight_progress(self):
        events = []
        with xpos.estimate_crl_focus(config(), progress=events.append) as result:
            report = result.numerical_report
            self.assertIsInstance(report, xpos.CRLNumericalReport)
            self.assertEqual(len(report.estimated_phase_step_rad), 1)
            self.assertAlmostEqual(report.detector_points_per_fwhm[0],
                                   result.fwhm_at_peak_m/result.detector_step_m)
            self.assertTrue(0 <= report.detector_boundary_power_fraction <= 1)
            self.assertTrue(0 <= report.coherent_exit_boundary_power_fraction <= 1)
            self.assertEqual(report.memory["source_cache_location"], "not_used")
            self.assertFalse(result.summary()["convergence_checked"])
            json.dumps(result.summary(), allow_nan=False)
        stages = [event.stage for event in events]
        self.assertLess(stages.index("memory"), stages.index("peak_search"))
        with xpos.estimate_crl_focus(config(geometry="two-dimensional", grid_points_y=32)) as result:
            self.assertEqual(len(result.numerical_report.estimated_phase_step_rad), 2)
            self.assertEqual(len(result.numerical_report.detector_points_per_fwhm), 2)
            self.assertAlmostEqual(result.numerical_report.estimated_phase_step_rad[1],
                result.numerical_report.estimated_phase_step_rad[0]*result.grid_step_y_m/result.grid_step_m)

    def test_boundary_power_is_union_including_small_and_dark_arrays(self):
        for shape in ((1,), (2,), (20,), (1, 2), (20, 30)):
            image = np.ones(shape)
            inner = tuple(slice(max(1, n//20), -max(1, n//20)) for n in shape)
            expected = 1-image[inner].size/image.size
            self.assertAlmostEqual(float(xpos._boundary_power_fraction(image, np)), expected)
        self.assertTrue(np.isnan(xpos._boundary_power_fraction(np.zeros((20, 30)), np)))

    def test_context_manager_preserves_exports_and_closes_scratch_file(self):
        cfg = config(source_type="gaussian", source_method="points", source_points=3)
        with patch.object(xpos, "_source_host_cache_budget", return_value=0):
            with xpos.estimate_crl_focus(cfg) as result:
                session = result._session
                scratch = session._mode_file
                self.assertIsNotNone(scratch)
                self.assertEqual(result.numerical_report.memory["source_cache_location"], "temporary storage")
                expected_summary = result.summary()
                image = result.best_intensity.copy()
                self.assertFalse(result.closed)
                result.plane_at(result.peak_z_m)
            self.assertTrue(scratch.closed)
            self.assertTrue(result.closed)
            self.assertIsNone(session.spectrum)
            self.assertIsNone(session.crl)
            self.assertEqual(session._mode_batches, [])
            np.testing.assert_array_equal(result.best_intensity, image)
            self.assertEqual(result.summary(), expected_summary)
            result.close()
            for action in (lambda: result.plane_at(.1), result.check_convergence, result.__enter__):
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    action()

    def test_exception_closes_prepared_and_partially_prepared_sessions(self):
        cfg = config(source_type="gaussian", source_method="points", source_points=3)
        original = xpos._CRLFocusSession.__init__
        for failed_stage in ("window_checks", "source_modes", "sampling", "finalizing", "complete"):
            sessions, files = [], []
            def capture(session, *args, sessions=sessions, files=files, **kwargs):
                sessions.append(session)
                original(session, *args, **kwargs)
                if session._mode_file is not None:
                    files.append(session._mode_file)
            def fail(event, failed_stage=failed_stage):
                if event.stage == failed_stage:
                    raise RuntimeError("cancelled")
            with self.subTest(stage=failed_stage), patch.object(xpos, "_source_host_cache_budget", return_value=0):
                with patch.object(xpos._CRLFocusSession, "__init__", capture), self.assertRaisesRegex(RuntimeError, "cancelled"):
                    xpos.estimate_crl_focus(replace(cfg, adaptive_window=True), progress=fail)
                self.assertTrue(sessions)
                self.assertTrue(all(session._closed for session in sessions))
                self.assertTrue(all(file.closed for file in files))

    def test_convergence_preserves_physical_windows_and_parity(self):
        for counts in ((64,), (65,), (33, 32)):
            options = {"grid_points": counts[0]}
            if len(counts) == 2:
                options.update(geometry="two-dimensional", grid_points_y=counts[1])
            cfg = config(**options)
            sessions = []
            original = xpos._estimate_crl_focus
            def capture(*args, original=original, sessions=sessions, **kwargs):
                result = original(*args, **kwargs)
                sessions.append(result._session)
                return result
            with xpos.estimate_crl_focus(cfg) as result:
                base_image = result.best_intensity.copy()
                with patch.object(xpos, "_estimate_crl_focus", side_effect=capture):
                    report = result.check_convergence(relative_tolerance=.05)
                self.assertEqual(result.configuration, cfg)
                np.testing.assert_array_equal(result.best_intensity, base_image)
                self.assertTrue(result.summary()["convergence_checked"])
                self.assertEqual(result.summary()["convergence_report"], asdict(report))
                self.assertTrue(all(session._closed for session in sessions))
                base_steps = (result.grid_step_m,) if len(counts) == 1 else (result.grid_step_m, result.grid_step_y_m)
                for case in report.comparisons:
                    shape = case["grid_shape"][::-1]
                    self.assertEqual(tuple(n % 2 for n in shape), tuple(n % 2 for n in counts))
                    if case["case"] == "refined_sampling":
                        np.testing.assert_allclose(np.array(shape)*case["grid_step_m"], np.array(counts)*base_steps)
                    else:
                        np.testing.assert_array_equal(case["grid_step_m"], base_steps)
                    self.assertEqual(case["stable"], all(change is not None and np.isfinite(change) and change <= .05
                                                        for change in case["relative_changes"].values()))
                self.assertEqual(report.stable, all(case["stable"] for case in report.comparisons))
                for invalid in (0., -1., np.nan, np.inf):
                    with self.assertRaises(ValueError):
                        result.check_convergence(relative_tolerance=invalid)

    def test_failed_convergence_leaves_original_result_usable(self):
        with xpos.estimate_crl_focus(config()) as result:
            def cancel(event):
                if event.stage == "sampling":
                    raise RuntimeError("cancelled")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                result.check_convergence(progress=cancel)
            self.assertIsNone(result.convergence_report)
            self.assertFalse(result.closed)
            result.plane_at(result.peak_z_m)

    def test_unresolved_width_cannot_certify_convergence(self):
        with xpos.estimate_crl_focus(config()) as result:
            with patch.object(xpos, "_convergence_observables", return_value={"fwhm_x_m":None}):
                report = result.check_convergence()
            self.assertFalse(report.stable)
            self.assertTrue(all(case["relative_changes"]["fwhm_x_m"] is None for case in report.comparisons))

    def test_plane_and_gaussian_summaries_survive_close(self):
        for cfg in (config(), config(source_type="gaussian")):
            result = xpos.estimate_crl_focus(cfg)
            expected = result.summary()
            result.close()
            self.assertEqual(result.summary(), expected)

    def test_optional_backend_reliability_matches_numpy(self):
        # CPU injection exercises real array operations without authorizing a
        # production CPU fallback for explicitly requested accelerator backends.
        import importlib
        checked = 0
        cfg = config(source_type="gaussian", source_method="points", source_points=3)
        with xpos.estimate_crl_focus(cfg) as reference:
            for backend, package in (("torch", "torch"), ("jax", "jax")):
                try:
                    module = importlib.import_module(package)
                except (ImportError, OSError):
                    continue
                xp = (xpos._torch_arrays(module.device("cpu")) if backend == "torch" else
                      xpos._jax_arrays(module.devices("cpu")[0]))
                with self.subTest(backend=backend), patch.object(xpos, f"_{backend}_gpu", return_value=xp):
                    with xpos.estimate_crl_focus(replace(cfg, backend=backend)) as result:
                        np.testing.assert_allclose(result.best_intensity, reference.best_intensity,
                                                   rtol=1e-8, atol=1e-12)
                        for name in ("estimated_phase_step_rad", "detector_points_per_fwhm",
                                     "detector_boundary_power_fraction", "coherent_exit_boundary_power_fraction"):
                            np.testing.assert_allclose(getattr(result.numerical_report, name),
                                                       getattr(reference.numerical_report, name),
                                                       rtol=1e-8, atol=1e-12)
                        report = result.check_convergence()
                        self.assertEqual(len(report.comparisons), 2)
                        expected = result.summary()
                    self.assertTrue(result.closed)
                    self.assertEqual(result.summary(), expected)
                checked += 1
        if not checked:
            self.skipTest("Optional Torch/JAX packages unavailable")
