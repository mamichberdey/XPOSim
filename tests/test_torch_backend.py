"""Backend parity on identical fields and optical configurations."""
from dataclasses import replace
from unittest import TestCase, skipUnless
from unittest.mock import patch

import numpy as np
import XPOSim as xpos

try:
    import torch
except (ImportError, OSError):
    torch = None


def compare_fields(backend, reference_backend="numpy"):
    """Compare complex amplitudes, including phase, at fixed detector positions."""
    rng = np.random.default_rng(2026)
    for grid_type, args, zoom in (
        (xpos.Grid1D, (1e-7, 33), (8e-8, 24)),
        (xpos.Grid2D, (1e-7, 2e-7, 32, 27), (8e-8, 1.5e-7, 25, 19)),
    ):
        reference = grid_type(*args, backend=reference_backend)
        grid = grid_type(*args, backend=backend)
        field = rng.normal(size=grid.shape) + 1j*rng.normal(size=grid.shape)
        data = grid.xp.asarray(field)
        ref_field = reference.xp.asarray(field)
        distances = [-0.012, 0, 0.023]
        wave_number = 6e10
        for expected, actual in (
            (reference.fft(ref_field), grid.fft(data)),
            (reference.propagate_many(ref_field, distances, wave_number, batch_size=2),
             grid.propagate_many(data, distances, wave_number, batch_size=2)),
            (reference.propagate_to_grid(ref_field, distances[-1], wave_number,
                                         grid_type(*zoom, backend=reference_backend)),
             grid.propagate_to_grid(data, distances[-1], wave_number,
                                    grid_type(*zoom, backend=backend))),
        ):
            expected = xpos._to_numpy(expected)
            torch.testing.assert_close(actual, torch.as_tensor(expected, device=data.device),
                                       rtol=2e-10, atol=2e-12 * np.abs(expected).max())

        axes = np.array([-2., -0.8, 0., 1., 2.]) * 1e-6
        heights = rng.random((5,) * grid.dimension)
        surface = xpos.SurfaceError([heights, *([axes] * grid.dimension)])
        np.testing.assert_allclose(xpos._to_numpy(surface.on_grid(grid)),
                                   xpos._to_numpy(surface.on_grid(reference)), rtol=1e-12, atol=1e-14)


def compare_focus(backend, reference_backend="numpy"):
    for dimension, method in ((1, "points"), (2, "convolution")):
        config = xpos.CRLFocusConfig(
            backend=reference_backend, grid_dimension=dimension, grid_points=128,
            grid_points_y=128 if dimension == 2 else None, source_type="gaussian",
            source_method=method, source_points=3, adaptive_window=False, scan_points=7,
        )
        reference = xpos.estimate_crl_focus(config)
        actual = xpos.estimate_crl_focus(replace(config, backend=backend))
        for name in ("peak_z_m", "transmission", "z_m", "peak_intensity", "fwhm_m",
                     "best_intensity", "longitudinal_intensity"):
            np.testing.assert_allclose(getattr(actual, name), getattr(reference, name),
                                       rtol=1e-8, atol=1e-13, err_msg=name)


@skipUnless(torch is not None, "PyTorch is optional")
class TorchBackendTests(TestCase):
    def test_device_independent_kernels(self):
        # Exercise real Torch operations in CPU-only CI. Production selection requires CUDA.
        with patch.object(xpos, "_torch_gpu", return_value=xpos._torch_arrays(torch.device("cpu"))):
            compare_fields("torch")
            compare_focus("torch")

    def test_gpu_selection(self):
        if torch.cuda.is_available():
            self.assertEqual(xpos.select_backend("torch"), "torch")
            self.assertTrue(xpos.Grid1D(1e-7, 32, backend="torch").x.is_cuda)
        else:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                xpos.select_backend("torch")
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                xpos.Grid1D(1e-7, 32, backend="torch")

    @skipUnless(torch is not None and torch.cuda.is_available(), "CUDA GPU required")
    def test_torch_cuda_matches_numpy(self):
        compare_fields("torch")
        compare_focus("torch")

    @skipUnless(torch is not None and torch.cuda.is_available() and xpos.CUPY_AVAILABLE,
                "CUDA GPU and CuPy required")
    def test_torch_cuda_matches_cupy(self):
        compare_fields("torch", "cupy")
        compare_focus("torch", "cupy")
