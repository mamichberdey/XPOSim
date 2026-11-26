"""Backend dispatch and GPU-to-CPU fallback without optional GPU dependencies."""
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np

import XPOSim as xpos


class BackendSelectionTests(TestCase):
    def test_explicit_backend_dispatch(self):
        providers = {name: MagicMock() for name in ("cupy", "torch", "jax")}
        with (
            patch.object(xpos, "CUPY_AVAILABLE", True),
            patch.object(xpos, "cp", providers["cupy"]),
            patch.object(xpos, "_torch_gpu", return_value=providers["torch"]),
            patch.object(xpos, "_jax_gpu", return_value=providers["jax"]),
        ):
            self.assertIs(xpos._backend_module("numpy"), np)
            for name, provider in providers.items():
                with self.subTest(backend=name):
                    self.assertIs(xpos._backend_module(name.upper()), provider)
            self.assertEqual(xpos.select_backend("numpy"), "numpy")
            providers["cupy"].cuda.runtime.getDeviceCount.return_value = 1
            for name in providers:
                self.assertEqual(xpos.select_backend(name), name)

    def test_auto_priority_and_cpu_fallback(self):
        for expected in ("cupy", "torch", "jax", "numpy"):
            with self.subTest(expected=expected):
                device = MagicMock()
                device.cuda.runtime.getDeviceCount.return_value = 1
                torch_error = None if expected == "torch" else RuntimeError("GPU unavailable")
                jax_error = None if expected == "jax" else RuntimeError("GPU unavailable")
                with (
                    patch.object(xpos, "CUPY_AVAILABLE", expected == "cupy"),
                    patch.object(xpos, "cp", device),
                    patch.object(xpos, "_torch_gpu", side_effect=torch_error,
                                 return_value=device) as torch_probe,
                    patch.object(xpos, "_jax_gpu", side_effect=jax_error,
                                 return_value=device) as jax_probe,
                ):
                    self.assertEqual(xpos.select_backend("auto"), expected)
                    self.assertEqual(torch_probe.call_count, int(expected != "cupy"))
                    self.assertEqual(jax_probe.call_count, int(expected in {"jax", "numpy"}))

    def test_unknown_backend_is_rejected_at_public_entry_points(self):
        with self.assertRaisesRegex(ValueError, "auto, numpy, cupy, torch or jax"):
            xpos.select_backend("unsupported")
        with self.assertRaisesRegex(ValueError, "numpy, cupy, torch or jax"):
            xpos.Grid1D(1e-7, 32, backend="unsupported")
        with self.assertRaisesRegex(ValueError, "numpy, cupy, torch or jax"):
            xpos.Grid2D(1e-7, 2e-7, 32, 31, backend="unsupported")
        with self.assertRaises(ValueError):
            xpos.CRLFocusConfig(backend="unsupported")
