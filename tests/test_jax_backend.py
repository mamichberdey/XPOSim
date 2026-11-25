"""Check GPU selection and the device-independent JAX kernels."""
from unittest import TestCase, skipUnless
from unittest.mock import patch

import XPOSim as xpos
from backend_parity import compare_results

try:
    import jax
except (ImportError, OSError):
    jax = None


@skipUnless(jax is not None, "JAX is optional")
class JaxBackendTests(TestCase):
    def test_kernels(self):
        xp = xpos._jax_arrays(jax.devices("cpu")[0])
        with patch.object(xpos, "_jax_gpu", return_value=xp):
            compare_results(self, "jax")

    def test_gpu(self):
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                xpos.select_backend("jax")
            self.skipTest("JAX GPU unavailable")
        self.assertEqual(xpos.select_backend("jax"), "jax")
        compare_results(self, "jax")
