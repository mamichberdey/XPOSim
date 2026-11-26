"""Run the shared optical parity checks when a CuPy GPU is available."""
from unittest import TestCase

import XPOSim as xpos
from backend_parity import compare_results


class CuPyBackendTests(TestCase):
    def test_gpu(self):
        try:
            xpos.select_backend("cupy")
        except RuntimeError as exc:
            self.skipTest(str(exc))
        compare_results(self, "cupy")
