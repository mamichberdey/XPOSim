"""Run identical TensorFlow kernel checks in CPU CI and on a real GPU."""
from unittest import TestCase, skipUnless
from unittest.mock import patch

from backend_parity import compare_results
import XPOSim as xpos

try:
    import tensorflow as tf
except (ImportError, OSError):
    tf = None


@skipUnless(tf is not None, "TensorFlow is optional")
class TensorFlowBackendTests(TestCase):
    def test_kernels(self):
        with tf.device("/CPU:0"):
            xp = xpos._tensorflow_arrays(tf.constant(1).device)
            with patch.object(xpos, "_tensorflow_gpu", return_value=xp):
                compare_results(self, "tensorflow")

    def test_gpu(self):
        if not tf.config.list_physical_devices("GPU"):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                xpos.select_backend("tensorflow")
            self.skipTest("TensorFlow GPU unavailable")
        self.assertEqual(xpos.select_backend("tensorflow"), "tensorflow")
        compare_results(self, "tensorflow")
