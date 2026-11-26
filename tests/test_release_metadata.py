"""Release mistakes must fail before a PyPI upload is attempted."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_release", ROOT / "tools/check_release.py")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseMetadataTests(TestCase):
    def test_current_version_can_be_released_without_importing_runtime(self):
        import XPOSim
        self.assertEqual(RELEASE.check_release(ROOT, "v" + XPOSim.__version__), XPOSim.__version__)

    def test_invalid_tags_and_missing_notices_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "XPOSim.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
            for relative in ("LICENSE", "THIRD_PARTY_NOTICES.md", "xray_material_data/LICENSE",
                             "xray_material_data/README.md"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Release notice", encoding="utf-8")
            self.assertEqual(RELEASE.check_release(root, "v1.2.3"), "1.2.3")
            for tag in ("main", "1.2.3", "v1.2.4", "v1.2.3; echo bad"):
                with self.assertRaisesRegex(ValueError, "must match"):
                    RELEASE.check_release(root, tag)
            (root / "xray_material_data/LICENSE").unlink()
            with self.assertRaisesRegex(ValueError, "Missing release licensing"):
                RELEASE.check_release(root, "v1.2.3")

    def test_ambiguous_or_nonrelease_versions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for text in ('__version__ = "latest"', '__version__ = "1.2.3"\n__version__ = "1.2.4"'):
                (root / "XPOSim.py").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    RELEASE.check_release(root, "v1.2.3")
