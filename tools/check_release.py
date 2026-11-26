"""Validate a release tag without importing XPOSim or optional numerical packages."""
import ast
import re
import sys
from pathlib import Path


def check_release(root: Path, tag: str) -> str:
    tree = ast.parse((root / "XPOSim.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "__version__"
                        for target in node.targets)]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError("XPOSim.py must declare one literal __version__ string.")
    version = versions[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?", version):
        raise ValueError("Use a three-component release version, optionally with a/b/rc/post suffix.")
    if tag != f"v{version}":
        raise ValueError(f"Release tag {tag!r} must match module version v{version}.")
    for relative in ("LICENSE", "THIRD_PARTY_NOTICES.md", "xray_material_data/LICENSE",
                     "xray_material_data/README.md"):
        path = root / relative
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing release licensing/provenance file: {relative}")
    return version


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/check_release.py vMAJOR.MINOR.PATCH")
    try:
        version = check_release(Path(__file__).resolve().parents[1], sys.argv[1])
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(f"Release {version}: tag, code version and licensing files agree.")


if __name__ == "__main__":
    main()
