"""Validate built distributions and installation without dependency resolution.

Run with NumPy and SciPy already installed in this interpreter.
Usage: python tests/check_distribution.py dist
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import venv
from email.parser import BytesParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

DATA_FILES = (
    "AtomicConstants.dat",
    "CrossSec-Compton_McMaster.dat",
    "f1f2_Windt.dat",
)


def run(python, arguments, cwd, env):
    result = subprocess.run([str(python), *arguments], cwd=cwd, env=env,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def installed(python, cwd, env):
    rows = json.loads(run(python, ["-m", "pip", "list", "--format=json"], cwd, env))
    return {row["name"].lower().replace("_", "-"): row["version"] for row in rows
            if row["name"].lower() != "xposim"}


def check_install(distribution, version, directory, env, inherit):
    source_archive = distribution.suffix != ".whl"
    target = directory / ("source" if source_archive else "existing" if inherit else "bare")
    venv.EnvBuilder(with_pip=True, system_site_packages=inherit).create(target)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    before = installed(python, directory, env)
    # Deliberately omit --no-deps to verify the package's ordinary pip behavior.
    # No package index is available to either test installation.
    arguments = ["-m", "pip", "install", "--no-index"]
    if source_archive:
        arguments.append("--no-build-isolation")
    run(python, [*arguments, str(distribution)], directory, env)
    assert installed(python, directory, env) == before, "Other distributions changed"
    check = '''
import importlib.metadata as metadata
from pathlib import Path
import sys
assert not metadata.requires("XPOSim")
assert metadata.version("XPOSim") == sys.argv[1]
'''
    if inherit:
        check += '''
import numpy as np
import scipy
import XPOSim as xpos
assert Path(xpos.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert xpos.__version__ == metadata.version("XPOSim")
for formula, density in (("Si", 2.33), ("Be", 1.848)):
    delta, beta = xpos.Material(formula=formula, density=density).optical_constants(12.)
    assert np.isfinite(delta) and np.isfinite(beta) and delta > 0 and beta >= 0
assert xpos.select_backend("numpy") == "numpy"
grid = xpos.Grid1D(1e-7, 257, backend="numpy")
field = np.exp(-(grid.x/1e-6)**2).astype(complex)
actual = grid.propagate(field, .03, 2*np.pi/1e-10)
np.testing.assert_allclose(np.sum(abs(actual)**2), np.sum(abs(field)**2), rtol=1e-12)
assert np.isfinite(xpos.beam_metrics(actual, grid).max_intensity)
'''
    else:
        check += '''
from importlib.util import find_spec
assert find_spec("numpy") is None
assert find_spec("scipy") is None
try:
    import XPOSim
except ModuleNotFoundError as error:
    assert error.name == "numpy"
else:
    raise AssertionError("Expected the missing NumPy dependency to be reported")
'''
    run(python, ["-c", check, version], directory, env)
    assert installed(python, directory, env) == before, "Import changed the environment"


def main():
    artifacts = Path(sys.argv[1]).resolve()
    wheels = list(artifacts.glob("*.whl"))
    archives = list(artifacts.glob("*.tar.gz"))
    assert len(wheels) == len(archives) == 1
    wheel = wheels[0]
    source = Path(__file__).resolve().parents[1]
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        assert metadata["Name"].lower() == "xposim"
        assert not metadata.get_all("Requires-Dist"), "Wheel declares runtime dependencies"
        assert "XPOSim.py" in names
        for name in DATA_FILES:
            data = archive.read("xray_material_data/" + name)
            assert hashlib.sha256(data).digest() == hashlib.sha256(
                (source/"xray_material_data"/name).read_bytes()).digest()
    with tarfile.open(archives[0], "r:gz") as archive:
        names = archive.getnames()
        for name in ("XPOSim.py", "pyproject.toml", "tests/backend_parity.py",
                     *("xray_material_data/" + name for name in DATA_FILES)):
            assert any(path.endswith("/" + name) for path in names), name
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key != "PYTHONPATH"}
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    with TemporaryDirectory(prefix="xposim-distribution-") as temporary:
        directory = Path(temporary)
        for inherit in (False, True):
            check_install(wheel, metadata["Version"], directory, env, inherit)
        check_install(archives[0], metadata["Version"], directory, env, inherit=True)
    print("Wheel/sdist, material tables and all three offline installation scenarios passed")


if __name__ == "__main__":
    main()
