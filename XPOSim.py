"""Numerical wave-optics library for 1D and 2D X-ray optical schemes.

The module deliberately keeps the public model small:

* a :class:`Grid1D` or :class:`Grid2D` owns one common numerical grid;
* optical objects transform complex electric-field arrays ``E``;
* :meth:`OpticalElement.E` returns the field immediately after the element
  for ``z=0`` and Fresnel-propagates that field for ``z>0``;
* :class:`PointSource` creates a field and therefore is the only object that
  does not consume an input field;
* NumPy and CuPy are supported without changing the array type of a field;
* CRL surface errors are specified as sampled height deviations and interpolated
  onto the calculation grid.
* cached configuration parameters are read-only; CRL.with_parameters constructs
  a validated replacement without changing the original lens;
* detector planes can be propagated in bounded batches with shared spectra.

The file is intentionally self-contained. The material database is expected
in ``xray_material_data/`` next to this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator, Sequence
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:  # CuPy is optional; it is the preferred backend when available.
    import cupy as cp

    CUPY_AVAILABLE = True
except (ImportError, OSError, RuntimeError):  # Optional GPU libraries may be unusable.
    cp = None
    CUPY_AVAILABLE = False

__version__ = "0.7.0"


__all__ = [
    "__version__",
    "CUPY_AVAILABLE",
    "wavelength_energy",
    "Material",
    "Grid1D",
    "Grid2D",
    "ThicknessDefect",
    "FunctionThicknessDefect",
    "InterpolatedThicknessDefect",
    "SurfaceError",
    "ArrayObject",
    "OpticalObject",
    "OpticalElement",
    "PointSource",
    "Hole",
    "Blade",
    "CRL",
    "BeamMetrics",
    "FocusScanResult",
    "intensity",
    "full_width_half_max",
    "beam_metrics",
    "divergence",
    "gaussian_filter_1d",
    "gaussian_filter_2d",
    "select_backend",
    "CRLFocusConfig",
    "CRLFocusResult",
    "estimate_crl_focus",
]


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _array_module(array: Any):
    """Return the NumPy/CuPy module corresponding to ``array``.

    Parameters
    ----------
    array:
        NumPy or CuPy ndarray.

    Returns
    -------
    module
        ``numpy`` for NumPy arrays or ``cupy`` for CuPy arrays.

    Raises
    ------
    TypeError
        If ``array`` is not a supported ndarray type.
    """

    if isinstance(array, np.ndarray):
        return np
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp
    raise TypeError("Field must be a NumPy or CuPy ndarray.")


def _backend_name(array: Any) -> str:
    """Return ``'numpy'`` or ``'cupy'`` for a supported ndarray."""

    return "cupy" if CUPY_AVAILABLE and isinstance(array, cp.ndarray) else "numpy"


def _to_python_float(value: Any) -> float:
    """Convert a scalar NumPy/CuPy value to a Python ``float``."""

    if CUPY_AVAILABLE and isinstance(value, cp.ndarray):
        return float(value.get())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _to_numpy(value: Any) -> np.ndarray:
    """Return a NumPy representation of a scalar/array without changing the input."""

    if CUPY_AVAILABLE and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _validate_finite(name: str, value: float) -> None:
    """Reject non-finite scalar parameters before numerical calculation."""

    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar; got {value!r}.")


def _validate_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` when a numeric parameter is not strictly positive."""

    _validate_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value!r}.")


def _validate_count(name: str, value: int, minimum: int = 1) -> None:
    """Reject fractional counts and booleans rather than truncating them."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}.")


class _Configured:
    """Protect parameters whose reassignment would invalidate derived state."""

    _configuration_fields: frozenset[str] = frozenset()

    def __setattr__(self, name: str, value: Any) -> None:
        if (getattr(self, "_configuration_frozen", False)
                and name in self._configuration_fields and name in self.__dict__):
            raise AttributeError(
                f"{type(self).__name__}.{name} is read-only. "
                "Construct a new configuration"
                + (" with with_parameters(...)." if hasattr(self, "with_parameters") else ".")
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._configuration_fields:
            raise AttributeError(f"{type(self).__name__}.{name} is read-only.")
        object.__delattr__(self, name)

    def _freeze_configuration(self) -> None:
        for name in self._configuration_fields:
            value = self.__dict__.get(name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
        object.__setattr__(self, "_configuration_frozen", True)


def wavelength_energy(
    *, wavelength_m: float | None = None, energy_kev: float | None = None
) -> tuple[float, float]:
    """Resolve wavelength and photon energy from exactly one supplied quantity.

    Parameters
    ----------
    wavelength_m:
        Photon wavelength in metres.
    energy_kev:
        Photon energy in keV.

    Returns
    -------
    (float, float)
        ``(wavelength_m, energy_kev)``.

    Raises
    ------
    ValueError
        If both quantities or neither quantity are supplied.
    """

    if wavelength_m is None and energy_kev is None:
        raise ValueError("Specify either wavelength_m or energy_kev.")
    if wavelength_m is not None and energy_kev is not None:
        raise ValueError("Specify only one of wavelength_m and energy_kev.")

    hc_kev_m = 12.3984e-10
    if energy_kev is not None:
        _validate_positive("energy_kev", energy_kev)
        return hc_kev_m / energy_kev, float(energy_kev)

    _validate_positive("wavelength_m", wavelength_m)  # type: ignore[arg-type]
    return float(wavelength_m), hc_kev_m / float(wavelength_m)


# ---------------------------------------------------------------------------
# Material model
# ---------------------------------------------------------------------------


class _MaterialDatabase:
    """Internal parser/cache for the tabulated material data."""

    ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            package_dir = Path(__file__).resolve().parent
            self.data_dir = package_dir / "xray_material_data"
        self._f1f2_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._compton_cache: dict[str, np.ndarray] = {}
        self._mass_cache: dict[str, float] = {}

    @staticmethod
    def _section_lines(path: Path, element: str) -> list[str]:
        """Extract non-comment rows belonging to one ``#S`` section."""

        if not path.exists():
            raise FileNotFoundError(
                f"Material database file not found: {path}. "
                "Place the 'xray_material_data' directory next to XPOSim.py."
            )

        rows: list[str] = []
        active = False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#S"):
                    tokens = line[2:].split()
                    active = element in tokens
                    continue
                if active and not line.startswith("#"):
                    rows.append(line)
                elif active and line.startswith("#"):
                    # A new comment starts the next logical section/header.
                    continue
        return rows

    def f1f2(self, element: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and cache tabulated energy/f1/f2 values for one element."""

        element = element.strip()
        if element in self._f1f2_cache:
            return self._f1f2_cache[element]

        rows = self._section_lines(self.data_dir / "f1f2_Windt.dat", element)
        energies: list[float] = []
        f1: list[float] = []
        f2: list[float] = []
        for row in rows:
            parts = row.split()
            if len(parts) < 3:
                continue
            try:
                energies.append(float(parts[0]))
                f1.append(float(parts[1]))
                f2.append(float(parts[2]))
            except ValueError:
                continue

        if not energies:
            raise ValueError(f"No f1/f2 data found for element {element!r}.")

        result = np.asarray(energies), np.asarray(f1), np.asarray(f2)
        self._f1f2_cache[element] = result
        return result

    def compton_coefficients(self, element: str) -> np.ndarray:
        """Load and cache the four Compton correction coefficients."""

        element = element.strip()
        if element in self._compton_cache:
            return self._compton_cache[element]

        rows = self._section_lines(self.data_dir / "CrossSec-Compton_McMaster.dat", element)
        for row in rows:
            parts = row.split()
            if len(parts) < 4:
                continue
            try:
                result = np.asarray([float(parts[i]) for i in range(4)], dtype=float)
            except ValueError:
                continue
            self._compton_cache[element] = result
            return result
        raise ValueError(f"No Compton data found for element {element!r}.")

    def atomic_mass(self, element: str) -> float:
        """Load and cache the atomic mass for one element."""

        element = element.strip()
        if element in self._mass_cache:
            return self._mass_cache[element]

        rows = self._section_lines(self.data_dir / "AtomicConstants.dat", element)
        for row in rows:
            parts = row.split()
            if len(parts) < 3:
                continue
            try:
                mass = float(parts[2])
            except ValueError:
                continue
            self._mass_cache[element] = mass
            return mass
        raise ValueError(f"No atomic mass found for element {element!r}.")


_DEFAULT_MATERIAL_DATABASE = _MaterialDatabase()


class Material(_Configured):
    """Describe a material used by an X-ray optical element.

    Parameters
    ----------
    formula:
        Chemical formula such as ``"Si"`` or ``"AlSe8O15"``. Parentheses in
        chemical formulas are intentionally not parsed by this lightweight
        implementation.
    density:
        Material density in g/cm^3.
    database:
        Optional :class:`_MaterialDatabase` instance.
    delta, beta:
        Optional direct optical constants. Supplying both bypasses the tabulated
        material database and is useful for testing or custom materials.
    """

    _FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]+(?:[.,][0-9]+)?)?")
    _configuration_fields = frozenset({"formula", "density", "_database", "_delta", "_beta"})

    def __init__(
        self,
        formula: str | None = None,
        density: float | None = None,
        *,
        database: _MaterialDatabase | None = None,
        delta: float | None = None,
        beta: float | None = None,
    ) -> None:
        if (delta is None) != (beta is None):
            raise ValueError("delta and beta must be supplied together.")
        if formula is None and (delta is None or beta is None):
            raise ValueError("Specify formula+density or direct delta+beta.")
        if formula is not None and density is None:
            raise ValueError("density is required when formula is supplied.")

        self.formula = formula
        self.density = None if density is None else float(density)
        self._database = database or _DEFAULT_MATERIAL_DATABASE
        self._delta = None if delta is None else float(delta)
        self._beta = None if beta is None else float(beta)

        if self.density is not None:
            _validate_positive("density", self.density)
        if self._delta is not None:
            _validate_finite("delta", self._delta)
            _validate_finite("beta", self._beta)
            if self._beta < 0:
                raise ValueError("beta must be >= 0.")
        self._freeze_configuration()

    def composition(self) -> list[tuple[str, float]]:
        """Parse the chemical formula into ``(element, stoichiometry)`` pairs."""

        if self.formula is None:
            raise ValueError("Direct optical constants do not have a chemical composition.")

        formula = self.formula.replace(",", ".").strip()
        result: list[tuple[str, float]] = []
        position = 0
        for match in self._FORMULA_RE.finditer(formula):
            if match.start() != position:
                raise ValueError(f"Unsupported chemical formula syntax: {self.formula!r}.")
            element = match.group(1)
            count = 1.0 if match.group(2) is None else float(match.group(2))
            _validate_positive("stoichiometry", count)
            result.append((element, count))
            position = match.end()
        if position != len(formula) or not result:
            raise ValueError(f"Unsupported chemical formula syntax: {self.formula!r}.")
        return result

    def optical_constants(self, energy_kev: float) -> tuple[float, float]:
        """Return X-ray refractive-index decrement ``delta`` and absorption ``beta``.

        ``energy_kev`` is the photon energy in keV. Tabulated anomalous
        scattering factors are linearly interpolated in energy; values outside
        the tabulated range are rejected rather than extrapolated silently.
        """

        _validate_positive("energy_kev", energy_kev)
        if self._delta is not None and self._beta is not None:
            return self._delta, self._beta

        assert self.density is not None

        composition = self.composition()
        f1_values: list[float] = []
        f2_values: list[float] = []
        df2_values: list[float] = []
        atomic_masses: list[float] = []

        energy_ev = energy_kev * 1e3
        for element, _count in composition:
            energies, f1_table, f2_table = self._database.f1f2(element)
            if energy_ev < energies[0] or energy_ev > energies[-1]:
                raise ValueError(
                    f"Energy {energy_ev:g} eV is outside the f1/f2 table range for {element}."
                )
            f1 = float(np.interp(energy_ev, energies, f1_table))
            f2 = float(np.interp(energy_ev, energies, f2_table))

            coeffs = self._database.compton_coefficients(element)
            log_e = np.log(energy_kev)
            df2 = 1.4312e-5 * energy_kev * np.exp(
                coeffs[0]
                + coeffs[1] * log_e
                + coeffs[2] * log_e**2
                + coeffs[3] * log_e**3
            )

            f1_values.append(f1)
            f2_values.append(f2)
            df2_values.append(df2)
            atomic_masses.append(self._database.atomic_mass(element))

        counts = np.asarray([count for _, count in composition], dtype=float)
        molar_mass = float(np.sum(np.asarray(atomic_masses) * counts))
        scattering_sum = np.sum(
            (np.asarray(f1_values) - 1j * (np.asarray(f2_values) + np.asarray(df2_values))) * counts
        )
        chi0 = -8.3036e-4 * self.density / molar_mass / energy_kev**2 * scattering_sum
        return abs(float(np.real(chi0)) / 2.0), abs(float(np.imag(chi0)) / 2.0)


# ---------------------------------------------------------------------------
# Fourier/Fresnel grids
# ---------------------------------------------------------------------------


def _backend_module(backend: str):
    """Return the numerical module for a backend name."""

    backend = backend.lower()
    if backend == "numpy":
        return np
    if backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy backend requested, but CuPy is not installed.")
        return cp
    raise ValueError("backend must be 'numpy' or 'cupy'.")


def _centered_coordinates(n: int, step: float, xp) -> Any:
    """Create the same centered coordinate convention used by the original code."""

    left = -(n - 1) / 2
    right = n / 2
    return xp.arange(left, right, dtype=xp.float64) * step


def _centered_reciprocal_coordinates(n: int, step: float, xp) -> Any:
    """Create centered transverse-wavevector coordinates for an FFT grid."""

    dq = 2 * np.pi / (step * n)
    left = -(n - 1) / 2
    right = n / 2
    return xp.arange(left, right, dtype=xp.float64) * dq


def _fourier_phase(n: int, sign: int, xp) -> tuple[Any, complex]:
    """Create the phase vector and scalar used by the centered continuous FFT."""

    indices = xp.arange(n, dtype=xp.float64)
    phase = xp.exp(sign * 1j * np.pi * indices * (1 - 1 / n))
    scalar = np.exp(sign * 1j * np.pi * (1 - 1 / (2 * n) - n / 2))
    return phase, scalar


def _fft1d(array: Any, dx: float, phase_data: Any = None) -> Any:
    """Centered continuous FFT along the last axis, including field batches."""
    xp = _array_module(array)
    phase, scalar = phase_data or _fourier_phase(array.shape[-1], +1, xp)
    result = xp.fft.fft(array * phase, axis=-1)
    result *= phase
    result *= dx * scalar
    return result


def _ifft1d(array: Any, dx: float, phase_data: Any = None) -> Any:
    """Centered continuous inverse FFT along the last axis."""
    xp = _array_module(array)
    phase, scalar = phase_data or _fourier_phase(array.shape[-1], -1, xp)
    result = xp.fft.ifft(array * phase, axis=-1)
    result *= phase
    result *= scalar / dx
    return result


def _transform2d(array: Any, dx: float, dy: float, inverse: bool,
                 phase_data: Any = None) -> Any:
    """Transform the last two axes without allocating a full 2D phase map."""
    xp = _array_module(array)
    ny, nx = array.shape[-2:]
    if phase_data is None:
        px, sx = _fourier_phase(nx, -1 if inverse else +1, xp)
        py, sy = _fourier_phase(ny, -1 if inverse else +1, xp)
    else:
        px, sx, py, sy = phase_data
    work = array * px
    work *= py[:, None]
    transform = xp.fft.ifft2 if inverse else xp.fft.fft2
    result = transform(work, axes=(-2, -1))
    result *= px
    result *= py[:, None]
    result *= sx * sy / (dx * dy) if inverse else dx * dy * sx * sy
    return result


def _fft2d(array: Any, dx: float, dy: float) -> Any:
    return _transform2d(array, dx, dy, False)


def _ifft2d(array: Any, dx: float, dy: float) -> Any:
    return _transform2d(array, dx, dy, True)


class BaseGrid(_Configured, ABC):
    """Immutable numerical-grid parameters and shared CPU/GPU operations.

    Use dataclasses.replace(grid, dx=...) to construct a different grid. Arrays
    describing the coordinates must also be treated as read-only on the GPU.
    """

    dimension: int
    _configuration_fields = frozenset({
        "dx", "dy", "nx", "ny", "backend", "shape", "dimension", "x", "y",
        "qx", "qy", "x_axis", "y_axis", "qx_axis", "qy_axis", "dqx", "dqy",
    })

    @property
    def xp(self):
        return _backend_module(self.backend)

    def _validate_field_backend(self, field: Any) -> None:
        _array_module(field)
        if _backend_name(field) != self.backend:
            raise TypeError(f"Field backend is {_backend_name(field)!r}, but the grid uses {self.backend!r}.")
        if tuple(field.shape) != self.shape:
            raise ValueError(f"Expected field shape {self.shape}, got {field.shape}.")
        if not np.issubdtype(field.dtype, np.number):
            raise TypeError("Field arrays must have a numeric dtype.")

    def _transform_batch(self, fields: Any, inverse: bool = False) -> Any:
        key = -1 if inverse else +1
        if key not in self._phase_cache:
            px, sx = _fourier_phase(self.nx, key, self.xp)
            if self.dimension == 1:
                self._phase_cache[key] = (px, sx)
            else:
                py, sy = _fourier_phase(self.ny, key, self.xp)
                self._phase_cache[key] = (px, sx, py, sy)
        phase = self._phase_cache[key]
        if self.dimension == 1:
            transform = _ifft1d if inverse else _fft1d
            return transform(fields, self.dx, phase)
        return _transform2d(fields, self.dx, self.dy, inverse, phase)

    def fft(self, field: Any) -> Any:
        """Return the centered continuous Fourier transform of one field."""
        self._validate_field_backend(field)
        return self._transform_batch(field)

    def ifft(self, field: Any) -> Any:
        """Return the centered continuous inverse Fourier transform."""
        self._validate_field_backend(field)
        return self._transform_batch(field, inverse=True)

    def _propagate_spectrum(self, spectrum: Any, z_values: np.ndarray, k: float) -> Any:
        """Batched inverse FFTs, with separable Fresnel factors along x and y."""
        xp = self.xp
        z = xp.asarray(z_values)[:, None]
        qx = self.qx if self.dimension == 1 else self.qx_axis
        phase_x = xp.exp((-0.5j / k) * z * qx**2)
        work = spectrum[None, ...] * (phase_x if self.dimension == 1 else phase_x[:, None, :])
        if self.dimension == 2:
            phase_y = xp.exp((-0.5j / k) * z * self.qy_axis**2)
            work *= phase_y[:, :, None]
        return self._transform_batch(work, inverse=True)

    def propagate(self, field: Any, z: float, k: float) -> Any:
        """Fresnel-propagate one field; signed distances are supported."""
        self._validate_field_backend(field)
        _validate_finite("z", z)
        _validate_positive("k", k)
        if z == 0:
            return field.copy()
        return self._propagate_spectrum(self.fft(field), np.asarray([z]), k)[0]

    def _batch_size(self, n_planes: int, dtype: Any, batch_size: int | None) -> int:
        if batch_size is not None:
            _validate_count("batch_size", batch_size)
            return min(n_planes, batch_size)
        if self.backend == "numpy":
            return 1  # GPU batching is optional; keep CPU working memory small.
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        field_bytes = int(np.prod(self.shape)) * np.dtype(dtype).itemsize
        # Reserve space for FFT workspace, spectra, intensity and retained output.
        return max(1, min(n_planes, 16, int(free_bytes * 0.25) // (6 * field_bytes)))

    def iter_propagate(self, field: Any, z_values: Sequence[float], k: float, *,
                       batch_size: int | None = None) -> Iterator[tuple[slice, Any]]:
        """Yield (plane_slice, fields) batches with one shared forward FFT.

        Each batch has shape (batch, *grid.shape). None selects a conservative
        GPU batch from free memory, or one plane on CPU. The iterator retains
        no detector stack. Explicit batch sizes override the memory heuristic.
        """
        self._validate_field_backend(field)
        _validate_positive("k", k)
        z_array = _validated_distances(z_values)
        spectrum = self.fft(field)
        yield from self._iter_spectrum(field, spectrum, z_array, k, batch_size)

    def _iter_spectrum(self, field: Any, spectrum: Any, z_array: np.ndarray,
                       k: float, batch_size: int | None) -> Iterator[tuple[slice, Any]]:
        size = self._batch_size(len(z_array), spectrum.dtype, batch_size)
        for start in range(0, len(z_array), size):
            stop = min(start + size, len(z_array))
            positions = z_array[start:stop]
            fields = self._propagate_spectrum(spectrum, positions, k)
            zero = np.flatnonzero(positions == 0)
            if zero.size:
                fields[self.xp.asarray(zero)] = field
            yield slice(start, stop), fields

    def propagate_many(self, field: Any, z_values: Sequence[float], k: float, *,
                       batch_size: int | None = None) -> Any:
        """Return a detector stack; use iter_propagate for bounded output memory."""
        self._validate_field_backend(field)
        z_array = _validated_distances(z_values)
        # Allocate only after validation, and before the iterator sizes its batches.
        _validate_positive("k", k)
        if batch_size is not None:
            _validate_count("batch_size", batch_size)
        dtype = self.xp.result_type(field.dtype, self.xp.complex128)
        result = self.xp.empty((len(z_array), *self.shape), dtype=dtype)
        for planes, fields in self.iter_propagate(field, z_array, k, batch_size=batch_size):
            result[planes] = fields
        return result

    def convolve(self, field_a: Any, field_b: Any) -> Any:
        """Convolve two fields using the continuous Fourier convention."""
        return self.ifft(self.fft(field_a) * self.fft(field_b))


def _validated_distances(z_values: Sequence[float], *, nonnegative: bool = False) -> np.ndarray:
    z = np.array(_to_numpy(z_values), dtype=float, copy=True)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("z_values must be a non-empty, finite 1D sequence.")
    if nonnegative and np.any(z < 0):
        raise ValueError("z_values must contain non-negative distances.")
    return z


@dataclass
class Grid1D(BaseGrid):
    """Uniform 1D grid: dx in metres, nx samples, NumPy or CuPy backend."""

    dx: float
    nx: int
    backend: str = "cupy" if CUPY_AVAILABLE else "numpy"

    def __post_init__(self) -> None:
        _validate_positive("dx", self.dx)
        _validate_count("nx", self.nx, 2)
        self.backend = self.backend.lower()
        xp = self.xp
        self.x = _centered_coordinates(self.nx, self.dx, xp)
        self.qx = _centered_reciprocal_coordinates(self.nx, self.dx, xp)
        self.dqx = 2 * np.pi / (self.dx * self.nx)
        self.shape = (self.nx,)
        self.dimension = 1
        self._phase_cache: dict[int, Any] = {}
        self._freeze_configuration()


@dataclass
class Grid2D(BaseGrid):
    """Uniform 2D grid; fields have shape (ny, nx), distances are in metres."""

    dx: float
    dy: float
    nx: int
    ny: int
    backend: str = "cupy" if CUPY_AVAILABLE else "numpy"

    def __post_init__(self) -> None:
        _validate_positive("dx", self.dx)
        _validate_positive("dy", self.dy)
        _validate_count("nx", self.nx, 2)
        _validate_count("ny", self.ny, 2)
        self.backend = self.backend.lower()
        xp = self.xp
        self.x_axis = _centered_coordinates(self.nx, self.dx, xp)
        self.y_axis = _centered_coordinates(self.ny, self.dy, xp)
        self.qx_axis = _centered_reciprocal_coordinates(self.nx, self.dx, xp)
        self.qy_axis = _centered_reciprocal_coordinates(self.ny, self.dy, xp)
        self.shape = (self.ny, self.nx)
        # Broadcast views preserve the public 2D interface without four dense maps.
        self.x = xp.broadcast_to(self.x_axis[None, :], self.shape)
        self.y = xp.broadcast_to(self.y_axis[:, None], self.shape)
        self.qx = xp.broadcast_to(self.qx_axis[None, :], self.shape)
        self.qy = xp.broadcast_to(self.qy_axis[:, None], self.shape)
        self.dqx = 2 * np.pi / (self.dx * self.nx)
        self.dqy = 2 * np.pi / (self.dy * self.ny)
        self.dimension = 2
        self._phase_cache: dict[int, Any] = {}
        self._freeze_configuration()


# ---------------------------------------------------------------------------
# Optical-object model
# ---------------------------------------------------------------------------


class OpticalObject(_Configured, ABC):
    """Base class for every object that can produce or transform ``E``."""

    c = 299_792_458.0
    _configuration_fields = frozenset({
        "grid", "wavelength_m", "energy_kev", "k", "angular_frequency",
        "x0", "y0", "radius", "center_offset",
    })

    def __init__(
        self,
        grid: BaseGrid,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a grid-bound optical object and resolve its photon energy."""

        self.grid = grid
        if wavelength_m is None and energy_kev is None:
            self.wavelength_m = None
            self.energy_kev = None
            self.k = None
            self.angular_frequency = None
        else:
            self.wavelength_m, self.energy_kev = wavelength_energy(
                wavelength_m=wavelength_m, energy_kev=energy_kev
            )
            self.k = 2 * np.pi / self.wavelength_m
            self.angular_frequency = 2 * np.pi * self.c / self.wavelength_m
        self._freeze_configuration()

    def propagate(self, field: Any, z: float) -> Any:
        """Fresnel-propagate ``field`` on this object's grid by ``z`` metres."""

        _validate_finite("z", z)
        if z < 0:
            raise ValueError("Propagation distance z must be >= 0.")
        if self.k is None:
            raise ValueError(
                "Propagation requires wavelength_m or energy_kev at construction time."
            )
        return self.grid.propagate(field, z, self.k)

    @abstractmethod
    def E(self, z: float = 0.0) -> Any:
        """Return the complex electric field at distance ``z`` from the object."""

    def I(self, z: float = 0.0) -> Any:
        """Return intensity ``|E|^2`` at distance ``z`` from the object."""

        field = self.E(z=z)
        return abs(field) ** 2

    def intensity(self, z: float = 0.0) -> Any:
        """Alias for :meth:`I`, kept with a descriptive English name."""

        return self.I(z=z)

    def divergence(self, z: float = 0.0) -> float | tuple[float, float] | None:
        """Estimate the far-field angular FWHM of the object's field."""

        if self.k is None:
            raise ValueError("Divergence calculation requires wavelength_m or energy_kev.")
        return divergence(self.E(z=z), self.grid, self.k)


class OpticalElement(OpticalObject):
    """Base class for an optical element consuming an input field array."""

    def __init__(
        self,
        input_field: Any | None = None,
        grid: BaseGrid | None = None,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Store an optional incident field and validate it when provided."""

        if grid is None:
            raise TypeError("grid is required.")
        super().__init__(
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        if input_field is not None:
            self.grid._validate_field_backend(input_field)
        self.input_field = input_field

    @property
    def input_field(self) -> Any:
        return self._input_field

    @input_field.setter
    def input_field(self, field: Any) -> None:
        if field is not None:
            self.grid._validate_field_backend(field)
        self._input_field = field

    def _require_input_field(self) -> Any:
        """Return the incident field or raise a clear error in geometry-only mode."""
        if self.input_field is None:
            raise ValueError(
                f"{type(self).__name__} field calculation requires input_field; "
                "geometry-only operations remain available without it."
            )
        self.grid._validate_field_backend(self.input_field)
        return self.input_field

    def E(self, z: float = 0.0) -> Any:
        """Return the element's transformed field and optionally propagate it by ``z``."""

        _validate_finite("z", z)
        if z < 0:
            raise ValueError("Propagation distance z must be >= 0.")
        self._require_input_field()
        if self.k is None and z > 0:
            raise ValueError("Propagation requires wavelength_m or energy_kev.")
        field_after_element = self._field_after_element()
        if z == 0:
            return field_after_element
        return self.propagate(field_after_element, z)

    @abstractmethod
    def _field_after_element(self) -> Any:
        """Return the field immediately after the element, before external propagation."""

    def clear_cache(self) -> None:
        """Clear cached element-specific arrays such as a CRL thickness map."""

        return None


class PointSource(OpticalObject):
    """Create a paraxial spherical-wave field on a 1D or 2D grid.

    ``position`` is a scalar on :class:`Grid1D` and ``(x0, y0)`` on
    :class:`Grid2D`. The field is defined for positive propagation distances;
    a point source at ``z=0`` is singular and is therefore intentionally not
    represented as an array.
    """

    def __init__(
        self,
        grid: BaseGrid,
        position: float | tuple[float, float] = 0.0,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a point source with a transverse source position."""

        super().__init__(
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        if grid.dimension == 1:
            if isinstance(position, Iterable) and not isinstance(position, (str, bytes)):
                position = tuple(position)  # type: ignore[assignment]
                if len(position) != 1:
                    raise ValueError("A 1D point source needs one transverse coordinate.")
                position = float(position[0])
            self.x0 = float(position)  # type: ignore[arg-type]
            self.y0 = None
        else:
            if not isinstance(position, Iterable) or isinstance(position, (str, bytes)):
                raise ValueError("A 2D point source position must be (x0, y0).")
            position_tuple = tuple(position)
            if len(position_tuple) != 2:
                raise ValueError("A 2D point source position must contain x0 and y0.")
            self.x0, self.y0 = float(position_tuple[0]), float(position_tuple[1])
        _validate_finite("source x0", self.x0)
        if self.y0 is not None:
            _validate_finite("source y0", self.y0)

    def E(self, z: float = 0.0) -> Any:
        """Create the source field at propagation distance ``z`` metres."""

        _validate_positive("z", z)
        if z <= 0:
            raise ValueError("PointSource.E requires z > 0 because the point source is singular at z=0.")
        if self.k is None:
            raise ValueError("PointSource.E requires wavelength_m or energy_kev.")

        if self.grid.dimension == 1:
            return self.grid.xp.exp(
                1j * self.k * (self.grid.x - self.x0) ** 2 / (2 * z)
            )

        return self.grid.xp.exp(
            1j
            * self.k
            * ((self.grid.x - self.x0) ** 2 + (self.grid.y - self.y0) ** 2)
            / (2 * z)
        )


class Hole(OpticalElement):
    """Apply two symmetric circular/linear openings and then propagate the field.

    On a 2D grid, ``center_offset=(x0, y0)`` creates holes at ``(+x0,+y0)``
    and ``(-x0,-y0)``. On a 1D grid, ``center_offset=x0`` creates intervals
    centered at ``+x0`` and ``-x0``. The mask is binary even when the two holes
    overlap.
    """

    def __init__(
        self,
        input_field: Any | None = None,
        grid: BaseGrid | None = None,
        center_offset: float | tuple[float, float] | None = None,
        radius: float | None = None,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a symmetric two-hole aperture."""

        if grid is None:
            raise TypeError("grid is required.")
        if center_offset is None:
            raise TypeError("center_offset is required.")
        if radius is None:
            raise TypeError("radius is required.")
        super().__init__(
            input_field,
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        _validate_positive("radius", radius)
        self.radius = float(radius)
        if grid.dimension == 1:
            if isinstance(center_offset, Iterable) and not isinstance(center_offset, (str, bytes)):
                center_offset = tuple(center_offset)  # type: ignore[assignment]
                if len(center_offset) != 1:
                    raise ValueError("A 1D Hole needs one center coordinate.")
                center_offset = float(center_offset[0])
            self.center_offset = float(center_offset)  # type: ignore[arg-type]
        else:
            if not isinstance(center_offset, Iterable) or isinstance(center_offset, (str, bytes)):
                raise ValueError("A 2D Hole center_offset must be (x0, y0).")
            center = tuple(center_offset)
            if len(center) != 2:
                raise ValueError("A 2D Hole center_offset must contain x0 and y0.")
            self.center_offset = (float(center[0]), float(center[1]))
        for offset in np.atleast_1d(self.center_offset):
            _validate_finite("center_offset", offset)

    def transmission_mask(self) -> Any:
        """Return the binary aperture mask for the two symmetric holes."""

        xp = self.grid.xp
        if self.grid.dimension == 1:
            x0 = float(self.center_offset)
            return (
                ((self.grid.x - x0) ** 2 <= self.radius**2)
                | ((self.grid.x + x0) ** 2 <= self.radius**2)
            ).astype(self.input_field.dtype if self.input_field is not None else float)

        x0, y0 = self.center_offset
        first = (self.grid.x - x0) ** 2 + (self.grid.y - y0) ** 2 <= self.radius**2
        second = (self.grid.x + x0) ** 2 + (self.grid.y + y0) ** 2 <= self.radius**2
        return (first | second).astype(self.input_field.dtype if self.input_field is not None else float)

    def _field_after_element(self) -> Any:
        """Multiply the input field by the hole mask."""

        return self.input_field * self.transmission_mask()


class Blade(OpticalElement):
    """Pass only the positive half-axis (1D) or first quadrant (2D).

    This preserves the original blade convention: in 2D the field survives
    only for ``x >= 0`` and ``y >= 0`` simultaneously.
    """

    def transmission_mask(self) -> Any:
        """Return the binary blade mask for the object's grid."""

        if self.grid.dimension == 1:
            return (self.grid.x >= 0).astype(self.input_field.dtype if self.input_field is not None else float)
        return ((self.grid.x >= 0) & (self.grid.y >= 0)).astype(self.input_field.dtype if self.input_field is not None else float)

    def _field_after_element(self) -> Any:
        """Multiply the input field by the blade mask."""

        return self.input_field * self.transmission_mask()


# ---------------------------------------------------------------------------
# CRL thickness defects
# ---------------------------------------------------------------------------


class ThicknessDefect(ABC):
    """Interface for a model that modifies a CRL thickness map.

    A defect receives the ideal thickness and may return a completely replaced
    thickness map or any additive/multiplicative modification of it. The
    returned array must have the same grid shape and backend as ``thickness``.
    """

    @abstractmethod
    def apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any:
        """Return a modified CRL thickness map for one element."""


class FunctionThicknessDefect(ThicknessDefect):
    """Adapt a user-supplied Python function to the thickness-defect interface."""

    def __init__(self, function: Callable[[Any, BaseGrid, int], Any]) -> None:
        """Store ``function(thickness, grid, element_index)`` as the defect model."""

        self.function = function

    def apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any:
        """Apply the wrapped function and validate its returned shape/backend."""

        result = self.function(thickness, grid, element_index)
        grid._validate_field_backend(result)
        return result


class InterpolatedThicknessDefect(ThicknessDefect):
    """Adds a custom thickness defect defined on its own coordinate grid,
    interpolated onto the CRL calculation grid.
    """

    def __init__(self, defect_array: Any, coords: Any, fill_value: float = 0.0) -> None:
        """Initialize the interpolated thickness defect.

        Parameters
        ----------
        defect_array:
            NumPy array containing the defect values (heights/thickness modifications).
            Shape must be (nx,) for 1D or (ny, nx) for 2D.
        coords:
            For 1D: a 1D array of x-coordinates ``x_def``.
            For 2D: a tuple of 1D arrays ``(x_def, y_def)`` defining the axes.
        fill_value:
            Value used outside the defect coordinate bounds (default is 0.0,
            meaning pure ideal thickness).
        """
        values = _to_numpy(defect_array)
        if values.ndim == 1:
            data = SurfaceError([values, coords])
            self.coords = data.x_axis
        elif values.ndim == 2 and isinstance(coords, (tuple, list)) and len(coords) == 2:
            data = SurfaceError([values, coords[0], coords[1]])
            self.coords = (data.x_axis, data.y_axis)
        else:
            raise ValueError("Defect data must be 1D with x coordinates, or 2D with (x, y) axes.")
        self.defect_array = data.error
        _validate_finite("fill_value", fill_value)
        self.fill_value = float(fill_value)

    def apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any:
        """Interpolate and add the sampled defect on the calculation backend."""
        grid._validate_field_backend(thickness)
        if self.defect_array.ndim != grid.dimension:
            raise ValueError("Defect dimensionality must match the calculation grid.")
        xp = grid.xp
        coords = (self.coords,) if grid.dimension == 1 else self.coords
        axes = (grid.x,) if grid.dimension == 1 else (grid.x_axis, grid.y_axis)
        if any(_to_python_float(axis[0]) < points[0] or
               _to_python_float(axis[-1]) > points[-1]
               for axis, points in zip(axes, coords)):
            warnings.warn(
                "The defect coordinates do not fully cover the calculation grid. "
                "Regions outside the defect bounds will use fill_value.",
                RuntimeWarning, stacklevel=2,
            )
        if grid.dimension == 1:
            values = xp.interp(grid.x, xp.asarray(self.coords), xp.asarray(self.defect_array),
                               left=self.fill_value, right=self.fill_value)
        else:
            values = _interpolate_map(self.defect_array, *self.coords, grid, self.fill_value)
        return thickness + values.astype(thickness.dtype, copy=False)


def _interpolate_map(values: Any, x_axis: Any, y_axis: Any,
                     grid: BaseGrid, fill_value: float) -> Any:
    """Interpolate a 2D map on its calculation backend when CuPy supports it."""
    xp = grid.xp
    interpolator_type = RegularGridInterpolator
    if grid.backend == "cupy":
        try:
            from cupyx.scipy.interpolate import RegularGridInterpolator as interpolator_type
        except ImportError:
            # Older CuPy releases can still use the CPU interpolation fallback.
            xp = np
    interpolator = interpolator_type(
        (xp.asarray(y_axis), xp.asarray(x_axis)), xp.asarray(values),
        method="linear", bounds_error=False, fill_value=fill_value,
    )
    if xp is np and grid.backend == "cupy":
        x_map, y_map = np.meshgrid(_to_numpy(grid.x_axis), _to_numpy(grid.y_axis))
    else:
        x_map, y_map = grid.x, grid.y
    points = xp.stack((y_map.ravel(), x_map.ravel()), axis=-1)
    return grid.xp.asarray(interpolator(points).reshape(grid.shape))


class SurfaceError(_Configured):
    """Surface-height error sampled on an independent regular 1D/2D grid.

    Input is ``[error, x]`` or ``[error, x, y]``. For 2D data, ``x`` and
    ``y`` may be 1D axes or 2D ``meshgrid`` coordinate arrays. Values outside
    the supplied coordinate range are zero. Data are copied into an immutable
    snapshot, so later edits of the supplied arrays cannot invalidate lens caches.
    """

    _configuration_fields = frozenset({"error", "x", "y", "x_axis", "y_axis"})

    def __init__(self, data: Sequence[Any]) -> None:
        if not isinstance(data, (list, tuple)) or len(data) not in (2, 3):
            raise ValueError("surface_error must be [error, x] or [error, x, y].")
        self.error = _to_numpy(data[0]).astype(float, copy=True)
        self.x = _to_numpy(data[1]).astype(float, copy=True)
        self.y = _to_numpy(data[2]).astype(float, copy=True) if len(data) == 3 else None
        if not all(np.all(np.isfinite(a)) for a in (self.error, self.x, self.y) if a is not None):
            raise ValueError("Surface values and coordinates must be finite.")
        if self.error.ndim not in (1, 2):
            raise ValueError("surface_error values must be a 1D or 2D array.")
        if self.error.ndim == 1:
            if self.y is not None or self.x.ndim != 1 or self.error.size != self.x.size:
                raise ValueError("1D surface_error must be [error_1d, x_1d].")
            if self.x.size < 2 or np.any(np.diff(self.x) <= 0):
                raise ValueError("surface_error x coordinates must be strictly increasing.")
            self.x_axis, self.y_axis = self.x, None
        else:
            if self.y is None:
                raise ValueError("2D surface_error must be [error_2d, x, y].")
            self.x_axis = self._axis_from_coordinates(self.x, "x")
            self.y_axis = self._axis_from_coordinates(self.y, "y")
            if self.error.shape != (self.y_axis.size, self.x_axis.size):
                raise ValueError("2D surface_error must have shape (len(y), len(x)).")
            if (self.x_axis.size < 2 or self.y_axis.size < 2 or
                np.any(np.diff(self.x_axis) <= 0) or np.any(np.diff(self.y_axis) <= 0)):
                raise ValueError("surface_error coordinate axes must be strictly increasing.")
        self._freeze_configuration()

    @staticmethod
    def _axis_from_coordinates(coords: np.ndarray, name: str) -> np.ndarray:
        if coords.ndim == 1:
            return coords
        if coords.ndim != 2:
            raise ValueError(f"surface_error {name} coordinates must be 1D or 2D meshgrid arrays.")
        axis = coords[0, :] if name == "x" else coords[:, 0]
        expected = np.broadcast_to(axis[None, :] if name == "x" else axis[:, None], coords.shape)
        if not np.allclose(coords, expected, rtol=1e-10, atol=1e-14):
            raise ValueError(f"surface_error {name} coordinates must form a regular meshgrid.")
        return np.asarray(axis, dtype=float)

    def on_grid(self, grid: BaseGrid) -> Any:
        """Linearly interpolate the surface error onto a calculation grid."""
        if grid.dimension == 1:
            if self.error.ndim != 1:
                raise ValueError("A 2D surface_error cannot be used with Grid1D.")
            return grid.xp.interp(grid.x, grid.xp.asarray(self.x_axis),
                                  grid.xp.asarray(self.error), left=0.0, right=0.0)
        if self.error.ndim != 2 or self.y_axis is None:
            raise ValueError("A 1D surface_error cannot be used with Grid2D.")
        return _interpolate_map(self.error, self.x_axis, self.y_axis, grid, 0.0)


class ArrayObject(OpticalElement):
    """Homogeneous-material optical object defined by a sampled thickness map.

    Supply ``[thickness_1d, x]`` for 1D or ``[thickness_2d, x, y]`` for 2D.
    For 2D, coordinates may be 1D axes or 2D regular ``meshgrid`` arrays.
    Zero thickness means no material; negative thickness is invalid. Outside
    the supplied map the object is absent. The map is linearly interpolated
    onto the calculation grid.
    """

    _configuration_fields = OpticalObject._configuration_fields | frozenset({
        "material", "_thickness_data", "delta", "beta",
    })

    def __init__(
        self,
        input_field: Any | None = None,
        grid: BaseGrid | None = None,
        thickness: Sequence[Any] | None = None,
        material: Material | None = None,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        if grid is None:
            raise TypeError("grid is required.")
        if thickness is None:
            raise TypeError("thickness is required as [values, x] or [values, x, y].")
        if material is None:
            raise TypeError("material is required.")
        super().__init__(input_field, grid, wavelength_m=wavelength_m, energy_kev=energy_kev)
        self.material = material
        self._thickness_data = SurfaceError(thickness)
        if self._thickness_data.error.ndim != grid.dimension:
            raise ValueError("Thickness-map dimensionality must match the calculation grid.")
        if np.any(self._thickness_data.error < 0):
            raise ValueError("ArrayObject thickness cannot be negative.")
        self._warn_if_clipped()
        if self.energy_kev is None:
            if material._delta is not None and material._beta is not None:
                self.delta, self.beta = material.optical_constants(1.0)
            else:
                self.delta = self.beta = None
        else:
            self.delta, self.beta = material.optical_constants(self.energy_kev)
        if self.delta is not None:
            _validate_finite("material delta", self.delta)
        if self.beta is not None and self.beta < 0:
            raise ValueError("material beta must be >= 0.")
        self.clear_cache()

    def _warn_if_clipped(self) -> None:
        """Warn only when the supplied object extends beyond the calculation grid."""
        d = self._thickness_data
        x_axis = self.grid.x if self.grid.dimension == 1 else self.grid.x_axis
        axes = [("x", d.x_axis, _to_numpy(x_axis))]
        if self.grid.dimension == 2 and d.y_axis is not None:
            axes.append(("y", d.y_axis, _to_numpy(self.grid.y_axis)))
        for name, coords, calc_coords in axes:
            lo, hi = float(np.min(calc_coords)), float(np.max(calc_coords))
            tol = 1e-12 * max(1.0, abs(lo), abs(hi))
            if coords[0] < lo - tol or coords[-1] > hi + tol:
                warnings.warn(
                    f"ArrayObject extends beyond the calculation grid along {name}; "
                    "the outside part is clipped.", UserWarning, stacklevel=3,
                )

    def thickness_map(self) -> Any:
        """Return the object's thickness interpolated onto the calculation grid."""
        if self._thickness_cache is None:
            self._thickness_cache = self._thickness_data.on_grid(self.grid)
        return self._thickness_cache

    def transmission(self) -> Any:
        """Return the complex transmission function of the object."""
        if self.k is None:
            raise ValueError("Transmission calculation requires wavelength_m or energy_kev.")
        if self.delta is None or self.beta is None:
            raise ValueError("Optical constants require energy_kev or explicit material delta/beta.")
        if self._transmission_cache is None:
            thickness = self.thickness_map()
            self._transmission_cache = self.grid.xp.exp(
                -1j * self.k * (self.delta - 1j * self.beta) * thickness
            )
        return self._transmission_cache

    def clear_cache(self) -> None:
        self._thickness_cache = None
        self._transmission_cache = None

    def _field_after_element(self) -> Any:
        """Apply material transmission; zero thickness and out-of-map regions are vacuum."""
        return self.input_field * self.transmission()


# ---------------------------------------------------------------------------
# CRL element
# ---------------------------------------------------------------------------


class CRL(OpticalElement):
    """Compound refractive element for 1D and 2D paraxial wave-optics schemes.

    A CRL is built from two opposing parabolic surfaces and a central bridge.
    The complete material thickness of one element is

    ``T(x) = 2 * sag(x) + d``

    inside the optical aperture, where ``d`` is the distance between the two
    parabolic surfaces. A sequence of ``n_elements`` such elements is simulated
    numerically with Fresnel propagation between adjacent element centres.

    The same class is used for both genuinely one-dimensional schemes and for
    one-axis-focusing CRLs embedded in a two-dimensional scheme. In 1D the
    single transverse coordinate is the local focusing coordinate. In 2D, if
    ``transverse_length`` is omitted, the CRL is rotationally symmetric. If it
    is supplied, the CRL is cylindrically focusing: the local ``x`` coordinate
    is the focusing direction and the local ``y`` coordinate defines the finite
    transverse width. The orientation of that local frame is controlled by
    ``rotation`` (in-plane angle in radians).

    Parameters
    ----------
    input_field:
        Optional complex field incident on the first CRL element. Omit it to
        construct geometry and evaluate thickness maps without calculating a field.
    grid:
        :class:`Grid1D` or :class:`Grid2D` used by the field.
    radius:
        Radius of curvature of each parabolic surface, metres.
    aperture:
        Full optical aperture (diameter), metres.
    bridge_thickness:
        Distance ``d`` between the two opposing parabolic surfaces, metres.
    n_elements:
        Number of CRL elements in the stack.
    material:
        :class:`Material` describing the refractive medium.
    transverse_length:
        Optional transverse physical length of a cylindrically focusing CRL in
        2D. If omitted on a 2D grid the CRL is rotationally symmetric. Ignored
        on a 1D grid.
    rotation:
        In-plane rotation angle in radians for a cylindrical 2D CRL.
    surface_error:
        Optional ``[error, x]`` 1D profile or ``[error, x, y]`` 2D map.
        The error is added to one parabolic surface, so its contribution to
        total thickness is doubled. Linear interpolation maps it to the CRL grid.
    reuse_element_geometry:
        Reuse one calculated thickness/transmission map for every element.
        The current model uses identical geometry and surface error for all elements.
    wavelength_m, energy_kev:
        Optional photon wavelength or energy. Required for propagation and
        transmission calculations; not required for geometry-only use.
    """

    _configuration_fields = OpticalObject._configuration_fields | frozenset({
        "aperture", "bridge_thickness", "n_elements", "material", "transverse_length",
        "rotation", "reuse_element_geometry", "_surface_error", "delta", "beta",
    })

    def __init__(
        self,
        input_field: Any | None = None,
        grid: BaseGrid | None = None,
        *,
        radius: float,
        aperture: float,
        bridge_thickness: float,
        n_elements: int,
        material: Material,
        transverse_length: float | None = None,
        rotation: float = 0.0,
        surface_error: Sequence[Any] | SurfaceError | None = None,
        reuse_element_geometry: bool = True,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a CRL, its optional optical field and geometry."""

        if grid is None:
            raise TypeError("grid is required.")
        super().__init__(
            input_field,
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        _validate_positive("radius", radius)
        _validate_positive("aperture", aperture)
        _validate_positive("bridge_thickness", bridge_thickness)
        _validate_count("n_elements", n_elements)
        _validate_finite("rotation", rotation)

        self.radius = float(radius)
        self.aperture = float(aperture)
        self.bridge_thickness = float(bridge_thickness)
        self.n_elements = int(n_elements)
        self.material = material
        self.transverse_length = None if transverse_length is None else float(transverse_length)
        self.rotation = float(rotation)
        self.reuse_element_geometry = bool(reuse_element_geometry)
        self._surface_error = (
            surface_error if isinstance(surface_error, SurfaceError) or surface_error is None
            else SurfaceError(surface_error)
        )
        if self._surface_error is not None and self._surface_error.error.ndim != grid.dimension:
            raise ValueError("Surface-error dimensionality must match the calculation grid.")

        if self.transverse_length is not None:
            _validate_positive("transverse_length", self.transverse_length)
        if self.grid.dimension == 1 and self.transverse_length is not None:
            raise ValueError("transverse_length is only meaningful for a 2D one-axis CRL.")

        if self._surface_error is not None:
            self._warn_surface_error_extent()

        if self.energy_kev is None:
            if material._delta is not None and material._beta is not None:
                self.delta, self.beta = material.optical_constants(1.0)
            else:
                self.delta = self.beta = None
        else:
            self.delta, self.beta = material.optical_constants(self.energy_kev)
        if self.delta is not None:
            _validate_positive("material delta", self.delta)
        if self.beta is not None and self.beta < 0:
            raise ValueError("material beta must be >= 0.")

        self._thickness_cache: dict[int, Any] = {}
        self._transmission_cache: dict[int, Any] = {}

        if self.delta is not None and self.n_elements > self.max_element_count():
            warnings.warn(
                f"CRL contains {self.n_elements} elements, while the maximum valid "
                f"paraxial external-focus count is {self.max_element_count()}. "
                "The formal focus lies inside the CRL, so external focal-plane "
                "analysis is not physically meaningful.",
                RuntimeWarning,
                stacklevel=2,
            )

    def with_parameters(self, **changes: Any) -> "CRL":
        """Construct a validated CRL with fresh caches and selected replacements.

        The incident field is shared unless input_field is replaced. Surface maps
        are immutable snapshots. The original lens remains valid if validation of
        the replacement fails. Photon energy or wavelength can be replaced alone.
        """
        parameters = dict(
            input_field=self.input_field, grid=self.grid, radius=self.radius,
            aperture=self.aperture, bridge_thickness=self.bridge_thickness,
            n_elements=self.n_elements, material=self.material,
            transverse_length=self.transverse_length, rotation=self.rotation,
            surface_error=self._surface_error,
            reuse_element_geometry=self.reuse_element_geometry,
            energy_kev=self.energy_kev,
        )
        if "wavelength_m" in changes and "energy_kev" not in changes:
            parameters.pop("energy_kev")
        parameters.update(changes)
        return type(self)(**parameters)

    @property
    def element_length(self) -> float:
        """Axial extent derived from the immutable geometry."""
        return self.bridge_thickness + self.aperture**2 / (4 * self.radius)

    @property
    def element_period(self) -> float:
        """Axial period occupied by one CRL element, in metres."""
        return self.element_length

    @property
    def length(self) -> float:
        """Total axial length of the CRL stack, in metres."""
        return self.element_period * self.n_elements

    def _require_optical_constants(self) -> tuple[float, float, float]:
        """Return (k, delta, beta), or explain which optical input is missing."""
        if self.k is None:
            raise ValueError(
                "This calculation requires wavelength_m or energy_kev; "
                "geometry-only CRLs do not define a photon wavelength."
            )
        if self.delta is None or self.beta is None:
            raise ValueError(
                "Optical constants require energy_kev, unless Material was created "
                "with explicit delta and beta."
            )
        return self.k, self.delta, self.beta

    @property
    def characteristic_length(self) -> float:
        """Return the characteristic CRL length ``sqrt(p R / (2 delta))`` in metres."""

        if self.delta is None:
            raise ValueError("characteristic_length requires energy_kev or explicit material delta/beta.")
        return float(np.sqrt(self.element_period * self.radius / (2 * self.delta)))

    @property
    def characteristic_element_count(self) -> float:
        """Return the characteristic element count ``characteristic_length / element_period``."""

        return self.characteristic_length / self.element_period

    def critical_element_count(self) -> float:
        """Return the real-valued critical element count ``pi/2 * Lc / p``.

        At this value the formal paraxial focus reaches the exit reference
        plane. Beyond it the formal focus is inside the CRL and external focal
        plane analysis is not physically meaningful.
        """

        return float(np.pi / 2 * self.characteristic_element_count)

    def max_element_count(self) -> int:
        """Return the largest valid integer element count with an external focus."""

        critical = self.critical_element_count()
        return max(0, int(np.ceil(critical) - 1))

    def _local_coordinates(self) -> tuple[Any, Any | None]:
        """Return coordinates in the CRL's local transverse frame after in-plane rotation."""

        if self.grid.dimension == 1:
            return self.grid.x, None

        theta = self.rotation
        c, s = np.cos(theta), np.sin(theta)
        x_local = c * self.grid.x + s * self.grid.y
        y_local = -s * self.grid.x + c * self.grid.y
        return x_local, y_local

    def _warn_surface_error_extent(self) -> None:
        """Warn when the supplied defect coordinate extent differs from the CRL aperture."""
        assert self._surface_error is not None
        error = self._surface_error
        half_x = self.aperture / 2
        half_y = (self.transverse_length / 2
                  if self.transverse_length is not None else half_x)
        if self.grid.dimension == 2 and self.transverse_length is not None:
            c, s = abs(np.cos(self.rotation)), abs(np.sin(self.rotation))
            half_x, half_y = c * half_x + s * half_y, s * half_x + c * half_y
        self._warn_extent_axis("x", error.x_axis, -half_x, half_x)
        if error.error.ndim == 2 and error.y is not None:
            self._warn_extent_axis("y", error.y_axis, -half_y, half_y)

    @staticmethod
    def _warn_extent_axis(axis: str, coords: np.ndarray, lower: float, upper: float) -> None:
        tol = 1e-12 * max(1.0, abs(lower), abs(upper))
        supplied_lower, supplied_upper = float(coords[0]), float(coords[-1])
        if supplied_lower < lower - tol or supplied_upper > upper + tol:
            warnings.warn(
                f"surface_error is wider than the CRL geometry along {axis}; "
                "values outside the aperture will be ignored.",
                UserWarning, stacklevel=3,
            )
        elif supplied_lower > lower + tol or supplied_upper < upper - tol:
            warnings.warn(
                f"surface_error is narrower than the CRL geometry along {axis}; "
                "the error is zero outside its coordinate range.",
                UserWarning, stacklevel=3,
            )

    def _ideal_thickness(self) -> Any:
        """Build one defect-free CRL thickness map."""

        xp = self.grid.xp
        x_local, y_local = self._local_coordinates()
        h_max = self.aperture**2 / (8 * self.radius)

        if self.grid.dimension == 1 or self.transverse_length is not None:
            sag = xp.minimum(x_local**2 / (2 * self.radius), h_max)
            if self.grid.dimension == 2:
                transverse_mask = abs(y_local) < self.transverse_length / 2
                sag = xp.where(transverse_mask, sag, h_max)
        else:
            radial_sag = (x_local**2 + y_local**2) / (2 * self.radius)
            sag = xp.minimum(radial_sag, h_max)

        if self._surface_error is not None:
            error_map = self._surface_error.on_grid(self.grid).astype(sag.dtype, copy=False)
            # Sampled errors use laboratory coordinates; the aperture mask follows
            # the rotated lens geometry.
            if self.grid.dimension == 1:
                aperture_mask = abs(x_local) <= self.aperture / 2
            elif self.transverse_length is not None:
                aperture_mask = ((abs(x_local) <= self.aperture / 2) &
                                 (abs(y_local) <= self.transverse_length / 2))
            else:
                aperture_mask = (x_local**2 + y_local**2) <= (self.aperture / 2)**2
            error_map = xp.where(aperture_mask, error_map, 0.0)
            sag = sag + error_map
        return 2 * sag + self.bridge_thickness

    def element_thickness(self, element_index: int = 0) -> Any:
        """Return the complete material thickness map of one CRL element.

        The returned thickness includes both opposing parabolic surfaces and
        the central bridge ``d``. A configured defect is applied after the ideal
        geometry is constructed.
        """

        _validate_count("element_index", element_index, 0)
        if not 0 <= element_index < self.n_elements:
            raise IndexError("element_index is outside the CRL range.")
        if self.reuse_element_geometry and self._thickness_cache:
            return next(iter(self._thickness_cache.values()))
        if element_index in self._thickness_cache:
            return self._thickness_cache[element_index]

        thickness = self._ideal_thickness()

        self.grid._validate_field_backend(thickness)
        if not np.issubdtype(thickness.dtype, np.number):
            raise TypeError("CRL thickness defect returned a non-numeric array.")
        if bool(self.grid.xp.any(~self.grid.xp.isfinite(thickness) | (thickness <= 0)).item()):
            min_thickness = _to_python_float(self.grid.xp.min(thickness))
            raise ValueError(
                "surface_error removes too much material: the resulting CRL "
                f"thickness must stay positive (minimum thickness is {min_thickness:g} m)."
            )

        if self.reuse_element_geometry:
            self._thickness_cache[0] = thickness
        self._thickness_cache[element_index] = thickness
        return thickness

    def transmission(self, element_index: int = 0) -> Any:
        """Return the complex transmission function of one CRL element."""

        _validate_count("element_index", element_index, 0)
        if element_index >= self.n_elements:
            raise IndexError("element_index is outside the CRL range.")
        if self.reuse_element_geometry and self._transmission_cache:
            return next(iter(self._transmission_cache.values()))
        if element_index in self._transmission_cache:
            return self._transmission_cache[element_index]

        thickness = self.element_thickness(element_index)
        k, delta, beta = self._require_optical_constants()
        transmission = self.grid.xp.exp(
            -1j * k * (delta - 1j * beta) * thickness
        )
        if self.reuse_element_geometry:
            self._transmission_cache[0] = transmission
        self._transmission_cache[element_index] = transmission
        return transmission

    def clear_cache(self) -> None:
        """Clear cached thickness and transmission maps so they can be recomputed."""

        self._thickness_cache.clear()
        self._transmission_cache.clear()

    def _field_after_element(self) -> Any:
        """Propagate through the complete CRL stack and return its exit-plane field."""

        field = self.propagate(self._require_input_field(), self.element_period / 2)
        for element_index in range(self.n_elements):
            field = field * self.transmission(element_index)
            if element_index < self.n_elements - 1:
                field = self.propagate(field, self.element_period)
        return self.propagate(field, self.element_period / 2)

    def theoretical_focus(self, *, n_elements: int | None = None) -> float:
        """Return the formal paraxial focal distance from the CRL exit reference plane."""

        count = self.n_elements if n_elements is None else n_elements
        _validate_count("n_elements", count)
        if count < 1:
            raise ValueError("n_elements must be >= 1.")
        critical = self.critical_element_count()
        if count >= critical:
            warnings.warn(
                "The requested element count is at or beyond the critical count; "
                "the formal focus is not an external physical focus.",
                RuntimeWarning,
                stacklevel=2,
            )
        u = count / self.characteristic_element_count
        return float(self.characteristic_length / np.tan(u))

    def effective_focus(
        self,
        source_distance: float,
        *,
        n_elements: int | None = None,
    ) -> float:
        """Return the finite-object image distance from the CRL exit reference plane."""

        _validate_positive("source_distance", source_distance)
        count = self.n_elements if n_elements is None else n_elements
        _validate_count("n_elements", count)
        if count < 1:
            raise ValueError("n_elements must be >= 1.")
        if count > self.max_element_count():
            raise ValueError(
                "The requested element count exceeds max_element_count(); "
                "the formal focus lies inside the CRL."
            )

        lc = self.characteristic_length
        u = count / self.characteristic_element_count
        focal_length = lc / np.sin(u)
        back_principal_distance = focal_length * (1 - np.cos(u))
        object_distance = source_distance + back_principal_distance
        denominator = object_distance - focal_length
        if abs(denominator) < np.finfo(float).eps:
            return float(np.inf)
        image_distance = focal_length * object_distance / denominator
        return float(image_distance - back_principal_distance)

    def magnification(
        self,
        source_distance: float,
        image_distance: float,
        *,
        n_elements: int | None = None,
    ) -> float:
        """Return the signed analytical CRL transverse magnification."""

        _validate_positive("source_distance", source_distance)
        _validate_positive("image_distance", image_distance)
        count = self.n_elements if n_elements is None else n_elements
        _validate_count("n_elements", count)
        if count < 1:
            raise ValueError("n_elements must be >= 1.")
        lc = self.characteristic_length
        stack_length = count * self.element_period
        u = stack_length / lc
        sine = np.sin(u)
        cosine = np.cos(u)
        numerator = image_distance * (stack_length * cosine + lc * sine) + stack_length * lc * sine
        denominator = source_distance * (stack_length + lc * sine * cosine)
        return float(numerator / denominator)

    def image_scale(self, source_distance: float, *, n_elements: int | None = None) -> float:
        """Return the analytical image/source distance ratio for a finite object."""

        _validate_positive("source_distance", source_distance)
        count = self.n_elements if n_elements is None else n_elements
        _validate_count("n_elements", count)
        if count > self.max_element_count():
            raise ValueError("The requested element count exceeds max_element_count().")
        lc = self.characteristic_length
        u = count / self.characteristic_element_count
        focal_length = lc / np.sin(u)
        back_principal_distance = focal_length * (1 - np.cos(u))
        object_distance = source_distance + back_principal_distance
        image_distance = focal_length * object_distance / (object_distance - focal_length)
        return float(image_distance / object_distance)

    def is_focus_outside_crl(self, *, n_elements: int | None = None) -> bool:
        """Return whether the specified integer element count gives an external focus."""

        count = self.n_elements if n_elements is None else n_elements
        _validate_count("n_elements", count)
        return count <= self.max_element_count()

    def focus_scan(
        self,
        z_values: Sequence[float],
        *,
        roi: Any = None,
        return_fields: bool = False,
        calculate_divergence: bool = False,
        batch_size: int | None = None,
    ) -> "FocusScanResult":
        """Scan detector planes with one CRL evaluation and one forward FFT.

        Independent planes use batched inverse FFTs on the calculation backend.
        batch_size=None selects a conservative GPU memory heuristic; CPU defaults
        to one plane. Set batch_size=1 for minimum working memory. Metrics stay
        on the backend until reduction to a few scalars per plane. With
        return_fields=True, only the final cropped stack is retained, without
        views holding full-grid fields alive. Output storage is still O(nz).
        """
        z_array = _validated_distances(z_values, nonnegative=True)
        if batch_size is not None:
            _validate_count("batch_size", batch_size)
        slices = _roi_slices(self.grid, roi)
        if not self.is_focus_outside_crl():
            warnings.warn(
                "CRL focus_scan skipped because the theoretical focus lies inside the CRL.",
                RuntimeWarning, stacklevel=2,
            )
            return FocusScanResult.invalid()

        E0 = self.E(z=0)
        spectrum = self.grid.fft(E0)
        # Free-space propagation changes Fourier phase, not angular intensity.
        div = _divergence_from_spectrum(spectrum, self.grid, self.k) if calculate_divergence else None
        x_roi, y_roi = _coordinates_for_slices(self.grid, slices)
        roi_shape = (x_roi.size,) if self.grid.dimension == 1 else (y_roi.size, x_roi.size)
        E_array = (self.grid.xp.empty((len(z_array), *roi_shape), dtype=spectrum.dtype)
                   if return_fields else None)
        metrics: list[BeamMetrics] = []
        for planes, fields in self.grid._iter_spectrum(E0, spectrum, z_array, self.k, batch_size):
            metrics.extend(_beam_metrics_batch(fields, self.grid, slices, div))
            if return_fields:
                E_array[planes] = fields[(slice(None), *slices)]
            del fields

        peaks = np.asarray([m.max_intensity for m in metrics])
        peak_index = None if np.all(np.isnan(peaks)) else int(np.nanargmax(peaks))
        return FocusScanResult(
            metrics=metrics, peak_index=peak_index,
            peak_z=None if peak_index is None else float(z_array[peak_index]),
            E=E_array, x_roi=x_roi, y_roi=y_roi,
        )


# ---------------------------------------------------------------------------
# Beam analysis
# ---------------------------------------------------------------------------


@dataclass
class BeamMetrics:
    """Summary of scalar beam properties measured on one detector plane."""

    max_intensity: float
    center: float | tuple[float, float] | None
    fwhm: float | tuple[float | None, float | None] | None
    divergence: float | tuple[float, float] | None = None


@dataclass
class FocusScanResult:
    """Result of a multi-distance focus scan.

    ``E`` has shape ``(nz, nx_roi)`` for 1D grids and
    ``(nz, ny_roi, nx_roi)`` for 2D grids. Its array backend matches the grid.
    ``x_roi`` and, for 2D grids, ``y_roi`` contain the transverse coordinates
    corresponding to the cropped field. Coordinates are returned even when
    ``return_fields=False`` (in which case ``E`` is ``None``).
    """

    metrics: list[BeamMetrics]
    peak_index: int | None
    peak_z: float | None = None
    E: Any = None
    x_roi: Any = None
    y_roi: Any = None
    valid: bool = True

    @classmethod
    def invalid(cls) -> "FocusScanResult":
        """Create an invalid result when a focus scan is physically inapplicable."""

        return cls(metrics=[], peak_index=None, peak_z=None, valid=False)


    @property
    def fwhm_at_peak(self) -> float | tuple[float, float] | None:
        """Return the FWHM measured at the best-focus sampled plane."""

        if self.peak_index is None or not self.metrics:
            return None
        return self.metrics[self.peak_index].fwhm

    @property
    def divergence_at_peak(self) -> float | tuple[float, float] | None:
        """Return the divergence measured at the best-focus sampled plane."""

        if self.peak_index is None or not self.metrics:
            return None
        return self.metrics[self.peak_index].divergence


def intensity(field: Any) -> Any:
    """Return the scalar intensity array ``|field|²`` without changing its backend."""

    _array_module(field)
    return abs(field) ** 2


def _widths_batch(profiles: Any, coordinates: Any) -> Any:
    """Vectorized, peak-crossing FWHM for (batch, samples) profiles on CPU/GPU."""
    xp = _array_module(profiles)
    batch, n = profiles.shape
    if n < 2:
        return xp.full(batch, xp.nan)
    rows = xp.arange(batch)
    indices = xp.arange(n)[None, :]
    peak = xp.argmax(xp.where(xp.isnan(profiles), -xp.inf, profiles), axis=1)
    maxima = profiles[rows, peak]
    half = maxima[:, None] / 2
    left = xp.max(xp.where((indices < peak[:, None]) & (profiles < half), indices, -1), axis=1)
    right = xp.min(xp.where((indices > peak[:, None]) & (profiles < half), indices, n), axis=1)
    li = xp.clip(left, 0, n - 2)
    ri = xp.clip(right, 1, n - 1)

    def crossing(i0, i1):
        v0, v1 = profiles[rows, i0], profiles[rows, i1]
        denominator = v1 - v0
        fraction = (maxima / 2 - v0) / xp.where(denominator == 0, 1, denominator)
        fraction = xp.where(denominator == 0, 0.5, fraction)
        return coordinates[i0] + fraction * (coordinates[i1] - coordinates[i0])

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        widths = crossing(ri - 1, ri) - crossing(li, li + 1)
    valid = (xp.all(xp.isfinite(profiles), axis=1) & (maxima > 0)
             & (left >= 0) & (right < n) & (widths >= 0))
    return xp.where(valid, widths, xp.nan)


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def full_width_half_max(intensity_array: Any, coordinate: Any) -> float | None:
    """Return interpolated FWHM around the largest peak, or None.

    The profile stays on its backend; only the final scalar is transferred from
    a GPU. Both half-maximum crossings must lie inside the sampled coordinates.
    """
    xp = cp if CUPY_AVAILABLE and isinstance(intensity_array, cp.ndarray) else np
    arr = xp.asarray(intensity_array, dtype=float)
    if arr.ndim != 1:
        arr = arr.squeeze()
    coord = xp.asarray(_to_numpy(coordinate) if xp is np else coordinate, dtype=float)
    if arr.ndim != 1 or coord.ndim != 1 or arr.size != coord.size:
        raise ValueError("full_width_half_max expects matching one-dimensional arrays.")
    if not bool(xp.all(xp.isfinite(coord)).item()) or bool(xp.any(xp.diff(coord) <= 0).item()):
        raise ValueError("FWHM coordinates must be finite and strictly increasing.")
    return _optional_float(_to_python_float(_widths_batch(arr[None, :], coord)[0]))


def _roi_slices(grid: BaseGrid, roi: Any) -> tuple[slice, ...]:
    """Resolve physical bounds once using only 1D coordinate axes."""
    if roi is None:
        return (slice(None),) * grid.dimension
    if not isinstance(roi, Sequence) or len(roi) != 2:
        raise ValueError("ROI must be (xmin, xmax) or ((xmin, xmax), (ymin, ymax)).")
    bounds = [roi] if grid.dimension == 1 else [roi[1], roi[0]]
    axes = [grid.x] if grid.dimension == 1 else [grid.y_axis, grid.x_axis]
    result = []
    for axis, limits in zip(axes, bounds):
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[0] > limits[1]:
            raise ValueError("ROI bounds must be finite, ordered pairs.")
        coords = _to_numpy(axis)
        start = int(np.searchsorted(coords, limits[0], side="left"))
        stop = int(np.searchsorted(coords, limits[1], side="right"))
        if start >= stop:
            raise ValueError("ROI does not overlap the grid.")
        result.append(slice(start, stop))
    return tuple(result)


def _coordinates_for_slices(grid: BaseGrid, slices: tuple[slice, ...]) -> tuple[Any, Any]:
    if grid.dimension == 1:
        return grid.x[slices[0]], None
    return grid.x_axis[slices[1]], grid.y_axis[slices[0]]


def _roi_coordinates(grid: BaseGrid, roi: Any) -> tuple[Any, Any]:
    return _coordinates_for_slices(grid, _roi_slices(grid, roi))


def _crop_field(field: Any, grid: BaseGrid, roi: Any) -> Any:
    return field[_roi_slices(grid, roi)]


def _beam_metrics_batch(fields: Any, grid: BaseGrid, slices: tuple[slice, ...],
                        divergence_value: Any = None) -> list[BeamMetrics]:
    """Reduce intensities and peak-crossing slices without host image copies."""
    xp = grid.xp
    work = fields[(slice(None), *slices)]
    values = abs(work)**2
    batch = values.shape[0]
    rows = xp.arange(batch)
    flat = xp.where(xp.isnan(values), -xp.inf, values).reshape(batch, -1)
    peaks = xp.argmax(flat, axis=1)
    maxima = flat[rows, peaks]
    finite = xp.any(xp.isfinite(values).reshape(batch, -1), axis=1)
    x_axis, y_axis = _coordinates_for_slices(grid, slices)
    if grid.dimension == 1:
        widths = _widths_batch(values, x_axis)
        table = _to_numpy(xp.stack((maxima, x_axis[peaks], widths, finite), axis=1))
        return [BeamMetrics(float(m), float(c), _optional_float(w), divergence_value)
                if ok else BeamMetrics(np.nan, None, None, None) for m, c, w, ok in table]
    j, i = peaks // values.shape[-1], peaks % values.shape[-1]
    width_x = _widths_batch(values[rows, j, :], x_axis)
    width_y = _widths_batch(values[rows, :, i], y_axis)
    table = _to_numpy(xp.stack((maxima, x_axis[i], y_axis[j], width_x, width_y, finite), axis=1))
    return [BeamMetrics(float(m), (float(cx), float(cy)),
                        (_optional_float(wx), _optional_float(wy)), divergence_value)
            if ok else BeamMetrics(np.nan, None, None, None)
            for m, cx, cy, wx, wy, ok in table]


def beam_metrics(field: Any, grid: BaseGrid, *, roi: Any = None,
                 calculate_divergence: bool = False, k: float | None = None) -> BeamMetrics:
    """Measure peak, peak position and slice FWHM; divergence uses the full field."""
    grid._validate_field_backend(field)
    slices = _roi_slices(grid, roi)
    if calculate_divergence and k is None:
        raise ValueError("k is required when calculate_divergence=True.")
    div = divergence(field, grid, k) if calculate_divergence else None
    return _beam_metrics_batch(field[None, ...], grid, slices, div)[0]


def divergence(field: Any, grid: BaseGrid, k: float) -> float | tuple[float, float] | None:
    """Far-field angular FWHM, reducing the Fourier intensity on its backend."""
    _validate_positive("k", k)
    spectrum = grid.fft(field)
    return _divergence_from_spectrum(spectrum, grid, k)


def _divergence_from_spectrum(spectrum: Any, grid: BaseGrid, k: float) -> Any:
    xp = grid.xp
    values = abs(spectrum)**2
    qx = grid.qx if grid.dimension == 1 else grid.qx_axis
    valid_x = abs(qx) <= k
    if int(xp.count_nonzero(valid_x).item()) < 3:
        return None
    theta_x = xp.arcsin(xp.clip(qx[valid_x] / k, -1, 1))
    if grid.dimension == 1:
        return _optional_float(_to_python_float(_widths_batch(values[None, valid_x], theta_x)[0]))
    valid_y = abs(grid.qy_axis) <= k
    if int(xp.count_nonzero(valid_y).item()) < 3:
        return None
    theta_y = xp.arcsin(xp.clip(grid.qy_axis[valid_y] / k, -1, 1))
    peak = xp.argmax(values)
    j, i = peak // grid.nx, peak % grid.nx
    wx = _widths_batch(values[j, valid_x][None, :], theta_x)[0]
    wy = _widths_batch(values[valid_y, i][None, :], theta_y)[0]
    widths = _to_numpy(xp.stack((wx, wy)))
    return _optional_float(widths[0]), _optional_float(widths[1])


def gaussian_filter_1d(field: Any, coordinate: Any, sigma: float) -> Any:
    """Gaussian-filter a 1D array using the library's continuous FFT convention."""

    _validate_positive("sigma", sigma)
    xp = _array_module(field)
    coord = xp.asarray(coordinate)
    dx = float(_to_python_float(coord[1] - coord[0]))
    kernel = xp.exp(-((coord - 0.0) / sigma) ** 2 / 2) / (np.sqrt(2 * np.pi) * sigma)
    return _ifft1d(_fft1d(field, dx) * _fft1d(kernel, dx), dx).real


def gaussian_filter_2d(field: Any, x: Any, y: Any, sigma_x: float, sigma_y: float) -> Any:
    """Gaussian-filter a 2D array using continuous 2D Fourier convolution."""

    _validate_positive("sigma_x", sigma_x)
    _validate_positive("sigma_y", sigma_y)
    xp = _array_module(field)
    x = xp.asarray(x)
    y = xp.asarray(y)
    if x.ndim == 1 and y.ndim == 1:
        x_grid, y_grid = xp.meshgrid(x, y)
    else:
        x_grid, y_grid = x, y
    dx = _to_python_float(x_grid[0, 1] - x_grid[0, 0])
    dy = _to_python_float(y_grid[1, 0] - y_grid[0, 0])
    xp = _array_module(field)
    kernel = xp.exp(
        -(x_grid / sigma_x) ** 2 / 2 - (y_grid / sigma_y) ** 2 / 2
    ) / (2 * np.pi * sigma_x * sigma_y)
    return _ifft2d(_fft2d(field, dx, dy) * _fft2d(kernel, dx, dy), dx, dy).real


# ---------------------------------------------------------------------------
# Ready-to-run CRL focus workflow (SI parameters; host-side exportable results)
# ---------------------------------------------------------------------------


def select_backend(preferred: str = "auto") -> str:
    """Select a working backend, probing a small GPU FFT rather than import alone.

    auto falls back to NumPy. An explicit cupy request raises a diagnostic error
    when CUDA cannot execute. This does not install dependencies or change drivers.
    """
    if preferred not in {"auto", "numpy", "cupy"}:
        raise ValueError("backend must be auto, numpy or cupy.")
    if preferred == "numpy":
        return "numpy"
    failure = "CuPy is not installed or its CUDA libraries could not be imported."
    if CUPY_AVAILABLE:
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA device is visible.")
            probe = cp.fft.fft(cp.asarray([0.0, 1.0]))
            probe.get()  # Synchronize: asynchronous CUDA errors must be caught here.
            return "cupy"
        except Exception as exc:
            failure = f"CUDA probe failed: {exc}"
    if preferred == "cupy":
        raise RuntimeError(failure + " Use backend='auto' or 'numpy' for CPU execution.")
    return "numpy"


@dataclass(frozen=True)
class CRLFocusConfig:
    """Parameters for an ideal, centered CRL with an entrance aperture.

    All lengths are metres, energy is keV, density is g/cm^3. A cylindrical lens
    uses the 1D model; an axisymmetric lens uses the 2D circular-pupil model.
    A finite point source has unit amplitude at the entrance plane. None selects
    a plane wave. The incident field is clipped to the nominal lens aperture.
    Accuracy controls sampling, not the physical model. Automatic grids are
    heuristic and capped for memory; diagnostics do not establish convergence.
    """

    energy_kev: float = 12.0
    material_formula: str = "Be"
    density_g_cm3: float = 1.848
    radius_m: float = 50e-6
    aperture_m: float = 100e-6
    bridge_thickness_m: float = 30e-6
    n_elements: int = 10
    source_distance_m: float | None = 15.0
    geometry: str = "cylindrical"
    accuracy: str = "preview"
    backend: str = "auto"
    scan_range_m: tuple[float, float] | None = None
    scan_points: int = 41
    refine_points: int = 21
    grid_points: int | None = None
    window_factor: float = 2.0
    batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("energy_kev", "density_g_cm3", "radius_m", "aperture_m", "bridge_thickness_m"):
            _validate_positive(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        _validate_count("n_elements", self.n_elements)
        _validate_count("scan_points", self.scan_points, 5)
        _validate_count("refine_points", self.refine_points, 0)
        if self.refine_points in (1, 2):
            raise ValueError("refine_points must be zero or >= 3.")
        if self.source_distance_m is not None:
            _validate_positive("source_distance_m", self.source_distance_m)
            object.__setattr__(self, "source_distance_m", float(self.source_distance_m))
        if self.geometry not in {"cylindrical", "axisymmetric"}:
            raise ValueError("geometry must be cylindrical or axisymmetric.")
        if self.accuracy not in {"preview", "refined"}:
            raise ValueError("accuracy must be preview or refined.")
        if self.backend not in {"auto", "numpy", "cupy"}:
            raise ValueError("backend must be auto, numpy or cupy.")
        _validate_positive("window_factor", self.window_factor)
        if self.window_factor <= 1:
            raise ValueError("window_factor must be > 1 to contain the entrance aperture.")
        if self.grid_points is not None:
            _validate_count("grid_points", self.grid_points, 16)
        if self.batch_size is not None:
            _validate_count("batch_size", self.batch_size)
        for name in ("n_elements", "scan_points", "refine_points", "grid_points", "batch_size"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "window_factor", float(self.window_factor))
        if self.scan_range_m is not None:
            bounds = _validated_distances(self.scan_range_m, nonnegative=True)
            if bounds.size != 2 or bounds[1] <= bounds[0]:
                raise ValueError("scan_range_m must contain increasing start and stop distances.")
            object.__setattr__(self, "scan_range_m", tuple(map(float, bounds)))
        # Parse the composition before allocating any fields.
        Material(self.material_formula, self.density_g_cm3).composition()


@dataclass
class CRLFocusResult:
    """A focus scan and best-plane image as NumPy arrays, independent of CUDA.

    Distances are measured from the CRL exit. The best plane maximizes sampled
    peak intensity; this need not minimize FWHM. Intensities use relative units.
    Transmission is exit/input power on the finite grid, after the entrance pupil.
    longitudinal_intensity contains central x profiles, not stored 2D planes.
    """

    configuration: CRLFocusConfig
    backend: str
    analytical_focus_m: float
    z_m: np.ndarray
    peak_intensity: np.ndarray
    fwhm_m: np.ndarray
    peak_index: int
    x_m: np.ndarray
    y_m: np.ndarray | None
    best_intensity: np.ndarray
    longitudinal_intensity: np.ndarray
    transmission: float
    delta: float
    beta: float
    grid_step_m: float
    diagnostics: tuple[str, ...]

    @property
    def peak_z_m(self) -> float:
        return float(self.z_m[self.peak_index])

    @property
    def fwhm_at_peak_m(self) -> float | tuple[float | None, float | None] | None:
        widths = np.atleast_1d(self.fwhm_m[self.peak_index])
        values = tuple(_optional_float(v) for v in widths)
        return values[0] if self.configuration.geometry == "cylindrical" else values

    def summary(self) -> dict[str, Any]:
        """JSON-compatible metadata; arrays are saved separately by the caller."""
        return {
            "schema_version": 1, "xposim_version": __version__,
            "configuration": asdict(self.configuration), "backend": self.backend,
            "analytical_focus_m": self.analytical_focus_m,
            "peak_z_m": self.peak_z_m, "fwhm_at_peak_m": self.fwhm_at_peak_m,
            "peak_intensity": float(self.peak_intensity[self.peak_index]),
            "transmission_after_entrance_aperture": self.transmission,
            "delta": self.delta, "beta": self.beta,
            "grid_points": int(self.x_m.size), "grid_step_m": self.grid_step_m,
            "window_m": float(self.x_m.size * self.grid_step_m),
            "sampled_planes": int(self.z_m.size), "diagnostics": list(self.diagnostics),
            "focus_criterion": "maximum sampled peak intensity",
            "distance_reference": "CRL exit plane", "intensity_units": "relative",
            "convergence_checked": False,
        }


def estimate_crl_focus(configuration: CRLFocusConfig) -> CRLFocusResult:
    """Run a centered ideal-CRL calculation, coarse scan and local refinement.

    Only one exit field and its spectrum are calculated. Detector propagation
    uses bounded batches. The result retains a single best-plane image and
    central line profiles, without a full stack of 2D images. An automatic scan
    covers 0.5 to 1.5 times the analytical real-image distance; a boundary peak
    is reported as unresolved. Supply scan_range_m to inspect another interval.
    """
    if not isinstance(configuration, CRLFocusConfig):
        raise TypeError("Pass a CRLFocusConfig instance.")
    cfg = configuration
    backend = select_backend(cfg.backend)
    material = Material(cfg.material_formula, cfg.density_g_cm3)
    parameters = dict(radius=cfg.radius_m, aperture=cfg.aperture_m,
                      bridge_thickness=cfg.bridge_thickness_m,
                      n_elements=cfg.n_elements, material=material, energy_kev=cfg.energy_kev)
    # A two-point grid is used only for analytical geometry, never propagated.
    reference = CRL(grid=Grid1D(cfg.aperture_m, 2, backend="numpy"), **parameters)
    if not reference.is_focus_outside_crl():
        raise ValueError("The element count puts the formal focus inside the CRL; reduce n_elements.")
    analytical = (reference.theoretical_focus() if cfg.source_distance_m is None else
                  reference.effective_focus(cfg.source_distance_m))
    if not np.isfinite(analytical) or analytical <= 0:
        raise ValueError("This source/lens configuration has no finite real image after the CRL.")

    messages: list[str] = []
    window = cfg.window_factor * cfg.aperture_m
    wavelength, _ = wavelength_energy(energy_kev=cfg.energy_kev)
    # Slit and circular-pupil diffraction estimates provide a sampling heuristic.
    factor = 0.886 if cfg.geometry == "cylindrical" else 1.028
    expected_width = factor * wavelength * analytical / cfg.aperture_m
    samples_per_width = 6 if cfg.accuracy == "preview" else 12
    step_target = expected_width / samples_per_width
    if cfg.source_distance_m is not None:
        step_target = min(step_target, wavelength * cfg.source_distance_m / (2 * cfg.aperture_m))
    required = max(512, 2**int(np.ceil(np.log2(window / step_target))))
    if cfg.grid_points is None:
        cap = (65536 if cfg.geometry == "cylindrical" else
               (2048 if backend == "cupy" or cfg.accuracy == "refined" else 1024))
        n = min(required, cap)
        if n < required:
            messages.append("grid_cap: Automatic grid hit its memory cap; refine sampling before trusting spot size.")
    else:
        n = cfg.grid_points
    step = window / n
    grid = (Grid1D(step, n, backend=backend) if cfg.geometry == "cylindrical" else
            Grid2D(step, step, n, n, backend=backend))
    xp = grid.xp
    if cfg.source_distance_m is None:
        incident = xp.ones(grid.shape, dtype=xp.complex128)
    else:
        position = 0.0 if grid.dimension == 1 else (0.0, 0.0)
        incident = PointSource(grid=grid, position=position, energy_kev=cfg.energy_kev).E(cfg.source_distance_m)
    pupil = (abs(grid.x) <= cfg.aperture_m / 2 if grid.dimension == 1 else
             grid.x**2 + grid.y**2 <= (cfg.aperture_m / 2)**2)
    incident *= pupil
    input_power = _to_python_float(xp.sum(abs(incident)**2))
    if input_power <= 0:
        raise ValueError("The entrance aperture is not resolved by the grid; refine sampling or reduce the window.")
    crl = CRL(input_field=incident, grid=grid, **parameters)
    exit_field = crl.E()
    spectrum = grid.fft(exit_field)
    output_power = _to_python_float(xp.sum(abs(exit_field)**2))
    transmission = output_power / input_power
    bounds = cfg.scan_range_m or (0.5 * analytical, 1.5 * analytical)
    z_values = np.linspace(*bounds, cfg.scan_points)
    measurements: dict[float, tuple[BeamMetrics, np.ndarray]] = {}
    slices = (slice(None),) * grid.dimension

    def measure(positions: np.ndarray) -> None:
        for planes, fields in grid._iter_spectrum(exit_field, spectrum, positions, crl.k, cfg.batch_size):
            metrics = _beam_metrics_batch(fields, grid, slices)
            profiles = fields if grid.dimension == 1 else fields[:, n // 2, :]
            lines = _to_numpy(abs(profiles)**2)
            for z, metric, line in zip(positions[planes], metrics, lines):
                measurements[float(z)] = (metric, line.copy())
            del fields

    measure(z_values)
    coarse_peaks = np.asarray([measurements[float(z)][0].max_intensity for z in z_values])
    if not np.all(np.isfinite(coarse_peaks)):
        raise ArithmeticError("The calculated intensity is non-finite; review numerical parameters.")
    best_coarse = int(np.argmax(coarse_peaks))
    if cfg.refine_points and 0 < best_coarse < len(z_values) - 1:
        fine = np.linspace(z_values[best_coarse - 1], z_values[best_coarse + 1], cfg.refine_points)
        fine = fine[~np.any(np.isclose(fine[:, None], z_values[None, :], rtol=1e-12, atol=0), axis=1)]
        if fine.size:
            measure(fine)
    z_all = np.asarray(sorted(measurements))
    metrics = [measurements[float(z)][0] for z in z_all]
    peaks = np.asarray([m.max_intensity for m in metrics])
    if not np.all(np.isfinite(peaks)):
        raise ArithmeticError("The refined intensity is non-finite; review numerical parameters.")
    best = int(np.argmax(peaks))
    widths = np.asarray([np.nan if m.fwhm is None else m.fwhm for m in metrics], dtype=float)
    best_field = grid._propagate_spectrum(spectrum, np.asarray([z_all[best]]), crl.k)[0]
    if z_all[best] == 0:
        best_field = exit_field
    image = _to_numpy(abs(best_field)**2)
    if best in (0, len(z_all) - 1):
        messages.append("scan_boundary: Peak is at the scan boundary; widen the scan range.")
    best_widths = np.atleast_1d(widths[best])
    if np.any(~np.isfinite(best_widths)):
        messages.append("fwhm_missing: Half-maximum crossings are outside the measured window.")
    elif np.min(best_widths) / step < 6:
        messages.append("sampling: Fewer than six samples across the focal FWHM; use a finer grid.")
    max_phase_step = crl.k * crl.delta * cfg.n_elements * cfg.aperture_m / cfg.radius_m * step
    if cfg.source_distance_m is not None:
        max_phase_step += crl.k * cfg.aperture_m / (2 * cfg.source_distance_m) * step
    if max_phase_step > np.pi / 2:
        messages.append("phase_sampling: Lens/source phase changes too rapidly between samples; refine the grid.")
    rim = max(1, n // 20)
    if grid.dimension == 1:
        edge_power = np.sum(image[:rim]) + np.sum(image[-rim:])
    else:
        edge_power = (np.sum(image[:rim]) + np.sum(image[-rim:]) +
                      np.sum(image[rim:-rim, :rim]) + np.sum(image[rim:-rim, -rim:]))
    total = float(np.sum(image))
    if total > 0 and edge_power / total > 1e-3:
        messages.append("window_boundary: Noticeable power reaches the best-plane window edge; enlarge the window.")
    if transmission < 1e-12:
        messages.append("low_transmission: The transmitted power is very small; review material and lens thickness.")
    x_axis = grid.x if grid.dimension == 1 else grid.x_axis
    return CRLFocusResult(
        configuration=cfg, backend=backend, analytical_focus_m=float(analytical),
        z_m=z_all, peak_intensity=peaks, fwhm_m=widths, peak_index=best,
        x_m=_to_numpy(x_axis).copy(), y_m=None if grid.dimension == 1 else _to_numpy(grid.y_axis).copy(),
        best_intensity=image, longitudinal_intensity=np.stack([measurements[float(z)][1] for z in z_all]),
        transmission=float(transmission), delta=float(crl.delta), beta=float(crl.beta),
        grid_step_m=step, diagnostics=tuple(messages),
    )
