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

The file is intentionally self-contained. The material database is expected
in ``xray_material_data/`` next to this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:  # CuPy is optional; it is the preferred backend when available.
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local machine.
    cp = None
    CUPY_AVAILABLE = False

__version__ = "0.5.0"


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


def _validate_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` when a numeric parameter is not strictly positive."""

    if value <= 0:
        raise ValueError(f"{name} must be > 0; got {value!r}.")


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


class Material:
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

        if self._delta is not None and self._beta is not None:
            return self._delta, self._beta

        _validate_positive("energy_kev", energy_kev)
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


def _fft1d(array: Any, dx: float) -> Any:
    """Compute the centered continuous 1D Fourier transform."""

    xp = _array_module(array)
    n = array.shape[0]
    phase, scalar = _fourier_phase(n, +1, xp)
    return dx * scalar * phase * xp.fft.fft(array * phase)


def _ifft1d(array: Any, dx: float) -> Any:
    """Compute the centered continuous 1D inverse Fourier transform."""

    xp = _array_module(array)
    n = array.shape[0]
    phase, scalar = _fourier_phase(n, -1, xp)
    return scalar * phase * xp.fft.ifft(array * phase) / dx


def _fft2d(array: Any, dx: float, dy: float) -> Any:
    """Compute the centered continuous 2D Fourier transform."""

    xp = _array_module(array)
    ny, nx = array.shape
    px, sx = _fourier_phase(nx, +1, xp)
    py, sy = _fourier_phase(ny, +1, xp)
    phase = py[:, None] * px[None, :]
    return dx * dy * sx * sy * phase * xp.fft.fft2(array * phase)


def _ifft2d(array: Any, dx: float, dy: float) -> Any:
    """Compute the centered continuous 2D inverse Fourier transform."""

    xp = _array_module(array)
    ny, nx = array.shape
    px, sx = _fourier_phase(nx, -1, xp)
    py, sy = _fourier_phase(ny, -1, xp)
    phase = py[:, None] * px[None, :]
    return sx * sy * phase * xp.fft.ifft2(array * phase) / (dx * dy)


class BaseGrid(ABC):
    """Common numerical-grid interface shared by 1D and 2D calculations."""

    dimension: int

    def _validate_field_backend(self, field: Any) -> None:
        """Verify that a field uses the backend and shape expected by the grid."""

        _array_module(field)
        if _backend_name(field) != self.backend:
            raise TypeError(
                f"Field backend is {_backend_name(field)!r}, but the grid uses {self.backend!r}."
            )
        if tuple(field.shape) != self.shape:
            raise ValueError(f"Expected field shape {self.shape}, got {field.shape}.")

    @abstractmethod
    def propagate(self, field: Any, z: float, k: float) -> Any:
        """Fresnel-propagate a field by distance ``z``."""

    @abstractmethod
    def fft(self, field: Any) -> Any:
        """Return the centered continuous Fourier transform of a field."""

    @abstractmethod
    def ifft(self, field: Any) -> Any:
        """Return the centered continuous inverse Fourier transform of a field."""

    @abstractmethod
    def convolve(self, field_a: Any, field_b: Any) -> Any:
        """Convolve two fields using the grid's Fourier convention."""


@dataclass
class Grid1D(BaseGrid):
    """Uniform one-dimensional transverse grid.

    Parameters
    ----------
    dx:
        Spatial sampling step in metres.
    nx:
        Number of samples.
    backend:
        ``"cupy"`` or ``"numpy"``. CuPy is preferred for production runs.
    """

    dx: float
    nx: int
    backend: str = "cupy" if CUPY_AVAILABLE else "numpy"

    def __post_init__(self) -> None:
        _validate_positive("dx", self.dx)
        if self.nx < 2:
            raise ValueError("nx must be >= 2.")
        self.backend = self.backend.lower()
        xp = _backend_module(self.backend)
        self.x = _centered_coordinates(self.nx, self.dx, xp)
        self.qx = _centered_reciprocal_coordinates(self.nx, self.dx, xp)
        self.dqx = 2 * np.pi / (self.dx * self.nx)
        self.shape = (self.nx,)
        self.dimension = 1

    @property
    def xp(self):
        """Return the numerical array module used by the grid."""

        return _backend_module(self.backend)

    def fft(self, field: Any) -> Any:
        """Compute the centered continuous Fourier transform of a 1D field."""

        self._validate_field_backend(field)
        return _fft1d(field, self.dx)

    def ifft(self, field: Any) -> Any:
        """Compute the centered continuous inverse Fourier transform of a 1D spectrum."""

        self._validate_field_backend(field)
        return _ifft1d(field, self.dx)

    def propagate(self, field: Any, z: float, k: float) -> Any:
        """Fresnel-propagate a 1D field by ``z`` metres."""

        self._validate_field_backend(field)
        if z == 0:
            return field.copy()
        spectrum = self.fft(field)
        transfer = self.xp.exp(-1j * (self.qx**2) * z / (2 * k))
        return self.ifft(spectrum * transfer)

    def convolve(self, field_a: Any, field_b: Any) -> Any:
        """Convolve two 1D fields using the library's continuous FFT convention."""

        self._validate_field_backend(field_a)
        self._validate_field_backend(field_b)
        return self.ifft(self.fft(field_a) * self.fft(field_b))


@dataclass
class Grid2D(BaseGrid):
    """Uniform two-dimensional transverse grid.

    Parameters
    ----------
    dx, dy:
        Spatial sampling steps in metres.
    nx, ny:
        Number of samples along x and y.
    backend:
        ``"cupy"`` or ``"numpy"``.
    """

    dx: float
    dy: float
    nx: int
    ny: int
    backend: str = "cupy" if CUPY_AVAILABLE else "numpy"

    def __post_init__(self) -> None:
        _validate_positive("dx", self.dx)
        _validate_positive("dy", self.dy)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be >= 2.")
        self.backend = self.backend.lower()
        xp = _backend_module(self.backend)

        x_axis = _centered_coordinates(self.nx, self.dx, xp)
        y_axis = _centered_coordinates(self.ny, self.dy, xp)
        qx_axis = _centered_reciprocal_coordinates(self.nx, self.dx, xp)
        qy_axis = _centered_reciprocal_coordinates(self.ny, self.dy, xp)

        self.x, self.y = xp.meshgrid(x_axis, y_axis)
        self.qx, self.qy = xp.meshgrid(qx_axis, qy_axis)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.qx_axis = qx_axis
        self.qy_axis = qy_axis
        self.dqx = 2 * np.pi / (self.dx * self.nx)
        self.dqy = 2 * np.pi / (self.dy * self.ny)
        self.shape = (self.ny, self.nx)
        self.dimension = 2

    @property
    def xp(self):
        """Return the numerical array module used by the grid."""

        return _backend_module(self.backend)

    def fft(self, field: Any) -> Any:
        """Compute the centered continuous Fourier transform of a 2D field."""

        self._validate_field_backend(field)
        return _fft2d(field, self.dx, self.dy)

    def ifft(self, field: Any) -> Any:
        """Compute the centered continuous inverse Fourier transform of a 2D spectrum."""

        self._validate_field_backend(field)
        return _ifft2d(field, self.dx, self.dy)

    def propagate(self, field: Any, z: float, k: float) -> Any:
        """Fresnel-propagate a 2D field by ``z`` metres."""

        self._validate_field_backend(field)
        if z == 0:
            return field.copy()
        spectrum = self.fft(field)
        transfer = self.xp.exp(
            -1j * (self.qx**2 + self.qy**2) * z / (2 * k)
        )
        return self.ifft(spectrum * transfer)

    def convolve(self, field_a: Any, field_b: Any) -> Any:
        """Convolve two 2D fields using the library's continuous FFT convention."""

        self._validate_field_backend(field_a)
        self._validate_field_backend(field_b)
        return self.ifft(self.fft(field_a) * self.fft(field_b))


# ---------------------------------------------------------------------------
# Optical-object model
# ---------------------------------------------------------------------------


class OpticalObject(ABC):
    """Base class for every object that can produce or transform ``E``."""

    c = 299_792_458.0

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

    def propagate(self, field: Any, z: float) -> Any:
        """Fresnel-propagate ``field`` on this object's grid by ``z`` metres."""

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

    def _require_input_field(self) -> Any:
        """Return the incident field or raise a clear error in geometry-only mode."""
        if self.input_field is None:
            raise ValueError(
                f"{type(self).__name__} field calculation requires input_field; "
                "geometry-only operations remain available without it."
            )
        return self.input_field

    def E(self, z: float = 0.0) -> Any:
        """Return the element's transformed field and optionally propagate it by ``z``."""

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

    def E(self, z: float = 0.0) -> Any:
        """Create the source field at propagation distance ``z`` metres."""

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

    def transmission_mask(self) -> Any:
        """Return the binary aperture mask for the two symmetric holes."""

        xp = self.grid.xp
        if self.grid.dimension == 1:
            x0 = float(self.center_offset)
            return (
                ((self.grid.x - x0) ** 2 <= self.radius**2)
                | ((self.grid.x + x0) ** 2 <= self.radius**2)
            ).astype(self.input_field.dtype)

        x0, y0 = self.center_offset
        first = (self.grid.x - x0) ** 2 + (self.grid.y - y0) ** 2 <= self.radius**2
        second = (self.grid.x + x0) ** 2 + (self.grid.y + y0) ** 2 <= self.radius**2
        return (first | second).astype(self.input_field.dtype)

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
            return (self.grid.x >= 0).astype(self.input_field.dtype)
        return ((self.grid.x >= 0) & (self.grid.y >= 0)).astype(self.input_field.dtype)

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
        self.defect_array = _to_numpy(defect_array)
        self.fill_value = float(fill_value)

        if isinstance(coords, (tuple, list)):
            self.coords = tuple(_to_numpy(c) for c in coords)
        else:
            self.coords = _to_numpy(coords)

    def apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any:
        """Interpolate the defect onto the CRL grid and add it to the thickness."""
        xp = grid.xp

        if grid.dimension == 1:
            x_def = self.coords
            if not isinstance(x_def, np.ndarray) or x_def.ndim != 1:
                raise ValueError("1D defect coordinates must be a 1D array.")

            grid_x = _to_numpy(grid.x)

            if grid_x.min() < x_def.min() or grid_x.max() > x_def.max():
                warnings.warn(
                    "The defect coordinates do not fully cover the CRL grid range. "
                    "Regions outside the defect bounds will use fill_value.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            interpolated = np.interp(
                grid_x, x_def, self.defect_array, 
                left=self.fill_value, right=self.fill_value
            )

        else:
            if not (isinstance(self.coords, tuple) and len(self.coords) == 2):
                raise ValueError("2D defect coordinates must be a tuple (x_def, y_def) of 1D arrays.")
            
            x_def, y_def = self.coords
            if x_def.ndim != 1 or y_def.ndim != 1:
                raise ValueError("2D defect coordinate axes must be 1D arrays.")

            grid_x_axis = _to_numpy(grid.x_axis)
            grid_y_axis = _to_numpy(grid.y_axis)

            if (grid_x_axis.min() < x_def.min() or grid_x_axis.max() > x_def.max() or
                grid_y_axis.min() < y_def.min() or grid_y_axis.max() > y_def.max()):
                warnings.warn(
                    "The defect coordinates do not fully cover the CRL grid range. "
                    "Regions outside the defect bounds will use fill_value.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            interp_func = RegularGridInterpolator(
                (y_def, x_def),
                self.defect_array,
                method="linear",
                bounds_error=False,
                fill_value=self.fill_value
            )

            grid_x = _to_numpy(grid.x)
            grid_y = _to_numpy(grid.y)
            points = np.stack((grid_y.flat, grid_x.flat), axis=-1)
            interpolated = interp_func(points).reshape(grid.shape)

        interpolated_backend = xp.asarray(interpolated, dtype=thickness.dtype)
        return thickness + interpolated_backend


class SurfaceError:
    """Surface-height error sampled on an independent regular 1D/2D grid.

    Input is ``[error, x]`` or ``[error, x, y]``. For 2D data, ``x`` and
    ``y`` may be 1D axes or 2D ``meshgrid`` coordinate arrays. Values outside
    the supplied coordinate range are zero.
    """

    def __init__(self, data: Sequence[Any]) -> None:
        if not isinstance(data, (list, tuple)) or len(data) not in (2, 3):
            raise ValueError("surface_error must be [error, x] or [error, x, y].")
        self.error = _to_numpy(data[0]).astype(float, copy=False)
        self.x = _to_numpy(data[1]).astype(float, copy=False)
        self.y = _to_numpy(data[2]).astype(float, copy=False) if len(data) == 3 else None
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
            values = np.interp(_to_numpy(grid.x), self.x_axis, self.error, left=0.0, right=0.0)
            return grid.xp.asarray(values)
        if self.error.ndim != 2 or self.y_axis is None:
            raise ValueError("A 1D surface_error cannot be used with Grid2D.")
        interpolator = RegularGridInterpolator(
            (self.y_axis, self.x_axis), self.error, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        points = np.stack((_to_numpy(grid.y).ravel(), _to_numpy(grid.x).ravel()), axis=-1)
        return grid.xp.asarray(interpolator(points).reshape(grid.shape))


class ArrayObject(OpticalElement):
    """Homogeneous-material optical object defined by a sampled thickness map.

    Supply ``[thickness_1d, x]`` for 1D or ``[thickness_2d, x, y]`` for 2D.
    For 2D, coordinates may be 1D axes or 2D regular ``meshgrid`` arrays.
    Zero thickness means no material; negative thickness is invalid. Outside
    the supplied map the object is absent. The map is linearly interpolated
    onto the calculation grid.
    """

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
            _validate_positive("material delta", self.delta)
        if self.beta is not None and self.beta < 0:
            raise ValueError("material beta must be >= 0.")

    def _warn_if_clipped(self) -> None:
        """Warn only when the supplied object extends beyond the calculation grid."""
        d = self._thickness_data
        axes = [("x", d.x_axis, _to_numpy(self.grid.x))]
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
        values = self._thickness_data.on_grid(self.grid)
        values_np = _to_numpy(values)
        if np.any(values_np < 0):
            raise ValueError("Interpolated ArrayObject thickness cannot be negative.")
        return values

    def transmission(self) -> Any:
        """Return the complex transmission function of the object."""
        if self.k is None:
            raise ValueError("Transmission calculation requires wavelength_m or energy_kev.")
        if self.delta is None or self.beta is None:
            raise ValueError("Optical constants require energy_kev or explicit material delta/beta.")
        thickness = self.thickness_map()
        return self.grid.xp.exp(-1j * self.k * (self.delta - 1j * self.beta) * thickness)

    def _field_after_element(self) -> Any:
        """Apply material transmission where thickness is positive; block no material outside."""
        thickness = self.thickness_map()
        mask = (thickness > 0).astype(self.input_field.dtype)
        return self.input_field * self.transmission() * mask


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
        Reuse one calculated thickness/transmission map for every element when
        the element geometry is identical. Per-element defect sequences automatically
        disable this optimization.
    wavelength_m, energy_kev:
        Optional photon wavelength or energy. Required for propagation and
        transmission calculations; not required for geometry-only use.
    """

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
        surface_error: Sequence[Any] | None = None,
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
        if n_elements < 1:
            raise ValueError("n_elements must be >= 1.")

        self.radius = float(radius)
        self.aperture = float(aperture)
        self.bridge_thickness = float(bridge_thickness)
        self.n_elements = int(n_elements)
        self.material = material
        self.transverse_length = None if transverse_length is None else float(transverse_length)
        self.rotation = float(rotation)
        self.reuse_element_geometry = bool(reuse_element_geometry)
        self._surface_error = None if surface_error is None else SurfaceError(surface_error)

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

        self.element_length = self.bridge_thickness + self.aperture**2 / (4 * self.radius)
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
        self._warn_extent_axis("x", error.x, -half_x, half_x)
        if error.error.ndim == 2 and error.y is not None:
            half_y = (self.transverse_length / 2
                      if self.transverse_length is not None else self.aperture / 2)
            self._warn_extent_axis("y", error.y, -half_y, half_y)

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
            # A surface defect is defined only inside the CRL aperture. Coordinates
            # are interpreted in the CRL's local frame, including after rotation.
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
        thickness_values = _to_numpy(thickness)
        if np.any(thickness_values <= 0):
            min_thickness = float(np.min(thickness_values))
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

        count = self.n_elements if n_elements is None else int(n_elements)
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
        count = self.n_elements if n_elements is None else int(n_elements)
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
        count = self.n_elements if n_elements is None else int(n_elements)
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
        count = self.n_elements if n_elements is None else int(n_elements)
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

        count = self.n_elements if n_elements is None else int(n_elements)
        return count <= self.max_element_count()

    def focus_scan(
        self,
        z_values: Sequence[float],
        *,
        roi: Any = None,
        return_fields: bool = False,
        calculate_divergence: bool = False,
    ) -> "FocusScanResult":
        """Propagate one CRL exit field to many positions and collect beam metrics.

        The CRL itself is evaluated once. Only Fresnel propagation is repeated
        for each requested ``z``. When ``return_fields=False`` no detector-image
        stack is retained. ``roi`` is specified in physical transverse coordinates.
        """

        z_array = np.asarray(z_values, dtype=float)
        if z_array.ndim != 1 or len(z_array) == 0:
            raise ValueError("z_values must be a non-empty 1D sequence.")
        if np.any(z_array < 0):
            raise ValueError("z_values must contain non-negative distances.")

        if not self.is_focus_outside_crl():
            warnings.warn(
                "CRL focus_scan skipped because the theoretical focus lies inside the CRL.",
                RuntimeWarning,
                stacklevel=2,
            )
            return FocusScanResult.invalid()

        E0 = self.E(z=0)
        E_stack = []
        metrics: list[BeamMetrics] = []
        for z in z_array:
            E = self.propagate(E0, float(z))
            measured = beam_metrics(
                E,
                self.grid,
                roi=roi,
                calculate_divergence=calculate_divergence,
                k=self.k,
            )
            metrics.append(measured)
            if return_fields:
                E_stack.append(_crop_field(E, self.grid, roi))

        max_intensities = np.asarray([m.max_intensity for m in metrics], dtype=float)
        peak_index = (
            None
            if len(max_intensities) == 0 or np.all(np.isnan(max_intensities))
            else int(np.nanargmax(max_intensities))
        )
        E_array = self.grid.xp.stack(E_stack, axis=0) if return_fields else None
        x_roi, y_roi = _roi_coordinates(self.grid, roi)
        peak_z = None if peak_index is None else float(z_array[peak_index])
        return FocusScanResult(
            metrics=metrics,
            peak_index=peak_index,
            peak_z=peak_z,
            E=E_array,
            x_roi=x_roi,
            y_roi=y_roi,
        )


# ---------------------------------------------------------------------------
# Beam analysis
# ---------------------------------------------------------------------------


@dataclass
class BeamMetrics:
    """Summary of scalar beam properties measured on one detector plane."""

    max_intensity: float
    center: float | tuple[float, float] | None
    fwhm: float | tuple[float, float] | None
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


def _crossing_linear(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    """Linearly interpolate an x-coordinate where a sampled curve reaches ``target``."""

    if y1 == y0:
        return float((x0 + x1) / 2)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def full_width_half_max(
    intensity_array: Any,
    coordinate: Any,
) -> float | None:
    """Return FWHM of a 1D intensity profile or ``None`` if no valid pair exists.

    The profile may be any NumPy/CuPy array; ``axis`` identifies the 1D slice.
    For multidimensional input the remaining dimensions must already represent
    a single profile. The routine uses linear interpolation around the half-max
    crossings and does not require an exactly Gaussian beam.
    """

    arr = _to_numpy(intensity_array).astype(float, copy=False)
    coord = _to_numpy(coordinate).astype(float, copy=False)
    if arr.ndim != 1:
        arr = np.asarray(np.squeeze(arr))
    if arr.ndim != 1 or coord.ndim != 1 or len(arr) != len(coord):
        raise ValueError("full_width_half_max expects a one-dimensional profile and coordinate array.")
    if not np.all(np.isfinite(arr)):
        return None

    max_value = float(np.max(arr))
    if max_value <= 0:
        return None
    half = max_value / 2.0
    peak = int(np.argmax(arr))

    left_indices = np.where(arr[:peak] < half)[0]
    right_indices = np.where(arr[peak + 1 :] < half)[0]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return None

    left_i = int(left_indices[-1])
    right_i = int(peak + 1 + right_indices[0])
    left_cross = _crossing_linear(
        coord[left_i], arr[left_i], coord[left_i + 1], arr[left_i + 1], half
    )
    right_cross = _crossing_linear(
        coord[right_i - 1], arr[right_i - 1], coord[right_i], arr[right_i], half
    )
    width = right_cross - left_cross
    return float(width) if width >= 0 else None


def _roi_coordinates(grid: BaseGrid, roi: Any) -> tuple[Any, Any | None]:
    """Return backend-preserving transverse coordinate axes after ROI cropping."""

    if grid.dimension == 1:
        x = grid.x
        if roi is None:
            return x, None
        if not (isinstance(roi, Sequence) and len(roi) == 2):
            raise ValueError("1D roi must be (xmin, xmax).")
        xmin, xmax = map(float, roi)
        mask = (x >= xmin) & (x <= xmax)
        indices = _to_numpy(np.where(_to_numpy(mask))[0])
        if len(indices) == 0:
            raise ValueError("1D ROI does not overlap the grid.")
        return x[int(indices[0]) : int(indices[-1]) + 1], None

    x_axis, y_axis = grid.x_axis, grid.y_axis
    if roi is None:
        return x_axis, y_axis
    if not (isinstance(roi, Sequence) and len(roi) == 2):
        raise ValueError("2D roi must be ((xmin, xmax), (ymin, ymax)).")
    (xmin, xmax), (ymin, ymax) = roi
    xmask = (x_axis >= float(xmin)) & (x_axis <= float(xmax))
    ymask = (y_axis >= float(ymin)) & (y_axis <= float(ymax))
    xi = _to_numpy(np.where(_to_numpy(xmask))[0])
    yi = _to_numpy(np.where(_to_numpy(ymask))[0])
    if len(xi) == 0 or len(yi) == 0:
        raise ValueError("2D ROI does not overlap the grid.")
    return (
        x_axis[int(xi[0]) : int(xi[-1]) + 1],
        y_axis[int(yi[0]) : int(yi[-1]) + 1],
    )


def _crop_field(field: Any, grid: BaseGrid, roi: Any) -> Any:
    """Crop a field to a physical-coordinate ROI without changing its backend."""

    if roi is None:
        return field

    if grid.dimension == 1:
        if not (isinstance(roi, Sequence) and len(roi) == 2):
            raise ValueError("1D roi must be (xmin, xmax).")
        xmin, xmax = map(float, roi)
        mask = (grid.x >= xmin) & (grid.x <= xmax)
        indices = _to_numpy(np.where(_to_numpy(mask))[0])
        if len(indices) == 0:
            raise ValueError("1D ROI does not overlap the grid.")
        return field[int(indices[0]) : int(indices[-1]) + 1]

    if not (isinstance(roi, Sequence) and len(roi) == 2):
        raise ValueError("2D roi must be ((xmin, xmax), (ymin, ymax)).")
    (xmin, xmax), (ymin, ymax) = roi
    xmask = (grid.x_axis >= float(xmin)) & (grid.x_axis <= float(xmax))
    ymask = (grid.y_axis >= float(ymin)) & (grid.y_axis <= float(ymax))
    xi = _to_numpy(np.where(_to_numpy(xmask))[0])
    yi = _to_numpy(np.where(_to_numpy(ymask))[0])
    if len(xi) == 0 or len(yi) == 0:
        raise ValueError("2D ROI does not overlap the grid.")
    return field[int(yi[0]) : int(yi[-1]) + 1, int(xi[0]) : int(xi[-1]) + 1]


def beam_metrics(
    field: Any,
    grid: BaseGrid,
    *,
    roi: Any = None,
    calculate_divergence: bool = False,
    k: float | None = None,
) -> BeamMetrics:
    """Measure peak intensity, beam center, FWHM and optionally divergence.

    The optional ``roi`` is interpreted in physical coordinates and all peak/FWHM
    measurements are performed in the cropped coordinate system.
    """

    grid._validate_field_backend(field)
    if calculate_divergence and k is None:
        raise ValueError("k is required when calculate_divergence=True.")

    work = _crop_field(field, grid, roi)
    values_np = _to_numpy(intensity(work))
    if not np.any(np.isfinite(values_np)):
        return BeamMetrics(np.nan, None, None, None)

    max_flat = int(np.nanargmax(values_np))
    max_value = float(values_np.flat[max_flat])

    if grid.dimension == 1:
        if roi is None:
            coord = _to_numpy(grid.x)
        else:
            xmin, xmax = map(float, roi)
            full_coord = _to_numpy(grid.x)
            coord = full_coord[(full_coord >= xmin) & (full_coord <= xmax)]

        center_index = int(np.nanargmax(values_np))
        center = float(coord[center_index])
        fwhm_value = full_width_half_max(values_np, coord)
        div = divergence(field, grid, k) if calculate_divergence else None
        return BeamMetrics(max_value, center, fwhm_value, div)

    local_j, local_i = np.unravel_index(max_flat, values_np.shape)
    if roi is None:
        x_axis = _to_numpy(grid.x_axis)
        y_axis = _to_numpy(grid.y_axis)
    else:
        (xmin, xmax), (ymin, ymax) = roi
        full_x = _to_numpy(grid.x_axis)
        full_y = _to_numpy(grid.y_axis)
        x_axis = full_x[(full_x >= float(xmin)) & (full_x <= float(xmax))]
        y_axis = full_y[(full_y >= float(ymin)) & (full_y <= float(ymax))]

    slice_y = values_np[:, local_i]
    slice_x = values_np[local_j, :]
    fwhm_y = full_width_half_max(slice_y, y_axis)
    fwhm_x = full_width_half_max(slice_x, x_axis)
    center = (float(x_axis[local_i]), float(y_axis[local_j]))
    div = divergence(field, grid, k) if calculate_divergence else None
    return BeamMetrics(max_value, center, (fwhm_x, fwhm_y), div)


def divergence(
    field: Any,
    grid: BaseGrid,
    k: float,
) -> float | tuple[float, float] | None:
    """Estimate far-field angular FWHM when the object's wavenumber is known."""

    _validate_positive("k", k)
    grid._validate_field_backend(field)
    spectrum = grid.ifft(field)
    values = _to_numpy(intensity(spectrum))

    if grid.dimension == 1:
        q = _to_numpy(grid.qx)
        valid = np.abs(q) <= k
        if np.count_nonzero(valid) < 3:
            return None
        theta = np.arcsin(np.clip(q[valid] / k, -1, 1))
        return full_width_half_max(values[valid], theta)

    qx = _to_numpy(grid.qx_axis)
    qy = _to_numpy(grid.qy_axis)
    valid_x = np.abs(qx) <= k
    valid_y = np.abs(qy) <= k
    if np.count_nonzero(valid_x) < 3 or np.count_nonzero(valid_y) < 3:
        return None

    max_j, max_i = np.unravel_index(int(np.argmax(values)), values.shape)
    theta_x = np.arcsin(np.clip(qx[valid_x] / k, -1, 1))
    theta_y = np.arcsin(np.clip(qy[valid_y] / k, -1, 1))
    fx = values[max_j, valid_x]
    fy = values[valid_y, max_i]
    return full_width_half_max(fx, theta_x), full_width_half_max(fy, theta_y)


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