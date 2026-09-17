"""Numerical wave-optics library for 1D and 2D X-ray optical schemes.

The module deliberately keeps the public model small:

* a :class:`Grid1D` or :class:`Grid2D` owns one common numerical grid;
* optical objects transform complex electric-field arrays ``E``;
* :meth:`OpticalElement.E` returns the field immediately after the element
  for ``z=0`` and Fresnel-propagates that field for ``z>0``;
* :class:`PointSource` creates a field and therefore is the only object that
  does not consume an input field;
* NumPy and CuPy are supported without changing the array type of a field;
* CRL defects are represented as thickness modifiers instead of separate
  CRL subclasses.

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

try:  # CuPy is optional; it is the preferred backend when available.
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the local machine.
    cp = None
    CUPY_AVAILABLE = False

__version__ = "0.3.0"


__all__ = [
    "__version__",
    "CUPY_AVAILABLE",
    "wavelength_energy",
    "Material",
    "Grid1D",
    "Grid2D",
    "ThicknessDefect",
    "FunctionThicknessDefect",
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

    ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$" )

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
        self.wavelength_m, self.energy_kev = wavelength_energy(
            wavelength_m=wavelength_m, energy_kev=energy_kev
        )
        self.k = 2 * np.pi / self.wavelength_m
        self.angular_frequency = 2 * np.pi * self.c / self.wavelength_m

    def propagate(self, field: Any, z: float) -> Any:
        """Fresnel-propagate ``field`` on this object's grid by ``z`` metres."""

        if z < 0:
            raise ValueError("Propagation distance z must be >= 0.")
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

        return divergence(self.E(z=z), self.grid, self.k)


class OpticalElement(OpticalObject):
    """Base class for an optical element consuming an input field array."""

    def __init__(
        self,
        input_field: Any,
        grid: BaseGrid,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Store and validate the field immediately before the optical element."""

        super().__init__(
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        self.grid._validate_field_backend(input_field)
        self.input_field = input_field

    def E(self, z: float = 0.0) -> Any:
        """Return the element's transformed field and optionally propagate it by ``z``."""

        if z < 0:
            raise ValueError("Propagation distance z must be >= 0.")
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
        input_field: Any,
        grid: BaseGrid,
        center_offset: float | tuple[float, float],
        radius: float,
        *,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a symmetric two-hole aperture."""

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
    def apply(self, thickness: Any, grid: BaseGrid, lens_index: int) -> Any:
        """Return a modified CRL thickness map for one lens."""


class FunctionThicknessDefect(ThicknessDefect):
    """Adapt a user-supplied Python function to the thickness-defect interface."""

    def __init__(self, function: Callable[[Any, BaseGrid, int], Any]) -> None:
        """Store ``function(thickness, grid, lens_index)`` as the defect model."""

        self.function = function

    def apply(self, thickness: Any, grid: BaseGrid, lens_index: int) -> Any:
        """Apply the wrapped function and validate its returned shape/backend."""

        result = self.function(thickness, grid, lens_index)
        grid._validate_field_backend(result)
        return result


# ---------------------------------------------------------------------------
# CRL element
# ---------------------------------------------------------------------------


class CRL(OpticalElement):
    """Compound refractive lens for 1D and 2D paraxial wave-optics schemes.

    A CRL is built from two opposing parabolic surfaces and a central bridge.
    The complete material thickness of one lens is

    ``T(x) = 2 * sag(x) + d``

    inside the optical aperture, where ``d`` is the distance between the two
    parabolic surfaces. A sequence of ``n_lenses`` such elements is simulated
    numerically with Fresnel propagation between adjacent lens centres.

    The same class is used for both genuinely one-dimensional schemes and for
    one-axis-focusing CRLs embedded in a two-dimensional scheme. In 1D the
    single transverse coordinate is the local focusing coordinate. In 2D, if
    ``transverse_length`` is omitted, the CRL is rotationally symmetric. If it
    is supplied, the CRL is cylindrically focusing: the local ``x`` coordinate
    is the focusing direction and the local ``y`` coordinate defines the finite
    transverse width. The orientation of that local frame is controlled by
    ``rotation=(rx, ry, rz)``; the current paraxial transverse model implements
    the in-plane angle ``rz`` and deliberately rejects non-zero ``rx``/``ry``.

    Parameters
    ----------
    input_field:
        Complex field incident on the first CRL element.
    grid:
        :class:`Grid1D` or :class:`Grid2D` used by the field.
    radius:
        Radius of curvature of each parabolic surface, metres.
    aperture:
        Full optical aperture (diameter), metres.
    bridge_thickness:
        Distance ``d`` between the two opposing parabolic surfaces, metres.
    n_lenses:
        Number of CRL elements.
    material:
        :class:`Material` describing the refractive medium.
    transverse_length:
        Optional transverse physical length of a cylindrically focusing CRL in
        2D. If omitted on a 2D grid the CRL is rotationally symmetric. Ignored
        on a 1D grid.
    rotation:
        Three angles ``(rx, ry, rz)`` in radians. ``rz`` rotates a cylindrical
        2D CRL within the transverse plane. Non-zero ``rx``/``ry`` are reserved
        for a future tilted-element model and currently raise an error.
    thickness_defect:
        One :class:`ThicknessDefect`, a callable, or one defect per lens. A
        defect receives the complete ideal thickness map and may replace or
        modify it.
    reuse_lens_geometry:
        Reuse one calculated thickness/transmission map for every lens when
        the lens geometry is identical. Per-lens defect sequences automatically
        disable this optimization.
    wavelength_m, energy_kev:
        Exactly one optical-energy representation. If omitted, the energy and
        wavelength are inherited from the first optical field only indirectly
        by explicitly supplying one of them to the CRL.
    """

    def __init__(
        self,
        input_field: Any,
        grid: BaseGrid,
        *,
        radius: float,
        aperture: float,
        bridge_thickness: float,
        n_lenses: int,
        material: Material,
        transverse_length: float | None = None,
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        thickness_defect: ThicknessDefect
        | Callable[[Any, BaseGrid, int], Any]
        | Sequence[ThicknessDefect | Callable[[Any, BaseGrid, int], Any]]
        | None = None,
        reuse_lens_geometry: bool = True,
        wavelength_m: float | None = None,
        energy_kev: float | None = None,
    ) -> None:
        """Initialize a CRL, its optical constants, geometry and defect model."""

        super().__init__(
            input_field,
            grid,
            wavelength_m=wavelength_m,
            energy_kev=energy_kev,
        )
        _validate_positive("radius", radius)
        _validate_positive("aperture", aperture)
        _validate_positive("bridge_thickness", bridge_thickness)
        if n_lenses < 1:
            raise ValueError("n_lenses must be >= 1.")
        if len(rotation) != 3:
            raise ValueError("rotation must be (rx, ry, rz).")

        self.radius = float(radius)
        self.aperture = float(aperture)
        self.bridge_thickness = float(bridge_thickness)
        self.n_lenses = int(n_lenses)
        self.material = material
        self.transverse_length = None if transverse_length is None else float(transverse_length)
        self.rotation = tuple(float(v) for v in rotation)
        self.reuse_lens_geometry = bool(reuse_lens_geometry)

        if self.transverse_length is not None:
            _validate_positive("transverse_length", self.transverse_length)
        if self.grid.dimension == 1 and self.transverse_length is not None:
            raise ValueError("transverse_length is only meaningful for a 2D one-axis CRL.")
        if abs(self.rotation[0]) > 1e-15 or abs(self.rotation[1]) > 1e-15:
            raise NotImplementedError(
                "Non-zero rx/ry rotations are not implemented in the transverse paraxial model; "
                "use rz for in-plane CRL rotation."
            )

        self.delta, self.beta = material.optical_constants(self.energy_kev)
        _validate_positive("material delta", self.delta)
        if self.beta < 0:
            raise ValueError("material beta must be >= 0.")

        # The complete edge thickness of one CRL element determines the pitch
        # used by the paraxial stack model.
        self.edge_thickness = self.bridge_thickness + self.aperture**2 / (4 * self.radius)
        self._thickness_cache: dict[int, Any] = {}
        self._transmission_cache: dict[int, Any] = {}

        self._normalize_defect(thickness_defect)

        # The maximum valid integer lens count is checked immediately so a
        # physically questionable CRL cannot be created silently.
        if self.n_lenses > self.max_lens_count():
            warnings.warn(
                f"CRL contains {self.n_lenses} elements, while the maximum valid "
                f"paraxial external-focus count is {self.max_lens_count()}. "
                "The formal focus lies inside the CRL, so external focal-plane "
                "analysis is not physically meaningful.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _normalize_defect(self, defect: Any) -> None:
        """Normalize a single defect or a per-lens defect sequence."""

        self._single_defect: ThicknessDefect | None = None
        self._per_lens_defects: list[ThicknessDefect] | None = None

        if defect is None:
            return

        if isinstance(defect, Sequence) and not isinstance(defect, (str, bytes)):
            if len(defect) != self.n_lenses:
                raise ValueError("A per-lens defect sequence must have n_lenses elements.")
            self._per_lens_defects = [self._coerce_defect(item) for item in defect]
            self.reuse_lens_geometry = False
            return

        self._single_defect = self._coerce_defect(defect)

    @staticmethod
    def _coerce_defect(defect: Any) -> ThicknessDefect:
        """Convert a defect object or callable into a ``ThicknessDefect``."""

        if isinstance(defect, ThicknessDefect):
            return defect
        if callable(defect):
            return FunctionThicknessDefect(defect)
        raise TypeError(
            "thickness_defect must be a ThicknessDefect, callable, or sequence of them."
        )

    @property
    def lens_pitch(self) -> float:
        """Return the centre-to-centre distance between neighbouring CRL elements."""

        return self.edge_thickness

    @property
    def characteristic_length(self) -> float:
        """Return the characteristic CRL length ``sqrt(p R / (2 delta))`` in metres."""

        return float(np.sqrt(self.lens_pitch * self.radius / (2 * self.delta)))

    @property
    def characteristic_lens_count(self) -> float:
        """Return the characteristic lens count ``characteristic_length / lens_pitch``."""

        return self.characteristic_length / self.lens_pitch

    def critical_lens_count(self) -> float:
        """Return the real-valued critical lens count ``pi/2 * Lc / p``.

        At this value the formal paraxial focus reaches the exit reference
        plane. Beyond it the formal focus is inside the CRL and external focal
        plane analysis is not physically meaningful.
        """

        return float(np.pi / 2 * self.characteristic_lens_count)

    def max_lens_count(self) -> int:
        """Return the largest valid integer lens count with an external focus."""

        critical = self.critical_lens_count()
        # The focus must remain strictly before the critical point.
        return max(0, int(np.ceil(critical) - 1))

    def _local_coordinates(self) -> tuple[Any, Any | None]:
        """Return coordinates in the CRL's local transverse frame after ``rz`` rotation."""

        if self.grid.dimension == 1:
            return self.grid.x, None

        theta = self.rotation[2]
        c, s = np.cos(theta), np.sin(theta)
        x_local = c * self.grid.x + s * self.grid.y
        y_local = -s * self.grid.x + c * self.grid.y
        return x_local, y_local

    def _ideal_thickness(self) -> Any:
        """Build one defect-free CRL thickness map.

        For a rotationally symmetric 2D CRL the sag depends on ``x_local²+y_local²``.
        If ``transverse_length`` is supplied, the CRL is cylindrically focusing:
        sag depends only on ``x_local`` and the finite ``y_local`` width limits the
        lens outside its useful aperture. On a 1D grid the single coordinate is
        always the focusing coordinate.
        """

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

        return 2 * sag + self.bridge_thickness

    def thickness(self, lens_index: int = 0) -> Any:
        """Return the complete material thickness map of one CRL element.

        The returned thickness includes both opposing parabolic surfaces and
        the central bridge ``d``. A configured defect is applied after the ideal
        geometry is constructed.
        """

        if not 0 <= lens_index < self.n_lenses:
            raise IndexError("lens_index is outside the CRL range.")
        if self.reuse_lens_geometry and self._thickness_cache:
            return next(iter(self._thickness_cache.values()))
        if lens_index in self._thickness_cache:
            return self._thickness_cache[lens_index]

        thickness = self._ideal_thickness()
        defect = None
        if self._per_lens_defects is not None:
            defect = self._per_lens_defects[lens_index]
        elif self._single_defect is not None:
            defect = self._single_defect
        if defect is not None:
            thickness = defect.apply(thickness, self.grid, lens_index)

        self.grid._validate_field_backend(thickness)
        if not np.issubdtype(thickness.dtype, np.number):
            raise TypeError("CRL thickness defect returned a non-numeric array.")
        if np.any(_to_numpy(thickness) < 0):
            raise ValueError("CRL thickness must not contain negative values.")

        if self.reuse_lens_geometry:
            self._thickness_cache[0] = thickness
        self._thickness_cache[lens_index] = thickness
        return thickness

    def transmission(self, lens_index: int = 0) -> Any:
        """Return the complex transmission function of one CRL element."""

        if self.reuse_lens_geometry and self._transmission_cache:
            return next(iter(self._transmission_cache.values()))
        if lens_index in self._transmission_cache:
            return self._transmission_cache[lens_index]

        thickness = self.thickness(lens_index)
        transmission = self.grid.xp.exp(
            -1j * self.k * (self.delta - 1j * self.beta) * thickness
        )
        if self.reuse_lens_geometry:
            self._transmission_cache[0] = transmission
        self._transmission_cache[lens_index] = transmission
        return transmission

    def clear_cache(self) -> None:
        """Clear cached thickness and transmission maps so they can be recomputed."""

        self._thickness_cache.clear()
        self._transmission_cache.clear()

    def _field_after_element(self) -> Any:
        """Propagate through the complete CRL stack and return its exit-plane field."""

        field = self.propagate(self.input_field, self.lens_pitch / 2)
        for lens_index in range(self.n_lenses):
            field = field * self.transmission(lens_index)
            if lens_index < self.n_lenses - 1:
                field = self.propagate(field, self.lens_pitch)
        return self.propagate(field, self.lens_pitch / 2)

    def theoretical_focus(self, *, n_lenses: int | None = None) -> float:
        """Return the formal paraxial focal distance from the CRL exit reference plane."""

        count = self.n_lenses if n_lenses is None else int(n_lenses)
        if count < 1:
            raise ValueError("n_lenses must be >= 1.")
        critical = self.critical_lens_count()
        if count >= critical:
            warnings.warn(
                "The requested lens count is at or beyond the critical count; "
                "the formal focus is not an external physical focus.",
                RuntimeWarning,
                stacklevel=2,
            )
        u = count / self.characteristic_lens_count
        return float(self.characteristic_length / np.tan(u))

    def effective_focus(
        self,
        source_distance: float,
        *,
        n_lenses: int | None = None,
    ) -> float:
        """Return the finite-object image distance from the CRL exit reference plane."""

        _validate_positive("source_distance", source_distance)
        count = self.n_lenses if n_lenses is None else int(n_lenses)
        if count < 1:
            raise ValueError("n_lenses must be >= 1.")
        if count > self.max_lens_count():
            raise ValueError(
                "The requested lens count exceeds max_lens_count(); "
                "the formal focus lies inside the CRL."
            )

        lc = self.characteristic_length
        u = count / self.characteristic_lens_count
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
        n_lenses: int | None = None,
    ) -> float:
        """Return the signed analytical CRL transverse magnification."""

        _validate_positive("source_distance", source_distance)
        _validate_positive("image_distance", image_distance)
        count = self.n_lenses if n_lenses is None else int(n_lenses)
        if count < 1:
            raise ValueError("n_lenses must be >= 1.")
        lc = self.characteristic_length
        lens_length = count * self.lens_pitch
        u = lens_length / lc
        sine = np.sin(u)
        cosine = np.cos(u)
        numerator = image_distance * (lens_length * cosine + lc * sine) + lens_length * lc * sine
        denominator = source_distance * (lens_length + lc * sine * cosine)
        return float(numerator / denominator)

    def image_scale(self, source_distance: float, *, n_lenses: int | None = None) -> float:
        """Return the analytical image/source distance ratio for a finite object."""

        _validate_positive("source_distance", source_distance)
        count = self.n_lenses if n_lenses is None else int(n_lenses)
        if count > self.max_lens_count():
            raise ValueError("The requested lens count exceeds max_lens_count().")
        lc = self.characteristic_length
        u = count / self.characteristic_lens_count
        focal_length = lc / np.sin(u)
        back_principal_distance = focal_length * (1 - np.cos(u))
        object_distance = source_distance + back_principal_distance
        image_distance = focal_length * object_distance / (object_distance - focal_length)
        return float(image_distance / object_distance)

    def is_focus_outside_crl(self, *, n_lenses: int | None = None) -> bool:
        """Return whether the specified integer lens count gives an external focus."""

        count = self.n_lenses if n_lenses is None else int(n_lenses)
        return count <= self.max_lens_count()

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

        if not self.is_focus_outside_crl():
            warnings.warn(
                "CRL focus_scan skipped because the theoretical focus lies inside the CRL.",
                RuntimeWarning,
                stacklevel=2,
            )
            return FocusScanResult.invalid(np.asarray(z_values, dtype=float))

        z_array = np.asarray(z_values, dtype=float)
        if z_array.ndim != 1 or len(z_array) == 0:
            raise ValueError("z_values must be a non-empty 1D sequence.")
        if np.any(z_array < 0):
            raise ValueError("z_values must contain non-negative distances.")

        field0 = self.E(z=0)
        fields: list[np.ndarray] = []
        metrics: list[BeamMetrics] = []
        for z in z_array:
            field = self.propagate(field0, float(z))
            measured = beam_metrics(
                field,
                self.grid,
                roi=roi,
                calculate_divergence=calculate_divergence,
                k=self.k,
            )
            metrics.append(measured)
            if return_fields:
                fields.append(_to_numpy(_crop_field(field, self.grid, roi)))

        max_intensities = np.asarray([m.max_intensity for m in metrics], dtype=float)
        peak_index = (
            None
            if len(max_intensities) == 0 or np.all(np.isnan(max_intensities))
            else int(np.nanargmax(max_intensities))
        )
        return FocusScanResult(
            z=z_array,
            metrics=metrics,
            peak_index=peak_index,
            fields=fields if return_fields else None,
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
    """Result of a multi-distance focus scan."""

    z: np.ndarray
    metrics: list[BeamMetrics]
    peak_index: int | None
    fields: list[np.ndarray] | None = None
    valid: bool = True

    @classmethod
    def invalid(cls, z: np.ndarray) -> "FocusScanResult":
        """Create an invalid result when a focus scan is physically inapplicable."""

        return cls(z=z, metrics=[], peak_index=None, fields=None, valid=False)

    @property
    def peak_z(self) -> float | None:
        """Return the scanned z position with maximum measured intensity."""

        if self.peak_index is None:
            return None
        return float(self.z[self.peak_index])

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



# ---------------------------------------------------------------------------
# End-of-file note
# ---------------------------------------------------------------------------

# The old SiemensStar, CRL1Dm/CRL1Deq/CRL1Dcurv/CRL2Dm experimental classes,
# global grid state, commented-out implementations, and `focus_params_calc`
# are intentionally not carried into this module. Their place in the new
# architecture is either a generic thickness defect or the standalone analysis
# layer above.
