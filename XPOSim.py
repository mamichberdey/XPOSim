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
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path
import re
from time import perf_counter
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

__version__ = "0.18.0"


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
    "CRLAnalyticalResult",
    "analyze_crl",
    "CRLFocusResult",
    "estimate_crl_focus",
    "CRLProgressUpdate",
    "PeakIntensitySearchResult",
    "find_peak_intensity_distance",
    "NumericalFocusResult",  # Compatibility names; use the peak-intensity API.
    "find_numerical_focus",
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


class _InverseFourierAxis:
    """Cached Bluestein/CZT quadrature of a continuous inverse Fourier axis."""
    def __init__(self, n, q0, dq, output_axis, xp):
        from scipy.fft import next_fast_len
        self.xp=xp
        self.n=n
        self.m=int(output_axis.size)
        self.length=next_fast_len(n+self.m-1)
        first=float(_to_python_float(output_axis[0]))
        step=float(_to_python_float(output_axis[1]-output_axis[0]))
        angle=dq*step
        j=xp.arange(n,dtype=xp.float64)
        d=xp.arange(-(n-1),self.m,dtype=xp.float64)
        l=xp.arange(self.m,dtype=xp.float64)
        self.pre=xp.exp(1j*(dq*first*j+angle*j*j/2))
        self.kernel=xp.fft.fft(xp.exp(-.5j*angle*d*d),n=self.length)
        self.post=xp.exp(.5j*angle*l*l+1j*q0*output_axis)*(dq/(2*np.pi))

    def apply(self,array,axis=-1):
        xp=self.xp
        work=xp.moveaxis(array,axis,-1)
        work=xp.fft.fft(work*self.pre,n=self.length,axis=-1)
        work*=self.kernel
        result=xp.fft.ifft(work,axis=-1)[...,self.n-1:self.n-1+self.m]*self.post
        return xp.moveaxis(result,-1,axis)


class _DetectorFourierPlan:
    """Evaluate the propagated complex Fourier field on a dense local grid."""
    def __init__(self,grid,detector):
        self.grid=grid
        self.detector=detector
        xp=grid.xp
        x=detector.x if grid.dimension==1 else detector.x_axis
        qx=grid.qx if grid.dimension==1 else grid.qx_axis
        self.x=_InverseFourierAxis(grid.nx,_to_python_float(qx[0]),grid.dqx,x,xp)
        self.y=None
        if grid.dimension==2:
            self.y=_InverseFourierAxis(grid.ny,_to_python_float(grid.qy_axis[0]),
                                      grid.dqy,detector.y_axis,xp)

    def _inverse(self,work):
        work=self.x.apply(work,axis=-1)
        if self.y is not None:
            work=self.y.apply(work,axis=-2)
        return work

    def propagate_many(self,spectrum,z_values,k):
        grid=self.grid
        xp=grid.xp
        z=xp.asarray(z_values)[:,None]
        qx=grid.qx if grid.dimension==1 else grid.qx_axis
        phase=xp.exp((-.5j/k)*z*qx**2)
        work=spectrum[None,...]*(phase if grid.dimension==1 else phase[:,None,:])
        if grid.dimension==2:
            work*=xp.exp((-.5j/k)*z*grid.qy_axis**2)[:,:,None]
        return self._inverse(work)

    def propagate(self,spectra,z,k):
        grid=self.grid
        xp=grid.xp
        qx=grid.qx if grid.dimension==1 else grid.qx_axis
        work=spectra*xp.exp((-.5j*z/k)*qx**2)
        if grid.dimension==2:
            work*=xp.exp((-.5j*z/k)*grid.qy_axis**2)[:,None]
        return self._inverse(work)


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

    def _propagate_mode_spectra(self, spectra: Any, z: float, k: float) -> Any:
        """Propagate independent source modes to one plane in one FFT batch."""
        xp = self.xp
        qx = self.qx if self.dimension == 1 else self.qx_axis
        work = spectra * xp.exp((-0.5j * z / k) * qx**2)
        if self.dimension == 2:
            work *= xp.exp((-0.5j * z / k) * self.qy_axis**2)[:, None]
        return self._transform_batch(work, inverse=True)

    def propagate(self, field: Any, z: float, k: float) -> Any:
        """Fresnel-propagate one field; signed distances are supported."""
        self._validate_field_backend(field)
        _validate_finite("z", z)
        _validate_positive("k", k)
        if z == 0:
            return field.copy()
        return self._propagate_spectrum(self.fft(field), np.asarray([z]), k)[0]

    def propagate_to_grid(self, field: Any, z: float, k: float, detector: "BaseGrid") -> Any:
        """Evaluate Fresnel propagation on a local uniform detector grid.

        This samples the propagated complex Fourier field with chirp-z transforms.
        It does not interpolate a previously calculated intensity or create
        frequencies missing from the entrance grid. The entrance grid must
        still resolve the optical field and phase.
        """
        self._validate_field_backend(field)
        _validate_finite("z",z)
        _validate_positive("k",k)
        if detector.dimension!=self.dimension or detector.backend!=self.backend:
            raise ValueError("Detector dimension/backend must match the propagation grid.")
        return _DetectorFourierPlan(self,detector).propagate(self.fft(field),z,k)

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

    def apply_to_fields(self, fields: Any) -> Any:
        """Apply the stack to a bounded batch of mutually independent fields.

        The leading axis indexes fields; the caller controls batch memory.
        Intensities, rather than amplitudes, must be summed for incoherent light.
        """
        _array_module(fields)
        if (_backend_name(fields) != self.grid.backend or
                fields.ndim != self.grid.dimension + 1 or fields.shape[1:] != self.grid.shape):
            raise ValueError("Expected a source batch with shape (batch, *grid.shape) on the grid backend.")
        k, _, _ = self._require_optical_constants()

        def propagate_batch(field, distance):
            return self.grid._propagate_mode_spectra(
                self.grid._transform_batch(field), distance, k)

        field = propagate_batch(fields, self.element_period / 2)
        for index in range(self.n_elements):
            field *= self.transmission(index)
            if index < self.n_elements - 1:
                field = propagate_batch(field, self.element_period)
        return propagate_batch(field, self.element_period / 2)

    def ray_transfer_matrix(self, *, focusing: bool = True) -> np.ndarray:
        """ABCD matrix of the discrete compressed-lens model in a local axis.

        This uses exactly the half-period/lens/half-period sequence of the wave
        propagator. focusing=False selects the free, nonfocusing local axis.
        """
        if not focusing:
            return np.asarray([[1.0, self.length], [0.0, 1.0]])
        if self.delta is None:
            raise ValueError("Ray transfer requires optical constants.")
        drift = np.asarray([[1.0, self.element_period / 2], [0.0, 1.0]])
        lens = np.asarray([[1.0, 0.0], [-2 * self.delta / self.radius, 1.0]])
        return np.linalg.matrix_power(drift @ lens @ drift, self.n_elements)

    def source_magnification(self, z: float, source_distance: float, *,
                             focusing: bool = True) -> float:
        """Signed source-to-plane displacement scale -B(z)/source_distance.

        It is an image magnification at a conjugate image plane and a source
        displacement scale elsewhere. Coordinates refer to a local lens axis.
        """
        _validate_finite("z", z)
        _validate_positive("source_distance", source_distance)
        matrix = self.ray_transfer_matrix(focusing=focusing)
        return -float(matrix[0, 1] + z * matrix[1, 1]) / source_distance

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

    def effective_aperture(self) -> float | tuple[float | None, float | None] | None:
        """Numerical intensity FWHM at the CRL exit, in metres.

        Uses the full propagation grid and the current coherent incident field,
        including absorption and propagation through every element. On a 2D
        grid, returns (x, y) slice widths through the intensity maximum. The y
        width is None for a finite-depth, one-axis CRL; it is not measured.
        Missing half-maximum crossings give None, just as in beam_metrics.
        CRLFocusResult.effective_aperture_m additionally handles incoherent
        Gaussian sources, using their weighted exit intensity.
        """
        return _beam_metrics_batch(self.E()[None,...],self.grid,
            (slice(None),)*self.grid.dimension,
            measure_y_fwhm=self.transverse_length is None)[0].fwhm

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
        The reported peak plane maximizes sampled transverse peak intensity;
        it need not coincide with minimum width or a geometrical image plane.
        For a finite-depth, one-axis CRL, y FWHM is deliberately unmeasured
        (None in each metric); its full diffraction field is still propagated.
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
            metrics.extend(_beam_metrics_batch(fields, self.grid, slices, div,
                measure_y_fwhm=self.transverse_length is None))
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
    """Result of a multi-distance scan, selecting highest sampled peak intensity.

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
        """Return the FWHM at the highest-peak-intensity sampled plane."""

        if self.peak_index is None or not self.metrics:
            return None
        return self.metrics[self.peak_index].fwhm

    @property
    def divergence_at_peak(self) -> float | tuple[float, float] | None:
        """Return the divergence at the highest-peak-intensity sampled plane."""

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
                        divergence_value: Any = None, *, measure_y_fwhm: bool = True) -> list[BeamMetrics]:
    """Reduce intensities and peak-crossing slices without host image copies."""
    xp = grid.xp
    work = fields[(slice(None), *slices)]
    values = abs(work)**2
    return _intensity_metrics_batch(values, grid, slices, divergence_value,
        measure_y_fwhm=measure_y_fwhm)


def _intensity_metrics_batch(values: Any, grid: BaseGrid, slices: tuple[slice, ...],
                             divergence_value: Any = None, *, measure_y_fwhm: bool = True) -> list[BeamMetrics]:
    """Measure already-incoherent intensities without creating artificial fields."""
    xp = grid.xp
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
    width_y = (_widths_batch(values[rows, :, i], y_axis) if measure_y_fwhm else
               xp.full(batch,xp.nan))
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
class CRLProgressUpdate:
    """Synchronous calculation progress; ETA refers to the current stage only.

    Adaptive stages have total=None and eta_s=None. Timings are wall seconds;
    sampling updates follow completed batches, including metric reductions.
    """
    stage: str
    completed: int
    total: int | None
    elapsed_s: float
    stage_elapsed_s: float
    eta_s: float | None
    detail: str = ""


class _CRLProgressReporter:
    def __init__(self, callback, *, clock=perf_counter):
        if callback is not None and not callable(callback):
            raise TypeError("progress must be callable or None.")
        self.callback,self.clock=callback,clock
        self.started=self.stage_started=clock()
        self.stage,self.total,self.detail="preparing",None,""

    def start(self,stage,total=None,detail=""):
        self.stage,self.total,self.detail=stage,total,detail
        self.stage_started=self.clock()
        self.advance(0)

    def advance(self,completed,detail=None):
        if detail is not None:
            self.detail=detail
        if self.callback is None:
            return
        now=self.clock()
        elapsed=max(0.0,now-self.stage_started)
        eta=None if self.total is None or completed==0 else elapsed*max(0,self.total-completed)/completed
        self.callback(CRLProgressUpdate(self.stage,int(completed),self.total,
            max(0.0,now-self.started),elapsed,eta,self.detail))

    def finish(self):
        self.stage,self.total,self.detail="complete",1,"Calculation complete"
        self.stage_started=self.clock()
        self.advance(1)


@dataclass(frozen=True)
class PeakIntensitySearchResult:
    """Distance of a local peak-intensity maximum, independent of a display grid.

    This intensity criterion need not coincide with minimum beam width or the
    geometrical image plane. It does not identify a unique physical focus.
    """
    z_m: float
    intensity: float
    bracket_m: tuple[float, float]
    converged: bool
    evaluations: int
    message: str
    history: tuple[tuple[float, float], ...]


def find_peak_intensity_distance(peak_intensity: Callable[[float], float], initial_z_m: float,
                         *, bounds_m: tuple[float, float] | None = None,
                         initial_step_m: float | None = None,
                         tolerance_m: float | None = None,
                         max_evaluations: int = 80) -> PeakIntensitySearchResult:
    """Bracket a local maximum with growing steps, then use bounded Brent search.

    No precomputed z array is consumed. The initial guess selects a local basin;
    diffraction may have several maxima, so this is not a global-optimum claim.
    The returned history can be used to inspect the adaptive search.
    The maximized quantity is the transverse peak intensity at each distance,
    not minimum FWHM or a geometrical imaging condition.
    """
    from scipy.optimize import minimize_scalar
    _validate_positive("initial_z_m", initial_z_m)
    _validate_count("max_evaluations", max_evaluations, 8)
    bounds = _validated_distances(bounds_m or (0.0, 3 * initial_z_m), nonnegative=True)
    if bounds.size != 2 or bounds[1] <= bounds[0]:
        raise ValueError("Peak-intensity search bounds must be an increasing pair.")
    lo, hi = map(float, bounds)
    step = initial_step_m if initial_step_m is not None else 0.04 * initial_z_m
    tolerance = tolerance_m if tolerance_m is not None else 1e-5 * initial_z_m
    _validate_positive("initial_step_m", step)
    _validate_positive("tolerance_m", tolerance)
    center = float(np.clip(initial_z_m, lo + (hi-lo)*1e-6, hi - (hi-lo)*1e-6))
    cache: dict[float, float] = {}

    class BudgetReached(Exception):
        pass

    def evaluate(z):
        z = float(z)
        if z not in cache:
            if len(cache) >= max_evaluations:
                raise BudgetReached
            value = float(peak_intensity(z))
            if not np.isfinite(value):
                raise ArithmeticError("Peak-intensity objective returned non-finite intensity.")
            cache[z] = value
        return cache[z]

    a, b, c = max(lo, center-step), center, min(hi, center+step)
    bracketed = False
    success = False
    message = "The evaluation budget was exhausted."
    try:
        # Start at the requested analytical/reference distance, then expand
        # its bracket. It is cached when the middle point is visited again.
        evaluate(center)
        for _ in range(max_evaluations):
            ya, yb, yc = evaluate(a), evaluate(b), evaluate(c)
            scale = max(abs(ya), abs(yb), abs(yc), np.finfo(float).tiny)
            if max(ya, yb, yc)-min(ya, yb, yc) <= scale*1e-12:
                message = "The objective is locally flat; a peak-intensity distance was not resolved."
                break
            if yb > ya and yb > yc:
                bracketed = True
                break
            step *= 1.6
            if ya >= yc:
                if a == lo:
                    message = "The maximum lies at the lower search boundary."
                    break
                a, b, c = max(lo, a-step), a, b
            else:
                if c == hi:
                    message = "The maximum lies at the upper search boundary."
                    break
                a, b, c = b, c, min(hi, c+step)
        if bracketed:
            optimized = minimize_scalar(lambda z: -evaluate(z), bounds=(a, c),
                                        method="bounded", options={"xatol": tolerance,
                                        "maxiter": max_evaluations-len(cache)})
            success = bool(optimized.success)
            evaluate(optimized.x)
            message = "A local peak was bracketed and refined." if success else str(optimized.message)
    except BudgetReached:
        pass
    if not cache:
        raise ArithmeticError("The peak-intensity search made no evaluations.")
    best = max(cache, key=cache.get)
    return PeakIntensitySearchResult(best, cache[best], (float(a), float(c)), success,
                                len(cache), message, tuple(cache.items()))


NumericalFocusResult = PeakIntensitySearchResult


def find_numerical_focus(*args, **kwargs) -> PeakIntensitySearchResult:
    """Compatibility wrapper for :func:`find_peak_intensity_distance`."""
    warnings.warn("find_numerical_focus is deprecated; use find_peak_intensity_distance. "
                  "The objective maximizes peak intensity, not a unique focus criterion.",
                  DeprecationWarning, stacklevel=2)
    return find_peak_intensity_distance(*args, **kwargs)


def _peak_z_roi(objective: Callable[[float], float], peak: PeakIntensitySearchResult,
                  bounds: tuple[float, float], level: float) -> tuple[float, float]:
    """Find the connected peak-intensity ROI by expanding and root refinement."""
    from scipy.optimize import brentq
    if peak.intensity <= 0:
        return bounds
    target = level * peak.intensity
    initial = max(0.005*max(peak.z_m, 1e-9), 1e-8)
    crossings = []
    for direction, limit in [(-1, bounds[0]), (1, bounds[1])]:
        inside = peak.z_m
        step = initial
        while True:
            outside = float(np.clip(inside+direction*step, bounds[0], bounds[1]))
            if outside == inside:
                crossings.append(float(limit))
                break
            value = objective(outside)
            if value <= target:
                a, b = sorted((inside, outside))
                crossings.append(float(brentq(lambda z: objective(z)-target, a, b,
                                               xtol=max(1e-9, initial*1e-3))))
                break
            if outside == limit:
                crossings.append(float(limit))
                break
            inside = outside
            step *= 1.6
    return crossings[0], crossings[1]


@dataclass(frozen=True)
class CRLFocusConfig:
    """Ideal CRL workflow. Lengths are metres; Gaussian source sizes are FWHM.

    Focusing dimension and numerical-grid dimension are independent. A one-
    dimensionally focusing lens on a 2D grid requires its unfocused transverse
    structure depth. Gaussian source points are mutually incoherent. With
    source_type='plane', a uniform unit-amplitude, constant-phase field enters
    the pupil; source_distance_m is normalized to None and magnification is
    undefined. The analytical image distance then equals collimated BFL.
    peak_search_tolerance_m controls the axial peak-intensity optimizer;
    focus_tolerance_m is retained as a legacy keyword alias. material_delta and
    material_beta together select fixed direct optical constants instead of a
    formula/density; their formula and density metadata are normalized to None.
    grid_step_x_m/grid_step_y_m request transverse spacing instead of counts.
    Actual spacings are quantized to scale-adaptive multiples of five. The
    automatic axial spacing or scan_step_m uses 1/2.5/5 times a power of ten;
    scan_points is a target count. The z ROI expands outward to that uniform
    lattice anchored at the full-precision numerical peak distance. Peak
    optimization remains independent of the display spacing.
    """
    energy_kev: float = 12.0
    material_formula: str | None = "Be"
    density_g_cm3: float | None = 1.848
    radius_m: float = 50e-6
    aperture_m: float = 100e-6
    bridge_thickness_m: float = 30e-6
    n_elements: int = 10
    source_distance_m: float | None = 15.0
    geometry: str = "one-dimensional"
    grid_dimension: int | None = None
    structure_depth_m: float | None = 50e-6
    source_type: str = "point"
    source_fwhm_x_m: float = 5e-6
    source_fwhm_y_m: float = 5e-6
    source_method: str = "convolution"
    source_points: int = 7
    accuracy: str = "preview"
    backend: str = "auto"
    scan_range_m: tuple[float, float] | None = None
    scan_points: int = 51
    grid_points: int | None = None
    grid_points_y: int | None = None
    window_factor: float = 1.3
    adaptive_window: bool = True
    batch_size: int | None = None
    axial_roi_level: float = 0.2
    transverse_roi_level: float = 0.01
    focus_tolerance_m: float | None = None
    peak_search_tolerance_m: float | None = None
    material_delta: float | None = None
    material_beta: float | None = None
    grid_step_x_m: float | None = None
    grid_step_y_m: float | None = None
    scan_step_m: float | None = None

    def __post_init__(self):
        aliases = {"cylindrical": "one-dimensional", "axisymmetric": "two-dimensional"}
        object.__setattr__(self, "geometry", aliases.get(self.geometry, self.geometry))
        if self.geometry not in {"one-dimensional", "two-dimensional"}:
            raise ValueError("Choose one-dimensional or two-dimensional focusing.")
        dimension = self.grid_dimension if self.grid_dimension is not None else (2 if self.geometry == "two-dimensional" else 1)
        _validate_count("grid_dimension", dimension)
        if dimension not in (1, 2) or (self.geometry == "two-dimensional" and dimension != 2):
            raise ValueError("Two-dimensional focusing requires a 2D grid.")
        object.__setattr__(self, "grid_dimension", int(dimension))
        if self.source_type not in {"point", "gaussian", "plane"}:
            raise ValueError("source_type must be point, gaussian or plane.")
        if self.source_type == "plane":
            object.__setattr__(self,"source_distance_m",None)
        else:
            _validate_positive("source_distance_m",self.source_distance_m)
            object.__setattr__(self,"source_distance_m",float(self.source_distance_m))
        if (self.material_delta is None)!=(self.material_beta is None):
            raise ValueError("material_delta and material_beta must be supplied together.")
        if self.material_delta is not None:
            _validate_positive("material_delta",self.material_delta)
            _validate_finite("material_beta",self.material_beta)
            if self.material_beta<0:
                raise ValueError("material_beta must be non-negative.")
            object.__setattr__(self,"material_delta",float(self.material_delta))
            object.__setattr__(self,"material_beta",float(self.material_beta))
            object.__setattr__(self,"material_formula",None)
            object.__setattr__(self,"density_g_cm3",None)
        else:
            _validate_positive("density_g_cm3",self.density_g_cm3)
            object.__setattr__(self,"density_g_cm3",float(self.density_g_cm3))
        for name in ("energy_kev", "radius_m", "aperture_m", "bridge_thickness_m", "window_factor"):
            _validate_positive(name, getattr(self, name))
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.window_factor <= 1:
            raise ValueError("window_factor must exceed one.")
        if dimension == 2 and self.geometry == "one-dimensional":
            if self.structure_depth_m is None:
                raise ValueError("A 2D grid needs structure_depth_m for a one-dimensionally focusing CRL.")
            _validate_positive("structure_depth_m", self.structure_depth_m)
        for name, minimum in (("n_elements",1),("scan_points",5),("source_points",3)):
            _validate_count(name, getattr(self,name), minimum)
            object.__setattr__(self,name,int(getattr(self,name)))
        for name in ("grid_points", "grid_points_y", "batch_size"):
            value=getattr(self,name)
            if value is not None:
                _validate_count(name,value,1 if name=="batch_size" else 32)
                object.__setattr__(self,name,int(value))
        for name in ("grid_step_x_m","grid_step_y_m","scan_step_m"):
            value=getattr(self,name)
            if value is not None:
                _validate_positive(name,value)
                object.__setattr__(self,name,float(value))
        if self.grid_points is not None and self.grid_step_x_m is not None:
            raise ValueError("Choose either grid_points or grid_step_x_m, not both.")
        if self.grid_points_y is not None and self.grid_step_y_m is not None:
            raise ValueError("Choose either grid_points_y or grid_step_y_m, not both.")
        if dimension==1 and self.grid_step_y_m is not None:
            raise ValueError("grid_step_y_m requires a 2D grid.")
        if self.source_method not in {"convolution", "points"}:
            raise ValueError("source_method must be convolution or points.")
        for name in ("source_fwhm_x_m", "source_fwhm_y_m"):
            value=getattr(self,name)
            _validate_finite(name,value)
            if value<0: raise ValueError("Source FWHM must be non-negative.")
            object.__setattr__(self,name,float(value))
        if self.accuracy not in {"preview","refined"} or self.backend not in {"auto","numpy","cupy"}:
            raise ValueError("Invalid accuracy or backend.")
        for name in ("axial_roi_level","transverse_roi_level"):
            value=getattr(self,name)
            _validate_positive(name,value)
            if value>=1: raise ValueError("ROI levels must be between zero and one.")
        if self.focus_tolerance_m is not None:
            _validate_positive("focus_tolerance_m",self.focus_tolerance_m)
            if self.peak_search_tolerance_m is not None and self.peak_search_tolerance_m != self.focus_tolerance_m:
                raise ValueError("Specify only one peak-search tolerance; focus_tolerance_m is a legacy alias.")
            object.__setattr__(self,"peak_search_tolerance_m",self.focus_tolerance_m)
        if self.peak_search_tolerance_m is not None:
            _validate_positive("peak_search_tolerance_m",self.peak_search_tolerance_m)
        if self.scan_range_m is not None:
            bounds=_validated_distances(self.scan_range_m,nonnegative=True)
            if bounds.size!=2 or bounds[1]<=bounds[0]: raise ValueError("Invalid z ROI.")
            object.__setattr__(self,"scan_range_m",tuple(map(float,bounds)))
        if self.material_delta is None:
            self.create_material().composition()

    def create_material(self) -> Material:
        """Build the selected material; explicit coefficients bypass tables.

        Direct delta/beta are fixed values used at energy_kev. They are not
        rescaled automatically when photon energy changes.
        """
        if self.material_delta is not None:
            return Material(delta=self.material_delta,beta=self.material_beta)
        return Material(self.material_formula,self.density_g_cm3)


@dataclass(frozen=True)
class CRLAnalyticalResult:
    """Preflight optics/geometry, calculated without propagation or GPU use.

    magnification_at_image uses the signed discrete-stack source displacement
    scale at image_distance_m, with one entry per calculation axis. For an
    unfocused y axis this is a source displacement scale, not conjugate image
    magnification. Plane-wave, absent and infinite image planes return None.
    """
    delta: float
    beta: float
    collimated_focus_m: float | None
    image_distance_m: float | None
    max_element_count: int
    lens_length_m: float
    external_focus: bool
    real_image: bool
    magnification_at_image: tuple[float | None, ...] = ()


def analyze_crl(configuration: CRLFocusConfig) -> CRLAnalyticalResult:
    """Return cheap analytical material/lens parameters for any material mode.

    Focal and image distances use the CRL exit reference. Counts and geometry
    delegate to CRL.max_element_count and its existing analytical methods.
    For N above the external-focus limit, distances are None while material
    constants, maximum N and total length remain available. A finite source
    can produce a negative (virtual) or infinite image distance.
    """
    if not isinstance(configuration,CRLFocusConfig):
        raise TypeError("Pass a CRLFocusConfig instance.")
    cfg=configuration
    delta,beta=cfg.create_material().optical_constants(cfg.energy_kev)
    # One reference element avoids warnings from a temporarily excessive N
    # while editing the form. Even N=1 can be invalid for extreme constants;
    # report that case via external_focus/max_element_count instead of stdout.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        reference=CRL(grid=Grid1D(cfg.aperture_m,2,backend="numpy"),
            radius=cfg.radius_m,aperture=cfg.aperture_m,
            bridge_thickness=cfg.bridge_thickness_m,n_elements=1,
            material=Material(delta=delta,beta=beta),energy_kev=cfg.energy_kev)
    maximum=reference.max_element_count()
    external=cfg.n_elements<=maximum
    focal=reference.theoretical_focus(n_elements=cfg.n_elements) if external else None
    image=(focal if cfg.source_type=="plane" else
           reference.effective_focus(cfg.source_distance_m,n_elements=cfg.n_elements)) if external else None
    real=image is not None and np.isfinite(image) and image>0
    magnification=(None,)*cfg.grid_dimension
    if cfg.source_type!="plane" and image is not None and np.isfinite(image):
        lens=reference.with_parameters(n_elements=cfg.n_elements)
        values=[lens.source_magnification(image,cfg.source_distance_m)]
        if cfg.grid_dimension==2:
            values.append(lens.source_magnification(image,cfg.source_distance_m,
                focusing=cfg.geometry=="two-dimensional"))
        magnification=tuple(values)
    return CRLAnalyticalResult(float(delta),float(beta),focal,image,maximum,
        float(reference.element_period*cfg.n_elements),bool(external),bool(real),magnification)


@dataclass
class CRLPlaneResult:
    z_m: float
    intensity: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray | None
    metrics: BeamMetrics
    magnification: tuple[float | None, ...]


class _CRLFocusSession:
    """Reusable GPU/CPU propagation state; no files or full detector stacks."""
    _mode_cache_budget_bytes = 512*1024**2

    def __init__(self, cfg, grid, crl, incident, coherent_state=None, progress=None):
        self.cfg,self.grid,self.crl,self.incident=cfg,grid,crl,incident
        self.xp=grid.xp
        self.exit_field=crl.E() if coherent_state is None else coherent_state.exit_field
        self.spectrum=grid.fft(self.exit_field) if coherent_state is None else coherent_state.spectrum
        self.input_power=_to_python_float(self.xp.sum(abs(incident)**2))
        self.transmission=_to_python_float(self.xp.sum(abs(self.exit_field)**2))/self.input_power
        self._last_z,self._last_image=None,None
        self._peak_cache={}
        self.slices=(slice(None),)*grid.dimension
        self.detector_grid=grid
        self._zoom_plan=None
        self._exit_metrics=None
        self.diagnostics=[]
        self._modes=None
        sigma_factor=1/(2*np.sqrt(2*np.log(2)))
        self.source_sigma=(cfg.source_fwhm_x_m*sigma_factor,cfg.source_fwhm_y_m*sigma_factor)
        self._point_model=cfg.source_type=="gaussian" and cfg.source_method=="points"
        if self._point_model:
            from scipy.special import roots_hermitenorm
            def quadrature(sigma):
                if sigma==0: return np.array([0.0]),np.array([1.0])
                nodes,weights=roots_hermitenorm(cfg.source_points)
                return nodes*sigma,weights/np.sum(weights)
            sx,wx=quadrature(self.source_sigma[0])
            if grid.dimension==1:
                self.positions=np.column_stack((sx,np.zeros_like(sx)))
                self.weights=wx
            else:
                sy,wy=quadrature(self.source_sigma[1])
                xx,yy=np.meshgrid(sx,sy)
                self.positions=np.column_stack((xx.ravel(),yy.ravel()))
                self.weights=(wy[:,None]*wx[None,:]).ravel()
            self.mode_batch=grid._batch_size(len(self.weights),self.spectrum.dtype,cfg.batch_size)
            cache_budget=self._mode_cache_budget_bytes
            if grid.backend=="cupy":
                free,_=cp.cuda.runtime.memGetInfo()
                cache_budget=min(1024*1024**2,int(free*.25))
            needed=len(self.weights)*int(np.prod(grid.shape))*self.spectrum.dtype.itemsize
            self._source_transmission_ready=False
            if needed<=cache_budget:
                if progress is not None:
                    progress.start("source_modes",len(self.weights),
                        f"{grid.backend} · source modes in batches of {self.mode_batch}")
                self._modes=list(self._generate_modes(progress))
            else:
                self.diagnostics.append("source_streaming: Source spectra exceed the RAM/VRAM cache budget; modes are recomputed in bounded batches.")
        elif cfg.source_type=="gaussian":
            self.diagnostics.append("convolution_model: Gaussian intensity convolution assumes a shift-invariant pupil response; use point sampling to include source-dependent clipping and absorption.")
        if not self._point_model:
            exit_intensity=self._blur(abs(self.exit_field)**2,0.0,grid=self.grid)
            self._exit_metrics=_intensity_metrics_batch(exit_intensity[None,...],self.grid,
                (slice(None),)*self.grid.dimension,
                measure_y_fwhm=self.cfg.geometry=="two-dimensional")[0]

    def configure_refined_detector(self,analytic):
        """Resolve the detector peak without globally oversampling the lens plane."""
        cfg=self.cfg
        wavelength,_=wavelength_energy(energy_kev=cfg.energy_kev)
        factor=.886 if cfg.geometry=="one-dimensional" else 1.028
        coherent_width=factor*wavelength*analytic/cfg.aperture_m
        projected=np.zeros(self.grid.dimension)
        if cfg.source_type=="gaussian":
            projected=np.abs(self.magnification(analytic))*np.array(self.source_sigma[:self.grid.dimension])*2*np.sqrt(2*np.log(2))
        widths=[24*max(coherent_width,projected[0])]
        targets=[coherent_width/24]
        if self.grid.dimension==2:
            if cfg.geometry=="two-dimensional":
                widths.append(24*max(coherent_width,projected[1]))
                targets.append(coherent_width/24)
            else:
                widths.append(max(2*cfg.structure_depth_m,12*projected[1]))
                targets.append(min(cfg.structure_depth_m/192,np.sqrt(wavelength*analytic)/24))
        propagation_widths=[self.grid.nx*self.grid.dx]
        if self.grid.dimension==2:
            propagation_widths.append(self.grid.ny*self.grid.dy)
        for axis,(wanted,available) in enumerate(zip(widths,propagation_widths)):
            if wanted>available*.98:
                widths[axis]=available*.98
                self.diagnostics.append("detector_window: Refined detector ROI is limited by the propagation window; enlarge the propagation window at fixed step.")
        counts=[]
        spacings=[]
        for width,target,available in zip(widths,targets,propagation_widths):
            desired=max(257,int(np.ceil(width/target)))
            if desired%2==0: desired+=1
            step=_nice_grid_step(width/desired,"down")
            count=max(257,int(np.ceil(width/step-1e-10)))
            if count%2==0:count+=1
            if count>2049:
                step=_nice_grid_step(width/2049,"up")
                count=int(np.ceil(width/step-1e-10))
                if count%2==0:count+=1
            limit=min(2049,int(np.floor(.98*available/step)))
            if limit%2==0:limit-=1
            count=min(count,limit)
            if count<3:
                raise ValueError("The detector spacing does not fit the propagation window.")
            if count<desired:
                self.diagnostics.append("detector_cap: Refined detector sampling reached its cap; reduce source extent or compare a smaller detector ROI.")
            counts.append(count)
            spacings.append(step)
        detector=(Grid1D(spacings[0],counts[0],backend=self.grid.backend)
                  if self.grid.dimension==1 else Grid2D(spacings[0],spacings[1],counts[0],counts[1],backend=self.grid.backend))
        self.detector_grid=detector
        self._zoom_plan=_DetectorFourierPlan(self.grid,detector)
        self._last_z,self._last_image=None,None
        self._peak_cache.clear()
        self.slices=(slice(None),)*detector.dimension

    def _generate_modes(self,progress=None):
        total_power=0.0
        xp=self.xp
        extra=(None,)*self.grid.dimension
        exit_intensity=(xp.zeros(self.grid.shape,dtype=xp.float64)
                        if self._exit_metrics is None else None)
        for start in range(0,len(self.weights),self.mode_batch):
            stop=min(start+self.mode_batch,len(self.weights))
            points=xp.asarray(self.positions[start:stop])
            phase=points[:,0][(slice(None),*extra)]*self.grid.x
            if self.grid.dimension==2:
                phase=phase+points[:,1][(slice(None),*extra)]*self.grid.y
            fields=self.incident[None,...]*xp.exp(-1j*self.crl.k/self.cfg.source_distance_m*phase)
            fields=self.crl.apply_to_fields(fields)
            weights=xp.asarray(self.weights[start:stop])
            if not self._source_transmission_ready:
                values=abs(fields)**2
                axes=tuple(range(1,fields.ndim))
                total_power+=_to_python_float(xp.sum(weights*xp.sum(values,axis=axes)))
                if exit_intensity is not None:
                    exit_intensity+=xp.sum(values*weights[(slice(None),*extra)],axis=0)
                del values
            spectrum=self.grid._transform_batch(fields)
            del fields,phase
            yield spectrum,weights
            if progress is not None:
                progress.advance(stop)
        if not self._source_transmission_ready:
            self.transmission=total_power/self.input_power
            self._source_transmission_ready=True
        if exit_intensity is not None:
            self._exit_metrics=_intensity_metrics_batch(exit_intensity[None,...],self.grid,
                (slice(None),)*self.grid.dimension,
                measure_y_fwhm=self.cfg.geometry=="two-dimensional")[0]

    def magnification(self,z):
        if self.cfg.source_type=="plane":
            return (None,)*self.grid.dimension
        values=[self.crl.source_magnification(z,self.cfg.source_distance_m)]
        if self.grid.dimension==2:
            values.append(self.crl.source_magnification(z,self.cfg.source_distance_m,
                          focusing=self.cfg.geometry=="two-dimensional"))
        return tuple(values)

    def intensity(self,z):
        _validate_finite("z",z)
        if z<0: raise ValueError("Detector distance must be non-negative.")
        z=float(z)
        if z==self._last_z:
            return self._last_image
        if self._point_model:
            image=self.xp.zeros(self.detector_grid.shape,dtype=self.xp.float64)
            modes=self._modes if self._modes is not None else self._generate_modes()
            extra=(None,)*self.grid.dimension
            for spectra,weights in modes:
                fields=(self.grid._propagate_mode_spectra(spectra,z,self.crl.k) if self._zoom_plan is None else
                        self._zoom_plan.propagate(spectra,z,self.crl.k))
                image+=self.xp.sum(abs(fields)**2*weights[(slice(None),*extra)],axis=0)
                del fields
        else:
            field=(self._zoom_plan.propagate(self.spectrum,z,self.crl.k) if self._zoom_plan is not None else
                   (self.exit_field if z==0 else self.grid._propagate_spectrum(self.spectrum,np.asarray([z]),self.crl.k)[0]))
            image=abs(field)**2
            image=self._blur(image,z)
        self._last_z,self._last_image=z,image
        return image

    def _blur(self,image,z,*,grid=None):
        if self.cfg.source_type!="gaussian":
            return image
        if self.grid.backend=="cupy":
            from cupyx.scipy.ndimage import gaussian_filter
        else:
            from scipy.ndimage import gaussian_filter
        grid=self.detector_grid if grid is None else grid
        magnification=self.magnification(z)
        sigma_x=abs(magnification[0])*self.source_sigma[0]/grid.dx
        sigmas=sigma_x if self.grid.dimension==1 else (
            abs(magnification[1])*self.source_sigma[1]/grid.dy,sigma_x)
        return gaussian_filter(image,sigmas,mode="constant",cval=0.0,truncate=5.0)

    def iter_intensities(self,z_values,progress=None):
        if self._point_model:
            if progress is not None:
                progress.advance(0,f"{self.grid.backend} · {self.mode_batch} source modes per batch; z planes sequential")
            for index,z in enumerate(z_values):
                yield float(z),self.intensity(float(z))
                if progress is not None:
                    progress.advance(index+1)
        elif self._zoom_plan is not None:
            size=self.grid._batch_size(len(z_values),self.spectrum.dtype,self.cfg.batch_size)
            if progress is not None:
                progress.advance(0,f"{self.grid.backend} · up to {size} z planes per batch")
            for start in range(0,len(z_values),size):
                positions=z_values[start:start+size]
                fields=self._zoom_plan.propagate_many(self.spectrum,positions,self.crl.k)
                images=abs(fields)**2
                for z,image in zip(positions,images):
                    yield float(z),self._blur(image,float(z))
                if progress is not None:
                    progress.advance(min(start+size,len(z_values)))
        else:
            for planes,fields in self.grid._iter_spectrum(
                    self.exit_field,self.spectrum,z_values,self.crl.k,self.cfg.batch_size):
                images=abs(fields)**2
                for z,image in zip(z_values[planes],images):
                    yield float(z),self._blur(image,float(z))
                if progress is not None:
                    progress.advance(planes.stop,
                        f"{self.grid.backend} · {planes.stop-planes.start} z planes in the completed batch")

    def objective(self,z):
        z=float(z)
        if z not in self._peak_cache:
            self._peak_cache[z]=_to_python_float(self.xp.max(self.intensity(z)))
        return self._peak_cache[z]

    def metrics(self,image):
        return _intensity_metrics_batch(image[None,...],self.detector_grid,
                       (slice(None),)*self.grid.dimension,
                       measure_y_fwhm=self.cfg.geometry=="two-dimensional")[0]

    def plane(self,z):
        image=self.intensity(z)
        metrics=self.metrics(image)
        x=self.detector_grid.x if self.grid.dimension==1 else self.detector_grid.x_axis
        y=None if self.grid.dimension==1 else self.detector_grid.y_axis
        return CRLPlaneResult(float(z),_to_numpy(image[self.slices]).copy(),
            _to_numpy(x[self.slices[-1]]).copy(),
            None if y is None else _to_numpy(y[self.slices[0]]).copy(),
            metrics,self.magnification(z))


@dataclass
class CRLFocusResult:
    """Display data and a live session for arbitrary z planes.

    numerical_peak maximizes the central coherent source point for Gaussian
    illumination; displayed intensities/widths still include the whole source.
    It is a reference distance, not necessarily the Gaussian source's maximum.
    The finite-source
    analytical_image_distance_m and collimated_focus_m are separate ray-optical
    distances; none is implicitly identified with minimum wave-optical FWHM.
    z_m is a uniform display grid anchored at the numerical peak distance.
    peak_index indexes that exact reference plane. best_intensity,
    reference_metrics and fwhm_at_peak_m refer to the exact numerical distance.
    """
    configuration: CRLFocusConfig
    backend: str
    analytical_image_distance_m: float
    collimated_focus_m: float
    numerical_peak: PeakIntensitySearchResult
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
    grid_shape: tuple[int,...]
    grid_step_y_m: float | None
    detector_shape: tuple[int,...]
    detector_step_m: float
    detector_step_y_m: float | None
    lens_length_m: float
    diagnostics: tuple[str,...]
    _session: _CRLFocusSession
    reference_metrics: BeamMetrics | None = None

    @property
    def peak_z_m(self):
        return self.numerical_peak.z_m

    @property
    def peak_search_basis(self):
        return "plane_wave" if self.configuration.source_type=="plane" else "central_point"

    @property
    def sampled_peak_z_m(self):
        """Highest full-source intensity among the displayed samples only."""
        return float(self.z_m[int(np.argmax(self.peak_intensity))])

    @property
    def fwhm_axes(self):
        """Axes on which focal/exit intensity FWHM is physically reported."""
        return ("x","y") if self.configuration.geometry=="two-dimensional" else ("x",)

    @property
    def analytical_focus_m(self):
        """Legacy alias for the finite-source analytical image distance."""
        return self.analytical_image_distance_m

    @property
    def numerical_focus(self):
        """Legacy alias for the peak-intensity search result."""
        return self.numerical_peak

    @property
    def fwhm_at_peak_m(self):
        """Full-source FWHM at the numerical reference distance."""
        value=np.atleast_1d(self.reference_metrics.fwhm if self.reference_metrics is not None
                          else self.fwhm_m[self.peak_index])
        widths=tuple(None if v is None else _optional_float(v) for v in value)
        return widths[0] if self.configuration.grid_dimension==1 else widths

    @property
    def scan_step_m(self):
        """Spacing of the uniform overview z grid; the optimizer is separate."""
        return float(self.z_m[1]-self.z_m[0])

    @property
    def intensity_reference(self):
        """Common full-source peak for overview and selected-plane display."""
        reference=0.0 if self.reference_metrics is None else self.reference_metrics.max_intensity
        return max(float(np.max(self.peak_intensity)),float(reference))

    @property
    def effective_aperture_m(self):
        """Full-source exit-intensity FWHM on the native propagation grid.

        A scalar for a 1D grid, or (x, y) peak-crossing slice widths for 2D.
        The unfocused y width is None for one-axis focusing on a 2D grid.
        Gaussian point modes are summed incoherently at the exit, before any
        detector zoom/crop. The convolution model applies its native-grid blur
        at z=0. No extra source propagations or detector scans are required.
        This numerical definition is not an absorption-only aperture estimate.
        """
        return self._session._exit_metrics.fwhm

    @property
    def magnification_at_peak(self):
        return self._session.magnification(self.peak_z_m)

    @property
    def magnification_at_focus(self):
        """Legacy alias for source-position scaling at peak intensity."""
        return self.magnification_at_peak

    def plane_at(self,z_m: float) -> CRLPlaneResult:
        """Evaluate any detector z, reusing lens/source state on CPU or GPU."""
        return self._session.plane(z_m)

    def summary(self):
        return {"xposim_version":__version__,"configuration":asdict(self.configuration),
                "backend":self.backend,"analytical_image_distance_m":self.analytical_image_distance_m,
                "collimated_focus_m":self.collimated_focus_m,
                "peak_z_m":self.peak_z_m,"fwhm_at_peak_m":self.fwhm_at_peak_m,
                "peak_search_basis":self.peak_search_basis,
                "axial_roi_basis":"manual" if self.configuration.scan_range_m is not None else self.peak_search_basis,
                "peak_search_initial_z_m":self.numerical_peak.history[0][0],
                "sampled_peak_z_m":self.sampled_peak_z_m,
                "scan_step_m":self.scan_step_m,"scan_grid_uniform":True,
                "scan_anchor_m":self.peak_z_m,
                "fwhm_axes":list(self.fwhm_axes),
                "reference_scan_index":self.peak_index,
                "intensity_reference":self.intensity_reference,
                "magnification_at_peak":self.magnification_at_peak,
                "transmission":self.transmission,"grid_shape":self.grid_shape,
                "effective_aperture_m":self.effective_aperture_m,
                "effective_aperture_convention":"exit_intensity_FWHM",
                "grid_step_m":self.grid_step_m,"grid_step_y_m":self.grid_step_y_m,
                "detector_shape":self.detector_shape,"detector_step_m":self.detector_step_m,
                "detector_step_y_m":self.detector_step_y_m,
                "lens_length_m":self.lens_length_m,"diagnostics":list(self.diagnostics),
                "peak_search_evaluations":self.numerical_peak.evaluations,
                "peak_search_converged":self.numerical_peak.converged,
                "source_width_convention":"FWHM", "distance_reference":"CRL exit",
                "intensity_units":"relative", "convergence_checked":False}


def _display_slice(axis,profile,level,margin=0.12):
    """Bound the significant signal and leave a small transverse display margin."""
    profile=np.asarray(profile)
    maximum=float(np.max(profile))
    significant=np.flatnonzero(profile>=level*maximum) if maximum>0 else np.array([],dtype=int)
    if significant.size==0:
        return slice(None)
    first,last=int(significant[0]),int(significant[-1])
    padding=max(3,int(np.ceil((last-first+1)*margin)))
    return slice(max(0,first-padding),min(len(axis),last+padding+1))


def _fast_grid_size(required):
    """Use small FFT radices supported efficiently by both pocketfft and cuFFT."""
    from scipy.fft import next_fast_len
    n=next_fast_len(int(np.ceil(required)))
    while True:
        remainder=n
        for factor in (2,3,5,7):
            while remainder%factor==0:
                remainder//=factor
        if remainder==1:
            return n
        n=next_fast_len(n+1)


def _nice_grid_step(value,rounding="nearest"):
    """Quantize transverse spacing in SI to a scale-adaptive multiple of five.

    E.g. 31.74 nm becomes 30 nm and 126.95 nm becomes 125 nm.
    The quantum is 5 nm for larger spacings, 0.5 nm below 5 nm, etc.;
    no arbitrary minimum resolution is imposed.
    """
    _validate_positive("grid step",value)
    quantum=5e-9*10.0**min(0,np.floor(np.log10(value/(5e-9))))
    ratio=value/quantum
    integer=(np.floor(ratio+1e-10) if rounding=="down" else
             np.ceil(ratio-1e-10) if rounding=="up" else np.floor(ratio+.5))
    return float(max(1,integer)*quantum)


def _nice_scan_step(value,rounding="nearest"):
    """Preferred axial steps: 1, 2.5, 5 times a power of ten (SI)."""
    _validate_positive("scan step",value)
    scale=10.0**np.floor(np.log10(value))
    candidates=scale*np.array([.5,1.,2.5,5.,10.])
    if rounding=="up":
        return float(candidates[np.flatnonzero(candidates>=value*(1-1e-10))[0]])
    if rounding=="down":
        return float(candidates[np.flatnonzero(candidates<=value*(1+1e-10))[-1]])
    return float(candidates[np.argmin(abs(candidates-value))])


def _focus_axis_sampling(width,required,count,requested_step,cap,minimum):
    """Keep actual spacings readable while covering the full requested window."""
    if requested_step is not None:
        step=_nice_grid_step(requested_step)
        n=_fast_grid_size(max(minimum,np.ceil(width/step-1e-10)))
        manual_cap=65536 if cap==65536 else 4096
        if n>manual_cap:
            raise ValueError(f"Requested propagation spacing needs {n} points on one axis; "
                             f"the {manual_cap}-point spacing-control limit would be exceeded. "
                             "Increase the spacing or use an explicit point count.")
    elif count is not None:
        n=count
        step=_nice_grid_step(width/n,"up")
    else:
        step=_nice_grid_step(width/min(required,cap),"down")
        n=_fast_grid_size(max(minimum,np.ceil(width/step-1e-10)))
        # A small FFT-size margin preserves the rounded-down spacing instead
        # of making e.g. 31.74 nm coarser than 30 nm just to keep N=2048.
        if n>_fast_grid_size(cap*1.1):
            step=_nice_grid_step(width/cap,"up")
            n=_fast_grid_size(max(minimum,np.ceil(width/step-1e-10)))
    return int(n),float(step)


def _uniform_scan_positions(bounds,target_count,requested_step,accuracy,anchor_m):
    """Expand the ROI on z=anchor+k*step, retaining the exact optimizer result.

    Only the spacing is rounded. Absolute coordinates inherit the full
    precision of anchor_m; the numerical reference is a regular lattice point,
    not an extra sample inserted into a zero-anchored grid.
    """
    _validate_finite("scan anchor",anchor_m)
    if anchor_m<0:raise ValueError("The scan anchor must be non-negative.")
    target=(bounds[1]-bounds[0])/(target_count-1)
    step=_nice_scan_step(requested_step) if requested_step is not None else _nice_scan_step(
        target,"down" if accuracy=="refined" else "up")
    def indices(spacing):
        start=max(int(np.ceil(-anchor_m/spacing)),
                  int(np.floor((bounds[0]-anchor_m)/spacing+1e-10)))
        stop=int(np.ceil((bounds[1]-anchor_m)/spacing-1e-10))
        return start,stop
    start,stop=indices(step)
    if requested_step is None:
        while stop-start+1<5:
            step=_nice_scan_step(step*.9,"down")
            start,stop=indices(step)
    count=stop-start+1
    if count<2:
        raise ValueError("The axial grid must contain at least two planes.")
    if count>10001:
        raise ValueError(f"The requested z spacing needs {count} planes; increase it to use at most 10001.")
    positions=anchor_m+(start+np.arange(count,dtype=float))*step
    # Cancelled floating-point roundoff at z=0 must not reach the propagator
    # as a tiny negative distance. This does not round the lattice origin.
    if positions[0]<0 and abs(positions[0])<=8*np.spacing(max(anchor_m,step)):
        positions[0]=0.0
    return positions


def _edge_power_fractions(image,xp):
    total=_to_python_float(xp.sum(image))
    if total<=0:
        return (0.0,)*image.ndim
    rim_x=max(1,image.shape[-1]//20)
    edge_x=xp.sum(image[...,:rim_x])+xp.sum(image[...,-rim_x:])
    fractions=[_to_python_float(edge_x)/total]
    if image.ndim==2:
        rim_y=max(1,image.shape[0]//20)
        fractions.append(_to_python_float(xp.sum(image[:rim_y])+xp.sum(image[-rim_y:]))/total)
    return tuple(fractions)


def _build_focus_session(cfg,backend,parameters,analytic,progress=None):
    """Fit CRL geometry first, resolve its physics, then check propagation edges."""
    wavelength,_=wavelength_energy(energy_kev=cfg.energy_kev)
    factor=.886 if cfg.geometry=="one-dimensional" else 1.028
    target_x=factor*wavelength*analytic/cfg.aperture_m/6
    if cfg.source_type!="plane":
        target_x=min(target_x,wavelength*cfg.source_distance_m/(2*cfg.aperture_m))
    quick_cap=65536 if cfg.grid_dimension==1 else (2048 if backend=="cupy" else 1024)
    cap=65536 if cfg.grid_dimension==1 else (4096 if cfg.accuracy=="refined" else quick_cap)
    k=2*np.pi/wavelength
    delta=parameters['material'].optical_constants(cfg.energy_kev)[0]
    source_curvature=0.0 if cfg.source_type=="plane" else 1/cfg.source_distance_m
    phase_slope=k*delta*cfg.n_elements*cfg.aperture_m/cfg.radius_m+k*cfg.aperture_m*source_curvature/2
    phase_target=(np.pi/3)/phase_slope
    width_x=cfg.window_factor*cfg.aperture_m
    depth=cfg.aperture_m if cfg.geometry=="two-dimensional" else cfg.structure_depth_m
    width_y=None if cfg.grid_dimension==1 else cfg.window_factor*depth
    enlarged=[]
    probe=None
    for attempt in range(4):
        if progress is not None:
            progress.start("grid",detail=f"Window attempt {attempt+1}: preparing the CRL field")
        messages=[]
        required_x=_fast_grid_size(max(256,width_x/target_x))
        if cfg.accuracy=="refined":
            required_x=max(_fast_grid_size(1.25*min(required_x,quick_cap)),_fast_grid_size(width_x/phase_target))
        nx,dx=_focus_axis_sampling(width_x,required_x,cfg.grid_points,cfg.grid_step_x_m,cap,256)
        if cfg.grid_points is None and nx<required_x:
            messages.append("grid_cap: Automatic x sampling reached its memory cap; refine the grid to check spot size.")
        if cfg.grid_dimension==1:
            grid=Grid1D(dx,nx,backend=backend)
        else:
            if cfg.geometry=="two-dimensional":
                target_y=target_x
            else:
                target_y=min(depth/128,wavelength*analytic/(6*depth))
                if cfg.source_type!="plane":
                    target_y=min(target_y,wavelength*cfg.source_distance_m/(2*depth))
            required_y=_fast_grid_size(max(128,width_y/target_y))
            if cfg.accuracy=="refined":
                required_y=_fast_grid_size(1.25*min(required_y,quick_cap))
                if cfg.geometry=="two-dimensional":
                    required_y=max(required_y,_fast_grid_size(width_y/phase_target))
            ny,dy=_focus_axis_sampling(width_y,required_y,cfg.grid_points_y,cfg.grid_step_y_m,cap,128)
            if cfg.grid_points_y is None and ny<required_y:
                messages.append("grid_cap_y: Automatic y sampling reached its memory cap; refine the grid.")
            grid=Grid2D(dx,dy,nx,ny,backend=backend)
        xp=grid.xp
        position=0.0 if grid.dimension==1 else (0.0,0.0)
        incident=(xp.ones(grid.shape,dtype=xp.complex128) if cfg.source_type=="plane" else
                  PointSource(grid=grid,position=position,energy_kev=cfg.energy_kev).E(cfg.source_distance_m))
        lens_parameters=dict(parameters)
        if grid.dimension==1:
            pupil=abs(grid.x)<=cfg.aperture_m/2
        elif cfg.geometry=="one-dimensional":
            pupil=(abs(grid.x)<=cfg.aperture_m/2)&(abs(grid.y)<cfg.structure_depth_m/2)
            lens_parameters["transverse_length"]=cfg.structure_depth_m
        else:
            pupil=grid.x**2+grid.y**2<=(cfg.aperture_m/2)**2
        incident*=pupil
        if _to_python_float(xp.sum(abs(incident)**2))<=0:
            raise ValueError("The entrance aperture is not resolved by the grid.")
        crl=CRL(incident,grid,**lens_parameters)
        probe_cfg=replace(cfg,source_method="convolution") if cfg.source_type=="gaussian" else cfg
        probe=_CRLFocusSession(probe_cfg,grid,crl,incident)
        if not cfg.adaptive_window:
            break
        edge=np.zeros(grid.dimension)
        if progress is not None:
            progress.start("window_checks",3,f"{backend} · propagation grid {grid.shape}")
        for index,z in enumerate((.8*analytic,analytic,1.2*analytic)):
            edge=np.maximum(edge,_edge_power_fractions(probe.intensity(z),xp))
            if progress is not None:
                progress.advance(index+1)
        grow_x=edge[0]>1e-3 and cfg.grid_points is None
        grow_y=grid.dimension==2 and edge[1]>1e-3 and cfg.grid_points_y is None
        if not grow_x and not grow_y:
            break
        if attempt==3:
            messages.append("mesh_boundary: Geometry-fit window expansion reached its limit; check the propagation window.")
            break
        if grow_x:
            width_x*=1.5
            enlarged.append('x')
        if grow_y:
            width_y*=1.5
            enlarged.append('y')
        probe=None
    if enlarged:
        messages.append("mesh_window: Geometry-fit window was expanded along "+', '.join(sorted(set(enlarged)))+" to reduce FFT-edge power.")
    # Return the coherent state without preparing all incoherent source modes.
    # The workflow finds the central-point reference before that costly work.
    return grid,crl,probe,messages


def estimate_crl_focus(configuration: CRLFocusConfig, *,
                       progress: Callable[[CRLProgressUpdate], None] | None = None) -> CRLFocusResult:
    """Find a central-source reference distance, then sample the full source.

    Grid dimensionality and focusing dimensionality are independent. Extended
    Gaussian sources are specified by intensity FWHM and treated incoherently.
    Point sampling propagates Gaussian quadrature modes through the full stack;
    convolution uses a shift-invariant approximation with z-dependent ABCD scale.
    The analytical image distance seeds the search. Gaussian illumination uses
    only its central coherent point for peak search and automatic axial ROI;
    all Gaussian modes/convolution are applied afterwards for displayed planes.
    No calculations are saved to files. The retained session evaluates slider z.
    progress receives synchronous stage updates without threads or timers.
    Unknown adaptive lengths have no percentage or ETA. Sampling ETA uses
    completed batches, preserving GPU batching and bounding progress overhead.
    """
    if not isinstance(configuration,CRLFocusConfig):
        raise TypeError("Pass a CRLFocusConfig instance.")
    reporter=_CRLProgressReporter(progress)
    reporter.start("preparing",detail="Selecting backend and material properties")
    cfg=configuration
    backend=select_backend(cfg.backend)
    material=cfg.create_material()
    parameters=dict(radius=cfg.radius_m,aperture=cfg.aperture_m,
        bridge_thickness=cfg.bridge_thickness_m,n_elements=cfg.n_elements,
        material=material,energy_kev=cfg.energy_kev)
    reference=CRL(grid=Grid1D(cfg.aperture_m,2,backend="numpy"),**parameters)
    if not reference.is_focus_outside_crl():
        raise ValueError("The element count puts the formal focus inside the CRL.")
    analytic=(reference.theoretical_focus() if cfg.source_type=="plane" else
              reference.effective_focus(cfg.source_distance_m))
    if not np.isfinite(analytic) or analytic<=0:
        raise ValueError("This source/lens configuration has no finite real image after the CRL.")
    grid,crl,coherent,messages=_build_focus_session(cfg,backend,parameters,analytic,reporter)
    search_cfg=(replace(cfg,source_type="point",source_fwhm_x_m=0.0,source_fwhm_y_m=0.0)
                if cfg.source_type=="gaussian" else cfg)
    search_session=(_CRLFocusSession(search_cfg,grid,crl,coherent.incident,coherent_state=coherent)
                    if cfg.source_type=="gaussian" else coherent)
    reporter.start("detector",detail="Preparing the observation grid")
    if cfg.accuracy=="refined":
        search_session.configure_refined_detector(analytic)
    xp=grid.xp
    wavelength,_=wavelength_energy(energy_kev=cfg.energy_kev)
    factor=.886 if cfg.geometry=="one-dimensional" else 1.028
    search_bounds=cfg.scan_range_m or (max(1e-9,.05*analytic),3*analytic)
    basis="Plane wave" if cfg.source_type=="plane" else "Central point source"
    reporter.start("peak_search",detail=basis+" · starting at the analytical image distance")
    evaluations=0
    def search_objective(z):
        nonlocal evaluations
        value=search_session.objective(z)
        evaluations+=1
        reporter.advance(evaluations,f"{basis} · evaluating z = {z*1e3:.2f} mm")
        return value
    numerical=find_peak_intensity_distance(search_objective,analytic,bounds_m=search_bounds,
                   tolerance_m=cfg.peak_search_tolerance_m)
    if not numerical.converged:
        messages.append("peak_search: "+numerical.message)
    reporter.start("axial_roi",detail=basis+" · finding the axial display interval")
    roi_evaluations=0
    def roi_objective(z):
        nonlocal roi_evaluations
        value=search_session.objective(z)
        roi_evaluations+=1
        reporter.advance(roi_evaluations)
        return value
    bounds=cfg.scan_range_m or _peak_z_roi(roi_objective,numerical,search_bounds,cfg.axial_roi_level)
    if bounds[1]<=bounds[0]:
        raise ArithmeticError("The peak-intensity ROI has zero extent.")
    if cfg.scan_range_m is None and (bounds[0]==search_bounds[0] or bounds[1]==search_bounds[1]):
        messages.append("z_roi_boundary: The axial ROI reached a search boundary; inspect the displayed range.")
    if cfg.source_type=="gaussian":
        messages.extend(search_session.diagnostics)
        search_session=None  # Release the central detector before source-mode caching.
        reporter.start("source_preparation",detail="Preparing the full Gaussian source after central-point search")
        session=_CRLFocusSession(cfg,grid,crl,coherent.incident,coherent_state=coherent,progress=reporter)
        if cfg.accuracy=="refined":
            reporter.start("detector",detail="Preparing the full Gaussian-source detector")
            session.configure_refined_detector(analytic)
        messages.append("central_source_reference: Peak search and automatic z ROI use the central point; displayed intensities and widths include the full Gaussian source. Its maximum can occur at a different z.")
    else:
        session=search_session
    coherent=None
    detector=session.detector_grid
    messages.extend(session.diagnostics)
    positions=_uniform_scan_positions(bounds,cfg.scan_points,cfg.scan_step_m,cfg.accuracy,numerical.z_m)
    metrics=[]
    lines=[]
    transverse_x=xp.zeros(detector.nx,dtype=xp.float64)
    transverse_y=None if grid.dimension==1 else xp.zeros(detector.ny,dtype=xp.float64)
    reporter.start("sampling",len(positions),"Calculating detector planes and beam metrics")
    for z,image in session.iter_intensities(positions,reporter):
        metric=session.metrics(image)
        metrics.append(metric)
        if grid.dimension==1:
            line=image
            transverse_x=xp.maximum(transverse_x,image)
        else:
            flat=xp.argmax(image)
            j,i=flat//detector.nx,flat%detector.nx
            line=image[detector.ny//2,:]
            transverse_x=xp.maximum(transverse_x,xp.max(image,axis=0))
            transverse_y=xp.maximum(transverse_y,xp.max(image,axis=1))
        lines.append(_to_numpy(line).copy())
    reporter.start("finalizing",detail="Finalizing transverse ROI and beam metrics")
    x_full=_to_numpy(detector.x if grid.dimension==1 else detector.x_axis)
    x_slice=_display_slice(x_full,_to_numpy(transverse_x),cfg.transverse_roi_level)
    y_full=None if grid.dimension==1 else _to_numpy(detector.y_axis)
    y_slice=None if grid.dimension==1 else _display_slice(y_full,_to_numpy(transverse_y),cfg.transverse_roi_level)
    session.slices=(x_slice,) if grid.dimension==1 else (y_slice,x_slice)
    best=session.plane(numerical.z_m)
    best_index=int(np.argmin(abs(positions-numerical.z_m)))
    widths=np.asarray([np.nan if m.fwhm is None else m.fwhm for m in metrics],dtype=float)
    active_count=len(("x","y") if cfg.geometry=="two-dimensional" else ("x",))
    best_widths=np.asarray([np.nan if v is None else v for v in np.atleast_1d(best.metrics.fwhm)[:active_count]],dtype=float)
    steps=np.array([detector.dx] if active_count==1 else [detector.dx,detector.dy])
    if np.any(~np.isfinite(best_widths)):
        messages.append("fwhm_missing: Half-maximum crossings are outside the calculation window.")
    elif np.any(best_widths/steps<6):
        messages.append("sampling: Fewer than six points across a transverse FWHM; refine the grid.")
    slope=crl.k*crl.delta*cfg.n_elements*cfg.aperture_m/cfg.radius_m*grid.dx
    if cfg.source_type!="plane":
        slope+=crl.k*cfg.aperture_m/(2*cfg.source_distance_m)*grid.dx
    if slope>np.pi/2:
        messages.append("phase_sampling: Phase changes too rapidly between x samples; refine the grid.")
    image=session.intensity(numerical.z_m)
    rim_x=max(1,detector.nx//20)
    if grid.dimension==1:
        edge=xp.sum(image[:rim_x])+xp.sum(image[-rim_x:])
    else:
        rim_y=max(1,detector.ny//20)
        edge=(xp.sum(image[:rim_y])+xp.sum(image[-rim_y:])+
              xp.sum(image[rim_y:-rim_y,:rim_x])+xp.sum(image[rim_y:-rim_y,-rim_x:]))
    total=_to_python_float(xp.sum(image))
    if total>0 and _to_python_float(edge)/total>1e-3:
        messages.append(("detector_boundary: Significant power reaches the refined detector ROI edge." if cfg.accuracy=="refined" else "window_boundary: Significant power reaches the numerical window edge; enlarge the window at fixed step."))
    if session.transmission<1e-12:
        messages.append("low_transmission: The transmitted intensity is very small.")
    if cfg.source_type=="gaussian" and cfg.source_method=="points":
        source_sigma=np.array(session.source_sigma[:grid.dimension])
        shift=np.abs(session.magnification(numerical.z_m))*source_sigma
        point_width=factor*wavelength*analytic/cfg.aperture_m
        if np.max(shift)*np.pi/np.sqrt(2*cfg.source_points)>point_width/2:
            messages.append("source_quadrature: Gaussian source nodes may be too sparse; compare with a higher point count.")
    result=CRLFocusResult(cfg,backend,float(analytic),reference.theoretical_focus(),numerical,
        positions,np.asarray([m.max_intensity for m in metrics]),widths,best_index,
        best.x_m,best.y_m,best.intensity,np.stack(lines)[:,x_slice].copy(),
        session.transmission,float(crl.delta),float(crl.beta),grid.dx,grid.shape,
        None if grid.dimension==1 else grid.dy,detector.shape,detector.dx,
        None if grid.dimension==1 else detector.dy,crl.length,tuple(messages),session,best.metrics)
    reporter.finish()
    return result
