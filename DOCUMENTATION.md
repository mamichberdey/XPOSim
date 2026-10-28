# XPOSim documentation

User guide and API reference for **[XPOSim](https://github.com/mamichberdey/XPOSim) 0.5.0**. Installation is described in [README.md](README.md).

This document describes the single-file implementation in the project repository, checked against commit `a68d920239bf4e5cc0ed5dcd4e3ff68315da44b7`. Examples and reference entries distinguish implemented behavior from intended general optical conventions. All code uses `import XPOSim as xpos`.

## Contents

1. [Fields, grids, and optical planes](#1-fields-grids-and-optical-planes)
2. [Complete example: point source, CRL, and detector](#2-complete-example-point-source-crl-and-detector)
3. [Building optical schemes](#3-building-optical-schemes)
4. [Scanning and analysing the focus](#4-scanning-and-analysing-the-focus)
5. [Extended incoherent sources](#5-extended-incoherent-sources)
6. [Surface errors and sampled material objects](#6-surface-errors-and-sampled-material-objects)
7. [One-dimensional and CPU calculations](#7-one-dimensional-and-cpu-calculations)
8. [API reference](#8-api-reference)
9. [Numerical practice and troubleshooting](#9-numerical-practice-and-troubleshooting)
10. [Implementation notes and validation](#10-implementation-notes-and-validation)

## 1. Fields, grids, and optical planes


An optical scheme is built by passing **complex field arrays** between objects. The field contains both amplitude and phase; intensity alone does not contain enough information for further wave propagation.

- `PointSource` creates a field and therefore does not take an incident field.
- Every field-transforming optical element takes an incident complex amplitude through `input_field` and is associated with a calculation `grid`.
- `element.E()` returns the field immediately after the element, including the internal calculation of a CRL stack.
- `element.E(z=distance)` applies the element and then propagates its output by that distance in metres.
- `element.I(z=distance)` returns intensity at the same plane. Equivalently, use `abs(element.E(z=distance))**2`.

Some elements can be constructed without `input_field` to inspect geometry, but their field calculation still requires an incident field. `Material` and grids are supporting objects, rather than field-transforming elements.

Distances are **relative to the current object's output reference plane**, not absolute laboratory coordinates. For a CRL, this is the stack exit plane. A second element must receive the field propagated to its own entrance plane. The library does not assemble a global beamline or infer gaps automatically.

The examples below are sequential: run the setup once and then continue with the subsequent sections. Optional schemes reuse the named objects from that setup.


### Units and array conventions

| Quantity | Convention |
| --- | --- |
| Coordinates, distances, thickness, wavelength | metres |
| Photon energy | keV |
| Density | g/cm³ |
| Rotation and divergence | radians |
| Transverse angular spatial frequency | radians per metre |
| 1D field | `(nx,)` |
| 2D field | `(ny, nx)`; `field[j, i]` corresponds to `(y[j], x[i])` |
| 2D scan stack | `(nz, ny_roi, nx_roi)` |
| 1D ROI | `(xmin, xmax)` |
| 2D ROI | `((xmin, xmax), (ymin, ymax))` |

A grid describes the same transverse window in every propagated plane; XPOSim does not automatically change sampling with distance. Specify exactly one of `energy_kev` and `wavelength_m` for wave calculations. Material-backed objects need energy to calculate tabulated optical constants: if specifying only wavelength, either resolve the energy first or provide direct `delta` and `beta`.

NumPy arrays reside on the CPU, CuPy arrays on the GPU. A grid validates the backend and shape of incident arrays. It does not automatically infer the grid, resample fields, or transfer them to another backend.

## 2. Complete example: point source, CRL, and detector

This script is self-contained when `XPOSim.py` and the material tables are available as described in the README. It uses a 12 keV point source 15 m before a cylindrical silicon CRL and displays a detector plane 35 mm after the lens exit. Run it in the supplied environment; the 2048 × 2048 grid requires GPU memory for multiple arrays and FFT workspace.

```python
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import XPOSim as xpos

DX, DY = 5e-8, 5e-8       # metres: 50 nm sampling
NX, NY = 2**11, 2**11     # 2048 samples along each axis

grid = xpos.Grid2D(
    dx=DX, dy=DY,
    nx=NX, ny=NY,
    backend="cupy",
)

print(f"X window: approximately ±{NX * DX * 1e6 / 2:.3f} µm")
print(f"Y window: approximately ±{NY * DY * 1e6 / 2:.3f} µm")

# CPU coordinate axes for plotting only.
x_um = cp.asnumpy(grid.x_axis) * 1e6
y_um = cp.asnumpy(grid.y_axis) * 1e6

ENERGY_KEV = 12.0
Z0 = 15.0  # source-to-CRL entrance distance, metres

ps = xpos.PointSource(
    grid=grid,
    position=(0.0, 0.0),
    energy_kev=ENERGY_KEV,
)
E0 = ps.E(z=Z0)

N_EL = 26
A = 50e-6
R = 6.25e-6
D = 2e-6

MAT = xpos.Material(formula="Si", density=2.33)

lens_parameters = dict(
    grid=grid,
    rotation=np.pi / 2,
    radius=R,
    aperture=A,
    bridge_thickness=D,
    n_elements=N_EL,
    material=MAT,
    energy_kev=ENERGY_KEV,
    transverse_length=20e-6,
)
crl_ideal = xpos.CRL(input_field=E0, **lens_parameters)

Z_DETECTOR = 35e-3  # distance after the CRL exit, metres

E_exit = crl_ideal.E()
E_detector = crl_ideal.propagate(E_exit, Z_DETECTOR)
I_detector = abs(E_detector)**2

plt.figure(figsize=(6, 5))
plt.pcolormesh(x_um, y_um, cp.asnumpy(I_detector), shading="auto")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title(f"Intensity {Z_DETECTOR * 1e3:.1f} mm after the CRL")
plt.colorbar(label="Intensity (a.u.)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
```

### Understanding the calculation

1. The grid fixes the window and sampling: 50 nm pixels, 2048 pixels per axis, approximately ±51.2 µm. `x_axis` and `y_axis` are 1D axes; `x` and `y` are 2D maps.
2. `ps.E(z=Z0)` creates the complex entrance field. In this implementation it has unit modulus and spherical-wave phase; absolute flux and distance-dependent amplitude normalization are not included.
3. `Material` obtains phase and absorption constants. `density=2.33` is in g/cm³.
4. `CRL` stores the incident field and geometry. At `rotation=np.pi/2` the cylindrical lens focuses along laboratory `y`. Without `transverse_length`, a 2D CRL is rotationally symmetric.
5. `crl_ideal.E()` applies transmission and propagation through the stack. `propagate(E_exit, Z_DETECTOR)` propagates only the saved exit field. `crl_ideal.E(z=Z_DETECTOR)` is equivalent for a single plane, but would recalculate the stack on each call.
6. Only the displayed intensity is copied to the CPU. Fields remain complex on the GPU for further propagation.

The exact lens parameter definitions are in [CRL](#api-crl). With the code's profile convention, the thickness saturates beyond the nominal aperture; the outer region is attenuated material, rather than a hard aperture. Add an explicit aperture if complete blocking is required.

For a notebook one level below the module, add the module's parent directory to `sys.path` **before** importing XPOSim, or launch the notebook from the project root.

## 3. Building optical schemes

The examples in Chapters 3–7 continue the complete example unless stated otherwise. Each alternative scheme creates a new object rather than modifying `crl_ideal`.


### Circular aperture or two-hole diffraction

```python
aperture = xpos.Hole(
    input_field=E0,
    grid=grid,
    center_offset=(0.0, 0.0),
    radius=10e-6,
    energy_kev=ENERGY_KEV,
)
E_aperture_detector = aperture.E(z=0.1)
I_aperture_detector = abs(E_aperture_detector)**2
```

`Hole` implements two symmetric circular openings. With zero offset the openings coincide, creating a single centred aperture. A nonzero offset such as `(15e-6, 0.0)` creates openings centred at opposite coordinates. `radius` is the radius of each opening.

### Chain an aperture and a CRL

```python
GAP = 0.01  # aperture-to-CRL entrance distance, metres
E_lens_input = aperture.E(z=GAP)
crl_after_aperture = xpos.CRL(input_field=E_lens_input, **lens_parameters)
E_final = crl_after_aperture.E(z=Z_DETECTOR)
I_final = abs(E_final)**2
```

This is a separate scheme: the aperture is at `Z0` from the source, and the CRL entrance is at `Z0 + GAP`. `E_lens_input` is the complex amplitude in that entrance plane. Do not pass `abs(E_lens_input)**2` as the next object's input.


Keep the source-to-element entrance distance separate from each subsequent gap. No global beamline position is inferred. All objects in one monochromatic scheme should use the same photon energy and grid.

## 4. Scanning and analysing the focus


### 1. Choose detector positions and an observation region

```python
Z_ARR = np.arange(15, 56, 1) * 1e-3  # 15 through 55 mm, inclusive
ROI_HALF_WIDTH = 3.2e-6
ROI = (
    (-ROI_HALF_WIDTH, ROI_HALF_WIDTH),  # x limits, metres
    (-ROI_HALF_WIDTH, ROI_HALF_WIDTH),  # y limits, metres
)

scan_result = crl_ideal.focus_scan(
    Z_ARR,
    roi=ROI,
    return_fields=True,
)
if not scan_result.valid:
    raise RuntimeError("This CRL has no valid external-focus scan.")

fields = scan_result.E
x_roi = cp.asnumpy(scan_result.x_roi)
y_roi = cp.asnumpy(scan_result.y_roi)
I_dot = abs(fields)**2
```

A 2D ROI has the form `((xmin, xmax), (ymin, ymax))` in **metres**, not pixel indices. Here 3.2 µm is the half-width; the complete ROI is approximately 6.4 µm wide. Set `roi=None` to retain the full window.

The stack shape is **`(number_of_planes, number_of_y_pixels, number_of_x_pixels)`**. Each `I_dot[i]` belongs to `Z_ARR[i]`. The library evaluates the CRL exit field once and propagates it independently to every requested distance; it does not apply the lens again at each plane. ROI cropping occurs after propagation, so it reduces retained data rather than the FFT grid size.

The original full-grid complex128 stack for 41 planes would require about 2.56 GiB, excluding working arrays. A small ROI substantially reduces stored data. Use `return_fields=False` if only beam metrics are required; then `scan_result.E` is `None`.

### 2. Inspect the sampled best-focus plane

```python
best_index = scan_result.peak_index
if best_index is None:
    raise RuntimeError("No valid intensity maximum was found in the scan.")

print(f"Peak-intensity plane: {scan_result.peak_z * 1e3:.3f} mm")
print("FWHM at that plane (metres):", scan_result.fwhm_at_peak)

plt.figure(figsize=(6, 5))
plt.pcolormesh(
    x_roi * 1e6, y_roi * 1e6,
    cp.asnumpy(I_dot[best_index]), shading="auto",
)
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Point-source intensity near focus")
plt.colorbar(label="Intensity (a.u.)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
```

`peak_z` selects the sampled plane with the largest maximum intensity within the ROI. It is not a continuous optimization or necessarily the plane of minimum FWHM. Refine the axial step near the maximum if necessary. In 2D, FWHM is returned as `(FWHM_x, FWHM_y)`; a width may be unavailable if a half-maximum crossing lies outside the observation region.

Analytical estimates can help choose the scan range:

```python
print("Parallel-beam focus from the CRL exit (m):", crl_ideal.theoretical_focus())
print("Finite-distance source image estimate (m):", crl_ideal.effective_focus(Z0))
```

These estimates describe the ideal paraxial CRL model and do not replace inspecting the simulated field.


### Measuring a saved field

```python
metrics = xpos.beam_metrics(E_detector, grid, roi=ROI)
print("Peak:", metrics.max_intensity)
print("Peak coordinates (m):", metrics.center)
print("FWHM (m):", metrics.fwhm)
```

The centre is the coordinate of the maximum sample, not an intensity centroid. In 2D, widths are measured on horizontal and vertical cuts through that peak, not from integrated projections. FWHM can be `None` when the observation interval does not contain both half-maximum crossings. This is common in the nonfocusing direction of a cylindrical lens.

`calculate_divergence=True` additionally measures the full field's angular-spectrum widths; ROI cropping does not restrict the divergence calculation.

## 5. Extended incoherent sources


A point-source calculation describes one coherent illumination mode. For mutually incoherent source points, **intensities add**. If changing source position mainly translates the detector pattern without changing its shape, a finite source can be approximated by convolving the point-source **intensity** with a projected source distribution. This is a shift-invariance approximation, not a general partial-coherence solver; see the [MIT notes on incoherent imaging](https://live.ocw.mit.edu/courses/2-71-optics-spring-2014/ecd8b4c1c7717a31e961837de299879b_MIT2_71S14_lec25_notes.pdf).

### 1. Define the Gaussian source size

```python
FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))
SIG_X = 1.03e-3 / FWHM_TO_SIGMA  # source-plane horizontal sigma, metres
SIG_Y = 1.17e-4 / FWHM_TO_SIGMA  # source-plane vertical sigma, metres
DETECTOR_SIGMA = 2e-7           # optional independent detector blur sigma
```

The two source sizes above are interpreted as **FWHM**, then converted to standard deviations. If your measurements already give sigma, do not divide them again. `DETECTOR_SIGMA` is an additional model assumption; use zero if no detector blur is intended.

### 2. Implement Gaussian intensity convolution on the GPU

```python
def blur_gauss(image, sig_x, sig_y, dx, dy):
    qx = 2 * cp.pi * cp.fft.fftfreq(image.shape[1], d=dx)
    qy = 2 * cp.pi * cp.fft.fftfreq(image.shape[0], d=dy)
    transfer = cp.exp(-0.5 * (
        (qx[None, :] * sig_x)**2 + (qy[:, None] * sig_y)**2
    ))
    blurred = cp.fft.ifft2(cp.fft.fft2(image) * transfer).real
    return cp.maximum(blurred, 0)  # remove tiny negative roundoff values
```

Use CuPy for operations on GPU arrays, including frequency grids and exponentials. The transfer function is one at zero frequency, corresponding to a normalized Gaussian kernel. FFT convolution uses periodic boundaries; enlarge the window if the beam or its blurred tails approach the edges.

### 3. Blur on the full window, then crop

For the cylindrical CRL with `rotation=np.pi/2`, use free-propagation scaling in the nonfocusing `x` direction and the CRL analytical magnification estimate in the focusing `y` direction:

$$
\sigma_{x,\mathrm{source}}(z)\approx\sigma_{x,0}\frac{z}{Z_0},
\qquad
\sigma_{y,\mathrm{source}}(z)\approx\sigma_{y,0}|M(Z_0,z)|.
$$

The first expression neglects the small stack length relative to the propagation distances. For independent Gaussian source and detector blur, variances add:

$$
\sigma_{\mathrm{total}}=\sqrt{\sigma_{\mathrm{source}}^2+\sigma_{\mathrm{detector}}^2}.
$$

Adding sigma values directly, as in `sigma_source + detector_sigma`, defines a different empirical model. The code below uses independent Gaussian broadening.

```python
# Build contiguous crop slices matching the earlier ROI.
x_axis = cp.asnumpy(grid.x_axis)
y_axis = cp.asnumpy(grid.y_axis)
ix = np.flatnonzero((x_axis >= ROI[0][0]) & (x_axis <= ROI[0][1]))
iy = np.flatnonzero((y_axis >= ROI[1][0]) & (y_axis <= ROI[1][1]))
xs = slice(int(ix[0]), int(ix[-1]) + 1)
ys = slice(int(iy[0]), int(iy[-1]) + 1)

I_blurred = cp.empty(I_dot.shape, dtype=cp.float64)

# E_exit was calculated in the single-detector example.
for i, z in enumerate(Z_ARR):
    magnification = abs(float(crl_ideal.magnification(Z0, float(z))))
    sigma_x = np.hypot(SIG_X * z / Z0, DETECTOR_SIGMA)
    sigma_y = np.hypot(SIG_Y * magnification, DETECTOR_SIGMA)

    E_plane = crl_ideal.propagate(E_exit, float(z))
    I_full = abs(E_plane)**2
    I_full_blurred = blur_gauss(I_full, sigma_x, sigma_y, DX, DY)
    I_blurred[i] = I_full_blurred[ys, xs]
```

Blurring only the small stored ROI would discard contributions from outside it and periodically connect its opposite edges. Here each full frame is blurred before cropping, while only the cropped intensity stack is retained. Confirm that the full computational window is sufficiently large.

For `rotation=0`, interchange the focusing and nonfocusing axis models. For a rotationally symmetric lens, both transverse directions are focusing directions. An arbitrary lens rotation requires rotating the source-blur covariance or using a more general source integration.

The analytical magnification is an approximation for this model; off-axis aberrations, vignetting, and large source sizes may invalidate simple convolution. A more general calculation samples source positions, propagates each source field through the complete scheme, and sums the resulting intensities with source weights. Coherent source components require adding complex fields instead.


## 6. Surface errors and sampled material objects

### Sampled surface error


The following optional example uses an unrotated cylindrical CRL so that the sampled map and lens axes are aligned:

```python
error_x = np.linspace(-A / 2, A / 2, 101)
error_y = np.linspace(-20e-6 / 2, 20e-6 / 2, 51)
error_xx, error_yy = np.meshgrid(error_x, error_y)
height_error = 20e-9 * np.cos(2 * np.pi * error_xx / A)

real_parameters = dict(lens_parameters)
real_parameters["rotation"] = 0.0
crl_real = xpos.CRL(
    input_field=E0,
    surface_error=[height_error, error_x, error_y],
    **real_parameters,
)
```

Heights and coordinates are in metres. A 2D grid requires a 2D error map with shape `(len(error_y), len(error_x))`; a 1D error profile is not accepted on `Grid2D`. The map is linearly interpolated onto the calculation grid and is zero outside its supplied coordinate range.

`surface_error` is added to the sag before the total thickness `2 * sag + bridge_thickness` is formed, so its contribution to total thickness is **twice** the supplied height error. The final thickness must remain positive. In this interface the same map is reused for each lens element. When comparing an ideal and an imperfect lens, keep all other parameters, including rotation, identical.


In the current implementation, error interpolation uses the laboratory grid coordinates directly. Rotating the lens does not rotate the interpolated map automatically, even though the aperture mask is evaluated in lens-local coordinates. The unrotated example avoids this ambiguity. To use a rotated measured map, explicitly resample it into laboratory coordinates first.

### A material object from a thickness map

```python
thickness = np.full((len(error_y), len(error_x)), 2e-6)
material_object = xpos.ArrayObject(
    input_field=E0, grid=grid,
    thickness=[thickness, error_x, error_y],
    material=MAT, energy_kev=ENERGY_KEV,
)
E_object_detector = material_object.E(z=Z_DETECTOR)
```

A thickness map gives the total material path length, not a one-surface error. The map is interpolated onto the calculation grid. **Current field behavior blocks zero-thickness points and points outside the supplied map**, rather than transmitting vacuum there. This distinction is important for masks or objects surrounded by free space. See [ArrayObject](#api-arrayobject).

## 7. One-dimensional and CPU calculations

### One-dimensional optical scheme


```python
grid_1d = xpos.Grid1D(dx=DX, nx=NX, backend="cupy")
source_1d = xpos.PointSource(
    grid=grid_1d, position=0.0, energy_kev=ENERGY_KEV,
)
crl_1d = xpos.CRL(
    input_field=source_1d.E(z=Z0),
    grid=grid_1d,
    radius=R, aperture=A, bridge_thickness=D,
    n_elements=N_EL, material=MAT, energy_kev=ENERGY_KEV,
)
I_1d = crl_1d.I(z=Z_DETECTOR)
```

Use 1D when the physics can be reduced to one transverse coordinate. Its fields have shape `(NX,)` and its focus-scan ROI is `(xmin, xmax)`. It is not a substitute for a 2D calculation when coupling between transverse directions matters.


### CPU backend

For CPU calculations, construct grids with `backend="numpy"`, create arrays with NumPy, and plot NumPy results directly. For example, this separate minimal scheme requires no GPU or material database:

```python
import numpy as np
import XPOSim as xpos

grid_cpu = xpos.Grid1D(dx=1e-7, nx=1024, backend="numpy")
source_cpu = xpos.PointSource(grid=grid_cpu, position=0.0, energy_kev=12.0)
field_cpu = source_cpu.E(z=15.0)
aperture_cpu = xpos.Hole(
    input_field=field_cpu, grid=grid_cpu,
    center_offset=0.0, radius=10e-6, energy_kev=12.0,
)
intensity_cpu = aperture_cpu.I(z=0.1)
```

CuPy availability is detected at import time. A successful CuPy import selects the default CuPy grid backend even if GPU execution later fails; request NumPy explicitly when CPU execution is intended.

## 8. API reference

Reference entries use the structure Parameters, Returns, Attributes, Raises/Warns, Notes, Examples, and See also where applicable. Signatures are taken from the supplied implementation. Required parameters have no default; `*` marks keyword-only arguments. `Array` below means a NumPy or CuPy ndarray with the backend and shape required by its grid.

Inherited field methods are documented once under [OpticalObject and OpticalElement](#api-opticalobject), and grid transform methods once under [grid methods](#api-grid-methods). Concrete element entries inherit those methods unless an override is described.

### API index

| Area | Entries |
| --- | --- |
| Configuration | `__version__`, `CUPY_AVAILABLE` |
| Grids | `Grid1D`, `Grid2D`, transforms and propagation |
| Material | `Material`, `wavelength_energy` |
| Optical objects | `OpticalObject`, `OpticalElement`, `PointSource`, `Hole`, `Blade`, `ArrayObject`, `CRL` |
| Defects and maps | `SurfaceError`, `ThicknessDefect`, `FunctionThicknessDefect`, `InterpolatedThicknessDefect` |
| Analysis | `BeamMetrics`, `FocusScanResult`, `intensity`, `full_width_half_max`, `beam_metrics`, `divergence` |
| Filtering | `gaussian_filter_1d`, `gaussian_filter_2d` |

### Module constants

`__version__`: string, `"0.5.0"` in this file. `CUPY_AVAILABLE`: bool indicating whether the CuPy import succeeded. It does not confirm a usable device or compatible driver.

<a id="api-wavelength_energy"></a>

### `wavelength_energy`

Resolve wavelength and energy from exactly one input.

```text
wavelength_energy(*, wavelength_m: float | None = None, energy_kev: float | None = None) -> tuple[float, float]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Exactly one of the two arguments is required. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m.  Exactly one of the two arguments is required. |

**Returns**

tuple[float, float]: (wavelength_m, energy_kev).

**Raises / Warns**

ValueError when both/neither inputs are supplied or the supplied value is nonpositive.

**Notes**

Uses hc = 12.3984e-10 keV m. Unlike geometry-only object construction, this function always requires exactly one input.

**Examples**

```python
wavelength, energy = xpos.wavelength_energy(energy_kev=12.0)
```

**See also:** [`Material`](#api-material); [`PointSource`](#api-pointsource).

<a id="api-material"></a>

### `Material`

Optical constants from chemical composition and density, or from direct values.

```text
Material(formula: str | None = None, density: float | None = None, *, database: _MaterialDatabase | None = None, delta: float | None = None, beta: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `formula` | str or None | `None` | Chemical formula, e.g. "Si". Supported syntax: element symbols followed by optional numeric stoichiometric counts (use physically valid positive counts); parentheses and group syntax are not parsed. Density is required when formula is supplied. |
| `density` | float or None | `None` | Positive material density in g/cm³; required with formula. |
| `database` | database object or None | `None` | Optional internal material-database provider with f1f2(), compton_coefficients(), and atomic_mass(). None uses xray_material_data next to the module. No public database factory is exported. |
| `delta` | float or None | `None` | Refractive-index decrement. Supply together with beta to bypass tables. Material stores it as given; CRL and ArrayObject require delta > 0. |
| `beta` | float or None | `None` | Absorption index. Supply together with delta. Material stores it as given; CRL and ArrayObject require beta >= 0. |

**Returns**

Material storing formula and density; use optical_constants() to obtain (delta, beta).

**Raises / Warns**

ValueError if delta/beta are supplied incompletely, neither material mode is supplied, density is absent with formula, or supplied density is nonpositive. Table errors arise when constants are evaluated.

**Notes**

Supplying both direct constants bypasses tables. If formula is also supplied, density remains required. Direct constants are energy-independent values as supplied by the caller.

**Examples**

```python
mat = xpos.Material(delta=3e-6, beta=1e-8)
```

**See also:** [`Material.optical_constants`](#api-material-optical_constants); [`Material.composition`](#api-material-composition); [`CRL`](#api-crl).

<a id="api-material-composition"></a>

### `Material.composition`

Parse chemical composition.

```text
Material.composition(self) -> list[tuple[str, float]]
```

**Parameters:** none.

**Returns**

list[tuple[str, float]] of (element, stoichiometric count).

**Raises / Warns**

ValueError for direct-constant materials without formula or unsupported formula syntax.

**Notes**

Parentheses and grouped chemical formulae are unsupported. Commas are normalized to decimal points.

**Examples**

```python
composition = xpos.Material(formula="SiO2", density=2.2).composition()
```

**See also:** [`Material`](#api-material).

<a id="api-material-optical_constants"></a>

### `Material.optical_constants`

Calculate the refractive-index decrement and absorption index.

```text
Material.optical_constants(self, energy_kev: float) -> tuple[float, float]
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `energy_kev` | float or None | `required` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

tuple[float, float]: (delta, beta).

**Raises / Warns**

ValueError for nonpositive energy in table mode or energy outside any required f1/f2 range; FileNotFoundError for missing tables. Malformed/incomplete tables may also raise parsing errors.

**Notes**

Uses linear interpolation of tabulated f1/f2 and a Compton correction. Direct mode returns the stored pair without energy conversion or table access.

**Examples**

```python
delta, beta = xpos.Material(delta=3e-6, beta=1e-8).optical_constants(12.0)
```

**See also:** [`Material`](#api-material); [`wavelength_energy`](#api-wavelength_energy).

<a id="api-grid1d"></a>

### `Grid1D`

Uniform one-dimensional transverse grid.

```text
Grid1D(dx: float, nx: int, backend: str = 'cupy' if CUPY_AVAILABLE else 'numpy') -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dx` | float | `required` | Positive x sampling interval, metres. |
| `nx` | int | `required` | Number of x samples; at least 2. |
| `backend` | str | `CuPy if importable; else NumPy` | "numpy" or "cupy" (case-insensitive). Default is "cupy" if CuPy imported successfully, otherwise "numpy". Fixed at module import, not selected by a device test. |

**Returns**

Grid1D with x, qx, dqx, shape=(nx,), dimension=1, backend, and xp.

**Raises / Warns**

ValueError for nonpositive dx, nx < 2, or an unknown backend; RuntimeError when CuPy was requested but is unavailable.

**Notes**

Coordinates use (arange(nx) - (nx-1)/2)*dx. Even grids straddle zero. qx spacing is 2*pi/(nx*dx).

**Examples**

```python
g = xpos.Grid1D(dx=1e-7, nx=1024, backend="numpy")
```

**See also:** [`Grid2D`](#api-grid2d); grid methods; Chapter 7.

<a id="api-grid2d"></a>

### `Grid2D`

Uniform two-dimensional transverse grid.

```text
Grid2D(dx: float, dy: float, nx: int, ny: int, backend: str = 'cupy' if CUPY_AVAILABLE else 'numpy') -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dx` | float | `required` | Positive x sampling interval, metres. |
| `dy` | float | `required` | Positive y sampling interval, metres. |
| `nx` | int | `required` | Number of x samples; at least 2. |
| `ny` | int | `required` | Number of y samples; at least 2. |
| `backend` | str | `CuPy if importable; else NumPy` | "numpy" or "cupy" (case-insensitive). Default is "cupy" if CuPy imported successfully, otherwise "numpy". Fixed at module import, not selected by a device test. |

**Returns**

Grid2D with x/y maps, x_axis/y_axis, qx/qy maps, qx_axis/qy_axis, dqx/dqy, shape=(ny,nx), dimension=2, backend, and xp.

**Raises / Warns**

ValueError for nonpositive sampling, counts below 2, or an unknown backend; RuntimeError if CuPy is unavailable.

**Notes**

Axis coordinates follow the same centred convention as Grid1D. A 2D field uses (y, x) array order.

**Examples**

```python
g = xpos.Grid2D(dx=1e-7, dy=1e-7, nx=128, ny=128, backend="numpy")
```

**See also:** [`Grid1D`](#api-grid1d); grid methods; Chapter 2.

<a id="api-grid-methods"></a>
### Grid methods and attributes

Both grid classes expose the following methods. Parameters `field`, `field_a`, and `field_b` must be ndarrays with the grid's backend and exact shape. All return arrays on that backend. Spatial transforms use the library's centred, continuously scaled convention; do not replace one side with an unshifted raw FFT without matching its phases and normalization.

| Signature | Parameters and returned value | Validation / behavior |
| --- | --- | --- |
| `grid.fft(field)` | Complex spectrum; scaled by dx in 1D or dx*dy in 2D | TypeError for backend mismatch; ValueError for shape mismatch |
| `grid.ifft(field)` | Inverse transform, with reciprocal spatial scaling | Same array validation |
| `grid.propagate(field, z, k)` | z in metres; positive k in rad/m. Returns ifft(fft(field)*H) | z=0 returns a copy. Unlike optical-object wrappers, this low-level method does not reject negative z or validate positive k |
| `grid.convolve(field_a, field_b)` | ifft(fft(a)*fft(b)); same output shape | Both arrays validated; periodic convolution |
| `grid.xp` | Read-only property: numpy or cupy module | No arguments |

`Grid1D` attributes: `dx`, `nx`, `backend`, `x`, `qx`, `dqx`, `shape`, `dimension`. `Grid2D` adds `dy`, `ny`, `y`, `x_axis`, `y_axis`, `qy`, `qx_axis`, `qy_axis`, `dqy`. Coordinate and reciprocal-coordinate arrays use float64 and the grid backend.

**Example:**

```python
g_small = xpos.Grid1D(dx=1e-7, nx=64, backend="numpy")
a_small = np.ones(g_small.shape, dtype=np.complex128)
a_recovered = g_small.ifft(g_small.fft(a_small))
a_propagated = g_small.propagate(a_small, z=0.1, k=2*np.pi/1e-10)
```

**See also:** Chapter 1; `OpticalObject.propagate`.

<a id="api-opticalobject"></a>

### `OpticalObject`

Abstract base for sources and field-transforming objects.

```text
OpticalObject(grid: BaseGrid, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

An abstract object; instantiate concrete subclasses instead. Attributes: grid, wavelength_m, energy_kev, k, angular_frequency; optical values are None in geometry-only mode. c = 299792458 m/s.

**Raises / Warns**

TypeError on direct instantiation; wavelength/energy validation follows wavelength_energy when optical data are supplied.

**Notes**

Abstract E() is the field interface. Shared convenience methods are listed below.

**See also:** [`OpticalElement`](#api-opticalelement); [`PointSource`](#api-pointsource).

<a id="api-opticalelement"></a>

### `OpticalElement`

Abstract base for elements consuming an incident field.

```text
OpticalElement(input_field: Any | None = None, grid: BaseGrid | None = None, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `input_field` | Array or None | `None` | Incident complex amplitude at the element entrance. Must match grid backend and shape. None allows some geometry-only operations; E() requires a field. |
| `grid` | Grid1D or Grid2D | `None` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

Abstract element storing input_field and inherited optical attributes.

**Raises / Warns**

TypeError for missing grid or backend mismatch; ValueError for shape mismatch. Direct instantiation is prohibited by the abstract method.

**Notes**

Geometry-only construction does not guarantee that every inspection method works without an incident array; Hole/Blade masks use the input dtype.

**See also:** [`OpticalObject`](#api-opticalobject); [`Hole`](#api-hole); [`ArrayObject`](#api-arrayobject); [`CRL`](#api-crl).

<a id="api-opticalobject-e"></a>

### `OpticalObject.E`

Abstract field interface.

```text
OpticalObject.E(self, z: float = 0.0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Complex field on the object grid.

**Raises / Warns**

Abstract method: behavior is defined by a concrete source or element.

**Notes**

For concrete elements zero z means exit plane; PointSource requires positive z.

**See also:** [`OpticalElement.E`](#api-opticalelement-e); [`PointSource.E`](#api-pointsource-e).

<a id="api-opticalelement-e"></a>

### `OpticalElement.E`

Apply the complete optical element and optionally propagate its output.

```text
OpticalElement.E(self, z: float = 0.0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Complex field with grid.shape and grid backend.

**Raises / Warns**

ValueError for z < 0, missing incident field, or missing optical wavelength for propagation. Material-backed subclasses may require optical constants even at z=0.

**Notes**

This method recalculates the element on each call. For many observation planes save E() once and call propagate() on the saved field.

**See also:** [`OpticalObject.propagate`](#api-opticalobject-propagate); [`CRL.focus_scan`](#api-crl-focus_scan).

<a id="api-opticalobject-propagate"></a>

### `OpticalObject.propagate`

Propagate a supplied field without applying the object.

```text
OpticalObject.propagate(self, field: Any, z: float) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |
| `z` | float | `required` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Propagated field on the same grid; at z=0 a copy is returned.

**Raises / Warns**

ValueError for z < 0, unknown wavelength, or shape mismatch; TypeError for unsupported array/backend mismatch.

**Notes**

Distances are relative to the plane of the supplied field. This does not use or replace input_field.

**See also:** grid.propagate; [`OpticalElement.E`](#api-opticalelement-e).

<a id="api-opticalobject-i"></a>

### `OpticalObject.I`

Evaluate the intensity of E(z).

```text
OpticalObject.I(self, z: float = 0.0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Real intensity array, shape grid.shape, same backend.

**Raises / Warns**

Propagates errors from the concrete E() implementation.

**Notes**

Equivalent to abs(self.E(z=z))**2. No automatic peak normalization or absolute flux calibration is applied.

**See also:** [`intensity`](#api-intensity); [`OpticalObject.intensity`](#api-opticalobject-intensity).

<a id="api-opticalobject-intensity"></a>

### `OpticalObject.intensity`

Descriptive alias of I(z).

```text
OpticalObject.intensity(self, z: float = 0.0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Same intensity array as I(z).

**Raises / Warns**

Propagates errors from I()/E().

**Notes**

The alias recalculates E(z) through I(z).

**See also:** [`OpticalObject.I`](#api-opticalobject-i); [`intensity`](#api-intensity).

<a id="api-opticalobject-divergence"></a>

### `OpticalObject.divergence`

Evaluate the angular-spectrum FWHM of E(z).

```text
OpticalObject.divergence(self, z: float = 0.0) -> float | tuple[float, float] | None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

float in 1D; (x_width, y_width) in 2D; None or None components when crossings are unavailable.

**Raises / Warns**

ValueError when wavelength is unknown; other field errors propagate from E().

**Notes**

Returns radians; computes E(z) first. See divergence() for the algorithm.

**See also:** [`divergence`](#api-divergence); [`beam_metrics`](#api-beam_metrics).

<a id="api-opticalelement-clear_cache"></a>

### `OpticalElement.clear_cache`

Base cache-clearing hook.

```text
OpticalElement.clear_cache(self) -> None
```

**Parameters:** none.

**Returns**

None.

**Raises / Warns**

No explicit exception in the base implementation.

**Notes**

The base method does nothing; CRL overrides it to clear thickness/transmission caches.

**See also:** [`CRL.clear_cache`](#api-crl-clear_cache).

<a id="api-pointsource"></a>

### `PointSource`

Generate a paraxial spherical-wave phase profile.

```text
PointSource(grid: BaseGrid, position: float | tuple[float, float] = 0.0, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `position` | float or tuple | `0.0` | Transverse source position, metres: scalar in 1D, (x0, y0) in 2D. For 2D pass a tuple explicitly; the scalar default is not accepted. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

PointSource with x0 and y0 (y0=None in 1D) and inherited optical attributes.

**Raises / Warns**

ValueError for an incompatible position dimensionality; missing/invalid optical inputs fail when needed.

**Notes**

In 2D position must be (x0, y0). Source E() has unit modulus; it omits distance-dependent amplitude normalization.

**Examples**

```python
src = xpos.PointSource(grid=g, position=(0.0, 0.0), energy_kev=12.0)
```

**See also:** [`PointSource.E`](#api-pointsource-e); Chapter 2.

<a id="api-pointsource-e"></a>

### `PointSource.E`

Create the source field at a positive distance.

```text
PointSource.E(self, z: float = 0.0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z` | float | `0.0` | Propagation distance, metres. Optical-object methods reject negative z; zero means the exit plane. PointSource.E requires z > 0. Low-level grid propagation evaluates signed z without this validation. |

**Returns**

Complex phase array exp(i*k*r_transverse**2/(2*z)) with grid.shape and backend.

**Raises / Warns**

ValueError for z <= 0 or unspecified wavelength/energy.

**Notes**

This source does not consume input_field. It represents neither a singular source-plane array nor an absolutely normalized propagator.

**See also:** [`PointSource`](#api-pointsource); [`OpticalObject.I`](#api-opticalobject-i).

<a id="api-hole"></a>

### `Hole`

One opening at zero offset or two symmetric hard openings otherwise.

```text
Hole(input_field: Any | None = None, grid: BaseGrid | None = None, center_offset: float | tuple[float, float] | None = None, radius: float | None = None, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `input_field` | Array or None | `None` | Incident complex amplitude at the element entrance. Must match grid backend and shape. None allows some geometry-only operations; E() requires a field. |
| `grid` | Grid1D or Grid2D | `None` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `center_offset` | float or tuple or None | `None` | Required symmetric opening offset, metres: scalar x0 in 1D or (x0, y0) in 2D. Openings lie at opposite positions. Zero offset makes a single opening. |
| `radius` | float | `None` | Positive opening radius, metres. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

Hole element; radius and center_offset are stored.

**Raises / Warns**

TypeError when grid, center_offset, or radius is missing; ValueError for nonpositive radius or incompatible offset dimensionality.

**Notes**

In 1D openings are intervals; in 2D they are circles. Overlap remains a binary union. Inherits E/I/propagate.

**See also:** [`Hole.transmission_mask`](#api-hole-transmission_mask); Chapter 3.

<a id="api-hole-transmission_mask"></a>

### `Hole.transmission_mask`

Return the binary union of the opening masks.

```text
Hole.transmission_mask(self) -> Any
```

**Parameters:** none.

**Returns**

Array of zeros/ones, grid.shape, grid backend, input_field dtype.

**Raises / Warns**

Requires input_field for its dtype; geometry-only mask access may raise AttributeError.

**Notes**

The mask itself does not propagate radiation.

**See also:** [`Hole`](#api-hole); [`OpticalElement.E`](#api-opticalelement-e).

<a id="api-blade"></a>

### `Blade`

Block the negative half-axis in 1D or everything except the first quadrant in 2D.

```text
Blade(input_field: Any | None = None, grid: BaseGrid | None = None, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `input_field` | Array or None | `None` | Incident complex amplitude at the element entrance. Must match grid backend and shape. None allows some geometry-only operations; E() requires a field. |
| `grid` | Grid1D or Grid2D | `None` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

Blade element using the OpticalElement constructor.

**Raises / Warns**

Shared element construction errors; a mask requires input_field dtype.

**Notes**

This is an ideal binary mask, not a configurable material attenuation model or a movable arbitrary knife edge.

**See also:** [`Blade.transmission_mask`](#api-blade-transmission_mask); [`Hole`](#api-hole).

<a id="api-blade-transmission_mask"></a>

### `Blade.transmission_mask`

Return the fixed blade transmission mask.

```text
Blade.transmission_mask(self) -> Any
```

**Parameters:** none.

**Returns**

Array: x >= 0 in 1D; (x >= 0) AND (y >= 0) in 2D, input_field dtype.

**Raises / Warns**

Requires input_field; without it mask inspection can raise AttributeError.

**Notes**

No edge-position or rotation parameter is supplied.

**See also:** [`Blade`](#api-blade).

<a id="api-arrayobject"></a>

### `ArrayObject`

Homogeneous-material object from a sampled total-thickness map.

```text
ArrayObject(input_field: Any | None = None, grid: BaseGrid | None = None, thickness: Sequence[Any] | None = None, material: Material | None = None, *, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `input_field` | Array or None | `None` | Incident complex amplitude at the element entrance. Must match grid backend and shape. None allows some geometry-only operations; E() requires a field. |
| `grid` | Grid1D or Grid2D | `None` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `thickness` | Array or sequence | `None` | A total thickness array for defect.apply; for ArrayObject construction, [values, x] or [values, x, y] in metres. ArrayObject forbids negative samples. |
| `material` | Material | `None` | Required material supplying delta and beta. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

ArrayObject with material, delta, beta, and inherited field attributes.

**Raises / Warns**

TypeError for missing grid/thickness/material; ValueError for negative thickness or nonphysical constants; UserWarning if map coordinates extend outside the grid. Material/map errors propagate.

**Notes**

Input map follows SurfaceError data layout. Field calculation multiplies by a positive-thickness mask: zero thickness and out-of-map regions are opaque. Constructor with wavelength only does not derive energy for material lookup.

**See also:** [`ArrayObject.thickness_map`](#api-arrayobject-thickness_map); [`ArrayObject.transmission`](#api-arrayobject-transmission); Chapter 6.

<a id="api-arrayobject-thickness_map"></a>

### `ArrayObject.thickness_map`

Interpolate the supplied thickness onto the calculation grid.

```text
ArrayObject.thickness_map(self) -> Any
```

**Parameters:** none.

**Returns**

Nonnegative real thickness array, metres, grid.shape and backend.

**Raises / Warns**

ValueError if interpolated thickness becomes negative; interpolation errors propagate.

**Notes**

Outside the supplied map thickness is zero.

**See also:** [`ArrayObject`](#api-arrayobject); [`SurfaceError.on_grid`](#api-surfaceerror-on_grid).

<a id="api-arrayobject-transmission"></a>

### `ArrayObject.transmission`

Compute the material transmission factor from thickness.

```text
ArrayObject.transmission(self) -> Any
```

**Parameters:** none.

**Returns**

Complex array exp(-1j*k*(delta-1j*beta)*thickness).

**Raises / Warns**

ValueError if wavelength or optical constants are unknown.

**Notes**

This returns material transmission only. The additional zero-thickness blocking mask is applied by E(), so E() is not input_field*transmission() at zero-thickness pixels.

**See also:** [`ArrayObject`](#api-arrayobject); [`OpticalElement.E`](#api-opticalelement-e).

<a id="api-crl"></a>

### `CRL`

Compound refractive lens with one or two transverse dimensions.

```text
CRL(input_field: Any | None = None, grid: BaseGrid | None = None, *, radius: float, aperture: float, bridge_thickness: float, n_elements: int, material: Material, transverse_length: float | None = None, rotation: float = 0.0, surface_error: Sequence[Any] | None = None, reuse_element_geometry: bool = True, wavelength_m: float | None = None, energy_kev: float | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `input_field` | Array or None | `None` | Incident complex amplitude at the element entrance. Must match grid backend and shape. None allows some geometry-only operations; E() requires a field. |
| `grid` | Grid1D or Grid2D | `None` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `radius` | float | `required` | Positive radius of each parabolic surface, metres. |
| `aperture` | float | `required` | Positive full optical aperture, metres; not a radius. |
| `bridge_thickness` | float | `required` | Positive minimum material thickness between opposing parabolic surfaces, metres. |
| `n_elements` | int | `required` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |
| `material` | Material | `required` | Required material supplying delta and beta. |
| `transverse_length` | float or None | `None` | Positive full local nonfocusing width, metres. In 2D selects a cylindrical CRL; None selects rotational symmetry. Supplying it on Grid1D raises ValueError. |
| `rotation` | float | `0.0` | In-plane angle, radians. For cylindrical CRLs: local focusing coordinate is cos(rotation)*x + sin(rotation)*y. |
| `surface_error` | sequence or None | `None` | [height_error, x] in 1D or [height_error, x, y] in 2D. Heights are metres and are doubled in total thickness. See SurfaceError for array constraints. |
| `reuse_element_geometry` | bool | `True` | Reuse a calculated element thickness/transmission map across the stack. This constructor has no per-element defect-sequence parameter. |
| `wavelength_m` | float or None | `None` | Positive wavelength in metres. Mutually exclusive with energy_kev. Both may be None for geometry-only construction. |
| `energy_kev` | float or None | `None` | Positive photon energy in keV. Mutually exclusive with wavelength_m. Needed for material-table evaluation. |

**Returns**

CRL with radius, aperture, bridge_thickness, n_elements, material, rotation, transverse_length, reuse_element_geometry, delta, beta, and shared optical attributes.

**Raises / Warns**

TypeError for missing grid; ValueError for invalid positive geometry/count, transverse_length on 1D, malformed errors, or nonphysical optical constants; UserWarning for error-map extent mismatch; RuntimeWarning if count exceeds the external-focus limit.

**Notes**

A full element thickness is 2*sag + bridge_thickness. Surface errors are added to sag and therefore doubled. Outside the aperture sag saturates; no hard aperture is applied. Stack period is bridge_thickness + aperture**2/(4*radius).

**See also:** [`CRL.focus_scan`](#api-crl-focus_scan); Chapter 2; Chapter 6.

<a id="api-crl-element_thickness"></a>

### `CRL.element_thickness`

Return one full material-thickness map.

```text
CRL.element_thickness(self, element_index: int = 0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `element_index` | int | `0` | Zero-based element index. CRL map methods require 0 <= element_index < n_elements. Defect adapters pass it to the user callback or ignore it. |

**Returns**

Positive real thickness array in metres, grid.shape and backend.

**Raises / Warns**

IndexError outside the stack range; ValueError if any resulting thickness is <= 0; TypeError for a nonnumeric map.

**Notes**

Caches the map. With reuse_element_geometry=True the same cached geometry is returned for each index. surface_error is the supported constructor error input.

**See also:** [`CRL.transmission`](#api-crl-transmission); [`CRL.clear_cache`](#api-crl-clear_cache).

<a id="api-crl-transmission"></a>

### `CRL.transmission`

Return the complex material transmission of one lens element.

```text
CRL.transmission(self, element_index: int = 0) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `element_index` | int | `0` | Zero-based element index. CRL map methods require 0 <= element_index < n_elements. Defect adapters pass it to the user callback or ignore it. |

**Returns**

Complex array, grid.shape/backend.

**Raises / Warns**

ValueError if wavelength or optical constants are missing; map/index errors propagate.

**Notes**

Caches transmission. No stack propagation is included in this one-element map.

**See also:** [`CRL.element_thickness`](#api-crl-element_thickness); [`OpticalElement.E`](#api-opticalelement-e).

<a id="api-crl-clear_cache"></a>

### `CRL.clear_cache`

Clear thickness and transmission caches.

```text
CRL.clear_cache(self) -> None
```

**Parameters:** none.

**Returns**

None.

**Raises / Warns**

No explicit exception.

**Notes**

For changed input_field alone, geometry caches need not be cleared. If changing energy/material/geometry substantially, reconstruct the object: clear_cache does not recompute stored k, delta, beta or other derived state.

**See also:** [`CRL.element_thickness`](#api-crl-element_thickness); [`CRL.transmission`](#api-crl-transmission).

<a id="api-crl-theoretical_focus"></a>

### `CRL.theoretical_focus`

Analytical parallel-beam focal distance from the stack exit.

```text
CRL.theoretical_focus(self, *, n_elements: int | None = None) -> float
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `n_elements` | int | `None` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |

**Returns**

float in metres: Lc/tan(N/Nc).

**Raises / Warns**

ValueError if n_elements < 1; RuntimeWarning at or above critical_element_count(); ValueError when delta is unknown.

**Notes**

The formula is still returned after the warning and may not represent an external physical focus.

**See also:** [`CRL.effective_focus`](#api-crl-effective_focus); CRL.characteristic_length.

<a id="api-crl-effective_focus"></a>

### `CRL.effective_focus`

Analytical finite-source image distance from the CRL exit.

```text
CRL.effective_focus(self, source_distance: float, *, n_elements: int | None = None) -> float
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source_distance` | float | `required` | Positive source-to-CRL entrance distance, metres. |
| `n_elements` | int | `None` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |

**Returns**

float in metres; infinity when the image-distance denominator is numerically zero.

**Raises / Warns**

ValueError for nonpositive source_distance, invalid count, or count above max_element_count().

**Notes**

Includes principal-plane offsets. A negative returned distance represents a formal virtual image, not a positive detector plane.

**See also:** [`CRL.theoretical_focus`](#api-crl-theoretical_focus); [`CRL.focus_scan`](#api-crl-focus_scan).

<a id="api-crl-magnification"></a>

### `CRL.magnification`

Evaluate the implemented analytical transverse magnification expression.

```text
CRL.magnification(self, source_distance: float, image_distance: float, *, n_elements: int | None = None) -> float
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source_distance` | float | `required` | Positive source-to-CRL entrance distance, metres. |
| `image_distance` | float | `required` | Positive CRL-exit-to-observation-plane distance, metres. |
| `n_elements` | int | `None` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |

**Returns**

float, dimensionless.

**Raises / Warns**

ValueError for nonpositive source/image distances or count < 1; missing constants also prevent calculation.

**Notes**

The expression uses the supplied source and image distances, stack length, and characteristic length. It does not propagate fields or enforce an imaging condition; use abs(result) for Gaussian blur widths. Validate its applicability away from focus.

**See also:** [`CRL.image_scale`](#api-crl-image_scale); Chapter 5.

<a id="api-crl-image_scale"></a>

### `CRL.image_scale`

Evaluate the finite-object conjugate-distance ratio in the analytical model.

```text
CRL.image_scale(self, source_distance: float, *, n_elements: int | None = None) -> float
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source_distance` | float | `required` | Positive source-to-CRL entrance distance, metres. |
| `n_elements` | int | `None` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |

**Returns**

float, dimensionless image_distance/object_distance including principal-plane offsets.

**Raises / Warns**

ValueError for nonpositive source_distance or count above max_element_count(); a zero formula denominator is not specially handled.

**Notes**

This is not an alias of magnification(source_distance, image_distance). Unlike the other analytical helpers, this method does not explicitly reject n_elements < 1; callers should pass positive integers.

**See also:** [`CRL.effective_focus`](#api-crl-effective_focus); [`CRL.magnification`](#api-crl-magnification).

<a id="api-crl-critical_element_count"></a>

### `CRL.critical_element_count`

Real-valued stack count where the formal back focus reaches the exit plane.

```text
CRL.critical_element_count(self) -> float
```

**Parameters:** none.

**Returns**

float: pi/2 * characteristic_length/element_period.

**Raises / Warns**

ValueError if required material constants are unknown.

**Notes**

Counts at or above this value do not have an external positive parallel-beam focus.

**See also:** [`CRL.max_element_count`](#api-crl-max_element_count); [`CRL.theoretical_focus`](#api-crl-theoretical_focus).

<a id="api-crl-max_element_count"></a>

### `CRL.max_element_count`

Largest integer stack count below the critical count.

```text
CRL.max_element_count(self) -> int
```

**Parameters:** none.

**Returns**

int: max(0, ceil(critical_element_count()) - 1).

**Raises / Warns**

Errors from critical_element_count() propagate.

**Notes**

This criterion belongs to the ideal parallel-beam analytical model.

**See also:** [`CRL.is_focus_outside_crl`](#api-crl-is_focus_outside_crl).

<a id="api-crl-is_focus_outside_crl"></a>

### `CRL.is_focus_outside_crl`

Check the stack-count external-focus criterion.

```text
CRL.is_focus_outside_crl(self, *, n_elements: int | None = None) -> bool
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `n_elements` | int | `None` | Positive number of CRL elements. In analytical methods None uses the stack count. Supply integer counts; constructor converts its argument with int(). |

**Returns**

bool: count <= max_element_count().

**Raises / Warns**

Errors from max_element_count() propagate.

**Notes**

Does not check positivity of an override count or the finite-source imaging condition. Use valid positive counts.

**See also:** [`CRL.focus_scan`](#api-crl-focus_scan); [`CRL.max_element_count`](#api-crl-max_element_count).

<a id="api-crl-focus_scan"></a>

### `CRL.focus_scan`

Evaluate the exit field once and propagate to all observation planes.

```text
CRL.focus_scan(self, z_values: Sequence[float], *, roi: Any = None, return_fields: bool = False, calculate_divergence: bool = False) -> "FocusScanResult"
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `z_values` | sequence of float | `required` | Nonempty 1D sequence of nonnegative distances from the CRL exit, metres. Order is preserved. |
| `roi` | sequence or None | `None` | Inclusive physical-coordinate limits: (xmin, xmax) in 1D, ((xmin, xmax), (ymin, ymax)) in 2D. None selects the full window. At least one grid sample must lie in each selected axis. |
| `return_fields` | bool | `False` | Retain cropped complex fields when True; when False E is None and metrics/coordinates are still returned. |
| `calculate_divergence` | bool | `False` | Also estimate angular FWHM from the full field, not the cropped ROI. |

**Returns**

FocusScanResult. E is (nz,nx_roi) in 1D or (nz,ny_roi,nx_roi) in 2D when return_fields=True, otherwise None. Metrics are Python/NumPy values; coordinate arrays use grid backend.

**Raises / Warns**

ValueError for empty/non-1D/negative distances or an invalid/nonoverlapping ROI; RuntimeWarning and invalid result when the stack-count criterion fails. Underlying field/material errors propagate.

**Notes**

No detector-stack storage unless requested. The peak index maximizes the per-plane maximum intensity within ROI, not a minimum-width criterion. Divergence uses the full field. ROI reduces retained data, not propagation grid size.

**See also:** [`FocusScanResult`](#api-focusscanresult); [`beam_metrics`](#api-beam_metrics); Chapter 4.

### CRL read-only properties

| Property | Type / units | Definition |
| --- | --- | --- |
| `element_period` | float, metres | Axial element length: bridge_thickness + aperture²/(4*radius) |
| `length` | float, metres | n_elements * element_period |
| `characteristic_length` | float, metres | sqrt(element_period*radius/(2*delta)); requires known delta |
| `characteristic_element_count` | float | characteristic_length/element_period |

**Examples:** use `crl_ideal.length` to inspect stack length and `crl_ideal.transmission()` to inspect a single-element map. Refer to Chapters 2, 4, and 6 for complete field, scan, and defect examples.

<a id="api-surfaceerror"></a>

### `SurfaceError`

Validated regular-grid sampled surface-height map.

```text
SurfaceError(data: Sequence[Any]) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | sequence | `required` | [values, x] or [values, x, y]. Values/coordinates in metres; axes strictly increasing, at least two points each. 2D values have shape (len(y), len(x)); axes or regular meshgrid coordinates are accepted. |

**Returns**

SurfaceError with error, x, y, x_axis, y_axis as CPU arrays (y None in 1D).

**Raises / Warns**

ValueError for invalid data layout, dimensionality, short/non-increasing axes, inconsistent shapes, or nonregular meshgrid coordinates.

**Notes**

A 2D grid requires a 2D map. Values outside coordinate bounds are zero. on_grid() interpolates laboratory coordinates, without automatically rotating the map with a CRL.

**See also:** [`SurfaceError.on_grid`](#api-surfaceerror-on_grid); [`CRL`](#api-crl); Chapter 6.

<a id="api-surfaceerror-on_grid"></a>

### `SurfaceError.on_grid`

Linearly interpolate sampled heights onto a grid.

```text
SurfaceError.on_grid(self, grid: BaseGrid) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |

**Returns**

Real array on grid backend with grid.shape.

**Raises / Warns**

ValueError when a 2D map is used on 1D or a 1D map on 2D; interpolation errors propagate.

**Notes**

Uses CPU NumPy/SciPy interpolation, then transfers the result to the grid backend.

**See also:** [`SurfaceError`](#api-surfaceerror).

<a id="api-thicknessdefect"></a>

### `ThicknessDefect`

Abstract interface for modifying a total-thickness array.

```text
ThicknessDefect()
```

**Parameters:** none.

**Returns**

An abstract adapter; subclass and implement apply().

**Raises / Warns**

TypeError on direct instantiation.

**Notes**

Exported but not accepted by the current CRL constructor. Use adapters explicitly on arrays; they are not a supported CRL defect= argument.

**See also:** [`FunctionThicknessDefect`](#api-functionthicknessdefect); [`InterpolatedThicknessDefect`](#api-interpolatedthicknessdefect).

<a id="api-thicknessdefect-apply"></a>

### `ThicknessDefect.apply`

Abstract total-thickness modification interface.

```text
ThicknessDefect.apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `thickness` | Array or sequence | `required` | A total thickness array for defect.apply; for ArrayObject construction, [values, x] or [values, x, y] in metres. ArrayObject forbids negative samples. |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `element_index` | int | `required` | Zero-based element index. CRL map methods require 0 <= element_index < n_elements. Defect adapters pass it to the user callback or ignore it. |

**Returns**

Modified total-thickness array with grid.shape and backend.

**Raises / Warns**

Defined by the concrete adapter.

**Notes**

element_index allows custom callbacks to vary behavior between elements, but the present CRL constructor does not integrate these adapters.

**See also:** [`FunctionThicknessDefect.apply`](#api-functionthicknessdefect-apply).

<a id="api-functionthicknessdefect"></a>

### `FunctionThicknessDefect`

Wrap a function implementing a thickness modification.

```text
FunctionThicknessDefect(function: Callable[[Any, BaseGrid, int], Any]) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `function` | callable | `required` | function(thickness, grid, element_index) returning a modified total thickness array with the original grid shape and backend. |

**Returns**

FunctionThicknessDefect with attribute function.

**Raises / Warns**

Callable is not validated at construction; a noncallable fails on apply().

**Notes**

The function may replace thickness or add/multiply corrections; it must return the complete resulting array.

**See also:** [`FunctionThicknessDefect.apply`](#api-functionthicknessdefect-apply); [`ThicknessDefect`](#api-thicknessdefect).

<a id="api-functionthicknessdefect-apply"></a>

### `FunctionThicknessDefect.apply`

Call the stored callback and validate returned shape/backend.

```text
FunctionThicknessDefect.apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `thickness` | Array or sequence | `required` | A total thickness array for defect.apply; for ArrayObject construction, [values, x] or [values, x, y] in metres. ArrayObject forbids negative samples. |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `element_index` | int | `required` | Zero-based element index. CRL map methods require 0 <= element_index < n_elements. Defect adapters pass it to the user callback or ignore it. |

**Returns**

Array returned by function(thickness, grid, element_index).

**Raises / Warns**

TypeError for unsupported/mismatched backend; ValueError for shape mismatch; callback errors propagate.

**Notes**

Does not check positivity or numeric physical validity beyond the shared grid validation.

**See also:** [`ThicknessDefect.apply`](#api-thicknessdefect-apply).

<a id="api-interpolatedthicknessdefect"></a>

### `InterpolatedThicknessDefect`

Interpolate an additive total-thickness correction.

```text
InterpolatedThicknessDefect(defect_array: Any, coords: Any, fill_value: float = 0.0) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `defect_array` | array-like | `required` | Additive total-thickness correction in metres, shape (nx,) or (ny, nx). Copied to CPU representation for interpolation. |
| `coords` | array-like or tuple | `required` | 1D: x axis as an ndarray (not a Python list, which is interpreted as an axis sequence). 2D: (x_axis, y_axis). Supply strictly increasing axes consistent with defect_array. |
| `fill_value` | float | `0.0` | Additive correction outside supplied defect coordinates, metres. |

**Returns**

Adapter with defect_array, coords, fill_value in CPU representation.

**Raises / Warns**

Most coordinate/shape constraints are checked during apply(), not construction.

**Notes**

Correction is added once to total thickness; unlike CRL surface_error it is not doubled. No CRL constructor integration is provided.

**See also:** [`InterpolatedThicknessDefect.apply`](#api-interpolatedthicknessdefect-apply); [`SurfaceError`](#api-surfaceerror).

<a id="api-interpolatedthicknessdefect-apply"></a>

### `InterpolatedThicknessDefect.apply`

Interpolate the correction and add it to the supplied thickness.

```text
InterpolatedThicknessDefect.apply(self, thickness: Any, grid: BaseGrid, element_index: int) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `thickness` | Array or sequence | `required` | A total thickness array for defect.apply; for ArrayObject construction, [values, x] or [values, x, y] in metres. ArrayObject forbids negative samples. |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `element_index` | int | `required` | Zero-based element index. CRL map methods require 0 <= element_index < n_elements. Defect adapters pass it to the user callback or ignore it. |

**Returns**

Array on the grid backend; correction is converted to thickness.dtype before addition.

**Raises / Warns**

ValueError for unsuitable coordinate structure/dimensionality; RuntimeWarning when the map does not cover the full grid. Interpolator/NumPy errors propagate for inconsistent samples.

**Notes**

Supply a floating-point thickness array to avoid truncating subpixel height corrections. element_index is unused. The output is not automatically checked for positive thickness.

**See also:** [`InterpolatedThicknessDefect`](#api-interpolatedthicknessdefect).

<a id="api-beammetrics"></a>

### `BeamMetrics`

Data container for measurements in one plane.

```text
BeamMetrics(max_intensity: float, center: float | tuple[float, float] | None, fwhm: float | tuple[float, float] | None, divergence: float | tuple[float, float] | None = None) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `max_intensity` | float | `required` | Maximum intensity in the measured ROI, in input-normalization units. |
| `center` | float, tuple or None | `required` | Peak sample coordinate in metres; (x, y) for 2D. Not a centroid. |
| `fwhm` | float, tuple or None | `required` | FWHM in metres; (FWHM_x, FWHM_y) for 2D. Components can be None. |
| `divergence` | float, tuple or None | `None` | Angular FWHM in radians, or None when not measured/unavailable. |

**Returns**

BeamMetrics with the constructor fields as attributes.

**Raises / Warns**

Dataclass construction does not validate values.

**Notes**

In 2D centre and widths are ordered (x,y). Width tuple components may themselves be None.

**See also:** [`beam_metrics`](#api-beam_metrics); [`FocusScanResult`](#api-focusscanresult).

<a id="api-focusscanresult"></a>

### `FocusScanResult`

Data container for axial scan results.

```text
FocusScanResult(metrics: list[BeamMetrics], peak_index: int | None, peak_z: float | None = None, E: Any = None, x_roi: Any = None, y_roi: Any = None, valid: bool = True) -> None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `metrics` | list of BeamMetrics | `required` | Per-plane metrics, in scan order. |
| `peak_index` | int or None | `required` | Index with the largest sampled peak intensity, not necessarily minimum width. |
| `peak_z` | float or None | `None` | Distance of peak_index from the CRL exit, metres. |
| `E` | Array or None | `None` | Cropped complex field stack, on the grid backend, or None when not retained. |
| `x_roi` | Array or None | `None` | 1D cropped x coordinates in metres, on the grid backend. |
| `y_roi` | Array or None | `None` | 1D cropped y coordinates in metres, or None for 1D/invalid result. |
| `valid` | bool | `True` | False for an invalid scan result. |

**Returns**

FocusScanResult with metrics, peak_index, peak_z, E, x_roi, y_roi, valid.

**Raises / Warns**

Dataclass construction does not validate cross-field consistency.

**Notes**

Normally created by CRL.focus_scan(). See its fields and peak convenience properties below.

**See also:** [`CRL.focus_scan`](#api-crl-focus_scan); [`BeamMetrics`](#api-beammetrics).

<a id="api-focusscanresult-invalid"></a>

### `FocusScanResult.invalid`

Create a sentinel for a skipped scan.

```text
FocusScanResult.invalid() -> "FocusScanResult"
```

**Parameters:** none.

**Returns**

FocusScanResult(metrics=[], peak_index=None, peak_z=None, E=None, x_roi=None, y_roi=None, valid=False).

**Raises / Warns**

No explicit exception.

**Notes**

Check valid before indexing the stack or using peak_z.

**See also:** [`CRL.focus_scan`](#api-crl-focus_scan).

<a id="api-intensity"></a>

### `intensity`

Compute the squared modulus.

```text
intensity(field: Any) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |

**Returns**

abs(field)**2 with the same shape and backend.

**Raises / Warns**

No explicit validation; the input must support abs() and exponentiation.

**Notes**

Does not normalize each frame or integrate over pixels.

**Examples**

```python
I = xpos.intensity(np.array([1 + 1j, 2 + 0j]))
```

**See also:** [`OpticalObject.I`](#api-opticalobject-i).

<a id="api-full_width_half_max"></a>

### `full_width_half_max`

Measure width between the nearest half-maximum crossings around the global peak.

```text
full_width_half_max(intensity_array: Any, coordinate: Any) -> float | None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `intensity_array` | array-like | `required` | Real 1D intensity profile. Singleton dimensions may be squeezed; must match coordinate length. |
| `coordinate` | array-like | `required` | 1D coordinates matching samples. For FWHM use increasing coordinates; for filtering use a uniform, centred axis with nonzero positive spacing. |

**Returns**

Python float in coordinate units, or None if no valid pair exists.

**Raises / Warns**

ValueError for incompatible profile/coordinate shapes; nonfinite intensity, nonpositive maximum, missing crossings, or negative measured width return None.

**Notes**

Linear interpolation locates crossings. No axis keyword exists despite a stale source docstring. Coordinates should be increasing; empty profiles are not specially handled.

**Examples**

```python
width = xpos.full_width_half_max(np.exp(-np.linspace(-5, 5, 101)**2 / 2), np.linspace(-5, 5, 101))
```

**See also:** [`beam_metrics`](#api-beam_metrics); [`divergence`](#api-divergence).

<a id="api-beam_metrics"></a>

### `beam_metrics`

Measure a field on a physical-coordinate ROI.

```text
beam_metrics(field: Any, grid: BaseGrid, *, roi: Any = None, calculate_divergence: bool = False, k: float | None = None) -> BeamMetrics
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `roi` | sequence or None | `None` | Inclusive physical-coordinate limits: (xmin, xmax) in 1D, ((xmin, xmax), (ymin, ymax)) in 2D. None selects the full window. At least one grid sample must lie in each selected axis. |
| `calculate_divergence` | bool | `False` | Also estimate angular FWHM from the full field, not the cropped ROI. |
| `k` | float or None | `None` | Wavenumber 2*pi/wavelength, radians per metre. Supply a positive value; required for divergence calculation. |

**Returns**

BeamMetrics: maximum, peak coordinate, FWHM, and optional divergence. All-nonfinite ROI returns (nan,None,None,None).

**Raises / Warns**

ValueError for wrong shape/ROI or missing k when divergence is requested; TypeError for unsupported/mismatched backend.

**Notes**

Copies ROI intensity to CPU. Centre is a peak coordinate. 2D widths use cuts through that peak; divergence uses the full uncropped field.

**See also:** [`BeamMetrics`](#api-beammetrics); [`full_width_half_max`](#api-full_width_half_max); [`CRL.focus_scan`](#api-crl-focus_scan).

<a id="api-divergence"></a>

### `divergence`

Estimate angular FWHM from the grid spectral distribution.

```text
divergence(field: Any, grid: BaseGrid, k: float) -> float | tuple[float, float] | None
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |
| `grid` | Grid1D or Grid2D | `required` | Grid defining coordinates, array shape, and backend. Required for field calculations. |
| `k` | float or None | `required` | Wavenumber 2*pi/wavelength, radians per metre. Supply a positive value; required for divergence calculation. |

**Returns**

float in 1D, (width_x,width_y) in 2D, radians; None or None components if unavailable.

**Raises / Warns**

ValueError for nonpositive k or wrong shape; TypeError for mismatched backend.

**Notes**

Uses grid.ifft(field), then squared modulus and theta=arcsin(q/k) on |q|<=k. At least three valid samples per required axis are needed. 2D uses cuts through the spectral maximum; this is FWHM, not RMS divergence.

**See also:** [`beam_metrics`](#api-beam_metrics); [`OpticalObject.divergence`](#api-opticalobject-divergence).

<a id="api-gaussian_filter_1d"></a>

### `gaussian_filter_1d`

Convolve a 1D numeric array with a sampled normalized Gaussian.

```text
gaussian_filter_1d(field: Any, coordinate: Any, sigma: float) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |
| `coordinate` | array-like | `required` | 1D coordinates matching samples. For FWHM use increasing coordinates; for filtering use a uniform, centred axis with nonzero positive spacing. |
| `sigma` | float | `required` | Positive Gaussian standard deviation in coordinate units, normally metres. |

**Returns**

Real array with the input backend and shape.

**Raises / Warns**

ValueError for nonpositive sigma; TypeError for an unsupported input array. Other shape/spacing problems are not explicitly validated.

**Notes**

Uses centred continuous FFT conventions and returns only the real part. A uniform centred axis is required. Finite sampling/truncation can affect normalization; not a general complex-field filter.

**See also:** [`gaussian_filter_2d`](#api-gaussian_filter_2d); Chapter 5.

<a id="api-gaussian_filter_2d"></a>

### `gaussian_filter_2d`

Convolve a 2D numeric array with an axis-aligned sampled Gaussian.

```text
gaussian_filter_2d(field: Any, x: Any, y: Any, sigma_x: float, sigma_y: float) -> Any
```

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `field` | Array | `required` | Field or numeric array on the indicated grid/backend. Propagation and beam analysis expect complex amplitudes; filters may take real intensities. |
| `x` | array-like | `required` | Centred uniform x axis or 2D regular meshgrid x coordinates, matching the image. |
| `y` | array-like | `required` | Centred uniform y axis or 2D regular meshgrid y coordinates, matching the image. |
| `sigma_x` | float | `required` | Positive Gaussian x standard deviation, normally metres. |
| `sigma_y` | float | `required` | Positive Gaussian y standard deviation, normally metres. |

**Returns**

Real array with the input backend and shape.

**Raises / Warns**

ValueError for nonpositive sigma_x or sigma_y; TypeError for an unsupported field. Other grid-shape issues propagate from indexing/FFT operations.

**Notes**

x/y may both be axes or matching meshgrids. Coordinates must be uniform and centred. Returns the real part only and uses periodic FFT convolution. Chapter 5 uses an analytic Gaussian transfer function instead.

**See also:** [`gaussian_filter_1d`](#api-gaussian_filter_1d); Chapter 5.

### Result properties and adapter examples

`FocusScanResult.fwhm_at_peak` and `FocusScanResult.divergence_at_peak` are read-only properties. Each returns the corresponding measurement in `metrics[peak_index]`, or `None` if there is no selected peak or the metric is unavailable. In 2D, width pairs may contain `None` components. They do not re-evaluate the field.

The adapter below modifies an array explicitly; it is not attached to a CRL constructor:

```python
g_def = xpos.Grid1D(dx=1e-7, nx=64, backend="numpy")
t_ideal = np.full(g_def.shape, 2e-6)
defect = xpos.FunctionThicknessDefect(
    lambda thickness, grid, element_index: thickness + 1e-9
)
t_modified = defect.apply(t_ideal, g_def, element_index=0)

map_adapter = xpos.InterpolatedThicknessDefect(
    defect_array=np.full(g_def.shape, 1e-9),
    coords=g_def.x,
)
t_interpolated = map_adapter.apply(t_ideal, g_def, element_index=0)
```

For directly supported lens errors, use `CRL(surface_error=...)` as shown in Chapter 6.

## 9. Numerical practice and troubleshooting

### Sampling and computational window

Choose a pixel size that resolves both the intensity structure and the phase variation introduced by lenses. Increasing NX at fixed DX widens the window; reducing DX at fixed NX improves sampling but narrows the window. To establish convergence, independently refine the sampling and increase the physical window until the measured result stabilizes.

FFT propagation assumes periodic boundaries. A propagated beam or blurred tail that reaches the window edges can re-enter from the opposite side. A small display ROI does not prevent this artifact. Propagate on a sufficiently large window and crop afterward.

### Memory and repeated calculations

A single 2048 × 2048 complex128 array takes 64 MiB; a 41-plane stack takes 2.56 GiB before working arrays and FFT buffers. Use `return_fields=False` for metrics-only scans, retain only needed ROIs, and avoid storing full-grid fields when only intensities are required.

`focus_scan` evaluates the CRL exit field once. For custom loops, calculate `E_exit = crl.E()` before the loop. The field backend is preserved, but material processing, interpolation, thickness validation, and metric extraction involve CPU operations or copies.

### Typical problems

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: XPOSim` | Run from the module directory or add its actual location to the Python import path; activate the intended interpreter |
| CuPy imports but the calculation fails | GPU visibility, NVIDIA driver compatibility, environment CUDA version; use explicit NumPy backend for CPU calculations |
| Backend or shape error | A CuPy grid needs CuPy arrays; 2D fields have (ny,nx), not (nx,ny) |
| Missing material file | Database directory and all three expected filenames next to XPOSim.py |
| Unknown optical constants | Supply energy_kev for formula materials, or provide delta/beta directly |
| Invalid point-source distance | PointSource.E needs z > 0 |
| `FWHM` is `None` | Both half-maximum crossings must be inside the measured region; a cylindrical lens can remain wide along its nonfocusing axis |
| Unexpectedly black map regions | ArrayObject blocks zero-thickness and out-of-map points in this implementation |
| Unexpected two openings or a quadrant mask | Hole creates symmetric openings; Blade passes the first quadrant in 2D |
| Surface error rejected | Grid2D requires a 2D map and increasing axes; resulting thickness must stay positive |
| `focus_scan.valid` is false | The ideal stack-count criterion places the formal parallel-beam focus inside the CRL; inspect the count and geometry |
| Edge artifacts after blur | Blur before cropping and increase the full computational window |
| Old geometry after editing a lens | Prefer reconstructing the object; clear_cache clears maps only, not optical constants or all derived state |

## 10. Implementation notes and validation

### Scope of the physical model

The calculation is scalar, monochromatic, forward, and paraxial. It does not solve vector polarization effects, back-reflections, or high-angle nonparaxial propagation. Material interactions use projected thickness; internal CRL propagation is handled between effective element planes. Extended-source convolution additionally assumes mutually incoherent source points and a sufficiently shift-invariant response.

### Behavior that differs from a generic optics API

- Source fields contain a phase profile with unit modulus, not a physically calibrated source amplitude.
- CRL surface errors are doubled in full material thickness, and maps are interpolated in laboratory coordinates without automatic rotation.
- `ArrayObject.E()` blocks zero-thickness regions even though `ArrayObject.transmission()` is one there.
- Thickness-defect adapters are exported helpers, but the current CRL constructor accepts only `surface_error`, not arbitrary defect objects or per-element defect lists.
- Formula-material lookup is not automatically triggered from wavelength alone. Resolve and pass photon energy or supply direct constants.
- Beam centre is the peak coordinate; 2D FWHM uses peak-crossing slices; `peak_z` selects maximum peak intensity among sampled planes.
- Low-level grid propagation accepts signed distances; optical-object propagation rejects negative distances.

These are documented implementation details, not changes to the library source. Internal docstrings occasionally describe broader intended behavior; the reference above follows executed code.

### Verification of these examples

All 25 Python example blocks were syntax-checked and executed on reduced NumPy grids using the repository code and its actual material tables for the silicon examples. A separate CPU-only aperture example also runs without material data. Checks cover finite outputs, field and scan shapes, transform round trips, and normalized Gaussian blurring. The checked source matches commit `a68d920239bf4e5cc0ed5dcd4e3ff68315da44b7`. These checks do not verify the exported environment installation, actual GPU execution, or convergence of the production-size optical calculation. Run the complete example in the intended environment and verify numerical convergence for your geometry.

*Additional physical-model references will be added by the author.*
