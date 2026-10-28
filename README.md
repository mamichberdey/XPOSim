# XPOSim (X-ray Paraxial Optical Simulation)




<picture>
  <source media="(prefers-color-scheme: dark)" srcset=<img width="3400" height="1013" alt="XPOSim_logo_dark" src="https://github.com/user-attachments/assets/41b705ef-8517-48c9-80dd-10bffce855b0" />>
  <source media="(prefers-color-scheme: light)" srcset=<img width="3400" height="1013" alt="XPOSim_logo_light" src="https://github.com/user-attachments/assets/e5d3943c-3e5e-4029-b22e-4f0dd36fe3c7" />>
  <img width="3400" height="1013" alt="XPOSim_logo" src=<img width="3400" height="1013" alt="XPOSim_logo_light" src="https://github.com/user-attachments/assets/e5d3943c-3e5e-4029-b22e-4f0dd36fe3c7" />>
</picture>




**GPU-accelerated paraxial wave-optics simulations for X-ray optical systems.**

XPOSim is a Python library for calculating the propagation and focusing of X-ray radiation in optical systems with one or two transverse dimensions. It models complex wavefields, diffraction, refraction, and absorption, with particular support for compound refractive lenses (CRLs), including lenses with surface errors.

The library can be used to study focal intensity distributions, evaluate the effect of optical imperfections, scan the field around a focus, and generate synthetic data for optical diagnostics and inverse problems.

## Why XPOSim?

XPOSim combines two complementary ways of reducing calculation time:

- **Paraxial wave optics.** Free-space propagation becomes a convolution with the Fresnel propagator. By the convolution theorem, this operation can be evaluated as a multiplication in Fourier space using fast Fourier transforms (FFTs), rather than by directly evaluating a diffraction integral at every observation point.
- **GPU acceleration with CuPy.** Large field arrays, FFTs, and elementwise operations can run on an NVIDIA GPU. Keeping field calculations on the GPU reduces repeated transfers between CPU and GPU memory.

For a grid containing $N$ samples in total, FFT-based propagation has an approximate computational complexity of $O(N\log N)$, compared with $O(N^2)$ for a direct discrete convolution. Actual performance depends on grid size, numerical precision, GPU hardware, and memory transfers; acceleration is generally most useful for sufficiently large calculations.

The supplied implementation also supports NumPy for CPU calculations. Material-table processing and some interpolation and validation operations use the CPU even when the field backend is CuPy.

## Main capabilities

- Uniform 1D and 2D transverse grids with Fresnel propagation.
- Point sources, apertures, absorbing blades, and objects defined by sampled material-thickness maps.
- Compound refractive lenses with propagation between individual lens elements.
- Rotationally symmetric and cylindrically focusing CRL geometries in 2D.
- Sampled surface-error profiles or maps interpolated onto the calculation grid.
- Material-dependent phase shifts and attenuation, using tabulated data or directly supplied optical constants.
- Intensity calculation, focus scans, beam-width and divergence estimates, and Gaussian filtering.

## Installation: Linux and Windows through WSL 2

The recommended setup uses Miniconda and a CuPy environment for GPU calculations. Run the Bash commands below in a Linux terminal, including the Linux terminal provided by WSL 2.

### 1. Prepare GPU access

You need an NVIDIA GPU supported by the selected CUDA/CuPy versions and a compatible NVIDIA driver.

- **Linux:** install the appropriate NVIDIA driver for your distribution.
- **Windows / WSL 2:** install a Windows NVIDIA driver with WSL CUDA support and use a WSL 2 Linux distribution. Do not install a Linux NVIDIA display driver inside WSL; GPU access is provided by the Windows driver.

Check that the GPU is visible from the Linux or WSL terminal:

```bash
nvidia-smi
```

Resolve driver or GPU-access errors before proceeding with GPU calculations. See the [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/) for Windows setup.

### 2. Install Miniconda

Skip this step if Miniconda is already installed and `conda` is available.

For Linux **x86_64**, including typical WSL 2 installations:

```bash
curl -L -o Miniconda3-latest-Linux-x86_64.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the downloaded installer:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the installer prompts and enable shell initialization when asked. Close and reopen the terminal, then check:

```bash
conda --version
```

For another CPU architecture, choose the corresponding installer from the [Miniconda downloads](https://www.anaconda.com/docs/getting-started/miniconda/install).

### 3. Download the project

Download the repository with Git:

```bash
git clone https://github.com/mamichberdey/XPOSim
```

### 4. Enter the project directory

Change to the directory created by Git:

```bash
cd XPOSim
```

### 5. Create the environment

Install the dependencies recorded in the supplied environment file:

```bash
conda env create -f environment.yml
```

The supplied export creates an environment named `cu_env`. It records Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, CuPy 14.2.0, and CUDA 13.4, together with plotting, IPython, and supporting packages. It targets Linux x86_64, including WSL 2 on that architecture. Its CUDA version requires compatible GPU hardware and an NVIDIA driver; the driver is not installed by this command.

This YAML records package versions without exact build identifiers. It is a description of the exported working environment, rather than a guarantee of identical builds or compatibility with every computer. Creation from this export must be verified on the intended target system.

### 6. Activate the environment

Select the environment before running Python:

```bash
conda activate cu_env
```

Repeat this activation in each new terminal session. If using a notebook or IDE, select this environment as the Python interpreter or notebook kernel.

Conda installs the CUDA dependencies listed in the file. A separate system-wide CUDA Toolkit is normally unnecessary for the conda-forge CuPy setup. See the [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html).

### 7. Check the module and material data

The supplied code is a standalone module. The repository should contain it as `XPOSim.py`; rename the uploaded `XPOSim(5).py` if preparing the repository from that file.

Place these database files next to the module:

- `xray_material_data/AtomicConstants.dat`
- `xray_material_data/CrossSec-Compton_McMaster.dat`
- `xray_material_data/f1f2_Windt.dat`

The tables must be supplied separately if absent from the repository. They are not installed by Conda. Calculating material properties from a chemical formula and density requires them; supplying both `delta` and `beta` directly to `Material` bypasses the database.

### 8. Verify the environment

Run this check from the directory containing `XPOSim.py`:

```bash
python - <<'PY'
import cupy as cp
import XPOSim

print("XPOSim version:", XPOSim.__version__)
print("CUDA devices:", cp.cuda.runtime.getDeviceCount())
grid = XPOSim.Grid1D(dx=1e-6, nx=1024, backend="cupy")
field = cp.ones(grid.shape, dtype=cp.complex128)
wavelength = 1e-10  # metres
propagated = grid.propagate(field, z=0.1, k=2 * cp.pi / wavelength)
cp.cuda.Stream.null.synchronize()
assert bool(cp.allclose(propagated, field).item())
print("GPU propagation check passed.")
PY
```

This checks that CuPy can execute a propagation calculation through XPOSim. Importing CuPy alone does not verify GPU execution.

Activate the environment in each new terminal session before running simulations:

```bash
conda activate cu_env
```

For CPU-only use, create an environment with `python=3.11 numpy scipy` and explicitly select `backend="numpy"` when constructing a grid.

## Basic principles for calculations in the paraxial approximation

### Field representation and assumptions

For monochromatic radiation propagating mainly along the optical axis $z$, write the scalar field as

$$
U(x,y,z)=E(x,y,z)e^{ikz}, \qquad k=\frac{2\pi}{\lambda},
$$

where $\lambda$ is the wavelength and $E$ is a slowly varying complex envelope. The time dependence $e^{-i\omega t}$ is implicit.

The paraxial approximation assumes small propagation angles, $|k_x|,|k_y|\ll k$, and neglects the second axial derivative of the envelope relative to its first derivative:

$$
\left|\frac{\partial^2 E}{\partial z^2}\right|
\ll
\left|2k\frac{\partial E}{\partial z}\right|.
$$

The scalar Helmholtz equation then reduces in free space to

$$
2ik\frac{\partial E}{\partial z}+\nabla_\perp^2E=0.
$$

This model describes forward propagation at small angles. It does not resolve back-reflections, polarization-dependent vector effects, or strongly nonparaxial propagation.

### Free-space propagation: the Fresnel propagator

For two transverse dimensions and a positive propagation distance $\Delta z$, the envelope propagator is

$$
P_{2D}(x,y,\Delta z)=\frac{1}{i\lambda\Delta z}
\exp\!\left[i\pi\frac{x^2+y^2}{\lambda\Delta z}\right].
$$

For one transverse dimension, it is

$$
P_{1D}(x,\Delta z)=\frac{1}{\sqrt{i\lambda\Delta z}}
\exp\!\left[i\pi\frac{x^2}{\lambda\Delta z}\right].
$$

The normalization differs between 1D and 2D. The carrier factor $e^{ik\Delta z}$ is omitted because XPOSim propagates the complex envelope; this common phase does not affect intensity.

### Propagation as a convolution

The field in the next plane is the transverse convolution

$$
E(x,y,z+\Delta z)=E(x,y,z)*P_{2D}(x,y,\Delta z).
$$

By the convolution theorem,

$$
E(z+\Delta z)=\mathcal{F}_\perp^{-1}
\left[\mathcal{F}_\perp\{E(z)\}\,
\mathcal{F}_\perp\{P(\Delta z)\}\right].
$$

In the implementation, the Fourier transform of the propagator is evaluated analytically:

$$
H(q_x,q_y,\Delta z)=
\exp\!\left[-i\frac{q_x^2+q_y^2}{2k}\Delta z\right],
$$

where $q_x$ and $q_y$ are transverse angular spatial frequencies in radians per metre. Thus each propagation step uses a forward FFT, multiplication by $H$, and an inverse FFT. For 1D propagation, omit the $q_y^2$ term.

### Interaction with a thin object

A thin optical object is represented by a complex transmission function:

$$
E_{\mathrm{out}}(x,y)=E_{\mathrm{in}}(x,y)T(x,y).
$$

This projection model assumes that diffraction inside each individual thin interaction can be neglected. An extended optical assembly can be modeled as a sequence of thin interactions separated by free-space propagation. XPOSim uses this approach for a stack of CRL elements.

### Transmission: phase shift and attenuation

For the refractive-index convention

$$
n=1-\delta+i\beta,
$$

the transmission through material thickness $t(x,y)$, relative to the same distance in vacuum, is

$$
T(x,y)=\exp\!\left[-ik(\delta-i\beta)t(x,y)\right]
=\exp[-ik\delta t(x,y)]\exp[-k\beta t(x,y)].
$$

Here $\delta$ is the refractive-index decrement and $\beta\geq0$ is the absorption index. The first factor describes phase shift, and the second describes amplitude attenuation. The corresponding intensity attenuation is $\exp[-2k\beta t(x,y)]$.

### Intensity

Intensity is proportional to the squared modulus of the complex amplitude. In the library's normalization,

$$
I(x,y)=|E(x,y)|^2.
$$

Absolute physical intensity requires a corresponding calibration of the input field.

### Numerical sampling

The transverse sampling must resolve both the wavefield and the phase variations introduced by optical elements. The computational window must be wide enough to contain the propagated beam. FFT propagation uses a periodic numerical domain, so insufficient window size can introduce wraparound artifacts. Check convergence by refining the sampling and increasing the window size for the geometry of interest.

Unless a parameter states otherwise, spatial quantities are in metres, angles are in radians, photon energies are in keV, and material density is in g/cm³.

### 3. Download the project

Download the repository with Git:

```bash
git clone https://github.com/mamichberdey/XPOSim
```

### 4. Enter the project directory

Change to the directory created by Git:

```bash
cd REPOSITORY
```

### 5. Create the environment

Install the dependencies recorded in the supplied environment file:

```bash
conda env create -f environment.yml
```

The supplied export creates an environment named `cu_env`. It records Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, CuPy 14.2.0, and CUDA 13.4, together with plotting, IPython, and supporting packages. It targets Linux x86_64, including WSL 2 on that architecture. Its CUDA version requires compatible GPU hardware and an NVIDIA driver; the driver is not installed by this command.

This YAML records package versions without exact build identifiers. It is a description of the exported working environment, rather than a guarantee of identical builds or compatibility with every computer. Creation from this export must be verified on the intended target system.

### 6. Activate the environment

Select the environment before running Python:

```bash
conda activate cu_env
```

Repeat this activation in each new terminal session. If using a notebook or IDE, select this environment as the Python interpreter or notebook kernel.

Conda installs the CUDA dependencies listed in the file. A separate system-wide CUDA Toolkit is normally unnecessary for the conda-forge CuPy setup. See the [CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html).

### 7. Check the module and material data

The supplied code is a standalone module. The repository should contain it as `XPOSim.py`; rename the uploaded `XPOSim(5).py` if preparing the repository from that file.

Place these database files next to the module:

- `xray_material_data/AtomicConstants.dat`
- `xray_material_data/CrossSec-Compton_McMaster.dat`
- `xray_material_data/f1f2_Windt.dat`

The tables must be supplied separately if absent from the repository. They are not installed by Conda. Calculating material properties from a chemical formula and density requires them; supplying both `delta` and `beta` directly to `Material` bypasses the database.

''' +s[b:].('### 5. Verify the environment','### 8. Verify the environment',1)
s=s.('conda activate xposim','conda activate cu_env')
a=s.index('## Code structure and usage workflow')
s=s[:a]+'''## Writing an optical simulation: the main logic

An optical scheme is built by passing **complex field arrays** between objects. The field contains both amplitude and phase; intensity alone does not contain enough information for further wave propagation.

- `PointSource` creates a field and therefore does not take an incident field.
- Every field-transforming optical element takes an incident complex amplitude through `input_field` and is associated with a calculation `grid`.
- `element.E()` returns the field immediately after the element, including the internal calculation of a CRL stack.
- `element.E(z=distance)` applies the element and then propagates its output by that distance in metres.
- `element.I(z=distance)` returns intensity at the same plane. Equivalently, use `abs(element.E(z=distance))**2`.

Some elements can be constructed without `input_field` to inspect geometry, but their field calculation still requires an incident field. `Material` and grids are supporting objects, rather than field-transforming elements.

Distances are **relative to the current object's output reference plane**, not absolute laboratory coordinates. For a CRL, this is the stack exit plane. A second element must receive the field propagated to its own entrance plane. The library does not assemble a global beamline or infer gaps automatically.

The examples below are sequential: run the setup once and then continue with the subsequent sections. Optional schemes reuse the named objects from that setup.

## Tutorial: point source and a cylindrical CRL

### 1. Import the libraries

Run Python from the repository root so that `XPOSim.py` can be imported:

```python
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import XPOSim as xpos
```

If a notebook is located one directory below `XPOSim.py`, add the parent directory before importing XPOSim:

```python
import sys
sys.path.append("../")
```

The relative path is resolved against the current working directory. Use this adjustment only if needed.

### 2. Create the calculation grid

```python
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
```

The window width is 102.4 µm along each axis. `grid.x` and `grid.y` are 2D coordinate maps, while `grid.x_axis` and `grid.y_axis` are 1D coordinate axes. Field arrays have shape **`(NY, NX)`**, so the first array index corresponds to `y`.

Use the same grid for elements in one scheme. Field arrays must have the correct shape and backend: a CuPy grid expects CuPy arrays. Convert arrays explicitly with `cp.asarray()` or `cp.asnumpy()` when transferring between devices, and keep calculation arrays on the GPU. See [CuPy array conversion](https://docs.cupy.dev/en/stable/reference/ndarray.html).

### 3. Create the incident spherical wave

```python
ENERGY_KEV = 12.0
Z0 = 15.0  # source-to-CRL entrance distance, metres

ps = xpos.PointSource(
    grid=grid,
    position=(0.0, 0.0),
    energy_kev=ENERGY_KEV,
)
E0 = ps.E(z=Z0)
```

`position` specifies the transverse source coordinates in metres. `ps.E(z=Z0)` creates the complex field in the plane where the first element is placed. A point source requires `z > 0`; calling `ps.E()` at its default zero distance is invalid.

In this implementation, the source produces a unit-modulus paraxial spherical-wave phase profile. It does not include absolute source flux or distance-dependent amplitude normalization. Use the resulting intensities as relative values unless the input is calibrated separately.

### 4. Define the material and lens

```python
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
```

| Parameter | Meaning |
| --- | --- |
| `input_field` | Complex amplitude at the CRL entrance |
| `radius` | Radius of curvature of each parabolic surface, metres |
| `aperture` | Full optical aperture, metres |
| `bridge_thickness` | Central material thickness between the two surfaces, metres |
| `n_elements` | Number of lens elements in the stack |
| `material` | Material used for phase shift and attenuation |
| `energy_kev` | Photon energy; keep it consistent throughout the scheme |
| `transverse_length` | Finite width in the local nonfocusing direction, metres |
| `rotation` | In-plane rotation of the cylindrical lens, radians |

With `transverse_length` specified, the 2D lens focuses along one local axis. At `rotation=0` this is laboratory `x`; at `rotation=np.pi/2` it is laboratory `y`. Omitting `transverse_length` creates a rotationally symmetric 2D CRL.

The CRL calculation alternates material transmission and propagation between element centres, including entrance and exit half-period propagation. Outside the nominal optical aperture the implemented profile saturates at the maximum material thickness; it is not a hard zero-transmission aperture. Add an aperture explicitly if complete blocking is required.

### 5. Calculate and display one detector plane

```python
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

For a single plane, `crl_ideal.E(z=Z_DETECTOR)` is equivalent. Calculating `E_exit` once is useful when propagating to many detector positions: repeatedly calling `crl_ideal.E(z=...)` also recalculates the passage through the stack.

## Scan the field around the focus

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

These estimates describe the ideal paraxial CRL model and do not  inspecting the simulated field.

## Approximate an extended incoherent source

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

## Other simple schemes and useful operations

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

### Add a sampled surface error

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

### A smaller 1D calculation

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

### Additional interfaces

| Interface | Purpose or convention |
| --- | --- |
| `grid.propagate(field, z, k)` | Fresnel propagation with an explicitly supplied wavenumber |
| `grid.convolve(a, b)` | FFT convolution on the grid |
| `xpos.intensity(field)` | Squared modulus, preserving the array backend |
| `xpos.beam_metrics(field, grid)` | Peak intensity, centre, and FWHM estimates |
| `focus_scan(..., calculate_divergence=True)` | Also calculate divergence estimates |
| `xpos.Blade` | Passes `x >= 0` in 1D; in 2D passes only `x >= 0` and `y >= 0` together |
| `xpos.ArrayObject` | Material object defined by a sampled thickness map; in this implementation zero-thickness pixels and points outside the map are blocked during field calculation |
| `Material(delta=..., beta=...)` | Supply optical constants directly instead of reading material tables |
| `crl.element_thickness()` / `crl.transmission()` | Inspect one element's thickness or complex transmission |
| `crl.clear_cache()` | Clear cached thickness and transmission maps after relevant changes |

For reproducible comparisons, keep the energy, grid, input illumination, and observation coordinates consistent. Check convergence of the diffraction calculation before interpreting small changes as physical lens defects.
