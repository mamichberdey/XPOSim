# XPOSim

XPOSim (X-ray Paraxial Optical Simulation) is a Python library for simulating monochromatic X-ray propagation, diffraction, and focusing. It propagates complex wavefields on one- and two-dimensional transverse grids using FFTs. Calculations run on the CPU with NumPy or on an NVIDIA GPU with CuPy, PyTorch or JAX.

Try XPOSim for X-ray focusing with a compound refractive lens.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamichberdey/XPOSim/blob/60049afd607a504065c3de014c06a70065b658fe/notebooks/CRL_Focus_Colab.ipynb)

[Installation](#installation) · [Quick start](#quick-start) · [Library guide](DOCUMENTATION.md) · [API reference](DOCUMENTATION.md#8-api-reference)

<a id="1-about-xposim"></a>

## Capabilities

- Build optical systems from compound refractive lenses (CRLs), apertures, and sampled material thickness maps.
- Model line-focus, point-focus, and crossed CRLs. Use tabulated material properties or supply the refractive index decrement and absorption index directly.
- Change illumination between a plane wave, a point source, and an incoherent Gaussian source in the CRL workflow.
- Calculate paraxial image distances, scan the propagated field in batches, and measure peak intensity, FWHM, and transmission. Use sampled surface errors to study departures from ideal lens geometry.

<a id="2-installation"></a>

## Installation

Use Python 3.10 or newer and activate the environment where you will run XPOSim.

### 1. Prepare NumPy

**NumPy is required for every backend, including GPU calculations.** XPOSim uses NumPy and the Python standard library for host-side interpolation, scalar searches, source quadrature and CPU calculations.

If NumPy is already installed, use the existing installation. In a new environment, install it with your package manager, for example with pip.

```sh
python -m pip install numpy
```

For an existing environment with a missing NumPy installation, follow its package manager's compatibility requirements. [NumPy installation guide](https://numpy.org/install/)

### 2. Install XPOSim

```sh
python -m pip install --no-deps "git+https://github.com/mamichberdey/XPOSim.git"
```

XPOSim leaves dependency management to your environment. Its package metadata declares no runtime dependencies, and `--no-deps` explicitly disables dependency resolution. NumPy from step 1 must be available when importing XPOSim.

The material tables are included, and the public import is `import XPOSim`. For wheel, local-checkout and offline source installations, see the installation guide. [Other installation methods](DOCUMENTATION.md#installation-into-an-existing-environment)

**Plotting is optional.** Install Matplotlib with `python -m pip install matplotlib` to use plotting helpers and plotting examples. The quick start below prints its results and works with NumPy from step 1.

### 3. Choose CPU or GPU execution

For **CPU calculations**, NumPy is sufficient. Select `backend="numpy"`.

For **GPU calculations**, additionally install **one** of CuPy, PyTorch or JAX with GPU support compatible with your system and driver. NumPy is still required. GPU frameworks have their own dependency requirements; JAX installations can require SciPy. Select the corresponding backend on a grid or in `CRLFocusConfig`.

| Compute backend | Setting | Installation |
| --- | --- | --- |
| NumPy on CPU | `backend="numpy"` | [Guide](https://numpy.org/install/) |
| CuPy on GPU | `backend="cupy"` | [Guide](https://docs.cupy.dev/en/stable/install.html) |
| PyTorch on GPU | `backend="torch"` | [Guide](https://pytorch.org/get-started/locally/) |
| JAX on GPU | `backend="jax"` | [Guide](https://docs.jax.dev/en/latest/installation.html) |

The high-level workflow defaults to `backend="auto"`. It probes CuPy, PyTorch and JAX in that order and uses NumPy if no GPU backend can run. An explicit GPU selection requires that backend to execute on a visible GPU. In Jupyter, restart the kernel after installing packages.

<a href="docs/assets/backend-benchmark.png"><img src="docs/assets/backend-benchmark.png" width="720" alt="XPOSim CPU and GPU benchmark on Intel Core i5-12450H and NVIDIA GeForce RTX 4060 Laptop GPU. Logarithmic scan time above, GPU speedup relative to NumPy below. NumPy is blue, CuPy green, and PyTorch orange."></a>

In this workload, both measured GPU backends have lower median times from 32,768 points per plane onward. [Benchmark setup and results](docs/GPU_VALIDATION.md#extended-grid-comparison).

GPU fields stay on the selected device during propagation, filtering and metric reduction. `focus_scan` transfers one compact metric table after the scan. The high-level workflow exports profiles in bounded chunks and returns NumPy arrays for plotting. Adaptive searches read scalar results to choose their next evaluation point. Retained CRL results support `with estimate_crl_focus(config) as result:` or explicit `result.close()`. Use `result.numerical_report` for sampling indicators and `result.check_convergence()` for optional grid/window comparisons. See [device memory and transfers](DOCUMENTATION.md#device-memory-and-transfers) for cache limits and synchronization details.

## Quick start

Run this example after installation, or from the repository root. It probes each GPU backend with a small FFT and lists the working backends in selection order. Leave `requested = "auto"` to use the first available backend, or set it to a backend name to require that backend.

```python
from importlib import import_module
import platform

import numpy as np
import XPOSim as xpos

requested = "auto"
available = []
for name in ("cupy", "torch", "jax", "numpy"):
    try:
        available.append(xpos.select_backend(name))
    except (ImportError, OSError, RuntimeError):
        continue

backend = available[0] if requested == "auto" else xpos.select_backend(requested)
device = "CPU" if backend == "numpy" else "GPU"
print(f"XPOSim {xpos.__version__} | Python {platform.python_version()} | {platform.system()}")
print(f"Core package: NumPy {np.__version__}")
print("Available backends:", ", ".join(available))
print(f"Selected: {backend} {import_module(backend).__version__} ({device})")

# Illuminate a silicon lens with a 12 keV point source 15 m upstream.
energy_kev, distance_m = 12.0, 15.0
grid = xpos.Grid1D(dx=20e-9, nx=4096, backend=backend)
source = xpos.PointSource(grid=grid, energy_kev=energy_kev)
lens = xpos.CRL(
    grid=grid, input_field=source.E(z=distance_m), energy_kev=energy_kev,
    material=xpos.Material(formula="Si", density=2.33),
    radius=6.25e-6, aperture=50e-6, bridge_thickness=2e-6, n_elements=26,
)

# Measure the beam at the predicted image plane, downstream of the lens exit.
z = lens.effective_focus(distance_m)
beam = xpos.beam_metrics(lens.E(z=z), grid)
if beam.fwhm is None or not (0 < beam.fwhm < lens.aperture and 1 < beam.max_intensity < np.inf):
    raise RuntimeError("Expected a resolved focal spot brighter than the incident beam.")
print(f"Image plane: {z * 1e3:.2f} mm from lens exit")
print(f"Spot FWHM: {beam.fwhm * 1e9:.1f} nm | Peak intensity: {beam.max_intensity:.1f} × incident")
print("XPOSim lens check passed.")
```

## Examples and reference

- Build an optical system from grids, a source, and individual elements. [Worked Python example](DOCUMENTATION.md#2-complete-example-point-source-crl-and-detector)
- Configure a CRL scan, choose the illumination model, and compare focus metrics. [High-level CRL workflow](DOCUMENTATION.md#high-level-crl-workflow)
- Explore a CRL focus interactively and export a simulation report. [Notebook source](notebooks/CRL_Focus_Colab.ipynb) · [Notebook guide](COLAB_GUIDE.md)

<a id="3-physical-model"></a>

## Physical model

The formulation below follows the wave-propagation framework described by **V. G. Kohn**. [Kohn (2022)][kohn2022], Section 2, presents the paraxial equation, Fresnel propagation, FFT evaluation, and the compressed-lens approximation. The analytical theory of continuously refractive parabolic lenses is developed in [Kohn (2003)][kohn2003]; diffraction and imaging with CRLs are discussed by [Kohn, Snigireva & Snigirev (2003)][kohn-imaging2003].

### 3.1 Complex amplitude and the paraxial approximation

XPOSim describes monochromatic radiation: a calculation uses a single photon energy and its corresponding wavelength. The beam travels mainly along the optical axis $z$, while $x$ and $y$ are transverse coordinates. All coordinates and distances are expressed in metres.

The calculated quantity is the **complex field envelope**

$$
E(x,y,z)=A(x,y,z)\exp[i\varphi(x,y,z)]
$$

where $A\geq0$ is the local amplitude, $\varphi$ is the phase in radians, and $i$ is the imaginary unit ($i^2=-1$). Keeping both amplitude and phase allows the model to describe interference, diffraction, and focusing. The common longitudinal carrier phase is omitted from this envelope; it does not affect the calculated intensity.

The **paraxial approximation** assumes propagation at small angles to $z$ and a field envelope that changes slowly along that axis compared with the wavelength. XPOSim uses a scalar field: polarization effects and back-reflections are outside this model. The following relations describe propagation between planes and interaction with material, without solving the full electromagnetic problem. See [Kohn (2022)][kohn2022], equations (1)–(2), for the field representation and paraxial equation.

### 3.2 Free-space propagation: the Fresnel propagator

Propagation over a positive distance $\Delta z$ is described by the Fresnel propagator. With two transverse coordinates,

$$
P_{2D}(x,y,\Delta z)=\frac{1}{i\lambda\Delta z}
\exp\left[\frac{i\pi(x^2+y^2)}{\lambda\Delta z}\right]
$$

where $\lambda$ is the radiation wavelength in metres. Here $x$ and $y$ represent transverse separations between a point in the input plane and a point in the observation plane. The propagator describes the complex contribution transferred between these points, including its phase.

For a model with one transverse coordinate, the normalization changes; this is the propagator given in [Kohn (2022)][kohn2022], equation (3):

$$
P_{1D}(x,\Delta z)=\frac{1}{\sqrt{i\lambda\Delta z}}
\exp\left[\frac{i\pi x^2}{\lambda\Delta z}\right]
$$

These propagators act on the complex envelope and omit the common carrier factor $\exp(ik\Delta z)$, where $k=2\pi/\lambda$ is the vacuum wavenumber in radians per metre.

### 3.3 Propagation as a convolution

The field in the observation plane is obtained by summing the contributions from the entire input plane. Mathematically, this is a convolution over the transverse coordinates:

$$
E(x,y,z+\Delta z)=E(x,y,z)*P_{2D}(x,y,\Delta z)
$$

The symbol $*$ denotes convolution, not pointwise multiplication. By the convolution theorem,

$$
E_{z+\Delta z}=\mathcal{F}^{-1}\left[\mathcal{F}(E_z)\,\mathcal{F}(P_{\Delta z})\right]
$$

where $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the forward and inverse Fourier transforms over $x,y$ (or $x$ in 1D). The shorthand $E_z$ denotes the field in plane $z$, and $P_{\Delta z}$ denotes the propagator for the distance $\Delta z$. XPOSim uses the analytic Fourier transform of the Fresnel propagator and computes the field transforms with FFTs. This replaces direct evaluation of the diffraction integral with two FFTs and an elementwise multiplication, which are efficiently performed on the GPU. The convolution formulation and its FFT evaluation are described in [Kohn (2022)][kohn2022], equation (4) and the accompanying text.

### 3.4 Interaction with a thin object

A thin optical object modifies the field through its complex transmission function $T$:

$$
E_{\mathrm{out}}(x,y)=E_{\mathrm{in}}(x,y)T(x,y)
$$

Here $E_{\mathrm{in}}$ and $E_{\mathrm{out}}$ are the complex fields immediately before and after the interaction, and $T$ is dimensionless. Unlike propagation, this is pointwise multiplication: each sample is modified locally.

The thin-object approximation neglects diffraction within that individual interaction. A CRL stack is represented by a sequence of material interactions separated by propagation between effective lens planes, so diffraction through the assembly is still taken into account. See the compressed-lens approach in [Kohn (2022)][kohn2022], Section 2.1, equations (5)–(6).

### 3.5 Material transmission: phase shift and attenuation

For X-rays, the material refractive index is written as

$$
n=1-\delta+i\beta
$$

where $\delta$ is the dimensionless refractive-index decrement and $\beta\geq0$ is the dimensionless absorption index. Both depend on material and photon energy. For a homogeneous material with thickness $t(x,y)$ in metres, the transmission relative to the same path in vacuum is

$$
T(x,y)=\exp\left[-ik(\delta-i\beta)t(x,y)\right]
$$

This transmission contains a phase factor $\exp(-ik\delta t)$ and an amplitude attenuation factor $\exp(-k\beta t)$. The phase shift is $-k\delta t$; its variation across the beam produces refractive focusing or wavefront distortion. The corresponding intensity attenuation is $\exp[-2k\beta t]$. Here $t$ denotes material thickness, not time. This transmission function corresponds to [Kohn (2022)][kohn2022], equation (6), with $k=K$ and $\delta-i\beta=\eta$ in that article.

### 3.6 Intensity

Intensity is proportional to the squared modulus of the complex field. In the library's normalization,

$$
I(x,y,z)=|E(x,y,z)|^2=A(x,y,z)^2
$$

The modulus removes the phase from the displayed intensity, but the phase remains essential for calculating subsequent propagation. Therefore, optical elements must pass **complex fields**, rather than intensity arrays, to one another. Absolute intensity units require a calibrated input-field normalization; otherwise the results are relative intensities.

### 3.7 References

1. **Kohn, V. G. (2022).** On the theory of synchrotron radiation nanofocusing with planar compound refractive lenses. *Journal of Synchrotron Radiation*, **29**, 615–621. [DOI: 10.1107/S1600577522001345][kohn2022]. [Open full text](https://journals.iucr.org/s/issues/2022/03/00/mo5253/index.html).

2. **Kohn, V. G. (2003).** An Exact Theory of Imaging with a Parabolic Continuously Refractive X-ray Lens. *Journal of Experimental and Theoretical Physics*, **97**, 204–215. [DOI: 10.1134/1.1600812][kohn2003]. [Author's PDF](https://kohnvict.narod.ru/articles/133e.pdf).

3. **Kohn, V., Snigireva, I., & Snigirev, A. (2003).** Diffraction theory of imaging with X-ray compound refractive lens. *Optics Communications*, **216**, 247–260. [DOI: 10.1016/S0030-4018(02)02285-X][kohn-imaging2003].

[kohn2022]: https://doi.org/10.1107/S1600577522001345
[kohn2003]: https://doi.org/10.1134/1.1600812
[kohn-imaging2003]: https://doi.org/10.1016/S0030-4018(02)02285-X

## Scientific validation and development

The initial independent reference suite checks complex Gaussian propagation, thin-lens focusing, slit Fresnel diffraction, Fourier normalization, material attenuation and photon-energy units. Run `python -m unittest discover -s tests -p test_physics_validation.py` with NumPy installed. Optional SciPy reference checks include slit Fresnel integrals and comparisons of the numerical helpers. Tests use documented setups and numerical tolerances.

- [Physical and numerical conventions](docs/PHYSICAL_CONVENTIONS.md)
- [Reference cases and test tolerances](docs/VALIDATION.md)
- [RTX 4060 GPU validation and measurements](docs/GPU_VALIDATION.md)
- [Engineering audit and staged roadmap](docs/ENGINEERING_AUDIT.md)
- [Proposed API evolution](docs/API_DESIGN.md)

The conventions document explains finite-window behavior on even grids and the entrance-pupil difference between low-level CRLs and the high-level focus workflow. GPU validation requires target hardware.
