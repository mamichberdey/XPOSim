# XPOSim

**X-ray Paraxial Optical Simulation — wave optics with GPU acceleration.**

[Download ZIP](https://github.com/mamichberdey/XPOSim/archive/refs/heads/main.zip) · [Installation](#2-installation) · [Tutorial](DOCUMENTATION.md#2-complete-example-point-source-crl-and-detector) · [API reference](DOCUMENTATION.md#8-api-reference) · [Repository](https://github.com/mamichberdey/XPOSim)

**Contents**

1. [About XPOSim](#1-about-xposim)
2. [Installation](#2-installation)
3. [Physical model](#3-physical-model)

---

## 1. About XPOSim

XPOSim is a Python library for modelling X-ray propagation and focusing with one or two transverse dimensions. It calculates the complex wavefield and intensity through optical systems built from compound refractive lenses (CRLs), apertures, and sampled material-thickness maps.

Use it to:

- Study how lens geometry and surface errors change the focal intensity distribution.
- Scan the field along the optical axis and measure beam widths and peak intensity.
- Compare optical configurations and generate synthetic data for X-ray optical diagnostics.

### Why XPOSim?

Free-space propagation is evaluated with FFTs using the convolution theorem. CuPy accelerates the main field operations on NVIDIA GPUs; NumPy provides the same grid and optical-element interfaces on the CPU. Optical schemes are built by passing complex field arrays between elements.

> [!TIP]
> **Use GPU acceleration for large 2D grids and repeated simulations.** CPU execution is also supported with NumPy and SciPy; CUDA is optional. Performance depends on the grid, hardware, and calculation.

Start with the [complete point source → CRL → detector example](DOCUMENTATION.md#2-complete-example-point-source-crl-and-detector) after installation. Continue with focus scans, extended-source modelling, surface errors, and parameter descriptions in the [documentation](DOCUMENTATION.md).

---

## 2. Installation

The installation guide uses **Miniconda** and the supplied `environment.yml` for the recommended GPU setup. The commands below are for Linux or the Linux terminal in WSL 2. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) first if it is not already available. Git is optional; the project can also be downloaded as a ZIP archive.

For CPU-only use, the required dependencies are listed in [Section 2.3](#23-cpu-only-dependencies). Set up that environment yourself; the GPU environment file is not required.

### 2.1 Download the project

**Option A — download an archive.**

[Download and extract the project](https://github.com/mamichberdey/XPOSim/archive/refs/heads/main.zip). Git is not required.

**Option B — clone the repository.**

If [Git](https://git-scm.com/downloads) is installed, download the repository with:

```bash
git clone https://github.com/mamichberdey/XPOSim.git
```

Enter the project folder. For a Git clone:

```bash
cd XPOSim
```

For a ZIP archive, use the extracted folder name, normally `XPOSim-main`, instead. The current directory should contain `XPOSim.py` and `environment.yml`.

### 2.2 GPU installation with Miniconda (recommended)

This GPU option requires an NVIDIA GPU and a compatible driver. The supplied environment targets **Linux x86_64 / WSL 2 x86_64** and specifies **CUDA 13.4**. Under WSL 2, GPU access is provided by the Windows NVIDIA driver; see [NVIDIA's setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

**Step 1 — create the environment.**

Use the supplied dependency file to create `xposim_gpu_env`:

```bash
conda env create -f environment.yml
```

**Step 2 — activate the environment.**

Run this before working with XPOSim:

```bash
conda activate xposim_gpu_env
```

The environment includes CuPy and CUDA runtime dependencies. The NVIDIA driver must be installed separately. Select `backend="cupy"` when creating a calculation grid. Repeat the activation command when opening a new terminal session.

### 2.3 CPU-only dependencies

CPU calculations require **NumPy** and **SciPy**. Installing SciPy normally installs a compatible NumPy version automatically, if needed; see the [official SciPy installation guide](https://scipy.org/install/). Prepare this environment yourself rather than using the GPU `environment.yml`.

Select `backend="numpy"` when creating grids and use NumPy arrays. CuPy, CUDA, and an NVIDIA driver are not required. **Matplotlib** is optional and needed only for plotting examples.

### 2.4 Run your first calculation

Run scripts from the project root so that `XPOSim.py` can be imported. In a notebook or IDE, select the environment in which you installed the dependencies.

> [!NOTE]
> Keep the included `xray_material_data/` directory next to `XPOSim.py`. Its tables are needed for formula-based materials on both CPU and GPU. Materials specified directly by `delta` and `beta` do not need these tables.

**Next:** run the [complete worked example](DOCUMENTATION.md#2-complete-example-point-source-crl-and-detector), or the [CPU-only example](DOCUMENTATION.md#7-one-dimensional-and-cpu-calculations).

---

## 3. Physical model

### 3.1 Complex amplitude and the paraxial approximation

XPOSim describes monochromatic radiation: a calculation uses a single photon energy and its corresponding wavelength. The beam travels mainly along the optical axis $z$, while $x$ and $y$ are transverse coordinates. All coordinates and distances are expressed in metres.

The calculated quantity is the **complex field envelope**

$$
E(x,y,z)=A(x,y,z)\exp[i\varphi(x,y,z)]
$$

where $A\geq0$ is the local amplitude, $\varphi$ is the phase in radians, and $i$ is the imaginary unit ($i^2=-1$). Keeping both amplitude and phase allows the model to describe interference, diffraction, and focusing. The common longitudinal carrier phase is omitted from this envelope; it does not affect the calculated intensity.

The **paraxial approximation** assumes propagation at small angles to $z$ and a field envelope that changes slowly along that axis compared with the wavelength. XPOSim uses a scalar field: polarization effects and back-reflections are outside this model. The following relations describe propagation between planes and interaction with material, without solving the full electromagnetic problem.

### 3.2 Free-space propagation: the Fresnel propagator

Propagation over a positive distance $\Delta z$ is described by the Fresnel propagator. With two transverse coordinates,

$$
P_{2D}(x,y,\Delta z)=\frac{1}{i\lambda\Delta z}
\exp\left[\frac{i\pi(x^2+y^2)}{\lambda\Delta z}\right]
$$

where $\lambda$ is the radiation wavelength in metres. Here $x$ and $y$ represent transverse separations between a point in the input plane and a point in the observation plane. The propagator describes the complex contribution transferred between these points, including its phase.

For a model with one transverse coordinate, the normalization changes:

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

where $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the forward and inverse Fourier transforms over $x,y$ (or $x$ in 1D). The shorthand $E_z$ denotes the field in plane $z$, and $P_{\Delta z}$ denotes the propagator for the distance $\Delta z$. XPOSim uses the analytic Fourier transform of the Fresnel propagator and computes the field transforms with FFTs. This replaces direct evaluation of the diffraction integral with two FFTs and an elementwise multiplication, which are efficiently performed on the GPU.

### 3.4 Interaction with a thin object

A thin optical object modifies the field through its complex transmission function $T$:

$$
E_{\mathrm{out}}(x,y)=E_{\mathrm{in}}(x,y)T(x,y)
$$

Here $E_{\mathrm{in}}$ and $E_{\mathrm{out}}$ are the complex fields immediately before and after the interaction, and $T$ is dimensionless. Unlike propagation, this is pointwise multiplication: each sample is modified locally.

The thin-object approximation neglects diffraction within that individual interaction. A CRL stack is represented by a sequence of material interactions separated by propagation between effective lens planes, so diffraction through the assembly is still taken into account.

### 3.5 Material transmission: phase shift and attenuation

For X-rays, the material refractive index is written as

$$
n=1-\delta+i\beta
$$

where $\delta$ is the dimensionless refractive-index decrement and $\beta\geq0$ is the dimensionless absorption index. Both depend on material and photon energy. For a homogeneous material with thickness $t(x,y)$ in metres, the transmission relative to the same path in vacuum is

$$
T(x,y)=\exp\left[-ik(\delta-i\beta)t(x,y)\right]
$$

This transmission contains a phase factor $\exp(-ik\delta t)$ and an amplitude attenuation factor $\exp(-k\beta t)$. The phase shift is $-k\delta t$; its variation across the beam produces refractive focusing or wavefront distortion. The corresponding intensity attenuation is $\exp[-2k\beta t]$. Here $t$ denotes material thickness, not time.

### 3.6 Intensity

Intensity is proportional to the squared modulus of the complex field. In the library's normalization,

$$
I(x,y,z)=|E(x,y,z)|^2=A(x,y,z)^2
$$

The modulus removes the phase from the displayed intensity, but the phase remains essential for calculating subsequent propagation. Therefore, optical elements must pass **complex fields**, rather than intensity arrays, to one another. Absolute intensity units require a calibrated input-field normalization; otherwise the results are relative intensities.

<!-- Physical-model references to be added by the author. -->
