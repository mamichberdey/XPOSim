# XPOSim

**X-ray Paraxial Optical Simulation — wave optics with GPU acceleration.**

[Repository](https://github.com/mamichberdey/XPOSim) · [Documentation and examples](DOCUMENTATION.md)

**Contents**

1. [About XPOSim](#1-about-xposim)
2. [Installation](#2-installation)
3. [Physical model](#3-physical-model)

---

## 1. About XPOSim

XPOSim is a Python library for simulating X-ray propagation and focusing in one or two transverse dimensions. It calculates complex wavefields and intensity distributions for compound refractive lenses (CRLs), apertures, and objects described by material-thickness maps, including lens surface errors.

The library is intended for studying focal distributions, evaluating optical imperfections, and generating synthetic data for X-ray optical diagnostics.

### Main advantages

- **Efficient propagation:** the paraxial approximation reduces free-space propagation to a convolution, evaluated using FFTs through the convolution theorem.
- **GPU acceleration:** CuPy runs the main array operations on NVIDIA GPUs. This is the **recommended setup**, especially for large 2D grids and repeated calculations.
- **Optional CPU backend:** NumPy and SciPy are sufficient for the library itself; CUDA and CuPy are optional. CPU calculations are useful for small examples and checks, but large simulations will generally be slower.

> **First calculation:** after installation, follow the [complete source → CRL → detector example](DOCUMENTATION.md#2-complete-example-point-source-crl-and-detector).

---

## 2. Installation

The commands below are for Linux or the Linux terminal in WSL 2. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) first.

### 2.1 Download the project

Choose one option:

- **ZIP archive:** [download and extract the project](https://github.com/mamichberdey/XPOSim/archive/refs/heads/main.zip). Git is not required.
- **Git clone:** if [Git](https://git-scm.com/downloads) is installed, download the repository with:

```bash
git clone https://github.com/mamichberdey/XPOSim.git
```

Enter the project folder. For a Git clone:

```bash
cd XPOSim
```

For a ZIP archive, use the extracted folder name, normally `XPOSim-main`, instead. The current directory should contain `XPOSim.py` and `environment.yml`.

### 2.2 GPU installation — recommended

This option requires an NVIDIA GPU and a compatible driver. The supplied environment targets **Linux x86_64 / WSL 2 x86_64** and specifies **CUDA 13.4**. Under WSL 2, GPU access is provided by the Windows NVIDIA driver; see [NVIDIA's setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

Create the environment from the supplied dependency file:

```bash
conda env create -f environment.yml
```

Activate it before running the project:

```bash
conda activate cu_env
```

The environment includes CuPy and CUDA runtime dependencies. The NVIDIA driver must be installed separately. Select `backend="cupy"` when creating a calculation grid. Repeat the activation command when opening a new terminal session.

### 2.3 CPU installation — optional

Use this option if a compatible GPU is unavailable or you only need small calculations. **CUDA, an NVIDIA driver, and CuPy are not required.** Create a separate environment with Python and the two required library dependencies:

```bash
conda create -n xposim_cpu --override-channels -c conda-forge python=3.11 numpy scipy
```

Activate the CPU environment:

```bash
conda activate xposim_cpu
```

Select `backend="numpy"` when creating grids and use NumPy arrays in your code. The supplied GPU `environment.yml` is not needed for this option. Plotting examples additionally require Matplotlib; see the [CPU usage guide](DOCUMENTATION.md#7-one-dimensional-and-cpu-calculations) for backend conventions.

### 2.4 Project files and material data

Run scripts from the project root so that `XPOSim.py` can be imported. Keep the included `xray_material_data/` directory next to the module: its tables are used to calculate optical constants from chemical formulas and density. Materials specified directly by `delta` and `beta` do not need these tables. This applies to both CPU and GPU calculations.

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

*References for the physical model will be added by the author.*
