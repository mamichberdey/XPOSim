# XPOSim (X-ray Paraxial Optical Simulation)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/2e5871f4-7dce-4fb5-afea-e0f77d400280">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/4990525d-43d6-4419-b972-7389c84afb40">
  <img width="600" alt="XPOSim_logo" src="https://github.com/user-attachments/assets/4990525d-43d6-4419-b972-7389c84afb40">
</picture>

**GPU-accelerated paraxial wave optics for X-ray optical systems.**

XPOSim calculates complex wavefields and intensity distributions in optical systems with one or two transverse dimensions. It supports compound refractive lenses (CRLs), apertures, sampled material objects, and lens surface errors. Applications include studying focusing, evaluating optical imperfections, and generating simulated data for X-ray optical diagnostics.

In the paraxial approximation, free-space propagation is a convolution with the Fresnel propagator. The convolution theorem allows this operation to be evaluated efficiently using FFTs. CuPy accelerates the main array calculations on NVIDIA GPUs; NumPy is also supported for CPU calculations.

Project repository: [mamichberdey/XPOSim](https://github.com/mamichberdey/XPOSim).

See the [user guide and API reference](DOCUMENTATION.md) for a complete worked example, optical-scheme construction, and parameter descriptions.

## Installation

The following steps target **Linux x86_64 or WSL 2 on x86_64**. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) first. Git is optional and is needed only if you choose to clone the repository. GPU calculations require a compatible NVIDIA GPU and driver. Under WSL 2, use the Windows NVIDIA driver; see [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

**1. Obtain the project files using either option below.** You only need to choose one.

- **Download a ZIP archive:** [download the main branch](https://github.com/mamichberdey/XPOSim/archive/refs/heads/main.zip) and extract it. Alternatively, select **Code → Download ZIP** on the [repository page](https://github.com/mamichberdey/XPOSim). Git is not required.
- **Clone with Git:** install [Git](https://git-scm.com/downloads), then run:

```bash
git clone https://github.com/mamichberdey/XPOSim.git
```

Both options provide the project files needed to run XPOSim. Cloning also retains Git version history and supports fetching later updates; the ZIP archive is a snapshot without that history.

**2. Enter the project directory.** For a Git clone, run this command from the folder where you downloaded the project:

```bash
cd XPOSim
```

For a ZIP download, use the extracted directory name, normally `XPOSim-main`, instead of `XPOSim`. If it is elsewhere, supply its full path. Continue from the project root containing `environment.yml` and `XPOSim.py`.

**3. Create the environment** using the supplied dependency file.

```bash
conda env create -f environment.yml
```

The supplied file specifies CUDA 13.4 and requires compatible hardware and a driver. CUDA dependencies are installed inside the environment; the NVIDIA driver is installed separately.

**4. Activate the environment.** Repeat this step in each new terminal session before running the project.

```bash
conda activate cu_env
```

Run scripts from the repository root, where `XPOSim.py` is located. For formula-based materials, the following data files must be present next to the module:

- `xray_material_data/AtomicConstants.dat`
- `xray_material_data/CrossSec-Compton_McMaster.dat`
- `xray_material_data/f1f2_Windt.dat`

These material tables are included in the repository and in its ZIP archive; Conda does not install them. Materials specified directly by `delta` and `beta` do not require these tables.

## Physical model

### Complex amplitude and the paraxial approximation

XPOSim describes monochromatic radiation: a calculation uses a single photon energy and its corresponding wavelength. The beam travels mainly along the optical axis $z$, while $x$ and $y$ are transverse coordinates. All coordinates and distances are expressed in metres.

The calculated quantity is the **complex field envelope**

$$
E(x,y,z)=A(x,y,z)\exp[i\varphi(x,y,z)],
$$

where $A\geq0$ is the local amplitude, $\varphi$ is the phase in radians, and $i$ is the imaginary unit ($i^2=-1$). Keeping both amplitude and phase allows the model to describe interference, diffraction, and focusing. The common longitudinal carrier phase is omitted from this envelope; it does not affect the calculated intensity.

The **paraxial approximation** assumes propagation at small angles to $z$ and a field envelope that changes slowly along that axis compared with the wavelength. XPOSim uses a scalar field: polarization effects and back-reflections are outside this model. The following relations describe propagation between planes and interaction with material, without solving the full electromagnetic problem.

### Free-space propagation: the Fresnel propagator

Propagation over a positive distance $\Delta z$ is described by the Fresnel propagator. With two transverse coordinates,

$$
P_{2D}(x,y,\Delta z)=\frac{1}{i\lambda\Delta z}
\exp\left[\frac{i\pi(x^2+y^2)}{\lambda\Delta z}\right],
$$

where $\lambda$ is the radiation wavelength in metres. Here $x$ and $y$ represent transverse separations between a point in the input plane and a point in the observation plane. The propagator describes the complex contribution transferred between these points, including its phase.

For a model with one transverse coordinate, the normalization changes:

$$
P_{1D}(x,\Delta z)=\frac{1}{\sqrt{i\lambda\Delta z}}
\exp\left[\frac{i\pi x^2}{\lambda\Delta z}\right].
$$

These propagators act on the complex envelope and omit the common carrier factor $\exp(ik\Delta z)$, where $k=2\pi/\lambda$ is the vacuum wavenumber in radians per metre.

### Propagation as a convolution

The field in the observation plane is obtained by summing the contributions from the entire input plane. Mathematically, this is a convolution over the transverse coordinates:

$$
E(x,y,z+\Delta z)=E(x,y,z)*P_{2D}(x,y,\Delta z).
$$

The symbol $*$ denotes convolution, not pointwise multiplication. By the convolution theorem,

$$
E(z+\Delta z)=\mathcal{F}_{\perp}^{-1}
\left[\mathcal{F}_{\perp}\{E(z)\}\,
\mathcal{F}_{\perp}\{P(\Delta z)\}\right],
$$

where $\mathcal{F}_{\perp}$ and $\mathcal{F}_{\perp}^{-1}$ are the forward and inverse Fourier transforms over $x,y$ (or $x$ in 1D). Transverse arguments are suppressed in this expression for readability. XPOSim uses the analytic Fourier transform of the Fresnel propagator and computes the field transforms with FFTs. This replaces direct evaluation of the diffraction integral with two FFTs and an elementwise multiplication, which are efficiently performed on the GPU.

### Interaction with a thin object

A thin optical object modifies the field through its complex transmission function $T$:

$$
E_{\mathrm{out}}(x,y)=E_{\mathrm{in}}(x,y)T(x,y).
$$

Here $E_{\mathrm{in}}$ and $E_{\mathrm{out}}$ are the complex fields immediately before and after the interaction, and $T$ is dimensionless. Unlike propagation, this is pointwise multiplication: each sample is modified locally.

The thin-object approximation neglects diffraction within that individual interaction. A CRL stack is represented by a sequence of material interactions separated by propagation between effective lens planes, so diffraction through the assembly is still taken into account.

### Material transmission: phase shift and attenuation

For X-rays, the material refractive index is written as

$$
n=1-\delta+i\beta,
$$

where $\delta$ is the dimensionless refractive-index decrement and $\beta\geq0$ is the dimensionless absorption index. Both depend on material and photon energy. For a homogeneous material with thickness $t(x,y)$ in metres, the transmission relative to the same path in vacuum is

$$
T(x,y)=\exp\left[-ik(\delta-i\beta)t(x,y)\right]
=\exp[-ik\delta t(x,y)]\exp[-k\beta t(x,y)].
$$

The first factor introduces a phase shift $-k\delta t$; its variation across the beam produces refractive focusing or wavefront distortion. The second factor attenuates the amplitude. The corresponding intensity attenuation is $\exp[-2k\beta t]$. Here $t$ denotes material thickness, not time.

### Intensity

Intensity is proportional to the squared modulus of the complex field. In the library's normalization,

$$
I(x,y,z)=|E(x,y,z)|^2=A(x,y,z)^2.
$$

The modulus removes the phase from the displayed intensity, but the phase remains essential for calculating subsequent propagation. Therefore, optical elements must pass **complex fields**, rather than intensity arrays, to one another. Absolute intensity units require a calibrated input-field normalization; otherwise the results are relative intensities.

*References for the physical model will be added by the author.*
