# Physical model

These equations describe the scalar paraxial model. For units, finite-window
boundary conditions and numerical limitations, see
[physical and numerical conventions](PHYSICAL_CONVENTIONS.md).

The formulation below follows the wave-propagation framework described by **V. G. Kohn**. [Kohn (2022)][kohn2022], Section 2, presents the paraxial equation, Fresnel propagation, FFT evaluation, and the compressed-lens approximation. The analytical theory of continuously refractive parabolic lenses is developed in [Kohn (2003)][kohn2003]; diffraction and imaging with CRLs are discussed by [Kohn, Snigireva & Snigirev (2003)][kohn-imaging2003].

## Complex amplitude and the paraxial approximation

XPOSim describes monochromatic radiation: a calculation uses a single photon energy and its corresponding wavelength. The beam travels mainly along the optical axis $z$, while $x$ and $y$ are transverse coordinates. All coordinates and distances are expressed in metres.

The calculated quantity is the **complex field envelope**

$$
E(x,y,z)=A(x,y,z)\exp[i\varphi(x,y,z)]
$$

where $A\geq0$ is the local amplitude, $\varphi$ is the phase in radians, and $i$ is the imaginary unit ($i^2=-1$). Keeping both amplitude and phase allows the model to describe interference, diffraction, and focusing. The common longitudinal carrier phase is omitted from this envelope; it does not affect the calculated intensity.

The **paraxial approximation** assumes propagation at small angles to $z$ and a field envelope that changes slowly along that axis compared with the wavelength. XPOSim uses a scalar field: polarization effects and back-reflections are outside this model. The following relations describe propagation between planes and interaction with material, without solving the full electromagnetic problem. See [Kohn (2022)][kohn2022], equations (1)–(2), for the field representation and paraxial equation.

## Free-space propagation: the Fresnel propagator

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

## Propagation as a convolution

The field in the observation plane is obtained by summing the contributions from the entire input plane. Mathematically, this is a convolution over the transverse coordinates:

$$
E(x,y,z+\Delta z)=E(x,y,z)*P_{2D}(x,y,\Delta z)
$$

The symbol $*$ denotes convolution, not pointwise multiplication. By the convolution theorem,

$$
E_{z+\Delta z}=\mathcal{F}^{-1}\left[\mathcal{F}(E_z)\,\mathcal{F}(P_{\Delta z})\right]
$$

where $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the forward and inverse Fourier transforms over $x,y$ (or $x$ in 1D). The shorthand $E_z$ denotes the field in plane $z$, and $P_{\Delta z}$ denotes the propagator for the distance $\Delta z$. XPOSim uses the analytic Fourier transform of the Fresnel propagator and computes the field transforms with FFTs. This replaces direct evaluation of the diffraction integral with two FFTs and an elementwise multiplication, which are efficiently performed on the GPU. The convolution formulation and its FFT evaluation are described in [Kohn (2022)][kohn2022], equation (4) and the accompanying text.

## Interaction with a thin object

A thin optical object modifies the field through its complex transmission function $T$:

$$
E_{\mathrm{out}}(x,y)=E_{\mathrm{in}}(x,y)T(x,y)
$$

Here $E_{\mathrm{in}}$ and $E_{\mathrm{out}}$ are the complex fields immediately before and after the interaction, and $T$ is dimensionless. Unlike propagation, this is pointwise multiplication: each sample is modified locally.

The thin-object approximation neglects diffraction within that individual interaction. A CRL stack is represented by a sequence of material interactions separated by propagation between effective lens planes, so diffraction through the assembly is still taken into account. See the compressed-lens approach in [Kohn (2022)][kohn2022], Section 2.1, equations (5)–(6).

## Material transmission: phase shift and attenuation

For X-rays, the material refractive index is written as

$$
n=1-\delta+i\beta
$$

where $\delta$ is the dimensionless refractive-index decrement and $\beta\geq0$ is the dimensionless absorption index. Both depend on material and photon energy. For a homogeneous material with thickness $t(x,y)$ in metres, the transmission relative to the same path in vacuum is

$$
T(x,y)=\exp\left[-ik(\delta-i\beta)t(x,y)\right]
$$

This transmission contains a phase factor $\exp(-ik\delta t)$ and an amplitude attenuation factor $\exp(-k\beta t)$. The phase shift is $-k\delta t$; its variation across the beam produces refractive focusing or wavefront distortion. The corresponding intensity attenuation is $\exp[-2k\beta t]$. Here $t$ denotes material thickness, not time. This transmission function corresponds to [Kohn (2022)][kohn2022], equation (6), with $k=K$ and $\delta-i\beta=\eta$ in that article.

## Intensity

Intensity is proportional to the squared modulus of the complex field. In the library's normalization,

$$
I(x,y,z)=|E(x,y,z)|^2=A(x,y,z)^2
$$

The modulus removes the phase from the displayed intensity, but the phase remains essential for calculating subsequent propagation. Therefore, optical elements must pass **complex fields**, rather than intensity arrays, to one another. Absolute intensity units require a calibrated input-field normalization; otherwise the results are relative intensities.

## References

1. **Kohn, V. G. (2022).** On the theory of synchrotron radiation nanofocusing with planar compound refractive lenses. *Journal of Synchrotron Radiation*, **29**, 615–621. [DOI: 10.1107/S1600577522001345][kohn2022]. [Open full text](https://journals.iucr.org/s/issues/2022/03/00/mo5253/index.html).

2. **Kohn, V. G. (2003).** An Exact Theory of Imaging with a Parabolic Continuously Refractive X-ray Lens. *Journal of Experimental and Theoretical Physics*, **97**, 204–215. [DOI: 10.1134/1.1600812][kohn2003]. [Author's PDF](https://kohnvict.narod.ru/articles/133e.pdf).

3. **Kohn, V., Snigireva, I., & Snigirev, A. (2003).** Diffraction theory of imaging with X-ray compound refractive lens. *Optics Communications*, **216**, 247–260. [DOI: 10.1016/S0030-4018(02)02285-X][kohn-imaging2003].

[kohn2022]: https://doi.org/10.1107/S1600577522001345
[kohn2003]: https://doi.org/10.1134/1.1600812
[kohn-imaging2003]: https://doi.org/10.1016/S0030-4018(02)02285-X
