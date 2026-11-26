# Physical and numerical conventions

This document records XPOSim 0.34.0 behavior. Distances, radii, surface heights and wavelength are in metres. Photon energy is in keV, density in g/cm³, and angles in radians. A 2D array is indexed `[y, x]`. Intensity is `abs(E)**2` in input-dependent relative units. Integrated power uses `sum(I)*dx` in 1D, or `sum(I)*dx*dy` in 2D. A 1D model represents a transverse section per unit invariant transverse extent.

## Field and propagation

The scalar monochromatic envelope uses the carrier convention `exp(i*k*z-i*omega*t)`. The carrier is omitted from returned arrays. The paraxial model retains transverse wavevector terms through second order. It excludes polarization, back-reflections, vector focusing and nonlinear response. Check both angular support `|q|/k << 1` and spatial resolution for a proposed application. Having a resolved intensity alone does not establish that a phase profile is resolved.

In one dimension the Fourier convention is

\[
\widehat E(q)=\int E(x)e^{-iqx}\,dx,\qquad
E(x)=\frac{1}{2\pi}\int\widehat E(q)e^{iqx}\,dq.
\]

In two dimensions each inverse-transform axis contributes `1/(2*pi)`. The discrete forward transform includes `dx` or `dx*dy`, and the inverse includes reciprocal normalization. Consequently Parseval power is `sum(abs(Ehat)**2)*dq/(2*pi)` in 1D.

Free propagation multiplies the spectrum by

\[
H(q_x,q_y,z)=\exp[-iz(q_x^2+q_y^2)/(2k)].
\]

Negative distances invert free propagation within the discrete model. They do not invert an absorbing object. `OpticalElement.E(z=0)` returns the exit-plane field, while positive `z` is measured from that exit. `PointSource.E(z)` requires positive source distance and returns a unit-modulus quadratic phase. It omits geometrical spreading and absolute source flux normalization.

## Material and CRL

The refractive index convention is `n=1-delta+i*beta`. Material thickness `t` transmits `exp(-i*k*delta*t-k*beta*t)`. Intensity attenuation is `exp(-2*k*beta*t)`.

For a parabolic lens, two surfaces give `t=bridge+x_local**2/radius` within the aperture. A sampled surface-height error is doubled when constructing total thickness. The finite CRL model alternates compressed material interactions with vacuum drift, using half a period at each stack end. Analytical continuously refractive results and the finite discrete stack have distinct approximations. Agreement must be assessed in a regime where both apply.

Outside the nominal aperture the current CRL thickness is capped at the rim thickness. It transmits through a material slab with attenuation. The high-level focus workflow separately restricts incident illumination to the entrance pupil. Low-level schemes needing that stop should compose a `Hole` with the CRL. This difference must be included when comparing analytical aperture diffraction and the two entry points.

The tabulated material path currently takes the absolute value of the real susceptibility when reporting delta. Negative f1 entries near absorption features can therefore lose their dispersion sign. Direct `Material(delta=..., beta=...)` accepts signed delta. Missing-value metadata in the source tables also needs explicit interpretation. Treat near-edge material calculations as requiring an independent optical-constant check until that behavior has a tested migration. This issue does not justify changing Fourier or attenuation sign conventions.

## Sampling and boundaries

Both axes use `x_j=(j-(N-1)/2)*dx` and `q_l=(l-(N-1)/2)*2*pi/(N*dx)`. Even counts have half-integer frequencies and no DC frequency. The spectral basis obeys `E(x+N*dx)=-E(x)` for even counts, and periodic continuation for odd counts. In 2D this applies independently to each axis.

This is a phase-twisted finite-window spectral representation. A constant plane wave is an eigenmode for odd counts. For even counts it is incompatible with the continued boundary, and propagation introduces boundary diffraction. A localized beam with negligible field at the edges can approximate open-space propagation. Compare larger windows at fixed step and smaller steps at fixed window, inspecting complex fields as well as intensity. The current edge-power warnings provide diagnostics, without an accuracy guarantee. Changing these conventions requires an explicit boundary-mode API and migration tests.

Local-detector chirp-z evaluation samples the same discrete Fourier series at new coordinates. It cannot recover unresolved entrance-plane structure or remove finite-window effects. A requested ROI may clip half-maximum crossings. FWHM then returns `None`.

## Coherence and precision

Individual fields propagate coherently. Gaussian point-source quadrature sums independently propagated intensities with normalized weights. Gaussian intensity convolution is an approximation whose adequacy depends on displacement response and clipping. Compare quadrature convergence when that assumption matters. General cross-spectral-density and polarization objects are future work.

Coordinates and phases currently use double precision, producing predominantly complex128 fields. There is no public end-to-end float32 precision contract. JAX initialization enables x64 globally. Backend auto-selection tries GPU implementations then NumPy. Explicit accelerator requests require a working GPU.

`wavelength_energy` currently uses the rounded constant `hc=12.3984e-10 keV*m`. Its relative difference from SI-defined constants is below `2e-6`. Changing it will shift established results and needs a documented numerical migration.

## Sources

The Fresnel kernel and thin-lens relations follow scalar Fourier optics with the explicit transform convention above. See J. W. Goodman, *Introduction to Fourier Optics*, third edition, chapters 4 and 5. The Gaussian reference in the validation suite is obtained by substituting a Gaussian into that kernel and evaluating the Gaussian integral.

The CRL model and its scope are described in V. G. Kohn, “On the theory of synchrotron radiation nanofocusing with planar compound refractive lenses,” *Journal of Synchrotron Radiation* 29, 615–621 (2022). This document records implementation conventions. Reproduction of that paper's numerical figures remains a separate validation task. [DOI](https://doi.org/10.1107/S1600577522001345)
