# Reference validation suite, initial revision

Run from the repository root with NumPy and SciPy installed.

```sh
python -m unittest discover -s tests -p test_physics_validation.py
python -m pytest tests --cov=XPOSim --cov-report=term-missing
python -m ruff check --isolated tests/test_physics_validation.py
```

The first command has no optional GPU or test-framework dependency. The other commands require pytest, pytest-cov and Ruff. The CI workflow installs these development tools separately.

## Cases implemented in the first audit package

| Test | Setup and independent expected result | Acceptance |
| --- | --- | --- |
| `fourier_quadrature_and_parseval` | Seed 937, complex random fields, 32 and 33 nodes, step 200 nm. Explicit matrix `dx*exp(-i*q*x)` gives the reference DFT. Parseval includes `dq/(2*pi)`. | DFT relative `1e-11`, absolute `1e-18` in field-times-length units. Round trip absolute `1e-12`. Integrated power absolute `1e-17` for this setup. |
| `gaussian_complex_field` | 1D amplitude `exp(-x²/w²)`, w=2 um, lambda=0.1 nm, z=80 mm, 1024/1025 nodes at 40 nm. With b=1+2iz/(kw²), expected field is `exp(-x²/(w²*b))/sqrt(b)`. | Relative `1e-8`, absolute `2e-10` in input amplitude units, including phase. |
| `two_dimensional_gaussian_and_power` | Separable 2D Gaussian with w_x=1.4 um, w_y=1.8 um, lambda=0.1 nm, z=40 mm. Product of independent 1D analytic fields. Also reverse propagation. | Field absolute `2e-10`, power ratio within `1e-12`, round trip absolute `1e-12`. |
| `thin_lens_gaussian_focus` | Amplitude Gaussian w=3 um with phase `-kx²/(2f)`, f=50 mm, lambda=0.1 nm. Analytic complex Gaussian propagation. Focal intensity FWHM is `sqrt(2*ln(2))*2*f/(k*w)`. | Field absolute `2e-10`, FWHM relative `0.002` to allow sampled crossing interpolation. |
| `slit_fresnel_quadrature_convergence` | Uniform slit of width 4 um, lambda=0.1 nm, z=20 mm. Independent SciPy Fresnel integrals provide the complex reference at 61 detector points. Midpoint integration is repeated at 40/20 nm. The library evaluates propagation onto that detector from a 160 um window. | Chirp-z field absolute `0.004`. Independent midpoint integration error decreases by more than a factor of 3 when halving the step. |
| `crl_material_transmission` | Single parabolic interaction, R=50 um, bridge=2 um, delta=3e-6, beta=2e-8. Compute thickness directly as `bridge+x²/R`. Check complex phase and Beer–Lambert intensity loss independently. | Absolute `1e-14` in transmission and relative incident intensity units. |
| `sampled_plane_wave_eigenmode` | 64/65 nodes, a wavevector exactly on the grid's sampled frequency axis. Propagation must multiply the wave by `exp(-iz*q²/(2k))`. | Field absolute `1e-12`. This also documents even-grid half-integer frequency behavior. |
| `energy_wavelength_units` | Compare 12 keV conversion against `h*c/(energy in joules)` using SciPy SI constants. | Relative `2e-6`, reflecting the legacy rounded hc constant. |

Gaussian and thin-lens expressions follow directly from the Fresnel kernel by completing the square. For `E0=exp(-a*x²)`, propagation gives `E_z=(1+2iaz/k)^(-1/2)*exp[-a*x²/(1+2iaz/k)]`, with the square-root branch continuous from z=0. A thin lens adds `i*k/(2*f)` to a. These equations supply an independent calculation of the complex field.

The slit reference uses the integral from -a to a of `exp[i*pi*(x-x')²/(lambda*z)]/sqrt(i*lambda*z)`. Substitution `u=sqrt(2/(lambda*z))*(x'-x)` yields the difference of Fresnel C and S functions divided by `sqrt(2i)`. SciPy documents these functions and their normalization. [SciPy Fresnel definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel.html)

The reference physical framework is scalar Fresnel diffraction and thin-lens Fourier optics in J. W. Goodman, *Introduction to Fourier Optics*, third edition, chapters 4 and 5. Formula derivations above make the conventions explicit. These are analytic checks, not a reproduced experimental dataset.

Tolerances apply to the stated grids and normalizations. They are deliberately tighter for discrete algebraic identities than for finite-window diffraction of a hard edge. Passing one grid does not establish convergence for arbitrary user inputs. The slit convergence assertion currently concerns independent quadrature, while library window and step convergence remains a planned case.

## Existing backend regressions

`tests/backend_parity.py` checks complex fields, intensity, FWHM, filters, interpolation, divergence and high-level focus against NumPy. Torch, TensorFlow and JAX kernels can run in CPU-only validation using test-only device injection. Production accelerator selection continues to require a GPU. Actual-device tests run conditionally. CuPy requires a CUDA environment. Skipped tests must be reported alongside successful tests.

CPU emulation does not validate GPU placement, asynchronous timing, device peak memory or PCIe traffic. Those require the hardware jobs described in the roadmap.

## Exchange format for future reference cases

Each published case should contain a JSON manifest and an NPZ data file, loaded with `allow_pickle=False`. The manifest should identify the schema version, case ID, description, source DOI/URL and equation, SI parameters, phase convention, dimensionality, grid/window, approximation regime, reference method, reference-code version, data SHA256, observable definitions, absolute and relative tolerances with units, and convergence evidence.

Store complex reference fields or an explicitly justified intensity-only observable. Use dimensionless relative errors plus physically meaningful absolute floors. Compare phase only where reference amplitude is resolved. Separate an independent analytic/published reference from a frozen library regression output in the manifest. Golden data updates require scientific review and a recorded reason.

Later cases must include finite-stack CRL focus against independent ABCD/Gaussian propagation, aperture Airy/sinc limits, library grid/window convergence, partially coherent quadrature convergence and actual GPU parity. Zone plates and gratings need explicit sampled phase/transmission models before specialized public element classes are promised. A published-paper comparison needs verified parameters and reference data with redistribution rights.
