# Reference validation suite, initial revision

Run from the repository root with NumPy installed. SciPy is optional for the independent Fresnel-integral reference and numerical-helper comparisons. Those reference cases are skipped when SciPy is absent.

```sh
python -m unittest discover -s tests -p test_physics_validation.py
python -m pytest tests --cov=XPOSim --cov-report=term-missing
python -m ruff check --isolated tests/test_physics_validation.py
```

The first command uses unittest and can run without optional GPU packages. The Fresnel-integral case uses SciPy when available. The other commands require pytest, pytest-cov and Ruff. The CI workflow installs these development tools separately.

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
| `energy_wavelength_units` | Compare 12 keV conversion against `h*c/(energy in joules)` using the exact SI defining constants h, c and e. | Relative `2e-6`, reflecting the legacy rounded hc constant. |

Gaussian and thin-lens expressions follow directly from the Fresnel kernel by completing the square. For `E0=exp(-a*x²)`, propagation gives `E_z=(1+2iaz/k)^(-1/2)*exp[-a*x²/(1+2iaz/k)]`, with the square-root branch continuous from z=0. A thin lens adds `i*k/(2*f)` to a. These equations supply an independent calculation of the complex field.

The slit reference uses the integral from -a to a of `exp[i*pi*(x-x')²/(lambda*z)]/sqrt(i*lambda*z)`. Substitution `u=sqrt(2/(lambda*z))*(x'-x)` yields the difference of Fresnel C and S functions divided by `sqrt(2i)`. SciPy documents these functions and their normalization. [SciPy Fresnel definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.fresnel.html)

The reference physical framework is scalar Fresnel diffraction and thin-lens Fourier optics in J. W. Goodman, *Introduction to Fourier Optics*, third edition, chapters 4 and 5. Formula derivations above make the conventions explicit. These are analytic checks, not a reproduced experimental dataset.

Tolerances apply to the stated grids and normalizations. They are deliberately tighter for discrete algebraic identities than for finite-window diffraction of a hard edge. Passing one grid does not establish convergence for arbitrary user inputs. The slit convergence assertion currently concerns independent quadrature, while library window and step convergence remains a planned case.

## Existing backend regressions

`tests/backend_parity.py` checks complex fields, intensity, FWHM, filters, interpolation, divergence and high-level focus against NumPy. Torch and JAX kernels can run in CPU-only validation using test-only device injection. Production accelerator selection continues to require a GPU. Actual-device tests run conditionally. CuPy requires a CUDA environment. Skipped tests must be reported alongside successful tests.

CPU emulation does not validate GPU placement, asynchronous timing, device peak memory or PCIe traffic. Those require the hardware jobs described in the roadmap.

## Exchange format for future reference cases

Each published case should contain a JSON manifest and an NPZ data file, loaded with `allow_pickle=False`. The manifest should identify the schema version, case ID, description, source DOI/URL and equation, SI parameters, phase convention, dimensionality, grid/window, approximation regime, reference method, reference-code version, data SHA256, observable definitions, absolute and relative tolerances with units, and convergence evidence.

Store complex reference fields or an explicitly justified intensity-only observable. Use dimensionless relative errors plus physically meaningful absolute floors. Compare phase only where reference amplitude is resolved. Separate an independent analytic/published reference from a frozen library regression output in the manifest. Golden data updates require scientific review and a recorded reason.

Later cases must include finite-stack CRL focus against independent ABCD/Gaussian propagation, aperture Airy/sinc limits, library grid/window convergence, partially coherent quadrature convergence and actual GPU parity. Zone plates and gratings need explicit sampled phase/transmission models before specialized public element classes are promised. A published-paper comparison needs verified parameters and reference data with redistribution rights.


## NumPy numerical core in 0.37.0

The runtime uses device-side bilinear interpolation, radix-smooth FFT padding, separable zero-padded NumPy Gaussian convolution, bounded Brent minimization and safeguarded bracketed inverse interpolation for roots. Gaussian source modes use a Golub-Welsch Jacobi matrix with diagonal zero and off-diagonal entries `sqrt(1), ..., sqrt(n-1)`. Squared first-row eigenvector entries give normalized Gaussian weights. Equal-probability nodes use `statistics.NormalDist.inv_cdf` with explicit infinite endpoints and variance restoration.

`tests/test_numpy_numerics.py` checks these helpers against analytic invariants and, when SciPy is present, independent SciPy implementations. The source quadrature uses a dense NumPy symmetric eigensolve, whose preparation cost grows with the quadrature order. Source-node convergence should still be checked when increasing the mode count.

A comparison of 0.36.0 with the replacement implementation used five 128-sample configurations, including Gaussian point/convolution sources and crossed lenses. At common detector distances, relative intensity L2 differences were at most `3.2e-16`. Peak-distance shifts were at most `7.9e-11 m`, within the configured adaptive-search tolerance. These CPU checks assess numerical agreement. They do not establish GPU performance of the new interpolation path.

The energy-conversion reference uses the exact SI values h=6.62607015e-34 J s, c=299792458 m/s and e=1.602176634e-19 C. [SI defining constants](https://www.bipm.org/en/measurement-units/si-defining-constants)

## Numerical stability and quadrature reuse

`tests/test_numerical_stability.py` checks minimization under objective scales
`1e-250` through `1e250`, flat objectives, narrow peaks with exhausted budgets,
reversed root brackets, nonfinite objectives, odd/even propagation round trips,
and FWHM invariance under intensity scales `1e-240` through `1e240`. Minimizer
positions must agree with the known quadratic minimum within `5e-8`; round-trip
complex fields use relative/absolute `2e-12`. Undefined scalar objectives fail
explicitly. Flat public peak searches report lack of convergence.

`test_library_sampling_and_window_convergence_against_gaussian` compares the
library's local-detector field with the analytic Gaussian above, using w=2 um,
lambda=0.1 nm, z=50 mm and an 81-node detector at 80 nm. Sampling is refined at
fixed 16 um entrance width with 8/16/32 nodes. Separately, a fixed 125 nm entrance
step uses 32/64/128 nodes. Each error must decrease at least tenfold per step and
finish below relative complex-field L2 error `1e-6`. Observed final errors on the
CPU were `8.22e-9` and `2.20e-8`. The initial coarse grids deliberately expose
sampling and window errors. This is a library convergence test against an
independent expression, rather than only an independent quadrature test.

Standard source nodes/weights now use a 16-entry LRU keyed by count and method.
Tests check reuse across physical source sizes, Gaussian moments, mutation
isolation and eviction. The initial dense eigensolve is unchanged.

Preparation timings measured in the cloud on an Intel Xeon Platinum 8573C,
x86_64, Python 3.12.14, NumPy 2.4.6, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`:

| Gaussian order | Cold median (range), ms | Cached median (range), ms |
| --- | --- | --- |
| 7 | 0.0417 (0.0235–0.240) | 0.00102 (0.000933–0.00138) |
| 201 | 4.15 (3.55–10.2) | 0.00114 (0.00108–0.00140) |

Seven cold trials cleared the cache before preparing sigma=2 um. Seven warm
trials each averaged 1000 calls with sigma=3 um at the same order; timing includes
scaling and copying the returned arrays. These measure quadrature preparation,
not full propagation, GPU execution or performance on a laptop. Large first-call
orders still need a more scalable eigensolver before claiming improved complexity.


## Retained CRL reliability and resource contract in 0.38.0

`tests/test_focus_reliability.py` checks storage estimates against explicit byte
counts, preflight progress ordering, phase/FWHM report dimensions, boundary-rim
union fractions including single-pixel/dark arrays, and JSON-compatible finite
reports. Context-manager and explicit-close tests force source spectra into a
temporary file. They verify file closure, dropped array references, repeatable
close, surviving exported data/summaries and rejected post-close plane requests.
Progress exceptions are injected during source preparation, sampling, finalizing
and completion to check deterministic cleanup, including partial construction.

Convergence tests cover even and odd 1D grids and a nonsquare mixed-parity 2D
grid. They check retained physical widths/steps and parity, original-result
integrity, closure of both temporary sessions, and per-observable stability
criteria. Missing widths cannot certify stability. Failed refinement leaves the
original session usable. These tests verify the convergence API's contract; the
analytic Gaussian case above separately establishes numerical improvement for a
known physical solution. The API leaves source quadrature and axial search
convergence to separate tests by design.

Closing releases owned references; allocator-reserved VRAM is not a leak test.
Actual-GPU allocation and timing checks remain hardware-dependent. No GPU memory
or speed conclusion is inferred from CPU test injection.
