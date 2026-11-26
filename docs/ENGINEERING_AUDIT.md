# Engineering and scientific audit

Audited on 2026-11-26 against commit `d736efa` and XPOSim 0.34.0. Scope includes all 4,982 lines of the numerical module by subsystem, 40 public exports, backend adapters, material parsing, current tests, README, API guide, Colab workflow, environment specification and tracked infrastructure. This is a first engineering audit. Independent reproduction of published CRL results and execution on actual accelerators remain open.

## Current state

The core provides a useful scalar Fresnel model, 1D/2D grids, sources, apertures, sampled objects, finite CRL stacks, crossed lenses, partial-coherence approximations, batch propagation and local Fourier detectors. Public geometry parameters are protected against reassignment. The double-precision Fourier implementation has backend parity tests including complex fields. Source and detector batching already address substantial memory pressure. SI length conventions and material attenuation are documented, and the README contains a self-checking lens example.

The repository has one large implementation module and manually maintained documentation. Before this audit package there were no tracked build metadata, CI workflows, dedicated independent physics reference suite, benchmark harness, documentation build, license, citation file, contribution policy or issue templates. The conda file is a detailed Linux/CUDA environment snapshot. It is unsuitable as the sole portable installation contract.

Backend parity tests share the same propagation algorithm and cannot independently establish its physical validity. Current backend array signatures largely use `Any`. New field/system/backend abstractions, versioned scheme serialization, general precision selection and deterministic resource disposal require design work.

## Top ten issues by impact

| Priority | Finding and evidence | Consequence and next action |
| --- | --- | --- |
| 1, P0 | Independent analytic coverage was absent. Existing tests mostly compare shared code paths and ray/stack compositions. | A common normalization, phase or approximation error can pass all backend checks. The first package adds eight independent reference checks. Extend to finite-stack focus, open-aperture diffraction, convergence and published data. |
| 2, P0 | `_centered_reciprocal_coordinates` shifts even-grid frequencies by half a bin. The zero frequency is missing. | Even-grid continuation changes phase at the boundary. A spatially uniform field develops boundary diffraction. Document this now, then design explicit boundary modes and compatibility migration. |
| 3, P1 | `_ideal_thickness` caps CRL material thickness outside its aperture. `_build_focus_session` separately masks incident illumination. | Low- and high-level workflows with identical-looking lens settings can describe different illuminated objects. Add pupil-explicit comparison examples and regression tests before changing defaults. |
| 4, P1 | CI and release gates were absent, and the full module has 13 Ruff findings under isolated default rules. | Regression protection depends on local execution. Add CPU matrix and coverage reporting now. Address legacy lint in a separate behavior-preserving package, then gate it. |
| 5, P1 | Sampling diagnostics do not establish step/window convergence. Local-detector refinement cannot recover missing entrance frequencies. | A finely sampled focal image can conceal aliased entrance phase or finite-window contamination. Add step and window convergence cases, numerical uncertainty reporting and validated parameter ranges. |
| 6, P1 | GPU tests skip in this environment. There is no measured device memory/performance reference. TensorFlow uses graph caching, while much JAX work remains eager. | Speed rankings, device placement and memory lifetime need real-device evidence. Add hardware jobs, synchronized timings, traces and repeated-run memory checks. Avoid attributing timings to a backend architecture without measurements. |
| 7, P1 | Device arrays remain mutable despite configuration attribute guards. `CRLFocusResult` retains a session, source caches and potentially a temporary file without public close/context-manager methods. | Mutation may invalidate derived caches, and retaining results retains memory. Define ownership, mutation detection or copies, cache invalidation and explicit resource lifetime. Garbage collection alone is insufficient as a memory-use contract. |
| 8, P1 | No installable distribution or installed-resource test. Material files are resolved next to the module. Tables include DABAX/Windt/McMaster provenance, with no repository-wide redistribution policy. | Wheel migration can lose material resources. Licensing and data redistribution must be resolved before a distributable release. Add pyproject metadata and wheel tests, preserving the legacy import and single-file path. |
| 9, P2 | One module combines physics, array adapters, optimization, storage, plots and reporting. There is no scheme schema or plugin registry. Float32 is not an end-to-end contract. | Broad rewrites would risk API and physical behavior. Introduce additive typed layers and precision-specific validation in successive releases. |
| 10, P2 | Reproducibility metadata and project governance are incomplete. Material constants use rounded numerical factors and linear interpolation. No golden-case manifest or external benchmark evidence exists. | Scientific comparisons need data checksums, constants, backend/device versions, approximation regimes and provenance. Establish validation manifests, a version/deprecation policy, citation and maintainer-approved licenses. |

P0 identifies scientific confidence and boundary interpretation work that precedes a reference-quality claim. These priorities describe impact. They do not imply that existing localized-beam calculations are universally invalid.

## Confirmed physical conventions and limitations

The propagation multiplier has negative quadratic phase in reciprocal space, consistent with the positive-phase Fresnel kernel and a negative-sign forward Fourier transform. Material transmission attenuates for positive beta. CRL thickness uses both parabolic surfaces. Source quadrature sums intensities. `PointSource` supplies normalized phase illumination without absolute spreading amplitude. Formal focus, numerical peak intensity and minimum FWHM are different reported observables.

For the boundary diagnostic, propagating an all-ones field with dx=1 um, lambda=0.1 nm and z=10 m gave intensity range approximately 0.00452–2.43126 on 64 nodes. On 65 nodes the maximum complex-field change was about `1.2e-14` in relative L2 norm. This follows from the sampled basis. A convention change must have an explicit opt-in migration and a finite-window validation suite.

See the dedicated conventions document for Fourier normalization, coherence, precision and aperture semantics. Sources and approximation regimes are recorded alongside each new validation case. The current work has no independently reproduced published-paper curves or laboratory reference measurements. [Physical conventions](PHYSICAL_CONVENTIONS.md) · [Validation cases](VALIDATION.md)

## Measurements and reproducibility

The initial local regression run completed 25 tests with 19 passes and 6 skips. Two of these tests came from a pre-existing uncommitted edit to `tests/test_torch_backend.py`. That edit is excluded from this audit package. A separate copied test tree with the committed version of that file verifies the reviewable package independently. It completed 31 tests with 26 passes and 5 skips, with 80.56% statement coverage.

The full local suite with the eight new physical cases completed 33 tests with 27 passes and 6 skips, with 80.65% statement coverage of `XPOSim.py`. This coverage includes CPU execution of optional-framework kernels. It is not a coverage claim for a minimal NumPy-only environment or actual GPU execution. The CI reports its own coverage without imposing a fabricated threshold.

A local CPU timing baseline used an Intel Xeon Platinum 8573C, Linux x86_64, Python 3.12.14, NumPy 2.4.6 and SciPy 1.17.1. `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The input was a 5 um Gaussian amplitude on a 256×256 grid with 200 nm steps. The CRL had R=50 um, aperture=30 um, bridge=2 um, three elements, delta=3e-6, beta=2e-8 and energy=12 keV. Nine detector planes spanned 0.01–0.1 m, with batch size 3 and fields omitted.

After one warm-up, five wall times were 34.91, 31.47, 36.19, 33.98 and 32.71 ms. Median was 33.98 ms. One separate cProfile sample took approximately 34 ms. About 24 ms cumulative time was in the centered transform implementation, including 16 ms in raw FFTs. Timings overlap within the call tree. Shared-host timings characterize this setup only. They establish no cross-hardware performance guarantee.

Existing lint was measured with `ruff check --isolated --select E4,E7,E9,F XPOSim.py` using Ruff 0.17.6. It reports 9 one-line control statements, 2 unused locals and 2 ambiguous names. The new physical suite passes isolated Ruff checks. Compatibility function `I` needs an exemption or a deprecation decision rather than an automatic rename.

## Staged roadmap and acceptance

| Stage | Change and files | Verification and acceptance | Risks and rollback |
| --- | --- | --- | --- |
| A, first package | `docs/ENGINEERING_AUDIT.md`, `docs/PHYSICAL_CONVENTIONS.md`, `docs/VALIDATION.md`, `docs/API_DESIGN.md`, `tests/test_physics_validation.py`, `.github/workflows/validation.yml`, README links. | Eight analytic/discrete tests pass with documented tolerances. Existing tests pass. Lint new code. CPU CI configured for Linux Python 3.11/3.12/3.13 and Windows 3.12. CI execution itself requires the remote service. | Test tolerances may reveal platform effects. Diagnose against references before relaxing. Revert the additive package independently. |
| B, physics confidence | Extend physics tests, versioned JSON/NPZ fixtures and conventions. Add explicit boundary/pupil design. | Independent Gaussian/ABCD finite-stack CRL focus, slit/Airy limits, step and window convergence, coherence quadrature convergence. Reproduce at least one published case with checked parameters and rights. Golden manifests carry reference provenance and SHA256. | Formula/finite-model mismatch and aliased reference data. Keep existing conventions as a named legacy mode. Separate reference corrections from solver changes. |
| C, distributable engineering | `pyproject.toml`, package/resource layout, compatibility facade, type/lint config, pre-commit, CI wheel/sdist jobs. | Install wheel and sdist in clean environments outside the checkout. All 40 exports and material calculations work. GPU dependencies remain optional. Fix and gate baseline lint. Gate new typed modules and track whole-library coverage. | Import identity, material paths and notebook pins. Preserve old entry point and test both installed and single-file modes. Revert packaging separately. |
| D, measured performance | CPU/GPU benchmark cases, metadata schemas, profiler traces, memory lifetime methods and tests. | Warm and cold timings, compilation separated, explicit synchronization, median/dispersion, peak memory, device model/driver/versions. At least one real supported GPU backend initially, then all four. Flag stable-hardware median regressions above 10% with repeated confirmation. | Noisy runners and framework allocators. Treat threshold as investigation trigger. Keep physical reference gates mandatory and revert optimization independently. |
| E, additive API | New typed backend/field/propagator/system layers, precision settings, registry and JSON schema, optional YAML adapter. | Legacy parity, dtype-specific physical tolerances, serialization round trips, no implicit cross-device conversion, deterministic ordering and bounded-memory scans. | Hidden copies, stale caches and graph/JIT invalidation. Ship adapters in small releases and keep legacy facade covered. |
| F, user experience and adoption | MkDocs or Sphinx site, validated examples/notebooks, migration guide, CONTRIBUTING, code of conduct, templates, version policy, CITATION and licenses. | Fresh-user installation and checked lens example in 15 minutes on supported CPU setups. Executable documentation, link checking, clear backend failures, maintainer-approved authorship/licenses and data rights. | Incorrect attribution or promises of support. Obtain owner decisions on license, citation authors and support scope before publishing those policies. |

Each stage should be split into focused changes containing tests, documentation and a recorded result. Performance-changing packages include comparable before/after measurements. Never combine a Fourier-convention change with a packaging move or backend refactor.

## Criteria for a reference-quality candidate

Proposed criteria should become release gates after baseline measurements and maintainer review.

- Versioned public API with compatibility tests, documented deprecation windows and an explicit supported-version policy.
- Every supported optical model has at least one independent analytic reference and demonstrated convergence in a declared regime. Every production backend passes the same suite on actual target hardware with precision-specific tolerances.
- At least one independently reproduced publication case and a documented comparison with an established tool such as SRW or WOFRY in an overlapping scalar-paraxial regime. Record discrepancies and approximations.
- At least two independent research groups reproduce the exchangeable validation suite and review reference updates. Preserve their environment and case metadata.
- Installed-package, resource and documentation checks pass on declared Python/OS combinations. Suggested coverage targets are at least 90% for the new numerical core, plus reviewed physical invariants and explicit gap tracking.
- Public CPU/GPU benchmark histories identify hardware and compilation/warm-up costs. Sustained-memory tests establish resource release. Regression thresholds are evaluated on stable runners.
- A maintainer-approved citation, code/data licenses, contribution process, release notes and support policy exist. A proposed LTS commitment must identify maintainers and capacity before it is advertised.

These are measurable project goals. Current evidence supports an evolving scientific tool with an initial validation foundation. Standard status depends on independent adoption and sustained verification.
