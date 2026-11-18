# CRL focus explorer in Google Colab

[Open the notebook](https://colab.research.google.com/github/mamichberdey/XPOSim/blob/35e3727a90bfbc69dc654b3f78d2918fc89b896b/notebooks/CRL_Focus_Colab.ipynb).

Run all once to initialize two separate outputs: **2 · CRL and source parameters** contains the editable inputs; **3 · Results** contains Compute, the plots and the z slider. Enter the lens/source parameters in cell 2, then press **Compute** in cell 3. The parameter fields are always shown in their own output and are never automatically collapsed or hidden by a calculation. Reinitializing the parameter cell in the same live session preserves entered values. No optical calculation starts automatically. Results remain in memory; no export or save/restore mechanism is included.

## Start

1. Open the link and sign in to Colab. Optionally save a copy to your Drive.
2. For GPU execution, use **Runtime → Change runtime type → GPU → Save**, reconnect, and run all cells. CPU execution is also supported.
3. Setup checks dependencies and actual CUDA execution. Missing CuPy is installed only when a compatible CUDA 12/13 environment is detected. A small GPU FFT probe selects CuPy when CUDA works, otherwise NumPy. The notice reports the actual CUDA device, independent of the session's GPU/TPU label. CPU sessions display the GPU-enabling instructions. There is no backend picker.
4. Run **2 · CRL and source parameters** to display the editable fields below that cell. The form defaults to silicon at 12 keV, surface radius 6.25 µm, full aperture 50 µm, bridge thickness 2 µm, 26 elements and source distance 15 m. **Fast** sampling is selected initially.
5. Run **3 · Results**, then press **Compute** after setting the inputs. To change the lens or source, edit the still-visible fields in cell 2 and press **Compute** again. The results stay in their own compact workspace, so using the detector slider does not move the input form.

The form is ordered **Source → CRL — General → CRL element — Geometry / Material**. Source controls appear first; focusing/grid dimensions, sampling and element count are grouped separately. Geometry and material occupy adjacent cards. Fields keep fixed positions: unused fields are disabled, while their values remain stored. Changing source type, grid or material definition does not insert/remove inputs. Advanced fields also retain their positions when enabled/disabled.

## Three ways to define the material

- **Preset material:** choose Be, Si, Al or diamond. The density selector beside its field offers **Auto** or **Manual**. Auto displays the actual preset density: 1.848, 2.33, 2.70 or 3.515 g/cm³ respectively. The first switch to Manual starts from the displayed preset density; an edited manual override is retained when switching modes.
- **Chemical formula:** enter a formula such as SiO2 and a manual density. The preset selector and automatic-density choice are inactive.
- **Direct δ / β:** enter dimensionless coefficients in scientific notation, for example `3.4e-6` and `3e-8`, using n = 1 − δ + i β. Formula and density are inactive and unused. Delta must be positive for this focusing-CRL workflow; beta may be zero. Coefficients are used as entered at the selected photon energy and are not automatically rescaled when energy changes. On first entry, the fields are seeded from the current valid material preview; edited values then persist across mode changes.

The library accepts `CRLFocusConfig(material_delta=..., material_beta=...)`. In this mode `material_formula` and `density_g_cm3` are normalized to `None`; database tables are bypassed.

## Analytical preview before Compute

Committing an input edit automatically refreshes the preview without propagating the wavefield or probing a GPU. Geometry shows **Analytical focal distance**, **Analytical image distance**, **Maximum CRL elements** and **Full CRL length**. Material shows δ and β in separate readouts. The same preview works for all three material modes.

The preview delegates to the existing CRL focal-distance and maximum-element-count functions. Distances are measured from the CRL exit. Maximum N refers to the collimated external-focus limit; finite-source imaging additionally requires a positive, finite image distance. If the chosen N exceeds that limit, material constants, maximum N and total length remain visible while focal/image distances are unresolved. A virtual/infinite image is identified in the inline status.

Use `xpos.analyze_crl(config)` for the same CPU-only preflight API. Its immutable `CRLAnalyticalResult` contains `delta`, `beta`, `collimated_focus_m`, `image_distance_m`, `max_element_count`, `lens_length_m`, `external_focus` and `real_image`. It uses only geometry/optical constants, with no FFT or field propagation.

GPU availability and session limits are controlled by Colab: [official FAQ](https://research.google.com/colaboratory/faq.html).

## Focusing and calculation grid

Focusing dimensionality describes the parabolic lens surfaces; it is independent of numerical-grid dimensionality.

| Focusing | Grid | Model |
| --- | --- | --- |
| One-dimensional focusing | 1D | Parabolic profile focusing along x |
| One-dimensional focusing | 2D | The same x-focusing profile with finite structure depth along y |
| Two-dimensional focusing | 2D | Rotationally symmetric parabolic profile focusing along x and y |

The structure-depth input is enabled for one-dimensional focusing on a 2D grid. Two-dimensional focusing selects a 2D grid automatically.

CRL lengths are entered in µm, photon energy in keV, density in g/cm³, and source distance in metres. Source distance is measured to the CRL entrance. Detector distances are measured from the CRL exit; displayed units adapt to the range.

## Plane-wave, point and Gaussian illumination

**Plane wave at CRL entrance** supplies a coherent field with uniform unit amplitude and constant phase inside the entrance pupil. Source distance and FWHM inputs stay in place but are disabled and ignored. In the Python API use `source_type="plane"`; `source_distance_m` is normalized to `None`. The analytical reference is collimated BFL. Source magnification has no finite-object meaning here: the notebook displays a dash for magnification, and the API returns `None` for each magnification axis.

A point source has a finite source-to-CRL distance and spherical phase. A Gaussian source is spatially incoherent, centered, and specified by its intensity **FWHM**: x on a 1D grid, x and y on a 2D grid. Internally, sigma = FWHM / (2 sqrt(2 ln 2)). Zero width represents a point-like source along that axis.

Two methods are available:

- **Gaussian convolution:** propagates the on-axis coherent field, then convolves its intensity with the source projected to detector z using the discrete-stack ray-transfer scale. This assumes a shift-invariant pupil response; source-dependent clipping, vignetting and absorption changes are not fully reproduced.
- **Incoherent point sampling:** Gaussian quadrature splits the source into weighted points. Each field passes through the CRL, and weighted intensities are added. Independent modes use batched FFTs. N nodes per nonzero-width axis give N² modes for a 2D source. Increase the node count to check convergence.

Source spectra are cached within a RAM/VRAM budget. Larger problems stream bounded batches and recompute modes rather than allocating an unbounded source stack. No cache is written to disk.

## Geometry-based adaptive grid

The initial transverse window encloses the aperture and, where applicable, structure depth, with a default minimum width of 1.3 times each geometric extent. Point counts account for diffraction width and phase variation and use FFT-friendly factors 2, 3, 5 and 7. Actual spacings are selected first as readable multiples of 5 nm, with 0.5 nm, 0.05 nm, etc. quanta below 5 nm; the window/count then adapt to preserve coverage. Automatic counts allow a small FFT-friendly margin near 110% of the nominal cap when that preserves the finer rounded spacing. They do not silently change the displayed step while retaining an unrelated computational step.

**Fast** uses the propagation grid directly for detector samples. **Accurate** evaluates the propagated complex field on a dense local detector grid using inverse Fourier chirp-z transforms, targeting at least 24 samples across the coherent diffraction-width estimate, within bounded sizes. This calculates the field at detector coordinates; it does not smooth an intensity image. It avoids enlarging the entire 2D grid just to resolve a nanometre-scale intensity peak.

Accurate detector spacings are quantized separately, rounding down when possible while preserving a bounded detector count and its propagation-window limit. Dense detector sampling cannot recover frequencies missing from the entrance grid. Accurate also increases the propagation-grid budget. Actual propagation-grid point counts and steps are shown beside the computed CPU/GPU backend. Two-dimensional counts are displayed as x × y. When Accurate uses a different detector grid, its counts and steps are reported there separately as well. The library's corresponding accuracy values remain `preview` and `refined`.

With **Adapt propagation window**, edge-power probes can enlarge automatic axes independently while preserving the resolution target. Explicit point-count overrides retain user control. Window expansion and detector sizes have memory limits. Check boundary warnings and compare finer grids, larger windows and more source nodes when quantitative accuracy matters. Automatic settings do not certify convergence.

## Peak-intensity distance and display ROI

`find_peak_intensity_distance` accepts a callable giving the transverse peak intensity at distance z. It expands a three-point bracket with variable steps and refines a local maximum with bounded scalar optimization. It consumes no precomputed z array and returns a `PeakIntensitySearchResult`.

This criterion finds a **local maximum of peak intensity**. It does not necessarily locate minimum FWHM or the geometrical image plane. Accordingly, the notebook labels the result **Numerical peak intensity distance** and reports **Analytical focal distance · parallel beam** and **Analytical image distance · finite source** separately.

The finite-source analytical image distance is the **first evaluated point** and seeds the local basin. Default search bounds are 0.05–3 times that distance. Several diffraction maxima can exist; this is a local search, with no global-maximum guarantee. Boundary or unresolved maxima appear in diagnostics.

For a Gaussian source, the CRL workflow uses **only its central coherent source point** for the adaptive maximum search and automatic axial ROI. It reuses the coherent exit field/spectrum and delays preparation of the other source modes until those steps finish. It does not repeatedly sum all source nodes while choosing the reference distance. Convolution also uses the central-point search, so at a fixed propagation grid this reference is independent of source FWHM, quadrature order and extended-source method.

The output label adds **central point** for Gaussian illumination. This is a reference plane: the full Gaussian source can attain its maximum at a different distance. All displayed intensity maps, profiles, peak-intensity curves, FWHM values and transmission calculations still use the complete selected source model. **Model, mesh and diagnostics** also reports the highest full-source intensity among the displayed z samples; that sampled value is not a second iterative optimization.

After the central-point search:

1. The axial display ROI follows the central-point peak intensity down to **Axial ROI intensity fraction**, default 0.2 of the numerical peak.
2. The interval is sampled for visualization, with the numerical maximum inserted explicitly. **Displayed z samples** changes the visualization grid, independently of the preceding optimization.
3. Significant transverse intensity across that interval defines a padded x/y ROI, with default threshold 0.01 of its maximum.

Manual z bounds also bound the search. The analytical initial guess is clamped to them. **Peak search tolerance** controls the optimizer's axial tolerance.

## Calculation progress and time estimates

Compute displays a stage progress bar, completed work, elapsed wall time and **ETA for this stage**. Stages cover grid/window preparation, central-point peak search, axial ROI, preparation of the full Gaussian source and its cached modes, detector-plane sampling, final metrics and rendering. Completion collapses to a compact elapsed-time indicator so the results workspace keeps its height. Errors leave a visible stopped state. Updates run synchronously in the existing computation callback, without threads or timers.

For a stage with a known total, ETA is estimated from elapsed time per completed item. Detector-plane progress advances after a completed batch and its metric reductions, rather than forcing each plane into its own batch. This preserves GPU batching. Estimates can change as timings accumulate and do not promise an exact completion time.

Adaptive peak/ROI searches do not know their eventual number of evaluations. They show activity and evaluation counts with **ETA: estimating · adaptive stage**, without inventing a percentage or treating the maximum evaluation budget as a fixed workload. The ETA for a known stage excludes later stages; a precise whole-calculation forecast is not available before the adaptive work has resolved its grid and display interval.

The Python API accepts `estimate_crl_focus(config, progress=callback)`. The callback receives an immutable `CRLProgressUpdate` with `stage`, `completed`, `total`, `elapsed_s`, `stage_elapsed_s`, `eta_s` and `detail`. Unknown totals and estimates are `None`. Omitting the callback retains the existing numerical behavior. Observer exceptions propagate to the caller.

## What batches on the GPU

A successful CUDA probe selects CuPy. Field arithmetic, FFTs, Fourier zoom and Gaussian filtering then execute on that backend. The current runtime notice identifies the actual GPU; CPU fallback uses NumPy/SciPy.

- Point-source, plane-wave and Gaussian-convolution display scans process independent z planes in batches, sharing the exit spectrum. Both Fast and Accurate detector propagation support this.
- Direct incoherent point sampling batches the independent source modes through the CRL and to each detector plane, then sums weighted intensities. Its z planes are currently visited sequentially; mode batches provide the parallel work at each z.
- Automatic GPU batches use the available VRAM and are capped at 16. **Batch size** in advanced controls overrides the heuristic; CPU automatic batches default to one. Cached source spectra have a separate memory budget and larger mode sets stream bounded batches.
- Consecutive CRL elements depend on the preceding field, and adaptive peak/ROI steps depend on previous evaluations, so these are sequential. Each field evaluation can still use GPU-parallel array operations and source-mode batches. Plot rendering and scalar optimization orchestration stay on the CPU.

## Numerical effective aperture

The effective aperture is defined here as the **intensity FWHM at the exit of the complete CRL**. For x, find the nearest half-maximum crossings around the exit intensity peak:

$$D_{\mathrm{eff},x}=x_{\mathrm{right}}-x_{\mathrm{left}},\qquad I_{\mathrm{exit}}(x_{\mathrm{left/right}},y_{\mathrm{peak}})=\tfrac12 I_{\mathrm{exit,max}}.$$

For two-axis focusing the y width is measured analogously on a 2D grid. For one-axis focusing on a 2D grid, the unfocused y diffraction profile is retained but no y FWHM is computed or displayed. Crossings are linearly interpolated on the full propagation grid, before detector zoom or display cropping. This definition includes illumination, absorption, and propagation inside the stack. It is the full width, not HWHM, and is not clamped to the nominal geometric aperture. If crossings are missing from the numerical window, the affected width is unresolved.

For incoherent point sampling, the exit intensity is the weighted sum $I_{\mathrm{exit}}=\sum_j w_j |E_{j,\mathrm{exit}}|^2$, rather than an average of individual widths. The sum is accumulated during the existing source-mode batches, with no additional CRL pass. The convolution model applies its source blur at z=0 on the native grid; its shift-invariance approximation still applies.

Use `result.effective_aperture_m` for the full-source result: a scalar in 1D or an `(x, y)` tuple in 2D. For one-axis focusing on a 2D grid the tuple is `(width_x, None)`; this is intentional, rather than an unresolved width. `result.fwhm_axes` lists the measured focusing axes. It also appears in `result.summary()` with convention `exit_intensity_FWHM`. For a low-level coherent `CRL` object, `crl.effective_aperture()` measures the FWHM of its exit field. For a finite-depth, one-axis CRL, both `crl.effective_aperture()` and `crl.focus_scan()` skip y width. Generic `beam_metrics()` still measures both axes by default. This is a numerical result available after propagation, separate from the inexpensive analytical preview.

## Results and slider

The selected **On-image measures** layout groups the results with visible borders and high-contrast headings. The top **CRL · whole calculation** table reports numerical peak-intensity distance, transmission and separate x/y effective-aperture values, with units in headers. Analytical focal and image distances remain beside the parameter form.

The right **Selected transverse plane** panel shows FWHM measurement bars and values directly on the 2D intensity map. **Peak intensity is never annotated or marked on that map**; it has its own bordered readout below, beside the magnification table. On a **1D grid**, peak intensity is annotated at the profile maximum, and a mathematical $\mathrm{FWHM}_x$ ruler joins the interpolated half-maximum crossings at their actual height. These two values are not repeated below the plot; magnification keeps its separate readout. The map labels use $\mathrm{FWHM}_x$ and $\mathrm{FWHM}_y$ for the measured focusing axes. Labels and values inherit the chosen theme explicitly, including in notebook frontends with their own HTML styling. Changed inputs produce a notice until Compute is pressed again.

The left panel contains three plots with a shared horizontal z axis:

1. **Intensity distribution near the focus**, using the central y row for a 2D grid;
2. peak intensity versus z, in muted blue;
3. FWHM versus z, in muted green, with separate shades and solid x / dashed y lines when both focus. The solid trace is slightly wider, so both remain visible when they coincide.

Turquoise vertical cursors mark the selected plane, with matching turquoise intersection markers on the peak-intensity and FWHM curves. A dotted reference line marks the numerical peak-intensity distance. Hollow markers label the **sampled** peak-intensity maximum (`max`) and focusing-axis width minima (`min`); they do not repeat numeric readouts or run another optimizer. Coincident x/y minima share one visible marker. Single-curve plots omit a redundant legend; x/y keys sit outside the data axes so they cannot cover a curve.

Peak intensity and FWHM are measured over the full detector plane; the longitudinal map remains a fixed central slice. For one-dimensional focusing, both the width graph and transverse-map ruler report x only. Individual metrics contain `(width_x, None)` and the y column of `result.fwhm_m` contains NaN; `result.fwhm_axes` is `("x",)`. The y intensity is still propagated and displayed unchanged.

The right panel shows a blue transverse profile for a 1D grid. For every 2D grid it shows **a square transverse intensity map above both x/y profiles**, with no visualization selector. Equal x/y physical scales and square display bounds preserve geometry; the map never stretches to fill a rectangle. Unequal physical beam dimensions remain unequal. Profile cuts pass through the current transverse maximum; interpolated half-maximum crossings, rather than a symmetric Gaussian approximation, determine the FWHM measurement bars. The transverse map uses **horizontal y and vertical x**, so its vertical coordinate agrees with the longitudinal map; data, labels and FWHM rulers are transposed together. For two-axis focusing x is solid and y dashed. For **one-axis focusing on a 2D grid both profiles are solid**, in different colors and line widths; the legend identifies x as focusing and y as diffraction. Raw oscillations are preserved. The profile axis extends 3.5% of the reference peak below zero so zero-intensity segments remain visible.

A **single shared linear color scale is attached directly to the map axes**: to the transverse map for 2D, or the longitudinal map for 1D. Both maps use `result.intensity_reference`, the largest full-source peak among the uniform displayed planes and the independently evaluated numerical reference. It stays fixed during navigation. CMRmap remains white at maximum. No separate, distant color-scale strip is used.

The common z slider sits below both result panels. An integer slider selects indices of the **uniform z lattice**; track clicks, dragging and keyboard moves use a synchronous handler without timers. Physical z entry snaps to the nearest calculated plane. Previous/Next move one regular sample; **Peak** selects the exact numerical reference, which is a regular sample of the peak-anchored lattice. Physical z entry still snaps to the nearest sample. The counter, plots and selected-plane metrics always refer to the same sampled z. `result.plane_at(z_m)` remains available for any detector distance, including the exact numerical reference.

Primary labels use 14 px text and numeric readouts approximately 20–21 px. Section lines separate names from values, and unit headers prevent x/y values from merging visually. Axes use up to two decimals without redundant zeros. The physical z input keeps enough decimals for the actual axial step. The turquoise cursor marks selected z; plot legends stay outside the data axes. Text uses a monospace stack with Consolas when installed; plots fall back to the available DejaVu Sans Mono. No font download or extra dependency is needed.

### Themes

Use **Theme → Auto / Dark / Light** beside Compute. The complete form, tables, figure backgrounds, axes, curves, labels and color scale use the same palette. Manual theme changes redraw existing results without repeating wave propagation. Auto probes the Colab output theme at initialization and Compute. Other frontends, or a changed Colab theme after calculation, can use the explicit selector; if an Auto probe is unavailable the previous palette is retained. The actual Colab frontend remains a separate integration check.

### Setting physical grid steps

Open **Sampling, ROI and search**. **Transverse grid control** offers Auto, Set spacing or Set point counts. Spacing targets are entered in nm; inactive counts/spacings keep their values but are ignored. Targets and fixed-count windows use the same readable-spacing policy; actual propagation counts and steps are reported beside the backend. Explicit spacing control has a 4096-point per-axis limit in 2D or 65536 in 1D; larger requests fail clearly rather than overriding the target silently. Existing explicit point-count overrides remain available.

**Axial grid control → Set spacing** enables Target Δz in mm. Auto uses the target sample count; Fast rounds the target spacing upward and Accurate downward to the preferred sequence **1, 2.5, 5 × a power of ten**. A manual target uses the nearest preferred spacing, e.g. 0.04 mm → 0.05 mm. The origin is the **full-precision numerical peak-intensity distance**, so $z_k=z_{\mathrm{peak}}+k\Delta z$. Only Δz is rounded; z values retain the computed offset. With a reference of **30.93 mm** and a step of **0.05 mm**, the grid includes **30.83, 30.88, 30.93, 30.98, 31.03 mm**. The code uses the unrounded reference, even when its displayed label has fewer decimals. ROI bounds expand outward on this anchored lattice, with nonnegative z. The maximum is 10001 displayed planes; the exact reference is a regular sample, not an extra point inserted between samples. Thus Δz is the real common spacing of every adjacent overview plane. Actual Δz and the number of planes appear beside the backend, with propagation and separate Accurate detector grids.

In the API use `grid_step_x_m`, `grid_step_y_m` and `scan_step_m`; a transverse step and point count for the same axis are mutually exclusive. `scan_points` is a target count rather than an exact count after rounding. `result.scan_step_m` and summary metadata `scan_grid_uniform` and `scan_anchor_m` describe the actual lattice. `peak_index` indexes the **exact numerical reference** in that lattice. `reference_metrics`, `best_intensity` and `fwhm_at_peak_m` refer to the exact optimized distance, evaluated independently of the lattice. Search history and exact-reference metrics therefore do not depend on the chosen display step.


Magnification is the signed source-to-plane displacement scale −B(z)/source distance from the ray model. At a conjugate image plane it is image magnification; elsewhere it describes source-position shifts. The unfocused y axis uses free propagation through the stack length.

## Reusable Python API

```python
import XPOSim as xpos

config = xpos.CRLFocusConfig(
    geometry="one-dimensional",
    grid_dimension=2,
    structure_depth_m=50e-6,
    source_type="gaussian",
    source_fwhm_x_m=5e-6,
    source_fwhm_y_m=5e-6,
    source_method="convolution",  # or "points"
    peak_search_tolerance_m=1e-6,
    backend="auto",
)
result = xpos.estimate_crl_focus(config)
search = result.numerical_peak  # Central-point reference for a Gaussian source.
plane = result.plane_at(search.z_m * 1.01)
print(result.analytical_image_distance_m, result.magnification_at_peak)
print(result.effective_aperture_m)  # Exit-intensity FWHM on the propagation grid.
```

For Gaussian illumination, `result.numerical_peak.intensity` is the central coherent-point intensity; `result.peak_intensity` and `result.fwhm_m` describe the full source. `peak_z_m` and `fwhm_at_peak_m` refer to the numerical reference plane. `result.sampled_peak_z_m` gives the full-source maximum among displayed samples. Summary metadata records `peak_search_basis`, `peak_search_initial_z_m` and `axial_roi_basis`.

All API lengths are metres. `find_peak_intensity_distance(objective, initial_z_m, ...)` also works independently of the CRL workflow. The older `find_numerical_focus` wrapper emits a `DeprecationWarning`; `NumericalFocusResult`, `result.numerical_focus`, `result.analytical_focus_m`, `result.magnification_at_focus` and `focus_tolerance_m` remain compatibility aliases. New summary keys use `peak_search_evaluations`, `peak_search_converged`, `analytical_image_distance_m` and `magnification_at_peak`. The workflow names `CRLFocusConfig`, `CRLFocusResult` and `estimate_crl_focus` remain available.

`CRL.apply_to_fields` handles bounded batches of independent source fields. `CRL.source_magnification` and `CRL.ray_transfer_matrix` expose the discrete-stack ray model. Established low-level NumPy/CuPy interfaces are retained.

For a new public notebook revision, commit the tested library first and pin that commit and the module SHA-256 in setup. Commit the notebook afterwards, then update the immutable Colab link. Keep notebook outputs and execution counts cleared.

## Limits and validation

- Intensities are relative. Incident amplitude is one within the nominal entrance pupil. Transmission is finite-window output power divided by power after that pupil.
- The model is scalar, monochromatic and paraxial. It excludes polarization, back-reflections and high-angle propagation.
- For memory errors, reduce batch size first. Reducing grid sizes reduces accuracy. Direct source integration can be considerably more expensive than convolution.
- An unresolved FWHM means the required half-maximum crossings were not found within the numerical window.
- If inputs are missing, confirm that the introduction says **Peak-anchored z grid · on-profile 1D metrics · light and dark themes** and run cell 2 explicitly. A previously opened notebook or Drive copy does not automatically receive repository updates: use the current link at the top of this guide. If widgets themselves do not render, rerun setup in a fresh session and allow the standard custom-widget manager.
- Colab session memory is temporary. Reopening starts with the parameter form; restarting a kernel loses entered values.

Material/preflight checks cover tabulated/direct numerical equivalence, no database access for direct constants, fixed coefficients across energy edits, beta=0, invalid coefficient pairs, maximum-N/virtual-image states and the absence of propagation/CUDA probes in analytical preview. Form/browser checks verify fixed field positions, retained values, all three material modes, actual grid metadata beside the backend and larger result labels. Exit-aperture checks cover a known Gaussian FWHM, missing crossings, native-grid measurement with Accurate detectors, anisotropic 2D widths, a direct weighted source-mode sum, cached/streamed agreement, no repeat propagation on aperture access and native-grid convolution at the exit. Browser checks cover one attached shared color scale, separate peak readout, on-map FWHM measurements, square/equal-scale map bounds, inherited dark/light foreground colors, Peak navigation, changed-parameter notices and absence of internal horizontal scrollbars at widths 700, 1024, 1366 and 1920 px. Plot checks verify common intensity normalization and legends outside the data axes. Central-search validation checks the first analytical evaluation, guards against full-Gaussian evaluation during search/ROI, verifies deferred source modes and one coherent CRL evaluation, and compares reference distances across source sizes, quadrature orders and source methods on fixed Fast/Accurate grids. Progress validation checks deterministic stage ETA, unknown adaptive totals, observer error propagation, bit-identical results with/without reporting, preserved three-plane batches in Fast/Accurate and preserved source-mode batches. Live browser checks observe running progress/ETA updates and compact completion alongside the existing z controls. CPU validation also covers direct constant-phase plane-wave propagation on 1D/2D grids, the far-point-source limit, source-distance independence and Accurate plane-wave propagation, along with the existing optical/source/grid checks, Fourier-zoom equivalence and analytical Gaussian propagation, the renamed peak-search API and compatibility aliases, independent input/result outputs, first-run form visibility, density changes, selected-plane readouts, formatting/colors and discrete slider callbacks. In a local live browser, actual radius, aperture, source type, source FWHM and source distance fields were edited before Compute; the calculated configuration was checked against those edits. The fields stayed rendered after calculation, and integer-slider track clicks, presses before mouse release, repeated clicks, dragging, keyboard, direct z entry and Previous/Next buttons were exercised at desktop sizes. Notebook-kernel checks assert that output is updated before each event handler returns, even when a notebook event loop is active; the former delayed handler fails that check. Injected rendering errors are visible and later selections recover. The 2D map and both profiles are displayed together, with equal panel canvas sizes. The current revision additionally checks peak-anchored grids using the 30.93/0.05 mm example and an unrounded reference, optimizer invariance across display steps, deliberate omission of unfocused-y width reductions, transposed intensity/ruler coordinates, math labels, 1D annotations without duplicate readouts, sampled extrema and mono fonts. Actual CUDA hardware and the Google Colab frontend were not available for validation.

Runtime depends on hardware, actual grid counts, z spacing and source nodes. Smaller requested spacing can substantially increase the number of samples; use the reported actual grids when comparing speed and convergence.
