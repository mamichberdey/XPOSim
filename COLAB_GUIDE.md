# CRL focus explorer in Google Colab

[Open the notebook](https://colab.research.google.com/github/mamichberdey/XPOSim/blob/3a29efa1f58d46adbd4d207e069207416e527290/notebooks/CRL_Focus_Colab.ipynb).

Run all once to initialize two separate outputs: **2 · CRL and source parameters** contains the editable inputs; **3 · Results** contains Compute, the plots and the z slider. Enter the lens/source parameters in cell 2, then press **Compute** in cell 3. The parameter fields are always shown in their own output and are never automatically collapsed or hidden by a calculation. Reinitializing the parameter cell in the same live session preserves entered values. No optical calculation starts automatically. Results remain in memory; no export or save/restore mechanism is included.

## Start

1. Open the link and sign in to Colab. Optionally save a copy to your Drive.
2. For GPU execution, use **Runtime → Change runtime type → GPU → Save**, reconnect, and run all cells. CPU execution is also supported.
3. Setup checks dependencies and actual CUDA execution. Missing CuPy is installed only when a compatible CUDA 12/13 environment is detected. A small GPU FFT probe selects CuPy when CUDA works, otherwise NumPy. The notice reports the actual CUDA device, independent of the session's GPU/TPU label. CPU sessions display the GPU-enabling instructions. There is no backend picker.
4. Run **2 · CRL and source parameters** to display the editable fields below that cell. The form defaults to silicon at 12 keV, surface radius 6.25 µm, full aperture 50 µm, bridge thickness 2 µm, 26 elements and source distance 15 m. **Fast** sampling is selected initially.
5. Run **3 · Results**, then press **Compute** after setting the inputs. To change the lens or source, edit the still-visible fields in cell 2 and press **Compute** again. The results stay in their own compact workspace, so using the detector slider does not move the input form.

In **Auto** density mode, the read-only **Density (g/cm³)** field shows the value actually used, and its label identifies the selected material. Presets are Be: 1.848, Si: 2.33, Al: 2.70 and diamond: 3.515 g/cm³. **Manual** allows an override. A custom chemical formula requires a manual density.

GPU availability and session limits are controlled by Colab: [official FAQ](https://research.google.com/colaboratory/faq.html).

## Focusing and calculation grid

Focusing dimensionality describes the parabolic lens surfaces; it is independent of numerical-grid dimensionality.

| Focusing | Grid | Model |
| --- | --- | --- |
| One-dimensional focusing | 1D | Parabolic profile focusing along x |
| One-dimensional focusing | 2D | The same x-focusing profile with finite structure depth along y |
| Two-dimensional focusing | 2D | Rotationally symmetric parabolic profile focusing along x and y |

The structure-depth input appears for one-dimensional focusing on a 2D grid. Two-dimensional focusing selects a 2D grid automatically.

CRL lengths are entered in µm, photon energy in keV, density in g/cm³, and source distance in metres. Source distance is measured to the CRL entrance. Detector distances are measured from the CRL exit; displayed units adapt to the range.

## Plane-wave, point and Gaussian illumination

**Plane wave at CRL entrance** supplies a coherent field with uniform unit amplitude and constant phase inside the entrance pupil. Source distance and FWHM inputs are hidden and ignored. In the Python API use `source_type="plane"`; `source_distance_m` is normalized to `None`. The analytical reference is collimated BFL. Source magnification has no finite-object meaning here: the notebook displays **Not applicable · plane wave**, and the API returns `None` for each magnification axis.

A point source has a finite source-to-CRL distance and spherical phase. A Gaussian source is spatially incoherent, centered, and specified by its intensity **FWHM**: x on a 1D grid, x and y on a 2D grid. Internally, sigma = FWHM / (2 sqrt(2 ln 2)). Zero width represents a point-like source along that axis.

Two methods are available:

- **Gaussian convolution:** propagates the on-axis coherent field, then convolves its intensity with the source projected to detector z using the discrete-stack ray-transfer scale. This assumes a shift-invariant pupil response; source-dependent clipping, vignetting and absorption changes are not fully reproduced.
- **Incoherent point sampling:** Gaussian quadrature splits the source into weighted points. Each field passes through the CRL, and weighted intensities are added. Independent modes use batched FFTs. N nodes per nonzero-width axis give N² modes for a 2D source. Increase the node count to check convergence.

Source spectra are cached within a RAM/VRAM budget. Larger problems stream bounded batches and recompute modes rather than allocating an unbounded source stack. No cache is written to disk.

## Geometry-based adaptive grid

The initial transverse window encloses the aperture and, where applicable, structure depth, with a default width of 1.3 times each geometric extent. Point counts account for diffraction width and phase variation and use FFT-friendly factors 2, 3, 5 and 7.

**Fast** uses the propagation grid directly for detector samples. **Accurate** evaluates the propagated complex field on a dense local detector grid using inverse Fourier chirp-z transforms, targeting at least 24 samples across the coherent diffraction-width estimate, within bounded sizes. This calculates the field at detector coordinates; it does not smooth an intensity image. It avoids enlarging the entire 2D grid just to resolve a nanometre-scale intensity peak.

Dense detector sampling cannot recover frequencies missing from the entrance grid. Accurate also increases the propagation-grid budget. Both grids and their steps are shown under **Model, mesh and diagnostics**. The library's corresponding accuracy values remain `preview` and `refined`.

With **Adapt propagation window**, edge-power probes can enlarge automatic axes independently while preserving the resolution target. Explicit point-count overrides retain user control. Window expansion and detector sizes have memory limits. Check boundary warnings and compare finer grids, larger windows and more source nodes when quantitative accuracy matters. Automatic settings do not certify convergence.

## Peak-intensity distance and display ROI

`find_peak_intensity_distance` accepts a callable giving the transverse peak intensity at distance z. It expands a three-point bracket with variable steps and refines a local maximum with bounded scalar optimization. It consumes no precomputed z array and returns a `PeakIntensitySearchResult`.

This criterion finds a **local maximum of peak intensity**. It does not necessarily locate minimum FWHM or the geometrical image plane. Accordingly, the notebook labels the result **Numerical peak intensity distance** and reports **Analytical focal distance · parallel beam** and **Analytical image distance · finite source** separately.

The finite-source analytical distance seeds the local basin. Default search bounds are 0.05–3 times that distance. Several diffraction maxima can exist; this is a local search, with no global-maximum guarantee. Boundary or unresolved maxima appear in diagnostics.

After the search:

1. The axial display ROI follows peak intensity down to **Axial ROI intensity fraction**, default 0.2 of the numerical peak.
2. The interval is sampled for visualization, with the numerical maximum inserted explicitly. **Displayed z samples** changes the visualization grid, independently of the preceding optimization.
3. Significant transverse intensity across that interval defines a padded x/y ROI, with default threshold 0.01 of its maximum.

Manual z bounds also bound the search. The analytical initial guess is clamped to them. **Peak search tolerance** controls the optimizer's axial tolerance.

## Calculation progress and time estimates

Compute displays a stage progress bar, completed work, elapsed wall time and **ETA for this stage**. Stages cover grid/window preparation, cached Gaussian source modes, adaptive peak search, axial ROI, detector-plane sampling, final metrics and rendering. Completion collapses to a compact elapsed-time indicator so the results workspace keeps its height. Errors leave a visible stopped state. Updates run synchronously in the existing computation callback, without threads or timers.

For a stage with a known total, ETA is estimated from elapsed time per completed item. Detector-plane progress advances after a completed batch and its metric reductions, rather than forcing each plane into its own batch. This preserves GPU batching. Estimates can change as timings accumulate and do not promise an exact completion time.

Adaptive peak/ROI searches do not know their eventual number of evaluations. They show activity and evaluation counts with **ETA: estimating · adaptive stage**, without inventing a percentage or treating the maximum evaluation budget as a fixed workload. The ETA for a known stage excludes later stages; a precise whole-calculation forecast is not available before the adaptive work has resolved its grid and display interval.

The Python API accepts `estimate_crl_focus(config, progress=callback)`. The callback receives an immutable `CRLProgressUpdate` with `stage`, `completed`, `total`, `elapsed_s`, `stage_elapsed_s`, `eta_s` and `detail`. Unknown totals and estimates are `None`. Omitting the callback retains the existing numerical behavior. Observer exceptions propagate to the caller.

## What batches on the GPU

A successful CUDA probe selects CuPy. Field arithmetic, FFTs, Fourier zoom and Gaussian filtering then execute on that backend. The current runtime notice identifies the actual GPU; CPU fallback uses NumPy/SciPy.

- Point-source, plane-wave and Gaussian-convolution display scans process independent z planes in batches, sharing the exit spectrum. Both Fast and Accurate detector propagation support this.
- Direct incoherent point sampling batches the independent source modes through the CRL and to each detector plane, then sums weighted intensities. Its z planes are currently visited sequentially; mode batches provide the parallel work at each z.
- Automatic GPU batches use the available VRAM and are capped at 16. **Batch size** in advanced controls overrides the heuristic; CPU automatic batches default to one. Cached source spectra have a separate memory budget and larger mode sets stream bounded batches.
- Consecutive CRL elements depend on the preceding field, and adaptive peak/ROI steps depend on previous evaluations, so these are sequential. Each field evaluation can still use GPU-parallel array operations and source-mode batches. Plot rendering and scalar optimization orchestration stay on the CPU.

## Results and slider

The top row reports numerical peak-intensity distance, collimated analytical focal distance, finite-source analytical image distance and transmission. For plane-wave input, the finite-source card is replaced by the incident-field type. The second row reports selected z, FWHM, peak intensity and magnification; these four cards update with the slider. Two-dimensional values carry x/y labels.

The left panel contains three plots with a shared horizontal z axis:

1. **Intensity distribution near the focus**, using the central y row for a 2D grid;
2. peak intensity versus z, in red;
3. FWHM versus z, in green, with separate shades and solid x / dashed y lines when both focus. The solid trace is slightly wider, so both remain visible when they coincide.

Orange vertical cursors mark the selected plane, with orange intersection markers on the peak-intensity and FWHM curves. A dotted reference line marks the numerical peak-intensity distance. Single-curve plots omit a redundant legend.

Peak intensity and FWHM are measured over the full detector plane; the longitudinal map remains a fixed central slice. For one-dimensional focusing, the main width graph shows x; the cards also report y for a 2D grid.

The right panel shows a blue transverse profile for a 1D grid. For every 2D grid it shows **the intensity map on the left and both x/y profiles together on the right**, with no visualization selector. Both profiles pass through the current transverse maximum and share a physical coordinate scale and reference-intensity normalization. The x profile is blue and solid; y is purple and dashed, with different widths and mild transparency to keep identical traces visible. Matching guide lines identify their cuts on the map.

Both main panels have equal widths and the same slightly taller image canvas. The common z slider sits immediately below both panels, so there is no unmatched blank area under the right image. An integer slider selects indices of the calculated z planes, including the numerical maximum. Track clicks, dragging and keyboard moves update the plane through the same synchronous handler, without background timers or an asyncio dependency. A physical **z** entry and **Previous/Next** buttons share that handler; entered distances snap to the nearest calculated plane. The counter shows the current plane number. Errors appear next to the plots and do not disconnect subsequent selections. The displayed z value, both panels and the cards therefore refer to the same plane. The library's `result.plane_at(z_m)` still supports arbitrary detector distances independently of the notebook slider.

Images update in place. Intensity normalization is linear and fixed to the reference peak over the displayed interval, making dimming visible as z changes. CMRmap is white at maximum intensity. Axes and readouts use up to two decimals without redundant zeros: `60`, `12.3`, `12.34`. Length units adapt to the viewed range; common powers of ten are placed on axis labels. Scientific values include spaces around the multiplication sign, for example −2.4 × 10⁻³.

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
search = result.numerical_peak
plane = result.plane_at(search.z_m * 1.01)
print(result.analytical_image_distance_m, result.magnification_at_peak)
```

All API lengths are metres. `find_peak_intensity_distance(objective, initial_z_m, ...)` also works independently of the CRL workflow. The older `find_numerical_focus` wrapper emits a `DeprecationWarning`; `NumericalFocusResult`, `result.numerical_focus`, `result.analytical_focus_m`, `result.magnification_at_focus` and `focus_tolerance_m` remain compatibility aliases. New summary keys use `peak_search_evaluations`, `peak_search_converged`, `analytical_image_distance_m` and `magnification_at_peak`. The workflow names `CRLFocusConfig`, `CRLFocusResult` and `estimate_crl_focus` remain available.

`CRL.apply_to_fields` handles bounded batches of independent source fields. `CRL.source_magnification` and `CRL.ray_transfer_matrix` expose the discrete-stack ray model. Established low-level NumPy/CuPy interfaces are retained.

For a new public notebook revision, commit the tested library first and pin that commit and the module SHA-256 in setup. Commit the notebook afterwards, then update the immutable Colab link. Keep notebook outputs and execution counts cleared.

## Limits and validation

- Intensities are relative. Incident amplitude is one within the nominal entrance pupil. Transmission is finite-window output power divided by power after that pupil.
- The model is scalar, monochromatic and paraxial. It excludes polarization, back-reflections and high-angle propagation.
- For memory errors, reduce batch size first. Reducing grid sizes reduces accuracy. Direct source integration can be considerably more expensive than convolution.
- An unresolved FWHM means the required half-maximum crossings were not found within the numerical window.
- If inputs are missing, confirm that the introduction says **Interface revision: separate parameter and results cells** and run cell 2 explicitly. A previously opened notebook or Drive copy does not automatically receive repository updates: use the current link at the top of this guide. If widgets themselves do not render, rerun setup in a fresh session and allow the standard custom-widget manager.
- Colab session memory is temporary. Reopening starts with the parameter form; restarting a kernel loses entered values.

Progress validation checks deterministic stage ETA, unknown adaptive totals, observer error propagation, bit-identical results with/without reporting, preserved three-plane batches in Fast/Accurate and preserved source-mode batches. Live browser checks observe running progress/ETA updates and compact completion alongside the existing z controls. CPU validation also covers direct constant-phase plane-wave propagation on 1D/2D grids, the far-point-source limit, source-distance independence and Accurate plane-wave propagation, along with the existing optical/source/grid checks, Fourier-zoom equivalence and analytical Gaussian propagation, the renamed peak-search API and compatibility aliases, independent input/result outputs, first-run form visibility, density changes, selected-plane cards, formatting/colors and discrete slider callbacks. In a local live browser, actual radius, aperture, source type, source FWHM and source distance fields were edited before Compute; the calculated configuration was checked against those edits. The fields stayed rendered after calculation, and integer-slider track clicks, presses before mouse release, repeated clicks, dragging, keyboard, direct z entry and Previous/Next buttons were exercised at desktop sizes. Notebook-kernel checks assert that output is updated before each event handler returns, even when a notebook event loop is active; the former delayed handler fails that check. Injected rendering errors are visible and later selections recover. The 2D map and both profiles are displayed together, with equal panel canvas sizes. Actual CUDA hardware and the Google Colab frontend were not available for validation.

An earlier silicon 2D Accurate CPU run with 15 display planes took approximately 45 seconds on a 2688×2688 propagation grid and 577×577 detector, resolving FWHM with over 40 detector samples. Runtime depends on hardware, z count, source nodes and grid size; this is one CPU benchmark.
