# CRL focus explorer in Google Colab

[Open the notebook](https://colab.research.google.com/github/mamichberdey/XPOSim/blob/4c26eb102e96ae96f5d81177ea6493ff3deed92c/notebooks/CRL_Focus_Colab.ipynb).

Run all once, change the English controls, then press **Compute focus**. The dynamic form retains entered values when its cell is initialized again in the same live session. Results are shown in a compact two-column workspace; the z slider is directly below the detector image. No calculation files are saved or exported. The notebook loads a pinned XPOSim revision and material tables automatically.

## Start

1. Open the link and sign in to Colab. Optionally save a notebook copy to your own Drive.
2. For GPU execution, use **Runtime → Change runtime type → GPU**, then run all cells. A CPU session also works.
3. Setup checks the available dependencies and CUDA. Missing CuPy is installed only when a compatible CUDA 12/13 GPU environment is detected. The interface uses an actual GPU FFT probe and selects CuPy when CUDA works, otherwise NumPy. The hardware notice reports the actual CUDA device rather than relying on a GPU/TPU session label. For a CPU session it shows the Colab GPU-enabling steps. There is no backend picker to accidentally select unavailable hardware.
4. The default example runs immediately: silicon at 12 keV, surface radius 6.25 µm, full aperture 50 µm, bridge thickness 2 µm, 26 elements and source distance 15 m. Material density is automatic by default; choose **Manual** to override a preset density. Change parameters and press **Compute focus** to update the analysis and slider.

GPU availability, runtime limits and disconnections are controlled by Colab: [official FAQ](https://research.google.com/colaboratory/faq.html).

## Focusing and calculation grid

These are separate choices:

| Focusing | Grid | Model |
| --- | --- | --- |
| One-dimensional focusing | 1D | Parabolic profile focusing along x |
| One-dimensional focusing | 2D | The same x-focusing profile with finite structure depth along y |
| Two-dimensional focusing | 2D | Rotationally symmetric parabolic profile focusing along x and y |

The structure-depth input appears for a one-dimensionally focusing CRL on a 2D grid. Two-dimensional focusing automatically selects a 2D grid. The terms describe focusing dimensions; the lens surfaces remain parabolic.

Lengths in the CRL controls are µm, photon energy is keV, density is g/cm³, source distance is metres, and detector z is millimetres from the CRL exit. Source distance is measured to the CRL entrance. The material presets set formula and density together.

## Point and Gaussian sources

A point source has a finite source-to-CRL distance and spherical phase. The plane-wave option has been removed from this notebook.

A Gaussian source is **spatially incoherent**, centered, and specified by its intensity **FWHM**. A 1D grid uses FWHM x; a 2D grid uses FWHM x and y. Internally, sigma = FWHM / (2 sqrt(2 ln 2)). A zero width on one axis represents a point-like source along that axis.

Two methods are available:

- **Gaussian convolution:** propagates the on-axis coherent field, then blurs the intensity with the Gaussian source projected to the requested detector z. The projection uses the discrete stack ABCD matrix, including the exit drift. It is a shift-invariant approximation: source-dependent clipping, vignetting and absorption changes within the CRL are not fully reproduced.
- **Incoherent point sampling:** Gaussian quadrature splits the source into weighted points. Each field passes through the full CRL, and weighted **intensities** are added. Source modes use batched FFTs on the selected backend. For a 2D source with nonzero widths, N nodes per axis give N² modes. Increase the node count to test source-integration convergence.

The direct method caches source spectra within a RAM/VRAM budget. Larger problems stream and recompute bounded batches instead of allocating an unbounded source stack. No cache is written to disk.

## Geometry-based adaptive grid

The initial transverse window encloses the full aperture and, where applicable, structure depth. Its default width is **1.3 times the corresponding geometric extent**. Sampling is chosen from the expected diffraction width and source-phase variation; **Quick / Refined** controls the resolution target.

Point counts use FFT-friendly combinations of factors 2, 3, 5 and 7. Initial propagation probes check edge power. With **Adapt propagation window**, automatic axes can grow independently when their boundaries carry significant intensity. The number of points is recalculated to preserve the resolution target. Explicit point-count overrides retain user control.

**Quick** uses the propagation grid directly for detector samples. **Refined** also calculates the complex propagated field on a dense, local detector grid using inverse Fourier chirp-z evaluation. This is a new field calculation at the detector coordinates, not interpolation of an intensity image. Its target is at least 24 points across the coherent diffraction-width estimate, with bounded detector size. It avoids globally enlarging the full 2D propagation grid merely to display a nanometre-scale focus smoothly.

The entrance grid still has to resolve amplitude and phase. Dense detector evaluation cannot reconstruct missing entrance frequencies or repair an underresolved lens calculation. Refined increases the propagation-grid budget too; propagation and detector grids are shown separately under **Model, mesh and diagnostics**. Direct point-source integration can require substantially more work than convolution.

A geometry-fitting window alone does not prove sufficient padding for FFT propagation. Expansion and grid sizes have memory limits, and the final display ROI is checked too. A boundary warning requires inspecting a larger window at a comparable step. A finer grid, larger window and more source nodes should be compared when quantitative accuracy matters. Automatic settings do not certify convergence.

## Numerical focus and automatic display ROI

`find_numerical_focus` accepts a callable peak-intensity objective and an initial distance. It expands a three-point bracket with a variable step, then refines a local maximum with bounded scalar optimization. It does **not** take a precomputed array of z planes.

The analytical image distance seeds the local basin. The default search bounds are 0.05–3 times that distance. Several diffraction maxima may exist: this is a local search, not a guarantee of a global optimum. An unresolved or boundary maximum is reported.

After focus search:

1. The axial display interval is estimated by following the peak intensity down to the **Axial ROI intensity fraction**, default 0.2 of the focus peak.
2. The selected interval is sampled for visualization; changing **Displayed z samples** does not change the preceding focus search.
3. Significant transverse intensity across that interval defines a padded x/y display ROI, with default threshold 0.01 of its maximum.

Manual z bounds are available in the advanced controls. They also bound the local focus search. The initial analytical guess is clamped to the selected interval.

## Compact results workspace

Images use **linear intensity**, CMRmap and white at the maximum. The left side of the dashboard has three vertically aligned plots with a shared horizontal z axis:

1. x–z intensity map (the grid row closest to y = 0 for 2D), above the statistics;
2. peak intensity versus z;
3. transverse FWHM versus z.

Plane peak intensity and FWHM are measured over the full detector plane; the longitudinal map remains a fixed central slice rather than following a moving y maximum. The right side shows the transverse ROI or an x/y profile. Its z slider is below the image, next to the varying intensity/FWHM/magnification readout. Image widgets update in place so slider changes do not move the page. Intensity normalization remains fixed to the focus/reference peak as z changes, making dimming visible; a linked cursor marks the selected z on the left-hand plots. One-curve plots have no redundant legend.

For a 2D field, each x profile passes through that plane's transverse intensity maximum, so the map, peak statistic and FWHM x use the same slice. Its y position can vary with z. For one-dimensional focusing, the main width plot shows the focused x width; the interactive plane also reports y width.

The Matplotlib **CMRmap** colormap is used with logarithmic normalization. Its maximum is white. The full map uses a common normalization; the interactive image is normalized within its selected plane.

A separate **z slider** evaluates the requested detector plane, including distances between the displayed samples. It reuses the lens and source state. A 1D grid shows the transverse profile; a 2D grid shows the image and x/y slices through its peak. Release the slider to trigger recalculation.

The global table shows collimated analytical BFL, finite-source analytical image distance, numerical focus, source magnification at focus, CRL length, transmission, backend, grid and actual window. The interactive table shows current z, peak intensity, FWHM and z-dependent magnification.

Magnification is the signed source-to-plane displacement scale -B(z)/source distance from the ray model. At an image plane it is image magnification; away from conjugate planes it describes source-position shifts rather than a sharp image. The unfocused y axis uses free propagation through the stack length.

## Limits and troubleshooting

- Intensities are relative. The entrance amplitude is one and is clipped to the nominal aperture. Transmission is finite-window output power divided by incident power after the entrance pupil.
- The optical model is scalar, monochromatic and paraxial. Polarization, back-reflections and high-angle propagation are outside it.
- For memory errors, reduce the batch size first. Reducing point counts also reduces accuracy. An explicit source-node count can be expensive on 2D grids; convolution is faster under its stated assumptions.
- A missing FWHM indicates that the required half-maximum crossings were not resolved within the computational window.
- If widget controls do not appear, rerun setup in a fresh Colab session and allow the standard custom-widget manager. The notebook requires ipywidgets.
- Colab session memory is temporary. Reopening the notebook recomputes the result; there is no save/restore/export mechanism.

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
    backend="auto",
)
result = xpos.estimate_crl_focus(config)
plane = result.plane_at(result.peak_z_m * 1.01)
```

All API lengths are metres. `CRL.apply_to_fields` handles a bounded batch of independent source fields. `CRL.source_magnification` and `CRL.ray_transfer_matrix` expose the discrete-stack ray model. The established low-level NumPy/CuPy interfaces remain available.

For a new public notebook version, first commit the tested library, then pin that commit and its module SHA-256 in setup. Commit the notebook afterwards and point the Colab link to the immutable notebook commit. Notebook outputs and execution counts should remain cleared in GitHub.

Validation for this revision: 13 existing CPU regression checks and 6 focused source/search/grid checks pass. The notebook was executed end to end in a Jupyter kernel, including conditional controls and live-slider callbacks. A real Colab front end and CUDA hardware were not available for validation; GPU execution remains to be checked in a GPU session.

## Display conventions and validation

Axes use fixed two-decimal tick labels. Length units adapt to the viewed range; large/small intensity values carry a power-of-ten scale on the axis. Parameters collapse after a successful calculation, and less frequently needed model/mesh details remain in an accordion. The workspace is width-limited and responsive instead of stretching plots across a wide monitor.

The interface was exercised in a local browser at 1366×768 and 1920×1080, including a real frontend slider event and unchanged page position. Colab adds its own toolbar/sidebar, so available cell width can vary. The dynamic form requires one initialization; rerunning it preserves inputs and waits for Compute focus rather than automatically repeating a potentially expensive run.

Numerical validation covers native-grid equivalence of chirp-z propagation, an analytical Gaussian field, batched propagation and incoherent source limits, in addition to the earlier CPU checks. A silicon 2D Refined run with 15 displayed z planes took approximately 45 seconds on the validation CPU and resolved the focal FWHM with over 40 detector samples. Runtime depends on hardware, z sample count, source nodes and grid size; actual CUDA timing remains unverified in this environment.
