# README: plt_slip_mu_relation.ipynb

**Purpose:** Visualize the relationship between 3D shear modulus (μ) structure and slip inversion bias in coseismic modeling.

**Date Created:** 2025-12-03

---

## Overview

This notebook analyzes how 3D heterogeneous shear modulus structure affects slip inversions by comparing:
- **Forward modeling:** Ground truth slip with 3D heterogeneous μ structure
- **Inversion 1:** Using correct 3D μ structure
- **Inversion 2:** Using homogeneous μ structure (ignoring heterogeneity)

The key question: **How does the μ contrast across the fault (Δμ = μ_HW - μ_FW) correlate with slip bias?**

---

## Input Requirements

### Configuration (Cell 3)

```python
meshname = "nicoyaCK4"
slip_str_gt = "_stripe_x0_y-45_lx80_dx35_rot0_ms0.0785_pou"
het3d_str = "_DeShon3D_ref_4"  # 3D heterogeneous model
homo_str = "_mul40u40"          # Homogeneous reference
inv_str = "_synlockbd_w945390_gs1e+02_ds1e-07"
fault_id = 7  # Fault boundary ID in mesh
blockleft = 8   # Subducting plate (footwall)
blockright = 9  # Overriding plate (hanging wall)
```

### Required Files

**Mesh files** (in `meshpath = /home/staff/chao/SSEinv/Nicoya/mesh/`):
- `nicoyaCK4.xml` - Main mesh
- `nicoyaCK4_facet_region.xml` - Boundary markers
- `nicoyaCK4_physical_region.xml` - Subdomain markers

**Shear modulus XDMF files** (in `resultpath = /home/staff/chao/SSEinv/Nicoya/syn_slip/`):
- `mu_true_nicoyaCK4_DeShon3D_ref_4.xdmf` - 3D heterogeneous μ
- `mu_true_nicoyaCK4_mul40u40.xdmf` - Homogeneous μ

**Slip files**:
- `mtrue_s_fault_nicoyaCK4_stripe_x0_y-45_lx80_dx35_rot0_ms0.0785_pou.txt` - Ground truth slip
- `m_s_fault_nicoyaCK4_stripe_..._DeShon3D_ref_4_synlockbd_..._DeShon3D_ref_4.txt` - Inferred slip (3D inversion)
- `m_s_fault_nicoyaCK4_stripe_..._DeShon3D_ref_4_synlockbd_..._mul40u40.txt` - Inferred slip (homogeneous inversion)

**Displacement files**:
- `d_obs_nicoyaCK4_stripe_..._DeShon3D_ref_4.txt` - Synthetic displacement at stations (6 columns: x, y, z, ux, uy, uz)

**GNSS station files** (in `datadir = /home/staff/chao/SSEinv/Nicoya/data/`):
- `CKfig6_data_final.csv` - GNSS station locations (lon, lat) and uncertainties

**Geographic reference files**:
- `Kyriakopoulos2016JGR/trench_xyz.txt` - Trench location
- `Kyriakopoulos2016JGR/Nicoya_interface.e` - Plate interface for depth contours

---

## Notebook Structure

### Section 1-2: Setup and Mesh Loading (Cells 1-6)
- Import libraries
- Load FEniCS mesh
- **Load fault_geometry** - Uses fault vertices (3165 nodes) matching slip data locations
- **Compute vertex normals** - Averages normals from 5 nearest facets for each vertex

**Key functions:**
- `compute_vertex_normals()` - Computes normals at fault vertices by averaging nearby facet normals
- `determine_normal_direction()` - Auto-detects and corrects normal direction
- `sample_mu_across_fault()` - Samples μ on both sides of fault using scipy griddata

**Important:** Uses fault vertices (3165 nodes) instead of facet centers to ensure **perfect alignment** with slip data coordinates

### Section 3: Load μ from XDMF (Cells 7-10)
- Load 3D and homogeneous μ grids using PyVista (via meshio)
- **Use scipy griddata** to interpolate μ values at fault vertices (3165 nodes)
- Compute μ anomaly: `(μ_3D - μ_hom) / μ_hom × 100%`

**Key function:** `load_xdmf_as_pyvista()`
- Converts XDMF to PyVista UnstructuredGrid
- μ values interpolated using `scipy.interpolate.griddata(method='nearest')`
- Ensures exactly 3165 values matching fault vertices

### Section 4-5: Load Slip and Displacement Data (Cells 11-21)
- Load ground truth slip
- Load inferred slip from both inversions (3D vs homogeneous)
- Compute slip difference: `slip_3D - slip_hom` (positive = hom underestimates)
- Add geographic coordinates (lon, lat, depth) to slip data
- **Load displacement at GNSS stations**

**Key function (Cell 20):** `rot_xy(x, y, rot)`
- Rotates coordinates/vectors by angle rot (degrees)
- Used to convert displacement from mesh coordinates to geographic coordinates

**Displacement loading (Cell 21):**
- Load displacement file (6 columns: x, y, z, ux, uy, uz)
- **Apply reverse rotation:** `rot_xy(ux, uy, -rot)` to convert to geographic coordinates
- Load actual GNSS station lon/lat from `CKfig6_data_final.csv`
- Compute horizontal displacement magnitude

### Section 6-9: Visualizations (Cells 22-29)
- **Cell 22-23:** Plot μ on fault surface using matplotlib
  - `plot_fault_mu_matplotlib()` with customizable colormap
- **Cell 24-25:** Plot slip comparison (3-panel PyGMT)
  - `plot_slip_comparison_pygmt()` - 3D inv, Hom inv, Difference
- **Cell 26-29:** Plot displacement at stations (3-panel PyGMT)
  - Load trench and plate interface data
  - Define displacement errors from GNSS uncertainties
  - `build_disp_vector()` - Builds horizontal and vertical displacement DataFrames
  - `plot_slip_and_displacement()` - True slip, Horizontal disp, Vertical disp with error bars

### Section 10: **Δμ Across Fault Calculation** (Cells 30-38)

**This is the core analysis!**

**Cell 33:** Define offset distance
```python
offset_distance = 2500.0  # meters (2.5 km, half of typical 5km mesh size)
```

**Cell 34-35:** Note on geometry computation
- Fault vertex normals already computed in Cell 6
- Functions already defined: `compute_vertex_normals()`, `determine_normal_direction()`, `sample_mu_across_fault()`

**Cell 35:** Sample μ on both sides of fault
- Uses `sample_mu_across_fault()` with scipy griddata interpolation
- Samples at ±2.5 km offset from fault vertices
- Returns `mu_hw` (hanging wall) and `mu_fw` (footwall)

**Cell 36:** Compute Δμ statistics
```python
delta_mu = mu_hw - mu_fw
```
- Negative Δμ: HW softer than FW
- Positive Δμ: HW stiffer than FW

**Cells 37-38:** Visualize Δμ on fault surface

### Section 11: **Correlation Analysis** (Cells 39-41)

**Cell 40:** Spatial interpolation
- Interpolate Δμ from fault vertices to slip node locations
- Uses scipy griddata with nearest neighbor for robustness
- **Note:** Since both use 3165 nodes, interpolation is minimal

**Cell 41:** Correlation plots
- **Panel 1:** Scatter plot: Δμ vs slip bias, colored by depth
- **Panel 2:** Binned statistics with error bars
- Compute Pearson correlation coefficient

**Expected pattern:**
- Negative Δμ (HW softer) → Positive slip bias (homogeneous inversion overestimates)
- Positive Δμ (HW stiffer) → Negative slip bias (homogeneous inversion underestimates)

### Section 12: Final Summary Figure (Cell 43)

**5-panel comprehensive figure:**
- **(a)** μ anomaly on fault
- **(b)** Δμ across fault (HW - FW)
- **(c)** Slip difference (Hom - 3D)
- **(d)** Structure-slip trade-off (correlation scatter)
- **(e)** Binned average

---

## Output Files

All saved to `resultpath = /home/staff/chao/SSEinv/Nicoya/syn_slip/`:

1. `slip_comparison_3panel.png` - Slip comparison (3D inv, Hom inv, Difference)
2. `slip_displacement_3panel.png` - True slip and displacement (slip, horizontal, vertical)
3. `slip_mu_relation_combined.png` - Initial multi-panel overview
4. `delta_mu_fault.png` - Δμ across fault visualization
5. `mu_hw_fw_comparison.png` - Side-by-side μ_HW and μ_FW
6. `correlation_delta_mu_slip_bias.png` - Correlation analysis (2 panels)
7. `complete_mu_slip_analysis.png` - **Final comprehensive 5-panel figure**

---

## Key Implementation Details

### Why Vertices Instead of Facets?

The notebook uses **fault vertices** (3165 nodes) rather than facet centers (6185 points) because:
- **Perfect alignment** with slip data coordinates (both use same fault_geometry file)
- Eliminates interpolation errors between slip and μ data
- Normals computed by averaging nearby facet normals (K=5 nearest neighbors)
- All data (slip, μ, Δμ) share identical spatial coordinates

### Normal Direction Convention

**Normal points from footwall to hanging wall:**
- Footwall (FW) = Subducting plate = `blockleft = 8`
- Hanging wall (HW) = Overriding plate = `blockright = 9`

This convention ensures:
```python
delta_mu = mu_hw - mu_fw
```
is interpreted correctly for the structure-slip trade-off.

### Sampling Offset Distance

**2.5 km offset** chosen because:
- Typical fault mesh size is ~5 km
- Sampling at half the mesh size avoids sampling the fault itself
- Far enough to be clearly on one side or the other
- Close enough to represent near-fault μ structure

**Auto-verification:** Cell 6 tests the normal direction by sampling at small offset (100 m) and checking which subdomain each point falls into.

### Interpolation Method

**scipy.interpolate.griddata with method='nearest'** used for:
1. **μ sampling at fault vertices:** Interpolates from XDMF mesh grid to fault vertices
2. **μ sampling across fault:** Interpolates μ at ±2.5 km offset points
3. **Δμ to slip nodes:** Minimal interpolation since both use 3165 nodes

**Why scipy instead of PyVista:**
- PyVista's `sample()` returned incorrect results (all mesh points instead of query points)
- scipy griddata is more reliable for arbitrary point interpolation
- Nearest neighbor preserves discrete μ contrasts without smoothing

### Coordinate Rotation

**rot_xy(x, y, rot) function:**
- Displacement vectors stored in rotated mesh coordinates
- Must apply **reverse rotation** `-rot` to convert to geographic coordinates
- Formula: `x_rot = x*cos(rot) + y*sin(rot)`, `y_rot = -x*sin(rot) + y*cos(rot)`
- Applied to displacement before visualization

---

## Physics Background

### Structure-Slip Trade-off

When inverting for slip with the **wrong** μ structure (homogeneous instead of 3D):

**If μ_HW < μ_FW (softer hanging wall):**
- The homogeneous model underestimates compliance on HW side
- To match the same displacement, inversion compensates by **increasing slip**
- Result: **Positive slip bias** (overestimate)

**If μ_HW > μ_FW (stiffer hanging wall):**
- The homogeneous model overestimates compliance on HW side
- To match the same displacement, inversion compensates by **decreasing slip**
- Result: **Negative slip bias** (underestimate)

**Correlation coefficient** quantifies this trade-off strength.

### Why This Matters

For real earthquake studies:
- GPS/GNSS measures surface displacement
- We invert displacement → slip using Green's functions
- Green's functions depend on μ structure
- **Wrong μ → biased slip estimates**
- This notebook quantifies the bias magnitude and spatial pattern

---

## Related Documentation

- `fault_surface_visualization.md` - Theory and implementation details for Δμ calculation
- `visualization_strategy_forward_sensitivity.md` - Overall visualization strategy
- `3D_heterogeneity_and_slip_inversion.md` - Physics of structure-slip trade-off
- `understanding_shear_modulus_invariance.md` - Why μ contrasts matter

---

## Usage Example

```python
# 1. Configure parameters (Cell 3)
meshname = "nicoyaCK4"
het3d_str = "_DeShon3D_ref_4"
homo_str = "_mul40u40"
problem_type = 'locking'  # or 'coseismic'

# 2. Run all cells sequentially

# 3. Key outputs to examine:
# - Cell 36: Δμ statistics (mean, min, max)
# - Cell 41: Correlation coefficient and p-value
# - Cell 25: slip_comparison_3panel.png (3D vs Hom slip)
# - Cell 29: slip_displacement_3panel.png (slip + displacement)
# - Cell 43: complete_mu_slip_analysis.png (comprehensive 5-panel)

# 4. Interpret results:
# - Look for systematic patterns in Δμ spatial distribution
# - Check correlation strength (r value)
# - Identify regions of high slip bias
# - For locking: focus on s_dip component (downdip coupling)
```

---

## Troubleshooting

**Issue:** "Field 'shear modulus' not found in XDMF"
- **Fix:** Check field name in Cell 10, may be 'f_12' or other name

**Issue:** Normal direction incorrect (Cell 6)
- **Fix:** Auto-correction should handle this, but verify `blockleft` and `blockright` IDs are correct

**Issue:** NaN values in delta_mu
- **Fix:** Normal, some sampling points may be outside mesh bounds, filtered in correlation analysis

**Issue:** PyGMT plots not displaying
- **Fix:** Use `fig.savefig()` instead of `fig.show()` if running non-interactively

**Issue:** PyVista sampling returns wrong number of points
- **Fix:** Already resolved - notebook now uses scipy griddata instead of PyVista sampling

**Issue:** Displacement vectors show as NaN
- **Fix:** Ensure `rot_xy()` function is defined (Cell 20) and reverse rotation is applied (Cell 21)

**Issue:** Mismatch between slip and μ coordinates
- **Fix:** Already resolved - both use fault_geometry coordinates (3165 nodes)

---

## Author Notes

**Created:** 2025-12-03
**Updated:** 2025-12-04
**Case Study:** Nicoya Peninsula, Costa Rica subduction zone
**Mesh:** nicoyaCK4 (tetrahedral, ~13k vertices, ~69k cells)
**Slip Pattern:** 2-stripe along-strike pattern
**Structure:** DeShon et al. 3D seismic velocity model converted to μ
**Problem Type:** Locking (interseismic coupling)

**Key Changes (2025-12-04):**
- Switched from facet centers (6185) to fault vertices (3165) for coordinate consistency
- Changed from PyVista sampling to scipy griddata interpolation for reliability
- Added coordinate rotation for displacement vectors (mesh → geographic)
- Added 3-panel PyGMT visualizations for slip comparison and displacement
- Integrated actual GNSS station locations from `CKfig6_data_final.csv`

**Key Result Expected:** Negative correlation between Δμ and slip bias, confirming that homogeneous inversions systematically misestimate slip in regions with strong μ contrast across the fault.
