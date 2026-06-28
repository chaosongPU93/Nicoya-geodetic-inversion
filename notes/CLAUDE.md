# Project Memory for Claude Code

## CG2 Shear Modulus Visualization (2026-02-01)

### Current Status (2026-02-04): ✓ WORKING

CG2 visualization is fully functional:
- **Data loads correctly**: 212,634 DOF points (7.35x more than 28,940 mesh vertices)
- **File format**: `_dofs.h5` preserves all CG2 DOFs (vertices + edge midpoints)
- **Notebook complete**: `plt_mu_slices_CG2.ipynb` supports both CG1 and CG2

### What was done
1. Updated `codes/plt_mu_slices_CG2.ipynb` to support CG2 (quadratic element) visualization
2. **Fixed critical bug**: FEniCS `XDMFFile.write()` only exports vertex values, losing CG2 edge midpoint DOFs
3. **Fixed h5py version conflict**: Rebuilt h5py to match FEniCS HDF5 version (1.14.3)

### Critical Fix: CG2 File Format (2026-02-01)

**Problem discovered**: Standard XDMF export only saves mesh vertices (28,940 points), not CG2 DOFs (212,634 points).

**Solution**: Modified `test_higher_CGmu_synth_stripeslip_3DCK_noi.py` to save additional `_dofs.h5` file:
- Added `import h5py` at line 55
- Added `save_function_with_dofs()` helper function (lines 57-104)
- Function called at lines 535, 544, 551, 1051, 1059, 1065
- CG2 now outputs: `.xdmf` (for ParaView) + `_dofs.h5` (all DOF coordinates + values)

**Notebook update**: `load_xdmf_as_pyvista()` now loads from `_dofs.h5` for CG2 data.

### h5py Version Fix (2026-02-01)

h5py must be compiled against the same HDF5 version as FEniCS uses. Fix applied:
```bash
conda activate fenics
pip uninstall h5py -y
pip install h5py --no-binary h5py
# Verify: python -c "import h5py; print(h5py.version.hdf5_version)"  # Should show 1.14.3
```

### File outputs
- **Homogeneous/K_2LAYER (CG1)**: `mu_true_..._mul55u25.xdmf` + `.h5` (28,940 vertices)
- **3D Heterogeneous (CG2)**: `mu_true_..._DeShon3D_ref_4_hull_CG2.xdmf` + `.h5` + `_dofs.h5` (212,634 DOF points = 7.35x more)

### Related files
- `codes/test_higher_CGmu_synth_stripeslip_3DCK_noi.py` - Main script with adaptive CG degree
- `codes/utils.py` - Contains `process_velocity_models_hull()` function
- `codes/plt_mu_slices_CG2.ipynb` - CG2 visualization notebook (updated)
- `codes/plt_slip_mu_relation_CK.ipynb` - Original CG1 notebook (kept as reference)

### Workflow
1. Select model type in `test_higher_CGmu_synth_stripeslip_3DCK_noi.py`:
   - For 3D: `mtrue_mu_expr_for = mtrue_mu_expr_het` (where `mtrue_mu_expr_het = None`)
   - For homogeneous: `mtrue_mu_expr_for = mtrue_mu_expr_hom`
2. CG degree is set automatically (CG2 for 3D, CG1 for homogeneous)
3. Run the script to generate output files (including `_dofs.h5` for CG2)
4. Open `plt_mu_slices_CG2.ipynb` to visualize using CG2 + tolerance method

### CG1 vs CG2 approach
- **CG1**: Load from XDMF as UnstructuredGrid, use PyVista `.slice()`
- **CG2**: Load from `_dofs.h5` as PolyData point cloud (7.35x more sample points)

### Slicing Method Comparison (at 40 km depth, 1 km tolerance)

| Method | Points | Description |
|--------|--------|-------------|
| CG1 + `.slice()` | 10,329 | Interpolated at plane-mesh intersection |
| CG1 + tolerance | 391 | Actual vertices within tolerance slab |
| CG2 + tolerance | 3,004 | Actual DOFs within tolerance slab |

**Key insight**: PyVista's `.slice()` interpolates along cell edges (creating many points), but CG2's tolerance-based extraction uses actual DOF values without interpolation. CG2 provides 7.68x more real data points than CG1 tolerance method.

### Memory Fix: Avoid redundant dl.project() calls (2026-02-02)

**Problem**: Running with `CG_mu_deg = 2` caused UMFPACK out-of-memory error:
```
UMFPACK V5.7.9: ERROR: out of memory
RuntimeError: PETSc error code is: 76 (Error in external library)
```

**Root cause**: `dl.project(mtrue_mu_fun, CG_mu)` solves a mass matrix system. With CG2 (~180,000 DOFs vs ~29,000 for CG1), this becomes too large.

**Why projection was unnecessary**: `mtrue_mu_fun` is already created as `dl.Function(CG_mu)` (line 1036), so projecting it back to the same space is redundant.

**Fix applied**: Commented out all redundant `dl.project()` calls and use direct assignment:
```python
# OLD (causes memory error with CG2):
# m_mu_true = dl.project(mtrue_mu_fun, CG_mu)

# NEW (correct - mtrue_mu_fun already in CG_mu space):
m_mu_true = mtrue_mu_fun
```

**Locations fixed** in `test_higher_CGmu_synth_stripeslip_3DCK_noi.py`:
- Line 532: `m_mu_true` projection (commented out)
- Line 541: `vs_true` projection (commented out)
- Line 548: `den_true` projection (commented out)
- Line 728: `m_mu_true` projection (commented out)
- Line 1049: `m_mu_true` projection (commented out)
- Line 1058: `vs_true` projection (commented out)
- Line 1065: `den_true` projection (commented out)
- Line 1309: `m_mu_true` projection (commented out)

**Key insight**: When a FEniCS function is already in the target function space, direct assignment is correct and avoids expensive linear solves.

### Homogeneous vs Heterogeneous Fix: Direct computation (2026-02-02)

**Problem 1**: Running with homogeneous model (`mtrue_mu_expr_inv != None`) caused:
```
AttributeError: 'Product' object has no attribute 'rename'
```

**Problem 2**: Using `dl.project()` as a fix still caused out-of-memory with CG2:
```
UMFPACK V5.7.9: ERROR: out of memory
```

**Root cause**:
- **Heterogeneous case**: `mtrue_mu_fun = dl.Function(CG_mu)` → already a Function ✓
- **Homogeneous case**: `mtrue_mu_fun = mu_expression(mtrue_mu_expr_fun)` returns `20*(2.+ufl.tanh(m))` → UFL Product expression, not a Function ✗
- `dl.project()` solves a mass matrix system, too expensive for CG2 (212,634 DOFs)

**Fix applied**: Compute values directly with numpy (no linear solve needed):
```python
if isinstance(mtrue_mu_fun, dl.Function):
    m_mu_true = mtrue_mu_fun
else:
    # mtrue_mu_expr_fun is already in CG_mu space, compute mu = 20*(2+tanh(m)) directly
    m_mu_true = dl.Function(CG_mu)
    m_values = mtrue_mu_expr_fun.vector()[:]
    m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
```

**Locations fixed** in `test_higher_CGmu_synth_stripeslip_3DCK_noi.py`:
- `solveCoseismicForward`: lines 533-542
- `solveCoseismicInversion_TanhSlip`: lines 1053-1062

### 3D Velocity Model with Convex Hull Method (2026-02-04)

**New function**: `ut.process_velocity_models_hull()` in `codes/utils.py` (lines 4237-4452)

**Purpose**: Build 3D shear modulus model from DeShon 3D velocity model + 1D background, using convex hull to avoid extrapolation artifacts.

**Key improvement over original `process_velocity_models()`**:
- Uses **Delaunay triangulation** (convex hull) instead of rectangular bounding box
- Correctly handles rotated grids (e.g., 45° rotation) that would have ~50% empty corners in bounding box
- DOFs inside convex hull → 3D interpolated velocity
- DOFs outside convex hull → 1D layered background (no nearest-neighbor fallback)

**Workflow** (10 steps):
1. Clean data - remove NaN values
2. Sort 1D models by depth (shallow to deep)
3. Build Delaunay triangulation from 3D data points
4. Report 3D model extent
5. Create FEniCS functions (CG space)
6. Apply 1D background to ALL DOFs via `get_layered_1d_value()`
7. Classify DOFs using `hull.find_simplex() >= 0`
8. Override with 3D linear interpolation inside hull
9. Assign to FEniCS functions
10. Compute μ = ρ × vs² (in GPa)

**Verification test added** (lines 4375-4406): Checks lateral homogeneity of 1D assignment
```
Verification: checking lateral homogeneity of 1D assignment
  Checked 18845 unique depth levels
  Max vs variance within z-slices: 3.16e-30 (at z=-143095.8 m)
  Max density variance within z-slices: 0.00e+00 (at z=None m)
  ✓ PASSED: 1D assignment is laterally homogeneous
```

### Adaptive CG Degree for Shear Modulus (2026-02-04)

**Rationale**: CG2 (quadratic elements) benefits smooth 3D variations but is overkill for piecewise constant models.

| Model Type | `mtrue_mu_expr` | CG Degree | Reason |
|------------|-----------------|-----------|--------|
| Homogeneous/K_2LAYER | `K_2LAYER(...)` | 1 (default) | Piecewise constant values |
| 3D Heterogeneous | `None` | 2 | Smooth spatial variation benefits from edge midpoints |

**Implementation** in `test_higher_CGmu_synth_stripeslip_3DCK_noi.py` (lines 1784-1788):
```python
# Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
if mtrue_mu_expr_for is None:  # 3D heterogeneous model via process_velocity_models_hull
    CG_mu_deg = 2
else:  # Homogeneous or K_2LAYER
    CG_mu_deg = 1
```

**String naming convention**:
- Homogeneous/K_2LAYER: `_mul55u25` (no CG suffix, CG1 is default)
- 3D Heterogeneous: `_DeShon3D_ref_4_hull_CG2` (`_hull` = convex hull method, `_CG2` = quadratic elements)

### Visualization: CG2 + Tolerance is Best (2026-02-04)

**Problem with PyVista `.slice()`**: Interpolates within mesh cells, creating artifacts that reflect mesh geometry rather than actual model values.

**Comparison at 40 km depth**:
| Method | Points | Issue |
|--------|--------|-------|
| CG1 + `.slice()` | 10,329 | Mesh interpolation artifacts |
| CG1 + tolerance | 391 | Sparse → patchy appearance |
| CG2 + tolerance | 3,004 | ✓ Best: dense + actual DOF values |

**Recommendation**: Use CG2 + tolerance-based extraction for visualization. This shows actual DOF values without interpolation artifacts, and CG2 provides 8x more points than CG1 tolerance method.

### Vertical Dip Slice Plotting - Step 7.2 (2026-02-04)

Updated `plot_vertical_slices_dip_pygmt()` function in `plt_mu_slices_CG2.ipynb` with publication-quality settings matching step 10.6 in `plt_slip_mu_relation_CK.ipynb`.

**Key fixes applied:**

1. **Z-axis direction**: Changed `projection="X?/-?"` → `projection="X?"` so z=0 is at top, negative depths below
2. **Equal aspect ratio**: Panel width auto-calculated from x/z range ratio (1 km = 1 km on plot)
3. **Axis labels**: "Along-dip (km)" for x-axis, "Depth (km)" for y-axis
4. **Slice labels**: "Along-strike=X km" at top-right corner
5. **Adjustable margins**: `margin_x`, `margin_y` parameters (in cm)

**Flexible layout support:**

| Layout | Colorbar Position | Use Case |
|--------|------------------|----------|
| `ncols=1` | Shared vertical on right (`+jML`) | Wide aspect ratios |
| `ncols>1` | Shared horizontal at center (`+jTC`) | Standard 2×2 layout |

**Colorbar positioning (outside subplot loop):**
```python
# Single column: vertical colorbar on right
position=f"x{cbar_x}c/{cbar_y}c+w{cbar_length}c/{cbar_thickness}c+v+jML"

# Multi-column: horizontal colorbar centered
position=f"x{cbar_x}c/{cbar_y}c+w{cbar_width}c/{cbar_height}c+h+jTC"
```

**Frame settings:**
- `frame=["WSne"]` in subplot: W,S get annotations; n,e get ticks only
- `frame=["WS"]` would show only left and bottom axes (no ticks on top/right)

**Surface warnings (can be ignored):**
```
surface [WARNING]: XXX unusable points were supplied; these will be ignored.
```
These warnings occur due to floating-point precision causing duplicate grid points. They don't affect output quality. To suppress: `pygmt.config(GMT_VERBOSE="e")` or round coordinates before gridding.

**Usage example:**
```python
fig = plot_vertical_slices_dip_pygmt(
    mu_3d_grid,
    y_dip_km=[-10, 10, 30, 50],
    nrows=4, ncols=1,           # Single column layout
    x_range=(-100, 80),
    z_range=(-60, 0),
    depth_levels=[-20, -30, -40, -50],  # Must be negative!
    margin_x=0.15, margin_y=0.6,
    filename=resultpath + f'mu_vertical_dip_CG{CG_mu_deg}.pdf'
)
```

## Uneven Mesh Setup and Inversion Scripts (2026-02-02)

### Background
The original mesh `nicoyaCKden_sm` has both left and right blocks with top boundary at z=0 (surface). However, the plate interface from Kyriakopoulos et al. does not reach the surface on the left (trench) side, causing a sharp, nearly vertical transition that produces artifacts in synthetic ground displacements.

### New Meshes with Uneven Top Boundary
- `nicoyaCKden_une_sm` - Uneven top, smaller fault zone (counterpart to `nicoyaCKden_sm`)
- `nicoyaCKden_une_all` - Uneven top, all subduction interface (counterpart to `nicoyaCKden_all`)

**Characteristics:**
- Left block (trench side): top at median depth ~7 km
- Right block (land side): top at z=0 (surface)

### Synthetic Test Scripts (with L-curve)
These scripts use dense observation grids and require helper functions for uneven mesh handling.

| Script | Mesh | Het L-curve | Hom L-curve |
|--------|------|-------------|-------------|
| `synth_stripeslip_inv_hetmu_uneven_nicoyaCK_lock_noi.py` | `nicoyaCKden_une_sm` | Yes | Yes |
| `synth_stripeslip_inv_hetmu_uneven_nicoyaCK_lock_noi2.py` | `nicoyaCKden_une_sm` | No | Yes |
| `synth_stripeslip_inv_hetmu3D_uneven_nicoyaCK_lock_noi.py` | `nicoyaCKden_une_sm` | Yes | Yes |

**L-curve gammas (complete set):** `[4e1, 6e1, 8e1, 1e2, 1.8e2, 2e2, 2.2e2, 2.5e2, 3e2, 3.5e2, 4e2, 1e3, 1e4]`

**Key helper functions** (for synthetic tests with dense grids):
- `extract_top_boundary_surface()` - Extract top boundary vertices from mesh
- `assign_z_via_ray_casting()` - Assign z-coordinates via vertical ray casting
- `compute_disp_on_grid_uneven()` - Compute displacements handling out-of-mesh points

### Real Data Inversion Scripts
These scripts use actual GNSS station locations (all on land at z=0), so they don't need uneven mesh helper functions.

| Script | Mesh | Het L-curve | Hom L-curve |
|--------|------|-------------|-------------|
| `slip_inv_hetmu_nicoyaCK_locking_both.py` | `nicoyaCKden_une_all` | Yes | Yes |
| `slip_inv_hetmu3D_nicoyaCK_locking_both.py` | `nicoyaCKden_une_all` | Yes | No (intentional) |
| `slip_inv_hetmu_nicoyaCK_coseis.py` | `nicoyaCKden_une_all` | Yes | Yes |
| `slip_inv_hetmu3D_nicoyaCK_coseis.py` | `nicoyaCKden_une_all` | Yes | No (intentional) |

**Design pattern:** 3D versions (with DeShon 3D velocity model) have only Het L-curve; non-3D versions have both Het and Hom L-curve.

### Memory Management
All L-curve loops include:
```python
import gc
# ... inside loop ...
del results
gc.collect()
```

### Verification Scripts (updated with argparse)
These scripts now accept custom mesh names via `--mesh/-m` and `--meshpath/-p` arguments:
- `verify_mesh_blocks.py` - Verify block structure and top boundary depths
- `verify_mesh_top.py` - Verify top boundary statistics
- `check_points_in_mesh.py` - Check observation points inside/outside mesh
- `diagnose_excluded_points.py` - Diagnose excluded observation points

**Usage:** `python verify_mesh_blocks.py -m nicoyaCKden_une_sm`

### Output Format
L-curve results use 4-column format:
```python
csvoutput.write("%.6e %.6e %.1e %.0e\n" % (dmisfit[i], mmisfit[i], gammas_s[i], rho_s))
```

## Uneven Mesh Support in test_higher_CGmu Script (2026-02-05)

### Changes Made to `test_higher_CGmu_synth_stripeslip_3DCK_noi.py`

Integrated uneven mesh handling from `synth_stripeslip_inv_hetmu3D_uneven_nicoyaCK_lock_noi.py` to support both:
- CG2 for 3D models
- Meshes with uneven top boundary

#### 1. Added Helper Functions (lines 108-304)
```python
extract_top_boundary_surface(mesh, boundaries, top_id)  # Extract top boundary vertices
find_valid_surface_depths_raycast(x_query, y_query, mesh, ...)  # LEVEL 1: Ray casting for z-depths
extrapolate_displacements(targets, d_obs, valid_indices, ...)  # LEVEL 2: Extrapolate failed points
```

#### 2. Mesh Selection with Auto-Detection (lines 519-529)
```python
# Meshes with even top boundary at 0 depth
meshname = "nicoyaCKden_sm"
# meshname = "nicoyaCKden_all"

# Meshes with uneven top boundary
# meshname = "nicoyaCKden_une_sm"
# meshname = "nicoyaCKden_une_all"

use_uneven_mesh = "une" in meshname  # Auto-detected flag
```

#### 3. Deferred Z-Depth Assignment (lines 420-448)
For uneven meshes, z-depth assignment is deferred until after mesh loading. Dense grid created with placeholder z=0.

#### 4. Z-Depth Assignment via Ray Casting (lines 556-617)
After mesh loading, if `use_uneven_mesh=True`:
- Extract top boundary surface from mesh
- Ray casting (LEVEL 1) to find valid z-depths at each (x,y)
- Extrapolate z for points outside mesh horizontal footprint
- Update `dense_data['z']` with computed depths

#### 5. LEVEL 2 Extrapolation in Forward Functions
Both `solveCoseismicForward()` and `computeGridDisplacements()` now include:
- Check if all target points computed successfully
- Build bounding box tree to identify failed points
- Extrapolate displacements from nearby valid points
- Return `d_full` (all points, including extrapolated)

**Updated return values:**
- `solveCoseismicForward()`: returns `d_full` instead of `d_obs`
- `computeGridDisplacements()`: returns `(d_full, valid_indices)`

### Two-Level Approach for Uneven Mesh Handling

| Level | Purpose | When Points Fail |
|-------|---------|------------------|
| **LEVEL 1** (Ray Casting) | Find valid z-depth for each (x,y) | (x,y) outside mesh horizontal footprint |
| **LEVEL 2** (Displacement) | Compute displacements at (x,y,z) | Numerical precision / mesh topology edge cases |

Both levels use nearest-neighbor extrapolation to ensure continuous fields.

### Uneven Mesh Support in `plt_mu_slices_CG2.ipynb` (2026-02-05) ✓ IMPLEMENTED

Updated the visualization notebook to support meshes with uneven top boundary (e.g., `nicoyaCKden_une_sm`).

**Backward compatible**: For even top boundary meshes, results are identical (no masking, no boundary line).

#### Changes Made

**1. Imports (Cell 1)**
- Added `LinearNDInterpolator` from scipy.interpolate
- Added optional `dolfin` import for mesh loading

**2. Configuration (Cell 3)**
- Added `use_uneven_mesh = "une" in meshname` auto-detection flag

**3. New Cell 4: Top Boundary Extraction**
Functions added:
```python
extract_top_boundary_from_mesh(mesh, boundaries, top_id=1)  # Uses FEniCS boundary markers
build_z_top_interpolator(top_coords)  # Returns LinearNDInterpolator for z_top(x,y)
```
- Loads mesh only if `use_uneven_mesh=True` and dolfin available
- `z_top_interp` is `None` for even meshes (no masking)

**4. Slice Extraction Functions (Cell 12)**
All extraction functions now accept `z_top_interp` parameter:
```python
extract_horizontal_slice(grid, z_depth, ..., z_top_interp=None)
extract_vertical_slice_dip(grid, y_const, ..., z_top_interp=None)
extract_vertical_slice_strike(grid, x_const, ..., z_top_interp=None)
```
- When `z_top_interp` is provided, points where `z > z_top(x,y)` are set to NaN
- Helper functions for boundary line plotting:
  - `get_top_boundary_line_dip(y_const, x_range, z_top_interp)`
  - `get_top_boundary_line_strike(x_const, y_range, z_top_interp)`

**5. Horizontal Slice PyGMT (Cell 32)**
- `plot_horizontal_slices_pygmt()` accepts `z_top_interp` parameter
- **Two-level masking**: point-level NaN in extraction + grid-level NaN after `pygmt.surface()`
- Grid-level mask converts lon/lat grid nodes to local mesh coords: `x_rot - x0, y_rot - y0`
- `fig.grdimage(nan_transparent=True)` renders masked areas as transparent
- Uses `np.nanmean()` for anomaly calculation

**6. Vertical Slice PyGMT (Cell 36)**
- `plot_vertical_slices_dip_pygmt()` accepts `z_top_interp` parameter
- **Two-level masking**: point-level NaN in extraction + grid-level NaN after `pygmt.surface()`
- Grid-level mask queries `z_top_interp(x_km*1e3, y_m)` at each grid node
- `fig.grdimage(nan_transparent=True)` renders masked areas as transparent
- Draws top boundary as dashed black line: `pen="1.0p,black,--"`

**7. Function Call Cells (Cells 33, 37, 38)**
All plotting calls now pass `z_top_interp=z_top_interp`

#### Usage

**For even mesh (default):**
```python
meshname = 'nicoyaCKden_sm'  # use_uneven_mesh = False
# z_top_interp = None (no masking, no boundary line)
```

**For uneven mesh:**
```python
meshname = 'nicoyaCKden_une_sm'  # use_uneven_mesh = True
# z_top_interp is built from mesh top boundary
# Vertical slices show dashed boundary line
# Above-boundary regions left blank
```

#### Design Decisions
- **Blank (not gray)** for above-boundary regions
- **Dashed black line** for top boundary on vertical slices
- **Extract from mesh** each time (no separate file needed)
- **LinearNDInterpolator** for z_top(x,y) queries (can switch to ray casting if needed)

#### Critical Fix: Grid-Level Masking After `pygmt.surface()` (2026-02-05)

**Problem**: Point-level NaN masking alone was insufficient. `pygmt.surface()` is an interpolation/extrapolation algorithm that fills the **entire** plotting domain. After removing NaN points before gridding, `surface` extrapolated into the above-boundary region, producing fabricated values (saturated red in vertical slices, uniform dark red in horizontal slices).

**Solution**: Two-level masking approach:
1. **Point-level** (existing): Set values to NaN in `extract_*_slice()` functions
2. **Grid-level** (new): After `pygmt.surface()`, re-mask grid nodes above top boundary

**Vertical slices** (Cell 36) — grid nodes are in km (along-dip, depth):
```python
# After data_grid = pygmt.surface(...)
if z_top_interp is not None:
    xx_g, zz_g = np.meshgrid(data_grid.x.values, data_grid.y.values)
    xy_q = np.column_stack([xx_g.ravel() * 1e3, np.full(xx_g.size, y_m)])
    z_top_grid = z_top_interp(xy_q).reshape(xx_g.shape) / 1e3
    above = (zz_g > z_top_grid) | np.isnan(z_top_grid)
    data_grid.values[above] = np.nan
```

**Horizontal slices** (Cell 32) — grid nodes are in lon/lat, must convert to local mesh coords:
```python
# After data_grid = pygmt.surface(...)
if z_top_interp is not None:
    lon_gg, lat_gg = np.meshgrid(data_grid.x.values, data_grid.y.values)
    x_rot, y_rot = utp.LL2ckmd(lon_gg.ravel(), lat_gg.ravel(), lon0, lat0, rot)
    xy_q = np.column_stack([x_rot - x0, y_rot - y0])  # MUST subtract offset!
    z_top_at_grid = z_top_interp(xy_q).reshape(lon_gg.shape)
    above = (depth_m > z_top_at_grid) | np.isnan(z_top_at_grid)
    data_grid.values[above] = np.nan
```

**Key pitfall**: `utp.LL2ckmd()` returns rotated coordinates WITH `(x0, y0)` offset, but `z_top_interp` expects LOCAL mesh coordinates (without offset). Must subtract `x0, y0`. This is consistent with the pattern used everywhere: `vel3d['x'] = x_rot - x0`.

### PyGMT Horizontal Slice Fixes and flag_savefig (2026-02-05)

**Issues fixed in `plot_horizontal_slices_pygmt()` (Cell 32)**:
1. Removed dead `cbar_width`/`cbar_height` definitions (leftover from commented-out per-panel colorbar)
2. Moved `makecpt` before the subplot loop (was inside loop; if last panel had no data, shared colorbar had no CPT)
3. Added `ncols==1` handling for colorbar layout (matching `plot_vertical_slices_dip_pygmt`)
4. Fixed colorbar y-position (was hardcoded for 2-row layout)
5. Fixed misleading print statement

**`flag_savefig` added (Cell 3)**:
```python
flag_savefig = 1  # 1=save figures to file, 0=display only
```
- Cells 33, 37, 38 now pass `filename=... if flag_savefig else None`
- Both `plot_horizontal_slices_pygmt()` and `plot_vertical_slices_dip_pygmt()` already check `if filename:` before `fig.savefig()`, so `None` skips saving

## Notebook: plt_1Dmodel_profile.ipynb (2026-02-28)

### Purpose
Depth profiles of DeShon (2006) 1D/3D velocity, density, shear modulus, and mesh-projected
CG2 shear modulus. Output PDFs saved to `DeShon_2006GJI/`.

### Cell structure (10 cells, run in order)
| Index | Content |
|-------|---------|
| [0] | Markdown title |
| [1] | Imports: numpy, pandas, matplotlib, `utils_plot as utp` |
| [2] | Load 1D/3D velocity + density; reproject vel3d via `utp.LL2ckmd` |
| [3] | Load mesh-projected CG2 μ from `_dofs.h5` (h5py) |
| [4] | Delaunay hull classify DOFs (inside_3d mask) + depth bracket histogram |
| [5] | `layer_xy()` helper for staircase plotting |
| [6] | 4-panel 2×2 → `plt_1Dmodel_profile_4panel.pdf` |
| [7] | 2-panel: all DOFs colored 3D/1D + 3D-hull only → `plt_mu_mesh_projected.pdf` |
| [8] | Single-panel: all mesh DOFs μ vs depth |
| [9] | 3-panel: Vp/Vs, density, tabulated 3D μ → `plt_1Dmodel_profile.pdf` |

### CRITICAL: vel3d coordinate system
The CSV `x, y` columns are in a different system — always recompute from lon/lat:
```python
lon0, lat0 = -84, 7;  rot = 45;  x0, y0 = 130e3, 350e3  # metres
x_rot, y_rot = utp.LL2ckmd(vel3d['lon'], vel3d['lat'], lon0, lat0, rot)
vel3d['x'] = x_rot - x0   # metres, local mesh coords
vel3d['y'] = y_rot - y0
```
`LL2ckmd` is in `utils_plot` (not `utils`).

### Hull classification (cell [4]) — mirrors process_velocity_models_hull
```python
hull_3d = Delaunay(vel3d[['x', 'y', 'z']].values)  # all metres
inside_3d = hull_3d.find_simplex(dof_coords) >= 0
```
- vel3d includes 700 km depth boundary node → hull spans 0–700 km depth
- "3D hull" = within the 3D model's **horizontal footprint** at any depth (0–400 km)
- "1D background" = laterally outside the 3D model footprint only
- This is consistent with `process_velocity_models_hull` (intended behaviour)
- DOFs at 90–400 km inside the footprint get linearly interpolated values between
  90 km 3D data and 700 km boundary values (benign for Nicoya fault depths)

### zorder gotcha in scatter plots
Plot 1D background (steelblue) with HIGHER zorder than 3D hull (tomato), otherwise
the 1D background dots are invisible where they overlap the dense 3D hull cloud.
