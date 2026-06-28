# Notes: Visualizing CG2 Shear Modulus Data

**Date:** 2026-01-30
**Updated:** 2026-02-01 (file format fix for proper CG2 DOF export)

**Related files:**
- `plt_mu_slices_CG2.ipynb` (new notebook for CG2 visualization, with publication-quality plotting)
- `plt_slip_mu_relation_CK.ipynb` (original notebook for CG1 - kept as safe reference)
- `test_higher_CGmu_synth_stripeslip_3DCK_noi.py` (generates CG2 mu models)

---

## Background

The shear modulus (μ) model was previously generated and stored using CG1 (linear) elements:
```python
CG_mu = dl.FunctionSpace(mesh, "CG", 1)
```

To get a smoother representation and reduce "grainy" artifacts, we want to use CG2 (quadratic) elements:
```python
CG_mu = dl.FunctionSpace(mesh, "CG", 2)
vs_func, den_func, mu_func, _ = ut.process_velocity_models(vel3d, vel1d, den1d, mesh, CG_mu_degree=2, verbose=False)
```

---

## The Problem

When using CG2, the XDMF file contains data at **more locations** than CG1:

| Element | DOF Locations | Points per Tetrahedron |
|---------|---------------|------------------------|
| CG1 | Vertices only | 4 |
| CG2 | Vertices + edge midpoints | 10 |

The original `load_xdmf_as_pyvista()` function creates a PyVista `UnstructuredGrid` where:
- `points` array has all DOF coordinates (vertices + edge midpoints for CG2)
- `cells` array only references the 4 corner vertices (mesh topology unchanged)
- This creates a mismatch where edge midpoint values are "orphaned" (not connected to any cells)

When PyVista's `.slice()` method is called, it uses cell topology for intersection, so the edge midpoint DOF values are ignored - **defeating the purpose of using CG2**.

---

## The Solution

### Key Insight
The purpose of CG2 is to get **~6x more sample points** (vertices + edge midpoints). To preserve this benefit for visualization, we need to use ALL DOF points, not just the ones connected by cells.

### Approach

1. **Load CG2 data as a PolyData point cloud** (not UnstructuredGrid)
   - Preserves all DOF locations
   - No cell topology needed

2. **Extract slices using tolerance-based filtering** (instead of PyVista `.slice()`)
   - `.slice()` requires cell topology (won't work for point clouds)
   - Tolerance-based: extract all points within distance of slice plane
   - Result: denser sampling from CG2 DOFs

3. **Interpolate to regular grid using `griddata`** (same as original)
   - More input points → smoother interpolation result

### Unified Approach (works for both CG1 and CG2)

```python
def load_xdmf_as_pyvista(filepath, cg_degree=1):
    if cg_degree == 1:
        # CG1: UnstructuredGrid with cell topology
        grid = pv.UnstructuredGrid(cells, cell_types, points)
    else:
        # CG2+: PolyData point cloud (all DOFs)
        grid = pv.PolyData(points)
    return grid

def extract_horizontal_slice(grid, z_depth, tolerance=1000):
    if isinstance(grid, pv.UnstructuredGrid):
        # CG1: PyVista .slice() for exact plane intersection
        slice_obj = grid.slice(normal='z', origin=[0, 0, z_depth])
    else:
        # CG2: Tolerance-based filtering
        mask = np.abs(z - z_depth) < tolerance
    return x, y, values
```

---

## Why Not Just Extract Vertex Values from CG2?

**Question raised:** If CG2 data is generated but we only extract values at vertices (same points as CG1), does it make any difference?

**Answer:** No significant difference. The benefit of CG2 comes from:
1. Quadratic interpolation within elements (using vertex + midpoint values)
2. Denser sampling for visualization

If you only use vertex values, you lose both benefits. You must use ALL DOF points to see the improvement.

---

## Critical Fix: XDMF Export Only Saves Vertex Values (2026-02-01)

### The Problem Discovered

FEniCS's `XDMFFile.write()` method exports data for **visualization purposes**, which means:
- It writes values at **mesh vertices only** (the mesh geometry)
- For CG1: This works correctly because DOFs = vertices
- For CG2: The edge midpoint DOFs are **NOT exported** - they're lost!

**Evidence:**
```python
# Both files had identical structure:
# CG1: 28,940 points, 148,258 cells
# CG2: 28,940 points, 148,258 cells  <-- Should be ~180,000+ points!
```

### The Fix

Modified `test_higher_CGmu_synth_stripeslip_3DCK_noi.py` to save a separate `_dofs.h5` file containing ALL DOF coordinates and values:

```python
def save_function_with_dofs(function, filename_base, attr_name, cg_degree=1):
    # Always save standard XDMF (for ParaView compatibility)
    xdmf_file = dl.XDMFFile(filename_base + '.xdmf')
    function.rename(attr_name, attr_name)
    xdmf_file.write(function)

    # For CG2+, also save all DOF points to HDF5
    if cg_degree > 1:
        V = function.function_space()
        dof_coords = V.tabulate_dof_coordinates()  # ALL DOFs!
        dof_values = function.vector()[:]

        with h5py.File(filename_base + '_dofs.h5', 'w') as f:
            f.create_dataset('coordinates', data=dof_coords)
            f.create_dataset('values', data=dof_values)
            f.attrs['cg_degree'] = cg_degree
            f.attrs['n_dofs'] = len(dof_coords)
            f.attrs['n_vertices'] = V.mesh().num_vertices()
```

---

## File Naming Convention

When saving CG2 models, the suffix `_CGmu2` is added:
```python
if CG_mu_deg > 1:
    mu_str_het = mu_str_het + f"_CGmu{CG_mu_deg}"
```

**CG1 output files:**
- `mu_true_nicoyaCKden_sm_DeShon3D_ref_4.xdmf` + `.h5`

**CG2 output files:**
- `mu_true_nicoyaCKden_sm_DeShon3D_ref_4_CGmu2.xdmf` + `.h5` (vertex values only, for ParaView)
- `mu_true_nicoyaCKden_sm_DeShon3D_ref_4_CGmu2_dofs.h5` (**NEW**: all DOF coordinates + values)

---

## h5py Version Fix (2026-02-01)

### The Problem

When running the script with `import h5py`, you may encounter:
```
h5py is running against HDF5 1.14.3 when it was built against 1.14.6
ValueError: Not a datatype (not a datatype)
```

This happens because h5py was compiled against a different HDF5 version than FEniCS uses.

### The Fix

Rebuild h5py from source against your system's HDF5:
```bash
conda activate fenics
pip uninstall h5py -y
pip install h5py --no-binary h5py
```

### Verify

```bash
python -c "import h5py; print(f'h5py HDF5 version: {h5py.version.hdf5_version}')"
# Should output: h5py HDF5 version: 1.14.3
```

Also verify both work together:
```python
import dolfin as dl
import h5py
# If no errors, versions are compatible
```

---

## Summary

| Aspect | CG1 | CG2 |
|--------|-----|-----|
| DOFs per element | 4 (vertices) | 10 (vertices + midpoints) |
| Load as | UnstructuredGrid | PolyData point cloud |
| Slice method | PyVista `.slice()` | Tolerance-based filtering |
| Interpolation | `griddata` | `griddata` |
| Point density | Base | ~6x denser |
| Visual result | May have grainy artifacts | Smoother |

---

## Memory Fix: Avoid redundant dl.project() (2026-02-02)

### The Problem

Running with `CG_mu_deg = 2` caused UMFPACK out-of-memory error at `dl.project()` calls:
```
UMFPACK V5.7.9: ERROR: out of memory
```

### Why It Happened

`dl.project(mtrue_mu_fun, CG_mu)` solves a mass matrix linear system. With CG2:
- CG1: ~29,000 DOFs → manageable
- CG2: ~180,000 DOFs → 6x larger system → UMFPACK runs out of memory

### Why Projection Was Unnecessary

Looking at the code:
```python
mtrue_mu_fun = dl.Function(CG_mu)  # Already in CG_mu space!
new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)
```

`mtrue_mu_fun` is created directly as `dl.Function(CG_mu)`, so projecting it back to the same space is redundant.

### The Fix

Replace projection with direct assignment:
```python
# OLD (causes memory error):
# m_mu_true = dl.project(mtrue_mu_fun, CG_mu)

# NEW (correct):
m_mu_true = mtrue_mu_fun
```

Same fix applied to `vs_func` and `den_func` (also already in CG_mu space).

---

## Usage

### Step 1: Generate CG2 data

In `test_higher_CGmu_synth_stripeslip_3DCK_noi.py`, set:
```python
CG_mu_deg = 2  # Use CG2 elements
```

Run the script. This will generate:
- `mu_true_..._CGmu2.xdmf` + `.h5` (standard XDMF, vertex values only)
- `mu_true_..._CGmu2_dofs.h5` (NEW: all DOF coordinates + values)

### Step 2: Visualize in notebook

In `plt_mu_slices_CG2.ipynb`, set:
```python
CG_mu_deg = 2  # For CG2 data
# or
CG_mu_deg = 1  # For CG1 data
```

The notebook will automatically:
1. **CG1**: Load from XDMF as UnstructuredGrid, use `.slice()` method
2. **CG2**: Load from `_dofs.h5` as PolyData point cloud with ALL DOF points
   - Falls back to XDMF if `_dofs.h5` not found (warning issued)
3. Use appropriate slice extraction (`.slice()` vs tolerance-based)
4. Apply same downstream processing (griddata → plot)

---

## Notebook Structure (`plt_mu_slices_CG2.ipynb`)

The notebook is organized into 8 sections:

### 1. Configuration
- Paths to data directories
- Model identifiers (`het3d_str`, `homo_str`)
- CG degree setting (`CG_mu_deg`)
- **Coordinate transformation parameters** (matching original notebook):
  ```python
  lon0, lat0 = -84, 7      # Reference point
  rot = 45                  # Rotation angle (degrees, CCW)
  x0, y0 = 130e3, 350e3    # Offset (meters)
  region_fault = [-86.5, -84.5, 9, 11]  # Geographic region
  ```

### 2. Load μ from XDMF Files
- `load_xdmf_as_pyvista(filepath, cg_degree)`: Unified loader for CG1/CG2
- Loads both 3D heterogeneous and homogeneous models

### 2.5 Load Plate Interface and Trench
- Loads `Kyriakopoulos2016JGR/Nicoya_interface.e` for plate geometry
- Loads `Kyriakopoulos2016JGR/trench_xyz.txt` for trench line
- Converts to geographic coordinates using `utp.ckm2LLd()`
- Creates `plate_grd` for PyGMT contour plotting

### 3. Slice Extraction Functions
- `extract_horizontal_slice(grid, z_depth, tolerance)`: Unified for CG1/CG2
- `extract_vertical_slice_dip(grid, y_const, tolerance)`: Along dip (constant y)
- `extract_vertical_slice_strike(grid, x_const, tolerance)`: Along strike (constant x)
- `interpolate_to_grid(x, y, values)`: Scattered points → regular grid

### 4-6. Matplotlib Visualization
- Quick visualization using matplotlib
- Horizontal slices at multiple depths
- Vertical slices along dip and strike
- Both absolute μ and anomaly plots

### 7. PyGMT Publication-Quality Plots
**Key functions matching original notebook style:**

```python
extract_horizontal_slice_geo(grid, z_depth, tolerance)
# Extracts slice and converts to geographic coordinates (lon/lat)

plot_horizontal_slices_pygmt(grid, depth_levels_km, mu_ref,
                              tolerance, plot_type, nrows, ncols, filename)
# Multi-panel figure with:
# - Geographic coordinates
# - Plate interface depth contours
# - Trench line with teeth symbols
# - Panel labels (a), (b), (c), (d)

plot_vertical_slices_dip_pygmt(grid, y_slices_km, mu_ref,
                                tolerance, plot_type, nrows, ncols, filename)
# Vertical cross-sections with plate interface overlay
```

**Colormaps (matching original):**
- Absolute μ: `gist_rainbow_r`
- Anomaly: `cmc.roma` (from cmcrameri)

### 8. CG1 vs CG2 Comparison
- Loads both versions if available
- Compares point density at same depth
- Visual scatter plot comparison

---

## Key Functions Reference

| Function | Purpose |
|----------|---------|
| `load_xdmf_as_pyvista(filepath, cg_degree)` | CG1: Load XDMF as UnstructuredGrid; CG2: Load `_dofs.h5` as PolyData |
| `save_function_with_dofs(function, filename_base, attr_name, cg_degree)` | Save FEniCS function (in `test_higher_CGmu...py`) |
| `extract_horizontal_slice(grid, z_depth, tolerance)` | Extract horizontal slice (mesh coords) |
| `extract_horizontal_slice_geo(grid, z_depth, tolerance)` | Extract horizontal slice (geographic coords) |
| `extract_vertical_slice_dip(grid, y_const, tolerance)` | Vertical slice at constant y |
| `extract_vertical_slice_strike(grid, x_const, tolerance)` | Vertical slice at constant x |
| `interpolate_to_grid(x, y, values)` | Scattered → regular grid via griddata |
| `plot_horizontal_slices_pygmt(...)` | Multi-panel PyGMT horizontal slices |
| `plot_vertical_slices_dip_pygmt(...)` | Multi-panel PyGMT vertical slices |

---

## Coordinate Transformation

The notebook uses the same coordinate system as the original:

```python
# Mesh local coords → Geographic coords
lon, lat = utp.ckm2LLd(x + x0, y + y0, lon0, lat0, -rot)

# Geographic coords → Mesh local coords
x_rot, y_rot = utp.LL2ckmd(lon, lat, lon0, lat0, rot)
x = x_rot - x0
y = y_rot - y0
```

---

## Tolerance Values

For CG2 point cloud slicing:
- **Horizontal slices**: `tolerance = 1500 m` (1.5 km half-thickness)
- **Vertical slices**: `tolerance = 2000 m` (2 km half-thickness)

These values balance between capturing enough points and maintaining slice sharpness.
