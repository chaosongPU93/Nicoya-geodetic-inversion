# Fault Surface Visualization: μ and μ Contrast

**Date:** 2025-12-03

**Key Insight:** The fault surface is WHERE THE SLIP OCCURS - visualizing μ structure directly on/across this surface is crucial for understanding forward sensitivity and inverse trade-offs.

---

## Why Fault-Surface Plots Are Critical

### 1. Physical Relevance
- **Slip occurs on the fault** - this is the source location
- Green's function sensitivity is highest near the source
- μ at/near fault directly controls:
  - Stress required to accommodate slip
  - Displacement propagation pattern
  - Asymmetry in surface observations

### 2. Natural Coordinate System
For subduction zones (like Nicoya):
- Fault surface is curved (dipping interface)
- Natural coordinates:
  - **Along-strike distance** (parallel to trench)
  - **Downdip distance** (from shallow to deep)
- Same coordinate system used for:
  - Slip distribution
  - Coupling maps
  - Aftershock distributions

### 3. Direct Connection to Inverse Problem
From your Question 2:
- μ contrast across fault → largest slip bias (10-30%)
- Plotting this contrast directly shows WHERE bias will be worst
- Quantifies the structure trade-off spatially

### 4. Easier to Interpret Than Volume Slices
- 2D surface vs. multiple 3D slices
- All relevant information in one view
- Can directly compare to slip distribution

---

## Two Complementary Visualizations

### Option 1: μ ON the Fault Surface

**What to plot:**
```
μ_fault(s, d) where:
  s = along-strike coordinate
  d = downdip coordinate
```

**How to extract:**
- Evaluate μ at fault element centroids
- Or average μ from elements adjacent to fault
- Use fault mesh coordinates

**What it shows:**
- Absolute μ values where slip occurs
- Along-strike variations in fault zone properties
- Depth variations (downdip changes)
- Which parts of fault are stiff vs. soft

**Interpretation:**
- Low μ regions: easier to deform, amplify displacement
- High μ regions: resist deformation, reduce displacement
- For SAME slip, different μ → different surface displacement

### Option 2: μ Contrast ACROSS the Fault (RECOMMENDED!)

**What to plot:**
```
Δμ_fault(s, d) = μ_HW(s,d) - μ_FW(s,d)

Or as ratio:
R_μ(s,d) = μ_HW(s,d) / μ_FW(s,d)
```

Where:
- HW = hanging wall (typically overriding plate in subduction)
- FW = footwall (typically subducting plate)

**What it shows:**
- **Asymmetry driver** - this is what causes displacement asymmetry!
- Where structure is most different across fault
- Direct predictor of slip bias in inverse problem

**Why this is powerful:**
From previous analysis:
- Positive Δμ (HW stiffer): displacement bias toward FW side
- Negative Δμ (HW softer): displacement bias toward HW side
- Magnitude of Δμ → magnitude of displacement asymmetry

---

## How to Compute

### Computing μ on Fault Surface

**Approach 1: Direct evaluation at fault nodes/elements**
```python
# Assuming fault is identified by boundary marker
fault_dofs = fault_boundary.dofs()  # or similar
μ_fault = μ_function.vector()[fault_dofs]
coords_fault = Vh.tabulate_dof_coordinates()[fault_dofs]
```

**Approach 2: Interpolation onto fault mesh**
```python
# Create 2D fault mesh
fault_mesh = extract_fault_surface(mesh, fault_marker=7)

# Interpolate 3D μ function onto 2D fault mesh
μ_on_fault = interpolate_to_surface(μ_3D, fault_mesh)
```

### Computing μ Contrast Across Fault

**Challenge:** Fault is an internal boundary (zero thickness)

**Solution:** Sample μ at small offset on each side

```python
def compute_mu_contrast(mesh, μ_function, fault_mesh, offset=100.0):
    """
    Compute μ contrast across fault

    Parameters:
    -----------
    mesh : dolfin.Mesh
        Full 3D mesh
    μ_function : dolfin.Function
        Shear modulus field
    fault_mesh : 2D mesh or coordinate array
        Fault surface coordinates
    offset : float
        Distance from fault to sample (meters)
        Should be ~1-2 element sizes

    Returns:
    --------
    μ_HW, μ_FW, Δμ : arrays
        Shear modulus on hanging wall, footwall, and difference
    """

    μ_HW = []
    μ_FW = []

    for point in fault_mesh.coordinates():
        # Get fault normal vector at this point
        n = compute_normal(point)  # pointing from HW to FW

        # Sample points on each side
        point_HW = point - offset * n  # hanging wall side
        point_FW = point + offset * n  # footwall side

        # Evaluate μ at these points
        μ_HW.append(μ_function(point_HW))
        μ_FW.append(μ_function(point_FW))

    μ_HW = np.array(μ_HW)
    μ_FW = np.array(μ_FW)
    Δμ = μ_HW - μ_FW

    return μ_HW, μ_FW, Δμ
```

**Choosing offset distance:**
- Too small: numerical noise, mesh-dependent
- Too large: samples far from fault (less relevant)
- Recommended: 2-3 element sizes, or 100-500 m
- Could also average over a small volume on each side

### Alternative: Element-Based Approach

```python
# For each fault face:
for fault_element in fault_elements:
    # Get adjacent volume elements
    elem_HW = adjacent_element(fault_element, side='HW')
    elem_FW = adjacent_element(fault_element, side='FW')

    # Get μ in each element
    μ_HW = μ_function at elem_HW centroid
    μ_FW = μ_function at elem_FW centroid

    # Compute contrast
    Δμ[fault_element] = μ_HW - μ_FW
```

This uses actual mesh topology - may be more robust.

---

## Visualization Strategies

### Layout 1: Side-by-Side Comparison

```
Figure with 2 or 3 panels:

Panel 1: μ_HW (hanging wall)
Panel 2: μ_FW (footwall)
Panel 3: Δμ = μ_HW - μ_FW (contrast)

All plotted on fault surface with same coordinates
```

**Coordinates:**
- x-axis: Along-strike distance (km)
- y-axis: Depth or downdip distance (km)
- Color: μ value or Δμ

### Layout 2: Combined with Slip Distribution

```
Figure with 2 rows × 2 columns:

Row 1 (Structure):
  [μ_HW]  [μ_FW]

Row 2 (Slip):
  [slip_strike]  [slip_dip]

All on same fault surface coordinates
```

Shows direct spatial relationship between μ and slip.

### Layout 3: Contrast with Displacement Response

```
Two panels:

Panel 1: Δμ on fault surface
Panel 2: Δu at surface (difference between 3D and homogeneous)

Linked by:
- Vertical lines showing projection from fault to surface
- Or contour overlay
```

Visually connects cause (μ contrast) to effect (displacement asymmetry).

### Layout 4: Multiple Fault-Surface Quantities

```
Four panels on fault surface:

[μ_fault]  [Δμ across fault]
[slip]     [Δu_surface projected onto fault]

All sharing same coordinate system
```

Comprehensive view of structure-slip-displacement relationship.

---

## Specific Recommendations for Your Analysis

### For Forward Problem Sensitivity

**Essential plots:**

1. **μ on fault surface** (from 3D model)
   - Shows absolute heterogeneity at slip source
   - Compare to reference 1D model

2. **Δμ across fault**
   - Shows asymmetry driver
   - Expect correlation with surface Δu asymmetry

3. **Combined visualization:**
   ```
   Three panels:
   (a) Δμ on fault surface
   (b) Surface Δu magnitude (map view)
   (c) Vertical cross-section showing both
   ```

**Quantitative analysis:**

Plot correlation:
```python
x-axis: Δμ at each fault patch
y-axis: Asymmetry in surface displacement above that patch

Asymmetry metric:
  A = (u_left - u_right) / (u_left + u_right)
  measured perpendicular to fault
```

Expected: Strong correlation - where Δμ is large, asymmetry is large.

### For Inverse Problem (Your Question 2)

**Diagnostic plot:**

```
Fault surface with three quantities:

(a) Δμ = μ_HW - μ_FW (true structure)
(b) Δm_hom = m_hom - m_3D (slip bias from homogeneous inversion)
(c) Correlation plot: Δμ vs. Δm

Expected: Anti-correlation
  - Where μ_HW < μ_FW (negative Δμ): slip overestimated
  - Where μ_HW > μ_FW (positive Δμ): slip underestimated
```

This directly visualizes the structure-slip trade-off!

---

## Technical Implementation Details

### Fault-Centered Coordinate System

For subduction zones, define natural coordinates:

```python
def fault_coordinates(fault_points, reference_point, strike_direction):
    """
    Convert (x,y,z) to (along-strike, downdip, normal)

    Parameters:
    -----------
    fault_points : array (N, 3)
        3D coordinates of fault surface points
    reference_point : array (3,)
        Origin for fault coordinates (e.g., trench at surface)
    strike_direction : array (3,)
        Unit vector along strike (e.g., parallel to trench)

    Returns:
    --------
    s : array (N,)
        Along-strike distance (km)
    d : array (N,)
        Downdip distance (km)
    """

    # Compute dip direction (perpendicular to strike, in fault plane)
    # For subduction: usually toward land
    dip_direction = compute_dip_direction(fault_mesh)

    # Project each point onto strike and dip directions
    relative_pos = fault_points - reference_point
    s = np.dot(relative_pos, strike_direction)
    d = np.dot(relative_pos, dip_direction)

    return s, d
```

### Handling Curved Fault Surfaces

Subduction interfaces are curved, not planar.

**Option 1: Flatten to 2D**
```python
# Use (s, d) coordinates
# Plot as regular 2D image with along-strike and downdip axes
# This is standard in earthquake literature
```

**Option 2: 3D surface plot**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot fault as 3D surface colored by μ
ax.plot_surface(X_fault, Y_fault, Z_fault,
                facecolors=cm.viridis(μ_fault))
```

**Option 3: Use pyvista**
```python
import pyvista as pv

# Create fault surface mesh
fault_surf = pv.PolyData(fault_points, fault_faces)
fault_surf['mu'] = μ_fault

# Plot
plotter = pv.Plotter()
plotter.add_mesh(fault_surf, scalars='mu',
                 cmap='viridis', show_edges=True)
plotter.show()
```

For publication, I recommend **Option 1** (flattened 2D in fault coordinates) - it's clearest and standard.

---

## What These Plots Reveal

### From μ on Fault Surface

**Pattern 1: Depth variations**
```
Shallow (< 10 km): Low μ (sediments, fractured rock)
Mid-depth (10-30 km): Moderate μ (crystalline crust)
Deep (> 30 km): High μ (stronger materials, higher P)
```

→ Same slip at different depths produces different surface displacement

**Pattern 2: Along-strike variations**
```
Some segments: μ varies along strike
Others: relatively uniform
```

→ Indicates 2D vs. 3D problem

**Pattern 3: Absolute values**
```
Typical ranges: 20-40 GPa for crustal depths
Anomalies: ±20-50% from average
```

→ Context for how heterogeneous the structure is

### From Δμ Across Fault

**Pattern 1: Asymmetry zones**
```
Regions with |Δμ| > 5 GPa: Strong asymmetry
  → Expect strong displacement bias
  → Homogeneous inversion will have large errors here

Regions with |Δμ| < 2 GPa: Weak asymmetry
  → Homogeneous inversion may be adequate
```

**Pattern 2: Sign of contrast**
```
Δμ > 0 (HW stiffer than FW):
  → HW deforms less
  → Surface displacement bias toward FW side
  → Homogeneous inversion overestimates slip on FW side

Δμ < 0 (HW softer than FW):
  → HW deforms more
  → Surface displacement bias toward HW side
  → Homogeneous inversion overestimates slip on HW side
```

**Pattern 3: Spatial correlation**
```
If Δμ is smoothly varying:
  → Displacement pattern also smooth
  → Slip bias pattern predictable

If Δμ is patchy:
  → Complex displacement pattern
  → Slip bias harder to correct
```

---

## Integration with Other Visualizations

### How Fault-Surface Plots Complement Volume Slices

**Volume slices show:**
- μ structure throughout 3D domain
- Off-fault heterogeneity
- Regional context

**Fault-surface plots show:**
- μ structure exactly where slip occurs
- Direct asymmetry driver
- Natural slip-distribution coordinate system

**Use both:**
1. Volume slices for overall structure understanding
2. Fault-surface plots for source-specific analysis

### Recommended Figure Sequence

**Figure 1: Overview**
- Multi-depth horizontal slices of μ
- Shows regional structure

**Figure 2: Fault-specific** (NEW!)
- Panel (a): μ on fault surface
- Panel (b): Δμ across fault
- Panel (c): Slip distribution (for reference)

**Figure 3: Response**
- Surface displacement difference Δu
- Vertical cross-section through fault

**Figure 4: Correlation** (POWERFUL!)
- Scatter: Δμ vs. surface displacement asymmetry
- Shows quantitative relationship

---

## Example Analysis Workflow

### Step 1: Extract Data
```python
# Run forward models
u_3D, μ_3D = run_forward(slip, μ_3D_model)
u_hom, μ_hom = run_forward(slip, μ_hom_model)

# Compute differences
Δu = u_3D - u_hom
```

### Step 2: Compute Fault Quantities
```python
# Extract fault surface coordinates
fault_coords = extract_fault_coordinates(mesh, fault_id=7)

# Compute μ on fault
μ_fault = evaluate_on_surface(μ_3D, fault_coords)

# Compute μ contrast
μ_HW, μ_FW = sample_across_fault(μ_3D, fault_coords,
                                  offset=200)  # 200m offset
Δμ_fault = μ_HW - μ_FW
```

### Step 3: Compute Surface Response
```python
# For each fault patch, compute surface displacement asymmetry
for patch in fault_patches:
    # Find surface region above this patch
    surface_region = project_to_surface(patch)

    # Compute left-right asymmetry
    u_left = np.mean(Δu[surface_region_left])
    u_right = np.mean(Δu[surface_region_right])
    asymmetry[patch] = (u_left - u_right) / (u_left + u_right)
```

### Step 4: Visualize
```python
# Plot Δμ on fault surface
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: μ contrast
plot_fault_surface(axes[0], s, d, Δμ_fault,
                   title='μ contrast (HW - FW)')

# Panel 2: Surface displacement asymmetry
plot_fault_surface(axes[1], s, d, asymmetry,
                   title='Surface displacement asymmetry')

# Panel 3: Correlation
axes[2].scatter(Δμ_fault, asymmetry, c=depth, cmap='viridis')
axes[2].set_xlabel('Δμ (GPa)')
axes[2].set_ylabel('Displacement asymmetry')
```

### Step 5: Quantify
```python
# Compute correlation coefficient
from scipy.stats import pearsonr
r, p = pearsonr(Δμ_fault.flatten(), asymmetry.flatten())
print(f"Correlation: r = {r:.3f}, p = {p:.3e}")

# Sensitivity metric
sensitivity = np.std(asymmetry) / np.std(Δμ_fault)
print(f"Sensitivity: {sensitivity:.3f} (asymmetry per GPa)")
```

---

## Specific Recommendations for Nicoya

### Your Subduction Zone Geometry

**Fault orientation:**
- Strike: ~NW-SE (parallel to Middle America Trench)
- Dip: ~15-20° toward NE (subducting beneath Costa Rica)

**Key structural contrasts:**
- Oceanic crust (Cocos plate, FW): typically μ ~ 30-40 GPa
- Overriding plate (HW): more complex
  - Forearc wedge: low μ ~ 15-25 GPa (sediments, fractured)
  - Deeper: higher μ ~ 35-45 GPa (crystalline)

**Expected Δμ pattern:**
```
Shallow (< 10 km): Δμ < 0 (HW softer due to wedge)
  → Expect displacement bias toward HW (land side)
  → Large amplification in forearc

Deep (> 20 km): Δμ ≈ 0 or slightly positive
  → Less asymmetry
  → More symmetric displacement
```

### What to Highlight

1. **Forearc wedge effect**
   - Low μ in overriding plate creates strong Δμ < 0
   - Should see enhanced displacement on land side
   - Critical for hazard assessment

2. **Along-strike variations**
   - Does Δμ vary along strike?
   - If yes: 3D modeling essential
   - If no: 2D approximation may suffice

3. **Depth variations**
   - How does Δμ change downdip?
   - Correlates with seismogenic zone vs. aseismic regions?

---

## Summary

### YES - Both Are Very Helpful!

**Plot μ on fault surface:**
- ✅ Shows absolute values where slip occurs
- ✅ Natural slip-distribution coordinate system
- ✅ Easy to compare with slip patterns
- Useful for: understanding source-region heterogeneity

**Plot Δμ across fault:**
- ✅✅✅ **MOST DIRECTLY RELEVANT**
- ✅ Shows asymmetry driver
- ✅ Predicts displacement bias
- ✅ Quantifies structure trade-off
- Essential for: understanding forward sensitivity AND inverse trade-offs

### Recommended Minimal Addition to Your Plots

Add one figure with fault-surface plots:
```
2 panels:
(a) μ on fault surface (from 3D model)
(b) Δμ contrast across fault

Both in fault coordinates: along-strike × downdip
Overlay slip distribution contours
```

This single figure will be **extremely insightful** and complement your volume slices perfectly.

### Implementation Priority

1. **High priority:** Δμ across fault - most diagnostic
2. **Medium priority:** μ on fault - good context
3. **Nice to have:** Combined plots with slip and surface response

The contrast plot (Δμ) is what I'd start with - it directly shows the driver of displacement asymmetry and slip bias!
