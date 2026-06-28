# Visualization Strategy: μ Sensitivity in Forward Problem

**Goal:** Understand how 3D μ heterogeneity affects the displacement field

**Date:** 2025-12-03

---

## Overview

To visualize sensitivity of displacement to μ structure, you need to show:
1. Where μ varies (the input heterogeneity)
2. Where displacement changes (the response)
3. The relationship between the two

---

## Recommended Visualizations

### 1. μ Structure Visualization

#### A. μ Anomaly Maps
**What to plot:**
```python
μ_anomaly = (μ_3D - μ_ref) / μ_ref  # Fractional deviation
```

**Where:**
- Horizontal slices at key depths:
  - z = 0 km (surface)
  - z = -5 km (shallow crust)
  - z = -15 km (mid-crustal, typical seismogenic depth)
  - z = -30 km (deep)
- Vertical cross-sections:
  - Perpendicular to fault strike (show fault-normal contrast)
  - Parallel to fault strike (show along-strike variations)
  - Through regions of interest (e.g., where stations are)

**Reference μ:**
- Option 1: Homogeneous μ_hom (constant value)
- Option 2: Depth-averaged μ at each depth (removes 1D trend)
- I recommend Option 2 to highlight lateral variations

**Visualization tips:**
- Use diverging colormap (e.g., RdBu: red=high μ, blue=low μ)
- Show fault trace/projection
- Overlay station locations on surface slice
- Show percentage range (e.g., ±20% from reference)

---

### 2. Displacement Field Visualization

#### A. Surface Displacement (Most Important!)

This is what GPS/GNSS actually measures.

**Plot 1: Displacement vectors**
```
Surface (z=0):
- Background: displacement magnitude |u|
- Vectors: horizontal displacement (u_x, u_y)
- Separate plots for u_3D and u_hom
```

**Plot 2: Displacement difference**
```
Δu_surface = u_3D - u_hom  (at z=0)
- Magnitude: |Δu|
- Vectors: (Δu_x, Δu_y)
- This shows the EFFECT of heterogeneity
```

**Plot 3: Component-wise differences**
```
Three panels:
- Δu_x (East component)
- Δu_y (North component)
- Δu_z (Vertical component)
```

#### B. Subsurface Displacement

**Horizontal slices (same depths as μ slices):**
```
For each depth level:
- u_3D displacement magnitude
- u_hom displacement magnitude
- Δu = u_3D - u_hom
```

Shows how displacement field differs at depth, not just surface.

**Vertical cross-sections:**
```
Fault-perpendicular slice:
- Side-by-side: u_3D | u_hom | Δu
- Shows asymmetry across fault
```

```
Fault-parallel slice:
- Shows along-strike variations
- Reveals 3D effects
```

---

### 3. Direct μ-Displacement Sensitivity Visualization

These are the KEY plots for understanding sensitivity!

#### A. Side-by-side Comparison (Recommended!)

**Layout:**
```
Figure with 2 columns × N rows:

Row 1 (Surface):
  [μ_anomaly at z=0]  |  [Δu at z=0]

Row 2 (Shallow):
  [μ_anomaly at z=-5km]  |  [Δu at z=-5km]

Row 3 (Mid-depth):
  [μ_anomaly at z=-15km]  |  [Δu at z=-15km]

Row 4 (Deep):
  [μ_anomaly at z=-30km]  |  [Δu at z=-30km]
```

**Why this works:**
- Direct visual correlation between μ anomalies and displacement response
- Can see spatial offset between μ anomaly location and displacement effect
- Shows depth-dependent sensitivity

#### B. Overlay Visualization

```
For each slice:
- Background: μ_anomaly (color)
- Contours: |Δu| (displacement difference magnitude)
- Vectors: Δu direction
```

Shows directly where μ anomalies cause displacement changes.

#### C. Correlation Plots

**Scatter plot:**
```
x-axis: μ_anomaly at each point in domain
y-axis: |Δu| at same point
Color by: depth

Expected pattern:
- Points should scatter around a trend
- Larger |μ_anomaly| → larger |Δu| (generally)
- But with spatial lag/smearing due to elasticity
```

**Stratified by region:**
```
Separate scatter plots for:
- Near-fault points (< 10 km from fault)
- Surface points (z=0)
- Deep points (z < -20 km)
```

This reveals where sensitivity is strongest.

---

### 4. Vertical Cross-Sections (Critical for 3D Understanding)

#### A. Fault-Perpendicular Slice

**Setup:**
```
Choose Y coordinate perpendicular to fault strike
Create X-Z slice through the domain
```

**Three-panel figure:**
```
Panel 1: μ_3D(x,z) with fault trace
Panel 2: u_3D(x,z) - displacement magnitude
Panel 3: Δu(x,z) = u_3D - u_hom
```

**Overlay:**
- Fault plane location
- Surface (z=0) line
- Station positions if along this profile

**What to look for:**
- Asymmetry across fault in μ
- Corresponding asymmetry in displacement
- How deep anomalies affect surface displacement

#### B. Fault-Parallel Slice

Shows along-strike variations:
```
Same three-panel layout
But for Y-Z slice (along strike)
```

Reveals whether problem is 2D or truly 3D.

---

### 5. Quantitative Sensitivity Metrics

#### A. Sensitivity as Function of Distance

**Plot:**
```
x-axis: Distance from fault (km)
y-axis: |Δu| / |μ_anomaly| (normalized sensitivity)
Color by: depth
```

Shows how sensitivity decays with distance from fault.

#### B. Depth Profile

**Plot:**
```
x-axis: Depth (km)
y-axis: RMS(Δu) at that depth
Optional: separate curves for different lateral positions
```

Shows which depths contribute most to displacement differences.

#### C. Surface Displacement vs. Depth of μ Anomaly

**Experiment:**
```
Create synthetic cases:
- Case 1: μ anomaly only at 0-5 km depth
- Case 2: μ anomaly only at 5-15 km depth
- Case 3: μ anomaly only at 15-30 km depth
- etc.

Compare surface Δu for each case
```

**Plot:**
```
Bar chart:
- x-axis: Depth range of μ anomaly
- y-axis: RMS(Δu_surface)
```

Quantifies depth sensitivity directly.

---

### 6. Strain Field Comparison (Advanced)

Strain is MORE sensitive to μ than displacement (it's the gradient).

**Plot:**
```
ε_anomaly = ε_3D - ε_hom

Where: ε = sym(∇u)
```

**Useful components:**
- Maximum shear strain
- Volumetric strain
- Second invariant

**Why useful:**
- Strain localizes more than displacement
- Shows where material is "working harder" due to heterogeneity
- May reveal features invisible in displacement

---

## Recommended Plot Sequence

### Minimal Set (Quick Analysis)

1. **Surface displacement difference** (vectors + magnitude)
2. **Vertical cross-section** through fault (3-panel: μ, u_3D, Δu)
3. **Horizontal slices at 3 depths** (side-by-side: μ_anomaly | Δu)

### Comprehensive Set (Publication Quality)

1. **μ structure overview:**
   - Horizontal slices at 4 depths
   - 2 vertical cross-sections (parallel & perpendicular to fault)

2. **Displacement field comparison:**
   - Surface: u_3D, u_hom, Δu (three panels)
   - Same for 2-3 subsurface depths

3. **Direct sensitivity visualization:**
   - Side-by-side multi-depth comparison (μ_anomaly | Δu)
   - Overlay plot (μ as color, Δu as contours)

4. **Vertical cross-sections:**
   - Fault-perpendicular: μ, u, Δu
   - Fault-parallel: μ, u, Δu

5. **Quantitative analysis:**
   - Scatter: μ_anomaly vs. |Δu|
   - Depth profile: RMS(Δu) vs. depth
   - Distance decay: sensitivity vs. distance from fault

6. **Station-specific:**
   - For each GPS station, show vertical profile of μ beneath it
   - Compare to Δu at that station
   - Reveals "what this station is sensitive to"

---

## Specific Recommendations

### For Your Case (Nicoya Subduction)

**Critical slices:**
```
Depths:
- z = 0 km (surface, where GPS is)
- z = -5 km (shallow seismogenic zone)
- z = -15 km (peak slip depth for typical events)
- z = -30 km (deep subduction interface)
```

**Critical cross-sections:**
```
Perpendicular to trench:
- Show oceanic plate vs. continental crust contrast
- Forearc wedge heterogeneity
- Slab interface

Parallel to trench:
- Along-strike variations
- 3D forearc structure
```

**Key features to highlight:**
- Fault plane (subduction interface)
- GPS station locations
- Coastline (important for hazard)
- Trench location

### Plotting Tips

**Colormaps:**
- μ_anomaly: Diverging (RdBu, coolwarm) centered at zero
- Displacement magnitude: Sequential (viridis, plasma)
- Displacement difference: Diverging (RdBu) centered at zero

**Scaling:**
- Use consistent color scales across comparable plots
- Show colorbars with physical units
- Indicate scale of differences (e.g., "max Δu = 5 cm")

**Annotations:**
- Always show fault location
- Mark surface, coastline
- Indicate station positions
- Show coordinate system (strike direction)

---

## Expected Patterns to Look For

### 1. Near-Fault Asymmetry
If μ₁ (left) ≠ μ₂ (right):
- Displacement stronger on softer side
- Asymmetric surface displacement pattern
- Clear in fault-perpendicular cross-section

### 2. Depth-Dependent Sensitivity
- μ anomalies at fault depth → large Δu at surface
- μ anomalies far from fault → smaller Δu
- Deep anomalies → broader, smoother surface Δu patterns

### 3. Spatial Lag
- Δu maximum may not align exactly with μ_anomaly maximum
- Due to elastic stress redistribution
- Particularly true for deep anomalies

### 4. Amplification in Soft Regions
- Where μ is low (e.g., forearc wedge)
- Displacement is amplified
- Shows clearly in vertical cross-sections

### 5. 3D vs. 2D Effects
- If along-strike variations in μ are small → approximately 2D
- If large → need 3D model, can't use 2D approximation

---

## Implementation Checklist

### Data to Extract

From your FEniCS simulations:

```python
# For both μ_3D and μ_hom cases:
- [ ] u_3D (displacement field on entire mesh)
- [ ] u_hom (displacement field on entire mesh)
- [ ] μ_3D (shear modulus field on mesh)
- [ ] μ_hom (constant value or 1D profile)

# Derived quantities:
- [ ] Δu = u_3D - u_hom
- [ ] μ_anomaly = (μ_3D - μ_ref) / μ_ref
- [ ] ε_3D, ε_hom (if doing strain analysis)

# Geometric info:
- [ ] Mesh coordinates (x, y, z)
- [ ] Fault element locations
- [ ] Station coordinates
- [ ] Subdomain labels (fault, blocks, etc.)
```

### Interpolation Strategy

Your FEniCS mesh is likely unstructured. To create regular slices:

```python
# Create regular grid for visualization
xi = np.linspace(xmin, xmax, nx)
yi = np.linspace(ymin, ymax, ny)
zi = depth_level  # e.g., -15 km

# Interpolate FEniCS function onto regular grid
from dolfin import interpolate
# or use point evaluation

# For each point in regular grid:
for x, y, z in regular_grid:
    μ_val = μ_function(x, y, z)
    u_val = u_function(x, y, z)
```

### Plotting Tools

**Python libraries:**
```python
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

# For 3D visualization:
import pyvista as pv  # good for 3D mesh visualization

# For publication-quality:
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.dpi'] = 300
```

---

## Validation Questions to Answer

After making these plots, you should be able to answer:

1. **Where is sensitivity highest?**
   - Near fault? Surface? Deep?

2. **Does surface displacement "see" deep heterogeneity?**
   - How does surface Δu relate to deep μ anomalies?

3. **Is the problem 2D or 3D?**
   - Do along-strike variations in μ create along-strike Δu patterns?

4. **What spatial scale matters?**
   - Do small-scale μ variations create small-scale Δu?
   - Or does elastic smoothing filter out fine structure?

5. **Asymmetry magnitude?**
   - How much does fault-zone asymmetry in μ affect surface displacement?
   - Quantify the effect

6. **Which GPS stations are most sensitive to structure?**
   - Near-field stations? Far-field?
   - Over what depth range?

---

## Example Figure Caption

**Figure: Sensitivity of displacement field to 3D shear modulus structure**

*Top row: Shear modulus anomaly (μ₃ᴅ - μᵣₑf)/μᵣₑf at four depth levels (0, -5, -15, -30 km). Bottom row: Corresponding displacement difference Δu = u₃ᴅ - uₕₒₘ showing the response to heterogeneity. Black lines show fault trace. White triangles mark GPS stations. Note the strong correlation between near-fault μ contrasts and surface displacement asymmetry. The displacement response to deep heterogeneity (z = -30 km) is smoother and lower amplitude, demonstrating depth-dependent sensitivity. Same slip distribution used in both models; differences are purely due to μ structure.*

---

## Summary: What to Plot

**Minimum viable analysis:**
1. Surface Δu map with vectors
2. One vertical cross-section (μ, u, Δu)
3. Scatter plot: μ_anomaly vs |Δu|

**Recommended comprehensive analysis:**
1. Multi-depth horizontal slices (μ and Δu side-by-side)
2. Two vertical cross-sections (perpendicular and parallel to fault)
3. Surface displacement detailed comparison
4. Quantitative sensitivity metrics (depth profile, distance decay)
5. Station-specific analysis

**Publication-ready:**
- All of the above
- Plus strain field analysis
- Plus synthetic tests (depth-dependent μ anomalies)
- Plus detailed statistical correlation

Your initial idea was good - slices are definitely the way to go. But I'd emphasize:
- **Side-by-side comparison** (μ next to Δu) over separate figures
- **Surface displacement** gets special attention (what's measured)
- **Vertical cross-sections** are crucial for 3D understanding
- **Quantitative metrics** to complement visual inspection
