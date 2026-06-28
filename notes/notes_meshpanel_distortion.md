# Panel A Mesh Cross-Section "Distortion" — Investigation & Resolution

**File:** `codes/plt_mu_slices_CG2.ipynb` (3-panel figure, Panel A = mesh cross-section)
**Figure caption context:** "(a) Cross-sectional view of the fault and domain discretization
along the profile indicated by the white dashed line in (c)."

## The issue

Supervisor repeatedly unhappy with Panel A: the mesh elements look "distorted."

**Root cause (not a bug):** Panel A is a planar cut through the unstructured 3D tetrahedral
mesh (`_tet_poly` in cell 41 = marching-tetrahedra, mathematically exact). A plane slicing a
tet yields a triangle or quadrilateral — **neither is an actual finite element**, just the
intersection footprint. Tets meet the plane at arbitrary/grazing angles, so cut polygons look
skewed and slivered. This is the true cross-section geometry. The plotting code is correct.

**Fundamental limit:** A cross-section of an unstructured 3D tet mesh is not itself a 2D mesh.
No rendering trick makes the true cut look like a clean 2D triangular mesh, because it isn't one.

## Options considered

- **Option A — 3D cut-away perspective (PyVista/ParaView).** The only approach that shows
  genuine elements cleanly. **REJECTED by supervisor** (he saw the ParaView version).
- **Option B — cosmetic treatments of the flat cross-section.** Implemented two (below).
- **Option D — re-mesh with structured along-strike layer.** Too costly; changes sim mesh. Not done.

## What was implemented (both are honest, purely-visual)

### Approach 1 — sliver filter + softened edges (cell 42)
- Two-pass render: **fill pass** (all polygons colored → continuous region) + **edge pass**
  (outline only polygons with isoperimetric quotient `q = 4π·Area/Perimeter² ≥ _sliver_q`).
- Helpers added in cell 41: `_poly_iso_q`, `_write_gmt_poly_edges`.
- Edges recolored black → soft gray (`_elem_pen = '0.12p,gray35'`).
- Knob: `_sliver_q` (cell 42). `_sliver_q = 0` ≡ no filtering (all edges drawn).
- **Effect:** modest. Removes thinnest slivers; softens harshness. Tested 0.10 and 0.25 —
  difference subtle, because most "distortion" is skewed-but-not-thin triangles (q ≈ 0.3–0.4)
  that survive any reasonable threshold.

### Approach 2 — quad-triangulation (cell 43, separate figure `fig3p_tri`)
- Splits every cut-quad into triangles (`_triangulate_poly`, fan from vertex 0) → uniform
  all-triangle appearance. Justified: cut-quads aren't real elements, so splitting is no less
  faithful than the original tri/quad mix.
- Knob: `_sliver_q_tri` (default 0 = all triangle edges).
- **Effect:** tidier/uniform, but does NOT straighten skewed/obtuse triangles (intrinsic to
  any planar cut).

## Decision / status

Neither cosmetic approach fully removes the skewed-element appearance — that character is
inherent to sectioning an unstructured tet mesh. The one method that would (3D cut-away) was
rejected. **Fallback = caption acknowledgment**, which is standard and defensible.

### Caption sentence (tightened for precision)

Original draft:
> "The apparent distortion of mesh elements is a visualization artifact arising from oblique
> sectioning of the 3D mesh and does not reflect the underlying mesh quality or geometry."

Imprecision: "distortion of mesh elements" implies the elements are distorted; really the
polygons shown are *cross-sections* of tetrahedra, not the elements. A well-shaped tet produces
a skewed cut when grazed. Tightened version:

> "The apparent distortion is a sectioning effect: the polygons shown are oblique cross-sections
> of tetrahedra rather than the elements themselves, and do not reflect the quality or geometry
> of the underlying 3D mesh."

**Strengthener (optional):** the "does not reflect mesh quality" claim is most bulletproof if
backed by actual 3D element quality metrics (aspect ratio / radius-edge ratio from gmsh). Add a
brief parenthetical if those numbers are available.
