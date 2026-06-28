# Nicoya Mesh Cross-Section Workflow

This note documents a Python workflow for plotting a vertical cross-section of the Gmsh mesh [`nicoyaCKden_une_all.msh`](/home/staff/chao/SSEinv/Nicoya/mesh/nicoyaCKden_une_all.msh) at constant `y = 30 km`.

## Starting point

- Mesh file: [`nicoyaCKden_une_all.msh`](/home/staff/chao/SSEinv/Nicoya/mesh/nicoyaCKden_une_all.msh)
- Mesh format: Gmsh ASCII `2.2`
- Mesh contents observed from the file:
  - 32,021 nodes
  - 34,157 triangular surface elements (`gmsh` type `2`)
  - 163,209 tetrahedral volume elements (`gmsh` type `4`)
- Coordinate scale appears to be meters, so `y = 30 km` is implemented as `y = 30000.0`

## Workflow

1. Read the `.msh` file and extract node coordinates plus tetrahedral connectivity.
2. Define the slicing plane `y = y0`.
3. For each tetrahedron, intersect its six edges with the plane.
4. Deduplicate intersection points produced by numerical roundoff.
5. Project the slice onto the `x-z` plane.
6. Plot the resulting line segments with `matplotlib`.

This is the right workflow for a panel like the lower-left inset in your screenshot, because that panel is a geometric cut through the tetrahedral mesh rather than a scalar field plot.

## Prototype script

The prototype script is:

- [`plot_constant_y_slice.ipynb`](/home/staff/chao/SSEinv/Nicoya/code/plot_constant_y_slice.ipynb)

It uses `meshio` when available and falls back to a small built-in Gmsh `2.2` parser if needed.

## Run

Use the `fenics` conda environment so the installed mesh packages are available, then open the notebook in Jupyter:

```bash
source ~/.bashrc
conda activate fenics
jupyter notebook /home/staff/chao/SSEinv/Nicoya/code/plot_constant_y_slice.ipynb
```

The notebook is organized into separate cells for:

- imports and setup
- mesh loading
- tetrahedron-plane intersection helpers
- slice generation
- plotting
- saving the figure

## Notes on geometry

- Because the slice is `y = constant`, no general plane basis is needed; the 2-D cross-section is simply the `x-z` view of the intersection.
- Most tetrahedra intersect the plane as either:
  - one line segment, or
  - a small polygon with 3 or 4 vertices whose perimeter is drawn
- A small tolerance is used so nodes that lie numerically near the plane are handled consistently.

## Optional PyVista route

`pyvista` is also available in the `fenics` environment. It is useful for quick inspection and could be used later for interactive slicing, but for this prototype I kept the slicing logic explicit so you can extend it more easily to arbitrary A-B profiles.

A typical PyVista workflow would be:

1. Build an unstructured grid from the tetrahedra.
2. Slice with plane normal `(0, 1, 0)` and origin `(0, y0, 0)`.
3. Export or plot the slice edges.

## Natural next steps

- Restrict the `x` and `z` plotting window to the fault region so the refined mesh is larger in the frame.
- Color the slice by element size, physical region, or depth.
- Overlay the fault geometry or topography intersection.
- Generalize from `y = constant` to an arbitrary vertical profile defined by two surface points A and B.
