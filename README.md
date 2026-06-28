# Nicoya SSE Inversion

Finite-element slip inversion for the Nicoya Peninsula subduction zone (Costa Rica),
incorporating heterogeneous elastic structure (shear modulus μ). The code inverts
geodetic (GNSS) observations for interseismic locking / slip-deficit rate and for
coseismic slip, comparing homogeneous, layered (1D), and 3D μ models.

This repository accompanies a manuscript in preparation.

## Directory structure

| Path | Contents |
|---|---|
| `codes/` | Inversion, forward, and synthetic-test scripts (`.py`) and plotting notebooks (`.ipynb`); `utils.py`, `utils_plot.py` |
| `mesh/` | 3D tetrahedral FE meshes (gmsh `.geo`/`.msh` sources + FEniCS `.xml` + boundary/subdomain markers) for the paper models |
| `data/` | GNSS observation data |
| `plateinterface/` | Plate-interface geometry inputs |
| `Kyriakopoulos2016JGR/` | Plate-interface model and inputs from Kyriakopoulos et al. (2016, JGR) |
| `DeShon_2006GJI/` | 3D velocity model and derived shear-modulus inputs from DeShon et al. (2006, GJI) |

## Dependencies

The forward/inverse modeling uses [FEniCS](https://fenicsproject.org/) (legacy `dolfin`)
and [hIPPYlib](https://hippylib.github.io/). Plotting uses
[PyGMT](https://www.pygmt.org/), `pyvista`, `meshio`, `numpy`, `pandas`, `scipy`.

Meshes are loaded directly from the FEniCS `.xml` format (e.g. `dl.Mesh(meshpath + meshname + '.xml')`).
The gmsh `.geo`/`.msh` sources are also included for regeneration if needed.

## Mesh naming

The paper uses the `nicoyaCKden*` family (CK plate interface, densified):
`nicoyaCKden_sm`, `nicoyaCKden_all`, `nicoyaCKden_une_sm` / `nicoyaCKden_une_all`
(uneven top boundary), and the `nicoyaden*` regular-interface comparison meshes.

## Notebook outputs

Notebook outputs are stripped from version control via a git clean filter
(`.githooks/nbstrip.py`, configured through `.gitattributes`). Working copies retain
their rendered outputs locally. To enable the filter after cloning:

```bash
git config filter.nbstrip.clean "python3 .githooks/nbstrip.py"
git config filter.nbstrip.smudge cat
```
