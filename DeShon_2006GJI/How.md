# How DeShon et al. (2006) parameterized their Nicoya velocity model

The 1-D velocity model in Table 1 of DeShon et al. (2006) defines **velocity at discrete depth nodes with trilinear interpolation between them** — not constant-velocity layers. The study uses **SIMULPS13Q**, which parameterizes the 3-D velocity field on a grid of nodes where velocity varies continuously via linear B-spline interpolation. Below is a detailed breakdown addressing each of your five questions, drawn directly from the paper's methodology (Section 3) and the documented behavior of the SIMULPS code family.

## The code is SIMULPS13Q, not simul2000

DeShon et al. explicitly state: "In this study we use the LET algorithm **SIMULPS13Q**, an iterative damped least squares solution for local earthquake data that utilizes approximate ray tracing for computation of theoretical traveltimes (Thurber 1983; Eberhart-Phillips 1990; Evans et al. 1994)." This is the version of SIMULPS that added Q (attenuation) inversion capability, though the study only inverts for VP and VP/VS. The 1-D starting model was obtained separately using **VELEST** (Kissling et al. 1995), as reported in DeShon & Schwartz (2004).

## Depth values are node points, not layer tops

The depths in Table 1 are **grid node positions** at which velocity is specified, with **linear interpolation between adjacent nodes**. This is intrinsic to how the entire SIMULPS code family works, as documented in the SIMULPS12 manual (Evans et al. 1994) and confirmed in the simul2017/simul2023 documentation on Zenodo: "The Simul codes parameterize velocity on a 3-D grid of nodes with velocity linearly interpolated between nodes."

Concretely, if the model specifies VP = 5.35 km/s at 0 km depth and VP = 6.12 km/s at 5 km depth, the code computes VP = **5.735 km/s at 2.5 km** via linear interpolation — it does not hold a constant 5.35 km/s throughout that interval. The paper's own language confirms this: it consistently discusses "grid node spacing" and "cuboid, or node-bounded parallelepiped" volumes, never "layer thicknesses" or "constant-velocity layers." Only nodes whose surrounding cuboids are sampled by ≥8 rays are included in the velocity inversion; nodes outside station coverage remain fixed at their starting values.

## The inversion uses velocity at grid nodes with trilinear interpolation

The 3-D velocity model is parameterized as **velocity values at discrete 3-D grid nodes** with **trilinear interpolation** (linear in x, y, and z) between the eight surrounding nodes of any point in the model. This produces a **continuously varying velocity field**, not a blocky, stepped model with sharp discontinuities at cell boundaries. Traveltime derivatives are extracted incrementally along small segments of each ray path through this smoothly varying field.

The horizontal grid is tested at two scales: a **coarse grid** of 20 × 20 km² node spacing (roughly one station per cuboid) and a **fine grid** of 10 × 10 km² beneath the peninsula. The depth node spacing mirrors the 1-D model with minor adjustments to regularize ray coverage. A **progressive inversion scheme** moves from 1-D → coarse 3-D VP → coarse 3-D VP + VP/VS → fine 3-D VP + VP/VS, with each step's output serving as the next step's starting model.

## The −10 km depth node is 10 km above sea level

In SIMULPS, the depth datum is sea level (0 km), and **negative depths denote elevations above sea level**. The simul2017/simul2023 documentation states explicitly: "The model 0-km depth equates to a present sea-level datum, the z = −1 km depth grid is 1 km above sea-level." A node at **−10 km therefore sits 10 km above sea level** — far above any seismic station or topography in the study area. It serves as an **upper boundary constraint node** for the model. Its velocity (VP = 3.00 km/s, VP/VS = 1.78) is held fixed during inversion because no ray paths sample this region. This boundary node ensures the velocity field is properly defined above the surface and prevents artifacts from developing at the top of the model. Similarly, the deepest nodes (e.g., 90 km and 700 km) serve as lower boundary constraints.

## VS comes from a joint VP/VS inversion, not independent S-wave tomography

DeShon et al. do **not** solve for VS independently. They jointly invert for **VP** (using P arrival times) and **VP/VS** (using S−P traveltime differences), then derive VS = VP / (VP/VS). The paper states: "We solve for the compressional wave velocity, VP, and the ratio of the compressional and shear wave velocities, VP/VS, using the compressional wave arrival time, P, and the time difference between the shear and compressional waves, S–P, respectively."

The starting VP/VS is a **uniform 1.78** everywhere, representing the average of the minimum 1-D VP/VS model from VELEST and consistent with other VP/VS studies in Costa Rica. The VS values in Table 1 (e.g., 3.01, 3.44, 3.53 km/s, etc.) are simply VP/1.78 at each node. The rationale for this approach is explicit in the paper: "using S–P traveltimes to solve for VP/VS accounts for the inherent coupling between VP and VS across the same structure and minimizes induced artefacts due to differing resolution if VP and VS were calculated separately." When solving for VP/VS, VP damping is set high (**100**) while VP/VS damping is **50**, so VP is "slightly overdamped" to stabilize the joint inversion.

## Reconstructed Table 1 for reference

The complete initial 1-D model (modified from DeShon & Schwartz 2004) is:

| Depth (km) | VP (km/s) | VS (km/s) | VP/VS |
|-----------|-----------|-----------|-------|
| −10.0 | 3.00 | 1.69 | 1.78 |
| 0.0 | 5.35 | 3.01 | 1.78 |
| 5.0 | 6.12 | 3.44 | 1.78 |
| 9.0 | 6.12 | 3.44 | 1.78 |
| 13.0 | 6.28 | 3.53 | 1.78 |
| 16.0 | 6.46 | 3.63 | 1.78 |
| 20.0 | 6.72 | 3.78 | 1.78 |
| 25.0 | 7.01 | 3.94 | 1.78 |
| 30.0 | 7.39 | 4.15 | 1.78 |
| 35.0 | 7.55 | 4.24 | 1.78 |
| 40.0 | 8.14 | 4.57 | 1.78 |
| 50.0 | 8.14 | 4.57 | 1.78 |
| 65.0 | 8.26 | 4.64 | 1.78 |
| 90.0 | 8.26 | 4.64 | 1.78 |
| 700.0 | 10.3 | 5.79 | 1.78 |

Note that the 9–5 km interval has identical VP (6.12 km/s), creating a zero-gradient zone. The jump from **7.55 to 8.14 km/s between 35 and 40 km** captures the Moho transition. The 700 km node is a far-field lower boundary that remains fixed.

## Conclusion

The key insight is that this is a **node-based, continuously interpolated** model — not a layer-cake of constant velocities. Every depth in Table 1 is a grid node where velocity is specified; between nodes, SIMULPS13Q interpolates linearly. The −10 km node is an above-sea-level upper boundary, and VS everywhere derives from the jointly inverted VP/VS ratio (starting at a uniform 1.78), not from independent S-wave inversion. If you are building a reproduction of this model, the correct implementation is linear interpolation between the tabulated depth-velocity node pairs, not step functions.