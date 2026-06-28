# Comparing Synthetic Surface Displacements: Regular vs Uneven Plate Interface

## Context

A collaborator suggested comparing synthetic surface displacement fields between two mesh configurations:

- `nicoyaCKden_sm` — regular plate interface, top boundary at z=0 everywhere
- `nicoyaCKden_une_sm` — uneven/irregular plate interface, top boundary at z≈−7 km on the seaside (trench) and z=0 on the landward side

## Why the comparison is not trivial

Three layers of inequivalence exist between the two configurations:

### 1. Different fault extent
The `une` mesh omits the shallow trench portion of the fault (depth 0 to ~7 km on the seaside). Even if slip is prescribed as the same continuous spatial function and normalized to the same total potency, the two sources occupy different spatial volumes. They are not the same source.

### 2. Different fault node distribution
Within the shared fault extent, the two meshes triangulate the fault surface differently (different node counts, locations, and areas). The same continuous slip function sampled at different nodes integrates to slightly different potency, requiring normalization.

### 3. Different observation surface
For the `une` mesh, surface displacements are computed at the FE mesh top boundary, which is at z≈−7 km on the seaside. For the `sm` mesh, the top boundary is at z=0 everywhere. On the seaside, z=0 is **outside** the `une` FE domain — displacements cannot be evaluated there. So the two meshes do not share a common observation surface over the full domain.

## The only well-posed fair comparison

Restrict everything to the **landward side**, where both meshes have z=0 as the top boundary:

1. **Same slip function**: define a continuous slip field (e.g., Gaussian patch) in fault-local coordinates
2. **Same fault extent**: mask out fault nodes outside the `une` mesh's depth and horizontal range on both meshes; the `une` mesh's extent defines the common domain
3. **Same potency**: normalize total potency (Σ slip_i × area_i) to the same reference value, integrated over the common fault extent only
4. **Same observation grid**: restrict to land-based points (z=0), where both mesh top boundaries coincide

After these four conditions, the only remaining difference between the two synthetic displacement fields is the fault surface geometry within the shared extent — the genuine signal of interest.

## Scientific framing

The comparison is only well-posed on the landward side. This is also where GNSS observations actually are, so it is the scientifically relevant domain. The near-trench region, where the two meshes differ most, has no observations and involves incomparable quantities (different observation depths, different fault extents).

The honest framing for a paper or collaborator discussion: **the comparison is restricted to the landward surface displacement field**, under identical source conditions (same slip function, same potency, same fault extent). Any extension to the seafloor region involves fundamentally incomparable quantities.

## Practical implementation notes

- Use `extract_top_boundary_surface()` to obtain the `une` mesh's top boundary vertices, which define the spatial mask for step 2
- Ray casting (`find_valid_surface_depths_raycast()`) can identify which fault nodes of the `sm` mesh fall outside the `une` mesh's horizontal footprint
- Potency normalization: compute area-weighted slip sum for each mesh over the masked fault, then rescale
- Observation grid: use only land-based GNSS station locations or a dense grid restricted to the landward region (x > coast)
