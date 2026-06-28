# Slip Inversion Strategies — Nicoya Interseismic Locking

Three strategies have been implemented across the synthetic recovery test scripts.
They differ in how the slip parameter space is defined and constrained.

---

## Strategy 1 — Unconstrained Two-Component Inversion

**Parameter space**: (s_strike, s_dip) at each fault vertex, freely signed, no bounds.

**Regularization**: Smoothness (H1/bi-Laplacian) + L2 damping only.  
The L-curve selects the regularization weight γ.

**Relevant scripts**:
- `synth_checkslip_inv_hetmu_uneven_nicoyaCK_lock_noi2.py`
- `synth_stripeslip_inv_hetmu_uneven_nicoyaCK_lock_noi.py`

**When useful**: General-purpose coseismic slip inversion where the slip direction
is not known a priori and the dataset has sufficient resolving power to separate
strike and dip contributions.

**Limitation**: When the true slip is dominated by one direction (e.g., trench-normal
back-slip), the surface displacement field is also dominated by that direction.
The inversion then faces a rank-deficiency: many (s_strike, s_dip) pairs reproduce
the data equally well as long as their vector sum points roughly in the true direction.
Damping alone regularizes the magnitude but cannot fix the directional ambiguity.

---

## Strategy 2 — Bounded Two-Component Inversion (TanhSlip)

**Parameter space**: (s_strike, s_dip) at each fault vertex, but each component is
mapped through a tanh transformation to enforce finite bounds:

    s_phys = A * tanh(m)  ∈ (-A, +A)

Bounds can be set independently for strike and dip, or asymmetrically
(e.g., dip constrained to [0, amp] for pure reverse faulting).

**Relevant scripts**:
- `synth_checkslip_az_inv_hetmu_uneven_nicoyaCK_lock_noi2.py`
- `synth_stripeslip_az_inv_hetmu_uneven_nicoyaCK_lock_noi.py`

**When useful**: When physical constraints on the slip direction or magnitude are
available (e.g., plate convergence rate as an upper bound on back-slip).
Bounds reduce the feasible parameter space and help when damping alone is
insufficient to single out the true solution.

**Limitation**: Bounds shrink the feasible space but do not resolve the fundamental
rank-deficiency when slip is along a fixed azimuth. If the data can only constrain
the projected amplitude in one direction, the decomposition into strike and dip
components remains underdetermined within the bounded region. Adding an azimuth
prior on the slip direction (Strategy 3) is a cleaner fix for this specific case.

---

## Strategy 3 — Scalar Amplitude Inversion with Fixed Azimuth (AzSlip)

**Parameter space**: A single scalar amplitude a(x) ≥ 0 at each fault vertex.
The slip direction is fixed to the plate convergence azimuth (N33.5°E for Nicoya),
decomposed into per-facet (c_strike, c_dip) coefficients so that the horizontal
projection of the in-plane slip vector matches the prescribed azimuth:

    slip = a(x) * (c_strike(x), c_dip(x))

The amplitude is mapped through tanh to enforce positivity and an upper bound:

    a_phys = amp_max * (tanh(m) + 1) / 2  ∈ [0, amp_max]

where amp_max = plate convergence rate (~80 mm/yr for Nicoya).

**Relevant scripts**:
- `synth_checkslip_az_scalaramp_inv_hetmu_uneven_nicoyaCK_lock_noi2.py`
- `synth_stripeslip_az_scalaramp_inv_hetmu_uneven_nicoyaCK_lock_noi2.py`
- `synth_stripeslip_az_scalaramp_inv_hetmu_uneven_nicoyaCK_lock_noi.py`
- `synth_stripeslip_az_scalaramp_inv_hetmu3D_uneven_nicoyaCK_lock_noi.py`

**When useful**: Interseismic locking, where the physical model is back-slip
at a rate equal to the plate convergence rate scaled by the coupling coefficient.
The coupling coefficient is exactly the scalar amplitude normalized by amp_max.
Fixing the direction from plate kinematics removes the directional degree of
freedom entirely, reducing the parameter count from 2N to N and converting a
fundamentally underdetermined decomposition into a well-posed scalar problem.

**Key advantage over Strategy 2**: Strategy 2 bounds the magnitude of each
component independently; the slip direction is still free within the bounded
region. Strategy 3 hard-codes the direction from physics, so the data only
needs to constrain "how much" rather than "how much and in which direction."
This is a much stronger constraint when the true direction is known.

### Strategy 3 — Forward/Inversion Operator Mismatch (and fix)

**Problem**: The original Strategy 3 forward model (`create_fault_local_stripes_az`)
prescribes slip in the global geographic frame with `||slip|| = amp` exactly at every
fault point. But the inversion PDE (`PDEVarf_AzSlip`) applies slip as:

    slip = m_phys * (c_strike(x), c_dip(x))

where `(c_strike, c_dip)` are CG1-averaged (vertex-averaged) per-facet unit vectors.
At fault undulations, adjacent facets point in slightly different directions, so CG1
averaging produces `||c|| < 1` at shared vertices. The inversion then needs
`m_phys = amp / ||c|| > amp_max` to reproduce the forward data, but is bounded at
`amp_max = amp`. This systematic under-recovery appears as spots in the recovered
amplitude map at undulation vertices.

**Key insight**: Switching between global and local recovery ratio does NOT fix the
spots. For a uniform-amplitude stripe, `true_amp = amp` everywhere, so global ratio
= local ratio = `rec_amp / amp`. The spots exist in `rec_amp` itself.

**Fix — Consistent Forward (`PDEVarf_AzSlip_Direct`)**: Prescribe the true model
in the same parameterization the inversion uses: set `m_phys = amp` directly (scalar
CG1 function on the fault). The PDE then applies `slip = m_phys * (c_strike, c_dip)`,
exactly as in inversion. The inversion recovers `m_phys`, not `||slip||`, and since
the forward also speaks in `m_phys`, the operator mismatch is eliminated → no spots.

Note: `||slip||_physical = amp * ||c|| < amp` at undulation vertices (geometric spots
exist in the physical slip field). This is a property of the mesh, not of the locking.

**Scripts using consistent forward**:
- `synth_stripe_az_consistfwd_nicoyaCK.py` — stripe pattern, CK uneven mesh
- `synth_check_az_consistfwd_nicoyaCK.py` — checkerboard pattern, CK uneven mesh

Both use `PDEVarf_AzSlip_Direct` for forward and `PDEVarf_AzSlip` (tanh) for
inversion. Output data files are prefixed `d_obs_cf_*`; inversion output strings
are prefixed with `_cf_` to distinguish from original mismatched-forward results.

**What to display from inversion**: Show `m_phys / amp_max` (the coupling
coefficient). Do **not** show `m_phys * ||c|| / amp_max` — that quantity re-introduces
geometric spots from fault undulations that are a mesh artifact, not real locking
heterogeneity.

---

## Summary Table

| Strategy | Parameters | Direction | Magnitude | Best for |
|---|---|---|---|---|
| 1 — Unconstrained | (s_strike, s_dip) | Free | Free | General coseismic |
| 2 — Bounded (TanhSlip) | (s_strike, s_dip) | Free within bounds | Bounded | Known-direction slip, partial constraint |
| 3 — Scalar Amp (AzSlip) | a(x) scalar | Fixed to plate az. | Bounded [0, amp_max] | Interseismic locking |

---

## PDE / Code Implementation

| Strategy | PDE class | Inversion function |
|---|---|---|
| 1 | `PDEVarf` | `solveCoseismicInversion_TanhSlip` (unbounded variant) |
| 2 | `PDEVarf_TanhSlip` | `solveCoseismicInversion_TanhSlip` |
| 3 | `PDEVarf_AzSlip` | `solveCoseismicInversion_AzSlip` |
| 3 (consistent fwd) | `PDEVarf_AzSlip_Direct` (fwd) + `PDEVarf_AzSlip` (inv) | `solveCoseismicInversion_AzSlip` |

In Strategy 3, `compute_az_cg1_functions()` projects per-facet azimuth coefficients
(c_strike, c_dip) to CG1 vertex functions by averaging over adjacent fault facets.
These are precomputed once and passed into the PDE variational form.

`PDEVarf_AzSlip_Direct` (forward only): identical slip BC as `PDEVarf_AzSlip` but
takes `m_phys` directly as a CG1 Function rather than via tanh. Used only for
generating synthetic data in consistent-forward tests.
