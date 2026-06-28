# Slip Prescription and PDE Parameter Flow

## 1. Are `FaultLocalCheckerboard` / `FaultLocalStripes` consistent with the inversion?

**Yes. Slip direction is per-facet in both cases.**

### The key insight (from the 2D reference code)

The expression `mtrue_s_expr` does **not** encode slip direction. It only encodes
**scalar magnitudes** at each DOF:

```
values[0] = strike-slip magnitude   (= 0 for pure dip-slip patterns)
values[1] = dip-slip magnitude      (= amp * pattern)
```

These scalars are stored in a 2-component `CG1` vector function space `V_slip` via
a Dirichlet BC on the fault boundary:

```python
V_slip = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
bc_fault_slip = dl.DirichletBC(V_slip, mtrue_s_expr, boundaries, fault)
```

### Where the physical slip direction actually comes from

In the weak form (`PDEVarf.__call__`), the slip vector is reconstructed as:

```python
dir_strike(n('+')) * ufl.avg(m_strike)  +  dir_dip(n('+')) * ufl.avg(m_dip)
```

where `n = dl.FacetNormal(mesh)` — a UFL object evaluated **per facet** at each
quadrature point.

```python
def dir_strike(n):
    z_dir = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    return n_cross_z / ufl.sqrt(ufl.dot(n_cross_z, n_cross_z))

def dir_dip(n):
    return ufl.cross(dir_strike(n), n)
```

So:
- The expression sets **how much** slip (scalar magnitudes).
- The weak form determines **which direction** is "strike" and "dip", using the exact
  per-facet normal at every integration point.
- This is identical for both the forward prescription and the inversion.

### What the averaged normal in the expression classes IS used for

`FaultLocalCheckerboard` and `FaultLocalStripes` compute an averaged fault normal
in `_compute_fault_geometry()` to build a fault-local coordinate frame. This is used
**only** to decide the pattern geometry — which parts of the fault fall inside a stripe
or checkerboard cell. It does **not** affect the slip direction.

### Summary

| Concern | What determines it | Per-facet? |
|---|---|---|
| Slip direction (strike/dip) | `dir_strike(n)`, `dir_dip(n)` in weak form | **Yes** |
| Pattern geometry (which facets get slip) | Averaged normal in expression class | No (approximation) |

The approximation in pattern geometry is acceptable for a gently curved fault and does
not affect the physical consistency between prescribed and inverted slip.

---

## 2. How does `mtrue` enter the PDE?

**Answer: through the `x` list passed to `pde.solveFwd`, not through the `pde` constructor.**

### Construction

```python
pde_varf = PDEVarf(mtrue_mu_fun)   # only mu is stored here; slip is NOT
pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
```

`pde` stores the variational form structure and boundary conditions, but has no
knowledge of the slip parameter `mtrue` yet.

### Solve

```python
u_mtrue = pde.generate_state()
x = [u_mtrue, mtrue, None]         # [STATE, PARAMETER, ADJOINT]
pde.solveFwd(x[hp.STATE], x)
```

Hippylib's `PDEVariationalProblem.solveFwd(state, x)` internally calls:

```python
pde_varf(u, m, p)   # where m = x[hp.PARAMETER] = mtrue
```

Inside `PDEVarf.__call__(u, m, p)`:
```python
m_strike, m_dip = dl.split(m)   # m IS mtrue here
```

So `mtrue` flows in at solve time via the `x` list. The `pde` object is a reusable
container for the variational structure; it receives the specific parameter values
only when solving.

### Why this design matters

This is the standard hippylib pattern. The same `pde` object can be called with
different `m` vectors (different slip models) without reconstruction. During inversion,
hippylib passes the current iterate `m` at each optimization step through the same
`x = [state, m, adjoint]` interface.

---

## 3. Plate convergence rate `V_norm` and the coupling ratio

### What `V_norm` is in the Nicoya code

```python
V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm/yr
s_dip_max = V_norm
# coupling ratio = s_dip / V_norm
```

`V_norm` is the **horizontal** Cocos-Caribbean convergence rate projected onto the
trench-normal direction. It is a single scalar applied uniformly to all fault facets.

### Why the projection onto the local down-dip direction is NOT invariant

For a rigid subducting plate moving at 3D velocity **V** (same everywhere on the plate),
the tangential (slip) component at each facet is:

```
V_tangential = V − (V · n̂) n̂
```

The down-dip component is `V · d̂`, where **d̂** is the local dip vector. For a curved
fault where the dip angle δ varies with depth, this is **not** constant:

- Shallow near trench (δ ≈ 10°): down-dip component ≈ V_h · cos(10°) ≈ 0.985 V_h
- Steeper at depth   (δ ≈ 30°): down-dip component ≈ V_h · cos(30°) ≈ 0.866 V_h

The 3D plate velocity **V** is the invariant quantity; any projection of it — whether
horizontal or along local down-dip — varies across a curved fault as the normal **n̂**
changes.

### Standard practice and its approximation

Using a single scalar `V_norm` (horizontal trench-normal rate) as the coupling denominator
for all facets is an approximation that:
- Works well for shallow-dipping faults (cos(δ) correction is small)
- Avoids per-facet projection complexity
- Is consistent with the vast majority of geodetic coupling literature

The rigorous approach would compute per-facet: project **V** onto each facet's local
dip direction and use that as the coupling denominator. In practice this variation is
small and is rarely done.

---

## 4. Initial model and tanh linearization bias by slip pattern

### Starting point: m0 = (0, 0)

With tanh reparameterization `physical_dip = (dip_max) · (tanh(m_dip) + 1) / 2`, the
Jacobian scales as `sech²(m_dip)`, which is **maximum at m=0** and decays toward zero
in both tails. Setting `m0 = 0` places the CG linearization at the point of maximum
sensitivity — this is the best practical choice.

The CG solver here performs a single Gauss-Newton step: it linearizes the forward
operator at `m0` and solves once. The starting point does not shift the solution for a
linear forward operator, but the tanh introduces mild nonlinearity — the linearization
is most accurate near `m=0`.

### Why stripe and checkerboard patterns behave differently

**Stripe (binary truth):** True physical dip is either `V_norm` (fully locked) or `0`
(fully sliding) — both at the extremes of the tanh range. Recovering these would require
`m_dip → ±∞`, but the linearization at `m0=0` approximates the tails poorly. The
inversion systematically **underestimates** locking at fully-locked patches and
underestimates sliding at fully-sliding patches. This is a fundamental limitation of
single-step Gauss-Newton with tanh bounds when the truth is at extremes.

**Checkerboard (sinusoidal truth):** True physical dip follows
`0.5·(sin(·)+1)·V_norm`, ranging continuously from 0 to V_norm with mean `V_norm/2`.
That mean coincides almost exactly with the tanh midpoint (`m=0` maps to `V_norm/2`),
so the bulk of the truth lives near the linearization point. The CG approximation is
most accurate where it matters most. The extreme values (near 0 and V_norm) still
suffer some saturation error, but they affect fewer DOFs.

**Practical implication:** When comparing stripe vs checkerboard recovery quality,
some of the stripe's underperformance reflects this linearization saturation effect,
not purely resolution or data coverage limitations. The proper fix for stripe patterns
is iterative nonlinear Gauss-Newton (multiple outer iterations), which is a larger
architectural change not currently implemented.

---

## 5. Locking ratio definition with the Az slip class

For the back-slip method (Savage 1983), the prescribed slip IS the back-slip deficit.
The locking ratio is therefore:

```
LR = sqrt(m_strike_inv² + m_dip_inv²) / V_norm
```

- LR = 1 → full back-slip recovered → fully locked
- LR = 0 → no back-slip → freely sliding

This generalises the OLD (pure dip) class where `m_strike = 0`, giving `LR = |m_dip| / V_norm`.
The vector sum is necessary for the Az class because both components are non-zero.

The Az class preserves total amplitude by construction: `sqrt(c_strike² + c_dip²) = 1`
for every facet, so `||slip_true|| = V_norm` at a fully sliding patch regardless of the
strike/dip decomposition.

---

## 6. Inversion bounds for the Az slip class (tanh transformation)

### Dip component

```python
dip_bounds = (0.0, 1.1 * amp)   # 10% headroom above V_norm
```

The truth at a fully-sliding patch is `c_dip × amp ≈ 0.994 × amp` (for az=45° on Nicoya),
almost at the upper bound. The 10% headroom prevents the tanh gradient from vanishing
at the truth, improving convergence at fully-unlocked patches. The physical cost
(allowing LR slightly > 1) is acceptable for a synthetic test.

### Strike component — why ±16 mm/yr is too tight

The OLD class prescribed `m_strike = 0` everywhere, so ±16 mm/yr was a safe numerical
guardrail. For the Az class on the **curved** Nicoya fault, the true per-facet c_strike
values span nearly ±amp due to local geometry. Analysis of the sandbox output
(`slip_fault_nicoyaCKden_sm_stripe_AZ45.txt`, 2639 active DOFs) shows:

| |c_strike| threshold | DOFs clipped | Fraction |
|---|---|---|
| > 10 mm/yr | 1511 / 2639 | 57.3% |
| > 16 mm/yr |  938 / 2639 | **35.5%** |
| > 20 mm/yr |  671 / 2639 | 25.4% |
| > 30 mm/yr |  341 / 2639 | 12.9% |
| > 50 mm/yr |  128 / 2639 |  4.9% |

Percentiles of |c_strike| (active DOFs):
- 50th: 12.0 mm/yr
- 75th: 20.2 mm/yr
- 90th: 34.6 mm/yr
- 95th: 48.8 mm/yr

**Spatial analysis** (via `plt_syn_slip_inv_hetmu_nicoyaCK_locking.ipynb` approach,
using `fault_geometry_nicoyaCKden_sm.txt` for DOF coordinates) shows that the high
|c_strike| DOFs concentrate in the **0–30 km depth zone** and trace the curved portion
of the slab visible in the slab contour map — confirming these are real geometry effects,
not mesh boundary artifacts:

| Depth zone | Active DOFs | > ±16 mm/yr |
|---|---|---|
| 0 to -10 km  |  860 | 34.9% |
| -10 to -30 km |  947 | **55.4%** |
| -30 to -50 km |  530 | 20.2% |
| -50 to -70 km |  226 |  2.7% |

The 10–30 km seismogenic zone — the most important depth for locking — is the worst
affected. A ±16 mm/yr bound would clip the truth for more than half the DOFs there.

### Recommended bounds for BOUND_TYPE='both' with Az class

```python
strike_bounds = (-amp, +amp)         # ±78.5 mm/yr — covers full range for curved fault
dip_bounds    = (0.0, 1.1 * amp)    # small headroom above truth at fully-unlocked patches
```

The ±amp strike bound is the physically motivated maximum: no back-slip component in
any direction can exceed the total convergence rate. For the OLD class (m_strike_true = 0),
a tighter bound is fine as a guardrail; for the Az class on a curved fault it must be ±amp.

---

## 7. Back-slip slip direction prescription: Scenario 1 vs Scenario 2

### Context

The Az slip classes (`FaultLocalStripesAz`, `FaultLocalCheckerboardAz`) prescribe a slip
direction by geographic azimuth rather than pure dip-slip. Two physically distinct
strategies were considered:

---

### Scenario 1 (CHOSEN): In-plane slip, constant magnitude

The slip vector **lies in the fault plane** (zero normal component) with constant magnitude
`amp` regardless of local dip. Its horizontal projection points anti-convergence.

**Construction** in `_compute_az_per_facet_coeffs` (utils.py):
1. Form horizontal unit vector `h` pointing in the convergence azimuth direction (mesh coords).
2. Construct in-plane vector: `v = (h_x, h_y, vz)` with `v·n = 0` → `vz = -(h_x·nx + h_y·ny)/nz`.
3. Normalise: `v_hat = v / |v|` → unit vector in fault plane, `|v_hat| = 1`.
4. Decompose: `c_strike = -dot(v_hat, strike_vec)`, `c_dip = -dot(v_hat, dip_vec)`.
5. Slip = `amp × pattern × (c_strike, c_dip)` → total magnitude = `amp × pattern`.

The negation in step 4 implements back-slip convention: prescribed slip is anti-parallel to
plate convergence. `sqrt(c_strike² + c_dip²) = 1` per facet by construction.

**Decomposition note**: For slip in the dip direction (α ≈ φ_n), the horizontal fraction of
`v_hat` = cos δ and the vertical fraction = sin δ. For Nicoya (δ ≈ 20°) the back-slip is
~94% horizontal and ~34% vertical. See §8 for the full derivation.
The total amplitude stays `amp` regardless of dip angle.

---

### Scenario 2 (REJECTED): Horizontal slip projected onto fault plane

Define slip as a **horizontal** vector of magnitude `amp` pointing anti-convergence, then
project onto the fault plane: `v_proj = h - dot(h, n) n`.

This gives magnitude `|v_proj| = amp × cos(δ)` (dip-dependent). On a dipping fault,
the amplitude is reduced by `cos(δ)` relative to Scenario 1.

**Why rejected**:
1. Slip magnitude depends on local dip — inconsistent with a uniform convergence rate.
2. Not consistent with the Savage (1983) back-slip convention, which prescribes a fault-plane
   dislocation of fixed magnitude equal to the interseismic deficit rate.

**Numerical difference for Nicoya**: δ ≈ 10–20° → `cos(δ) ≈ 0.94–0.98`, so the two
scenarios differ by only ~2–6%. The physical argument still favours Scenario 1.

---

### Supporting reference

**Savage, J.C. (1983)**, "A Dislocation Model of Strain Accumulation and Release at a
Subduction Zone", *J. Geophys. Res.*, **88**(B6), 4984–4996.

The Savage back-slip model defines the interseismic strain accumulation as equivalent to
an elastic dislocation applied in reverse (back-slip) on the seismogenic portion of the
fault. The dislocation is prescribed **on the fault plane** (an in-plane slip vector),
not as a horizontal displacement. Scenario 1 directly implements this convention.

Subsequent interseismic coupling studies (e.g., Bürgmann et al. 2005 Cascadia,
Moreno et al. 2010 Chile, Loveless & Meade 2011 global) universally project the relative
plate motion onto the fault plane to define the coupling reference rate — consistent
with Scenario 1.

---

### Consistency across Az pattern classes

Both `FaultLocalStripesAz` and `FaultLocalCheckerboardAz` use `_compute_az_per_facet_coeffs`
and apply the same:

```python
values[0] = amp * pattern * c_strike
values[1] = amp * pattern * c_dip
```

The **only** difference between the two classes is the pattern scalar:
- **Stripes**: binary 0 or 1 (top-hat function of along-strike and along-dip position)
- **Checkerboard**: sinusoidal `0.5*(sin(ω_s·s)+1) × 0.5*(sin(ω_d·d)+1)` ∈ [0, 1]

The slip direction prescription (Scenario 1 via per-facet `c_strike, c_dip`) is identical
and fully consistent between the two classes.

---

## 8. Horizontal vs vertical decomposition of the in-plane slip vector

### Derivation

For a fault with dip angle δ (from horizontal), dip azimuth φ_n, and hanging-wall normal:

```
n = (−sin δ · cos φ_n,  −sin δ · sin φ_n,  cos δ)    →  nz = cos δ
```

Given prescribed horizontal unit vector `h` (azimuth α), the in-plane condition `v·n = 0`
with `v = (h_x, h_y, vz)` gives:

```
h · n_horiz = −sin δ · cos(α − φ_n)

vz = −(h · n_horiz) / nz = tan δ · cos(α − φ_n)
```

After normalising `v_hat = v / |v|` for the case α = φ_n (slip in the dip direction):

```
|v| = 1 / cos δ

Horizontal fraction of v_hat = cos δ       ← decreases with steeper dip
Vertical   fraction of v_hat = sin δ       ← increases with steeper dip
```

### Dependence on dip angle and azimuth offset

| Factor | Effect on vertical fraction |
|---|---|
| Steeper dip (larger δ) | **More vertical** (sin δ increases) |
| Slip in dip direction (α = φ_n) | Maximum vertical: sin δ |
| Slip along strike (α − φ_n = 90°) | **Zero vertical** (vz = 0), purely horizontal regardless of δ |

**Limit checks:**
- δ → 0° (flat fault): vz = tan(0°)·cos(·) = 0 → purely horizontal ✓
- δ → 90° (vertical fault): vz → ∞ before normalisation → vertical fraction → 1 ✓
- α − φ_n = 90° (strike-slip direction): vz = 0 regardless of δ ✓

### Implication for Nicoya back-slip

Convergence azimuth ≈ 45° ≈ fault dip azimuth → α − φ_n ≈ 0° (near-dip-direction slip).
With δ ≈ 20° in the seismogenic zone:

```
Horizontal fraction ≈ cos(20°) ≈ 0.94
Vertical   fraction ≈ sin(20°) ≈ 0.34
```

The back-slip at Nicoya is ~94% horizontal and ~34% vertical. The surface displacements
therefore have a strong horizontal component and a smaller but non-negligible Uz signal
(consistent with observed Uz ~ 35 mm vs horizontal ~ 45 mm in synthetic tests).

### Effect of varying plate morphology

For models with the same prescribed azimuth but different fault geometry (e.g., smooth vs
irregular plate interface), facets with different local dip angles δ produce different
horizontal/vertical splits of the physical slip vector. This changes both the amplitude and
spatial pattern of surface displacements, particularly in the shallow zone where dip varies
most between the two plate models.
