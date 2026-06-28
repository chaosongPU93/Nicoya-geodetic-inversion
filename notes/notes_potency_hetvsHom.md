# Potency difference: heterogeneous vs homogeneous μ models

## Observation

| Model     | Interseis rate (m³/yr) | vs Hom | Coseis pot (m³) | vs Hom  |
|-----------|------------------------|--------|-----------------|---------|
| Hom (H)   | 9.22×10⁸               | (ref)  | 1.23×10¹⁰       | (ref)   |
| SW (S)    | 7.44×10⁸               | −19.3% | 1.06×10¹⁰       | −13.8%  |
| 1D        | 9.33×10⁸               | +1.2%  | 1.15×10¹⁰       | −6.5%   |
| Orig. 3D  | 9.25×10⁸               | +0.3%  | 1.16×10¹⁰       | −5.7%   |
| Ampl. 3D  | 9.20×10⁸               | −0.2%  | 1.14×10¹⁰       | −7.3%   |

**Trend**: SW is consistently lower than Hom in both cases. DeShon-based het models
(1D, 3D) show ~−6% for coseismic but ~0% for interseismic — **opposite/absent trend**.

---

## Setup

Both inversions use the same mesh (`nicoyaCKden_une_all`, 9636 fault DOFs), same GNSS
stations, and nearly identical amplitude regularization:

- δ_coseis   = γ / ρ_s = 2.5×10² / 2×10⁷  = **1.25×10⁻⁵**
- δ_interseis = γ / ρ_s = 3×10³  / 2.5×10⁸ = **1.20×10⁻⁵**

γ is 12× larger for interseismic (controls spatial smoothness, not amplitude directly).
Potency computation is byte-identical in both scripts.

---

## μ field at the fault (GPa, nearest-vertex interpolation)

| Region                        | Hom   | 1D    | Orig 3D | 1D/Hom | 3D/Hom |
|-------------------------------|-------|-------|---------|--------|--------|
| All fault (9636 DOFs), mean   | 40.00 | 51.47 | 51.49   | 1.287  | 1.287  |
| Coseis patch (top-25% mag)    | 40.00 | 40.80 | 39.27   | 1.020  | 0.982  |

The DeShon models are globally much stiffer (+28.7% fault-wide average) but at the
coseismic rupture patch specifically, μ_het ≈ μ_Hom (±2%).

---

## Slip-weighted effective μ

Defined as μ_eff = Σ |slip_i| · μ_i / Σ |slip_i|, using each model's own inferred slip.

| | Hom | 1D | Orig 3D | 1D/Hom | 3D/Hom |
|---|---|---|---|---|---|
| Coseis (own slip-wt) | 40.00 | 42.53 | 40.73 | 1.063 | 1.018 |
| Interseis (own slip-wt) | 40.00 | 45.97 | 45.05 | 1.149 | 1.126 |
| DeShon μ at coseis location (coseis-slip-wt) | 40.00 | 40.47 | 38.90 | 1.012 | 0.973 |

Key: the interseismic back-slip is concentrated where DeShon μ is 12–15% above Hom
(deeper, stiffer fault region), whereas the coseismic rupture sits where DeShon μ ≈ Hom.

---

## Argument for coseismic (partially convincing)

Potency ∝ 1/μ_eff (approximate, from u ~ μ·slip → inferred slip ~ u/μ). Then:

- 1D: 1/1.063 → predicted −5.9%; observed −6.5% ✓ (close)
- 3D: 1/1.018 → predicted −1.8%; observed −5.7% ✗ (underestimates)

The residual for 3D likely comes from the volumetric Green's function: even though
μ_het ≈ μ_Hom at the slip patch, the surrounding volume is much stiffer (+28.7%
fault-wide), which stiffens the full 3D elastic medium and amplifies surface
displacements for the same slip → inversion needs less slip → lower potency.

---

## Argument for interseismic ~0% (less convincing — open question)

Despite the interseismic slip being concentrated where DeShon μ is 12–15% above Hom,
the potency barely changes. Two candidate mechanisms:

**Candidate 1 — δ anchors amplitude when data constraint is weak**:
δ is the same in both inversions, but the coseismic data (large offsets, concentrated
single event) have higher effective signal-to-noise than interseismic rates. When the
data term is weak relative to δ, the amplitude of the solution is dominated by δ
(same for both models), making potency insensitive to μ. For coseismic, the data
term is strong enough to override δ → amplitude (and potency) becomes μ-sensitive.

**Candidate 2 — γ reshapes slip pattern to compensate**:
The 12× stronger γ forces a spatially smooth, spread-out interseismic solution.
The redistributed slip pattern may sample regions where the net μ_het/μ_Hom ratio
is closer to 1, partially cancelling the μ effect on potency.

**Why not fully convincing**: with δ the same and data amplitudes in the same
mm-scale range, it is not obvious that the data term is categorically weaker for
interseismic. A more rigorous test would require comparing the actual misfit-to-
regularization ratio at the L-curve corner for both cases, and/or running a synthetic
experiment where μ is uniformly scaled.

---

## Summary

| | Coseis | Interseis |
|---|---|---|
| Slip-wt μ_het / μ_Hom | 1.02–1.06 | 1.13–1.15 |
| Simple potency prediction (1/μ_eff) | −2% to −6% | −11% to −13% |
| Observed potency change | −6% to −7% | ~0% |
| Data-term vs regularization | data-dominated | possibly δ-dominated |
| γ effect | smooth, concentrated | very smooth, spread |

The coseismic case is approximately explained by the slip-weighted μ argument plus
volumetric Green's function stiffening. The interseismic ~0% result is not yet fully
explained and remains an open question for further investigation.
