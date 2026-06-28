# Migration Plan: CG1 → CG2 for Shear-Modulus Function Space

## Motivation

Reviewer comment: use the same numerical approach for all structural models and change only material attributes.
Currently, hom and K2LAYER (het) models use CG1 for the μ function space while 1D-layered and 3D models use CG2.
This inconsistency is being resolved by upgrading hom/K2LAYER to CG2 as well.
The accuracy benefit is minimal (μ is smooth or piecewise-constant), but numerical consistency is the goal.

To avoid overwriting existing CG1 result files, `_CG2` must be appended to output filename strings
wherever CG2 is now used — including for hom and K2LAYER, which previously had no such tag.

**`utils.py` default values are NOT changed.** All affected scripts will pass `CG_mu_degree=2` explicitly.

---

## Files in scope

### Priority 1 — `*_az_consistfwd_*` synthetic scripts (9 files)

| File | Group | Current state |
|---|---|---|
| `synth_stripe_az_consistfwd_nicoyaCK.py` | A (hom) | CG1 hardcoded |
| `synth_stripe_az_consistfwd_nicoya.py` | A (hom) | CG1 hardcoded |
| `synth_check_az_consistfwd_nicoyaCK.py` | A (hom) | CG1 hardcoded |
| `synth_stripe_az_consistfwd_het_nicoyaCK.py` | B (het) | CG1 hardcoded |
| `synth_stripe_az_consistfwd_het_nicoya.py` | B (het) | CG1 hardcoded |
| `synth_check_az_consistfwd_het_nicoyaCK.py` | B (het) | CG1 hardcoded |
| `synth_stripe_az_consistfwd_3D_nicoyaCK.py` | C (3D) | CG2 for 3D, CG1 for hom branch |
| `synth_stripe_az_consistfwd_3D_nicoya.py` | C (3D) | CG2 for 3D, CG1 for hom branch |
| `synth_check_az_consistfwd_3D_nicoyaCK.py` | C (3D) | CG2 for 3D, CG1 for hom branch |

### Priority 2 — Real data inversion scripts (6 files)

| File | Group |
|---|---|
| `slip_inv_hetmu3D_nicoyaCK_coseis.py` | C (3D) |
| `slip_inv_hetmu3D_nicoyaCK_coseis_diffrs.py` | C (3D) |
| `slip_inv_hetmu3D_nicoyaCK_locking_both.py` | C (3D) |
| `slip_inv_hetmu3D_nicoyaCK_locking_both_az.py` | C (3D) |
| `slip_inv_hetmu3Dori_nicoyaCK_coseis.py` | C (3D) |
| `slip_inv_hetmu3Dori_nicoyaCK_locking_both_az.py` | C (3D) |

### Priority 3 — Hardcoded CG1 scripts (3 files)

| File | Group |
|---|---|
| `slip_inv_hetmu1D_nicoyaCK_locking_both.py` | D (1D) |
| `slip_inv_hetmu1D_nicoyaCK_locking_both_az.py` | D (1D) |
| `slip_inv_hetmu1D_nicoyaCK_coseis.py` | D (1D) |

### Not in scope

- `test_mu_scaling_*` and `test_higher_CGmu_*` scripts — excluded intentionally
- `plt_test_syn_slip_az_scalaramp_inv_hetmu_nicoyaCK_locking.ipynb` — test notebook, excluded
- Notebook versions of `.py` inversion scripts (e.g. `slip_inv_hetmu3D_nicoyaCK_coseis.ipynb`) — not in scope
- `bounded_inversion_corrected.py`, `solveCoseismicInversion_bounded_standalone.py` — receive
  `mu_str_for`/`mu_str_inv` as function arguments; no change needed, suffix propagates automatically
- `test*.py`, `testmesh_*.py` — exploratory scripts, not in scope

---

## Plotting and analysis notebooks in scope

These notebooks read result files whose names will change due to `_CG2` being appended.
The change in each notebook is the same: append `_CG2` to the same `mu_str_*` variable assignments
that are changed in the corresponding `.py` scripts (same pattern, same logic).

### DIRECT — construct filenames via `mu_str_*` variables

| Notebook | Corresponding scripts |
|---|---|
| `plt_syn_slip_az_consistfwd_nicoyaCK_locking.ipynb` | `*_az_consistfwd_*nicoyaCK*.py` |
| `plt_syn_slip_az_consistfwd_nicoya_locking.ipynb` | `*_az_consistfwd_*nicoya*.py` |
| `plt_syn_slip_az_consistfwd_nicoya_locking_lc.ipynb` | `*_az_consistfwd_*nicoya*.py` |
| `plt_slip_inv_hetmu_nicoyaCK_locking_both_az.ipynb` | `slip_inv_hetmu3D_nicoyaCK_locking_both_az.py` |

**Change pattern:** find all `mu_str_for = ...` and `mu_str_inv = ...` assignments and append
`+ "_CG2"` following the same Group A/B/C logic as the corresponding `.py` script.

### CHECK — hardcoded filenames containing `_hom_` or `_het_`

These notebooks use literal filename strings instead of `mu_str_*` variables.
The `_CG2` tag must be inserted into the hardcoded filename strings wherever they reference
result files produced by the scripts being changed.

| Notebook | Corresponding scripts |
|---|---|
| `plt_slip_inv_hetmu_nicoyaCK_coseis.ipynb` | `slip_inv_hetmu3D_nicoyaCK_coseis.py` |
| `plt_slip_inv_hetmu_nicoyaCK_locking_both.ipynb` | `slip_inv_hetmu3D_nicoyaCK_locking_both*.py` |
| `plt_syn_slip_az_consistfwd_nicoya_locking_summary.ipynb` | `*_az_consistfwd_*nicoya*.py` |
| `plt_syn_slip_az_inv_hetmu_nicoyaCK_locking.ipynb` | `slip_inv_hetmu3D_nicoyaCK_locking_both*.py` |
| `plt_syn_slip_az_inv_hetmu_nicoya_locking.ipynb` | `slip_inv_hetmu3D_nicoya*.py` |

**Change pattern:** search for hardcoded filename strings containing `_hom_` or `_het_` that
correspond to result files from the affected scripts. Insert `_CG2` into those strings just before
or after the `_mul*u*` mu-parameter segment, matching the convention used by the `.py` scripts.
Verify by checking what the `.py` script produces vs. what the notebook reads.

---

## Other files — flag for review

- `synth.py` — standalone script that sets `mu_str_hom` and `mu_str_het` itself and uses them in
  output filenames. Not in the `_az_consistfwd_` group. **Decide whether to include in scope.**
  If yes, apply same Group A/B logic.

---

## Change instructions by group

---

### Group A — hom-only files

These files only ever use one μ model (homogeneous), hardcode CG1 for the μ space, and have no
`CG_mu_deg` variable. All three `CG_mu` assignments correspond to: forward μ, grid-displacement μ,
and inversion μ — one per major function block.

**Change 1 — μ function space (3 occurrences per file):**
```python
# BEFORE
CG_mu = dl.FunctionSpace(mesh, "CG", 1)

# AFTER
CG_mu = dl.FunctionSpace(mesh, "CG", 2)
```
Note: only change lines where the variable is named `CG_mu`. Do NOT change other CG1 spaces
(`CG1`, `Vm_diag`, `Vm_amp`, `V_slip_2`) — those are for slip/displacement DOFs and are unrelated.

**Change 2 — forward μ string (1 occurrence per file, near top-level script body):**
```python
# BEFORE
mu_str_for = mu_str_hom

# AFTER
mu_str_for = mu_str_hom + "_CG2"
```

**Change 3 — inversion μ string (1 occurrence per file, inside inversion loop/block):**
```python
# BEFORE
mu_str_inv = mu_str_hom

# AFTER
mu_str_inv = mu_str_hom + "_CG2"
```

**Change 4 — explicit `CG_mu_degree` argument in utils calls:**
Search for calls to `process_velocity_models*` or `build_1d_shear_modulus` that do NOT already
pass `CG_mu_degree`. Add `CG_mu_degree=2` explicitly. (In Group A files these calls likely do not
exist since hom μ is an analytic expression, not loaded from velocity models — verify when editing.)

---

### Group B — het (K2LAYER) files

Structure is nearly identical to Group A but uses `mu_str_het` instead of `mu_str_hom`, and has
two `mu_str_inv` assignments: one for het inverse, one for hom inverse.

**Change 1 — μ function space:** same as Group A (3 occurrences, only `CG_mu` lines).

**Change 2 — forward μ string:**
```python
# BEFORE
mu_str_for = mu_str_het

# AFTER
mu_str_for = mu_str_het + "_CG2"
```

**Change 3 — inversion μ string, het branch:**
```python
# BEFORE
mu_str_inv = mu_str_het

# AFTER
mu_str_inv = mu_str_het + "_CG2"
```

**Change 4 — inversion μ string, hom branch** (het files also invert with hom μ as reference):
```python
# BEFORE
mu_str_inv = mu_str_hom

# AFTER
mu_str_inv = mu_str_hom + "_CG2"
```

**Change 5 — explicit `CG_mu_degree` argument:** same as Group A (verify whether calls exist).

---

### Group C — 3D model files (both synthetic and real data inversion)

These files already use CG2 when `mtrue_mu_expr_for/inv is None` (i.e., when μ is loaded from
velocity model data). The hom inverse branch still uses CG1. The conditional block and function
signature defaults need updating.

**Change 1 — top-level conditional block for forward μ degree:**
```python
# BEFORE
if mtrue_mu_expr_for is None:
    CG_mu_deg = 2
    mu_str_for = mu_str_for + f"_CG{CG_mu_deg}"
else:
    CG_mu_deg = 1

# AFTER
CG_mu_deg = 2
mu_str_for = mu_str_for + f"_CG{CG_mu_deg}"
```
The `print(f"Forward problem based on: {mu_str_for}, ...")` line that follows is kept unchanged.

**Change 2 — inversion hom branch μ string and CG degree:**
```python
# BEFORE  (hom inverse branch)
mu_str_inv = mu_str_hom
# ... followed by function call with:
    CG_mu_deg=1,

# AFTER
mu_str_inv = mu_str_hom + "_CG2"
# ... followed by:
    CG_mu_deg=2,
```

**Change 3 — function signature defaults** (each major helper function defined in the script,
typically `run_forward_cf`, `run_forward_grid`, `run_inversion` or equivalent):
```python
# BEFORE
def run_forward_cf(..., CG_mu_deg=1):
def run_forward_grid(..., CG_mu_deg=1):
def run_inversion(..., CG_mu_deg=1, ...):

# AFTER
def run_forward_cf(..., CG_mu_deg=2):
def run_forward_grid(..., CG_mu_deg=2):
def run_inversion(..., CG_mu_deg=2, ...):
```

**Change 4 — `CG_mu_degree` argument in utils calls inside each function:**
Any call to `process_velocity_models_hull(...)`, `process_velocity_models_hull_interp(...)`, or
`build_1d_shear_modulus(...)` inside these functions should already pass `CG_mu_degree=CG_mu_deg`.
Confirm this is the case; if any call omits the argument, add `CG_mu_degree=CG_mu_deg` explicitly.

---

### Group D — 1D hardcoded CG1 files

These scripts set `CG_mu_deg = 1` unconditionally (no conditional block) and have a single
`mu_str_inv` assignment.

**Change 1 — CG degree variable:**
```python
# BEFORE
CG_mu_deg = 1

# AFTER
CG_mu_deg = 2
```

**Change 2 — μ string:** check whether `_CG{CG_mu_deg}` is already appended to `mu_str_inv`.
If not (likely, since these always used CG1 and never tagged), add:
```python
# BEFORE
mu_str_inv = mu_str_1d   # (or whatever the 1D string variable is named)

# AFTER
mu_str_inv = mu_str_1d + "_CG2"
```

**Change 3 — utils call:** these scripts call `build_1d_shear_modulus(...)`. Confirm it already
passes `CG_mu_degree=CG_mu_deg`; if not, add it explicitly.

---

## What NOT to change

- Any `dl.FunctionSpace(mesh, "CG", 1)` where the variable is **not** `CG_mu` —
  e.g., `CG1`, `Vm_diag`, `Vm_amp`, `V_slip_2` etc. are slip/displacement spaces and must stay CG1.
- Default values in `utils.py` helper functions (`CG_mu_degree=1` in `process_velocity_models*`,
  `build_1d_shear_modulus`).
- `test_mu_scaling_*` and `test_higher_CGmu_*` scripts.

---

## Verification after each file

After editing each file, confirm:
1. `grep "CG_mu = dl.FunctionSpace" <file>` → all show `"CG", 2`
2. `grep "mu_str_for\s*=\|mu_str_inv\s*=" <file>` → all assignments end with `+ "_CG2"` or
   `+ f"_CG{CG_mu_deg}"` (where `CG_mu_deg` is now always 2)
3. `grep "CG_mu_deg" <file>` → no remaining `= 1` except inside comments

---

## Recommended execution order

Work through files in this order to catch issues early (synthetic scripts are faster to re-run
than real data inversions):

1. **Group C synthetics** (Priority 1, 3D files) — `synth_stripe_az_consistfwd_3D_*.py` and
   `synth_check_az_consistfwd_3D_nicoyaCK.py` — smallest change (remove else branch only)
2. **Group A synthetics** (Priority 1, hom files) — `synth_*_az_consistfwd_nicoya*.py` and
   `synth_check_az_consistfwd_nicoyaCK.py`
3. **Group B synthetics** (Priority 1, het files) — `synth_*_az_consistfwd_het_*.py`
4. **DIRECT plotting notebooks** for Priority 1 — `plt_syn_slip_az_consistfwd_*.ipynb` and
   `plt_slip_inv_hetmu_nicoyaCK_locking_both_az.ipynb`
5. **Group C real data inversion** (Priority 2) — `slip_inv_hetmu3D_*.py` and
   `slip_inv_hetmu3Dori_*.py`
6. **CHECK plotting notebooks** for Priority 2 — `plt_slip_inv_hetmu_nicoyaCK_coseis.ipynb`,
   `plt_slip_inv_hetmu_nicoyaCK_locking_both.ipynb`,
   `plt_syn_slip_az_inv_hetmu_nicoyaCK_locking.ipynb`,
   `plt_syn_slip_az_inv_hetmu_nicoya_locking.ipynb`,
   `plt_syn_slip_az_consistfwd_nicoya_locking_summary.ipynb`
7. **Group D** (Priority 3) — `slip_inv_hetmu1D_nicoyaCK_*.py`
8. **Decide**: `synth.py` — include or leave as-is

After completing all `.py` edits for a priority group, re-run the affected scripts on a small
test case before proceeding to the next group, to confirm output filenames are as expected.
