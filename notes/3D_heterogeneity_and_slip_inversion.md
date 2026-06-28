# 3D Heterogeneity Effects on Ground Displacement and Slip Inversion

**Date:** 2025-12-03
**Code:** `synth_checkslip_inv_hetmu3D_nicoyaCK_lock_noi.py`
**Context:** Forward modeling with 3D μ structure, inversion with correct vs. homogeneous structure

## The Setup

**Forward modeling:**
- 3D shear modulus field μ(x,y,z) from 3D seismic velocity model (μ = ρ·Vs²)
- Heterogeneity exists:
  - Across the fault plane (fault-parallel contrast)
  - Within each block (blockleft and blockright)
  - At all depths
- Prescribed slip pattern on fault
- Generate synthetic surface displacement

**Inversion (two approaches):**
1. **Correct structure:** Use true 3D μ(x,y,z)
2. **Homogeneous structure:** Use constant μ₀ (ignoring heterogeneity)

**Both inversions use the same synthetic surface displacement data**

## The Two Questions

### Question 1
**How does ground displacement correspond to the μ structure?**
- Does it only rely on μ at the ground surface?
- Or is it a compound effect from the entire 3D volume?

### Question 2
**How does the difference in inferred slip relate to the μ structure?**
- Does it only depend on μ contrast across the fault plane?
- Or is it a compound effect from the entire 3D μ field?

---

## Question 1: Ground Displacement and μ Structure

### Short Answer

Ground displacement is a **volume-integrated compound effect** of the entire 3D μ field, NOT just surface μ. Every point in the volume contributes to surface displacement through the solution of the static equilibrium equations.

### Detailed Explanation

#### The Physics: Green's Functions

Surface displacement at point **x_obs** due to slip on fault element at **x_src** is given by:

```
u(x_obs) = ∫∫_fault G(x_obs, x_src; μ(x,y,z)) · s(x_src) dA
```

Where:
- **G** = elastic Green's function (displacement at x_obs due to unit point force at x_src)
- **s** = slip vector
- **μ(x,y,z)** = 3D shear modulus field
- The integral is over the fault surface

**Critical point:** The Green's function **G** depends on the **ENTIRE 3D μ field**, not just values at x_obs or x_src!

#### Why the Entire Volume Matters

The static Green's function couples all spatial points through the equilibrium equations. This is NOT wave propagation - it's **spatial coupling through an elliptic PDE**.

**Physical processes involved:**

1. **Global equilibrium constraint**
   - The equation ∇·σ = 0 must be satisfied everywhere simultaneously
   - This creates non-local coupling between all points in the domain
   - Like solving Laplace's equation - changes anywhere affect solution everywhere

2. **Strain partitioning**
   - Fault slip creates displacement discontinuity
   - Surrounding material must deform to accommodate this
   - How strain distributes depends on μ(x,y,z) throughout the volume

3. **Interface conditions**
   - At μ boundaries, traction must be continuous: σ₁·n = σ₂·n
   - But constitutive law relates σ to ε through μ
   - This forces strain discontinuities: ε₁/ε₂ = μ₂/μ₁
   - Displacement remains continuous but gradients change

4. **Stress redistribution**
   - Elastic equilibrium requires force balance everywhere
   - Heterogeneities cause stress concentrations/reductions
   - These redistribute throughout the entire volume via equilibrium constraints
   - Like how pushing on a sponge deforms distant points

#### Mathematical Formulation

The static Green's function satisfies an **elliptic PDE** (NOT a wave equation):

```
∇·[C(x,y,z) : ∇G] + δ(x - x_src) = 0
```

Where:
- **C(x,y,z)** = 4th-order elasticity tensor (depends on μ(x,y,z) and ν)
- This is an **elliptic PDE** like Laplace's equation
- No time derivatives ∂²/∂t² - this is static, not dynamic
- The solution **G** couples all spatial points through C

**Key property of elliptic PDEs:**
- Changes anywhere affect the solution everywhere (non-local)
- Influence decays with distance but never exactly zero
- Like steady-state heat conduction or electrostatics

In weak form (like your code):

```
∫_V [C(x,y,z) : ∇G] : ∇φ dV = φ(x_src)
```

The integral is over the **entire volume V**, so μ everywhere contributes to the solution.

#### Depth Sensitivity

Surface displacement has **different sensitivities** to μ at different depths:

**Near-surface μ (0-5 km):**
- ✓ HIGH sensitivity
- Displacement field makes sharp gradients near surface
- Small volume but large strain

**Mid-depth μ (5-30 km, including fault zone):**
- ✓✓✓ HIGHEST sensitivity
- Where the fault slip occurs
- Largest deformation
- Controls primary displacement pattern

**Deep μ (>30 km):**
- ✓ MODERATE sensitivity
- Affects long-wavelength components
- Influences regional displacement field
- Like "background" elastic properties

**Analogy:** Like pressing down on a layered cake - the surface deformation depends on the stiffness of ALL layers, not just the top layer. Each layer's resistance contributes to how the surface deforms.

#### Lateral Sensitivity

Surface displacement is also sensitive to **lateral variations** in μ:

**On-fault vs. off-fault:**
- ✓✓✓ Highest sensitivity to μ immediately adjacent to fault
- Strain is largest near slip source
- But off-fault heterogeneities also matter!

**Near-field vs. far-field:**
- Near stations (< 50 km from fault): sensitive to local μ structure
- Far stations (> 100 km): sensitive to regional μ structure
- Different stations "see" different parts of the μ field

#### Practical Example

Consider three scenarios with identical slip:

**Scenario A: Homogeneous**
```
Surface: ====================================
         μ = 30 GPa everywhere
Fault:   ========[SLIP]==================
```
→ Symmetric displacement, smooth decay with distance

**Scenario B: Surface anomaly only**
```
Surface: ====[μ=15 GPa]====================  ← Low μ near surface
         μ = 30 GPa below
Fault:   ========[SLIP]==================
```
→ Enhanced displacement at surface (softer material)
→ But pattern still relatively symmetric

**Scenario C: 3D heterogeneity**
```
Surface: ====================================
         [μ varies in 3D, 15-45 GPa]
Fault:   ===[μ₁]==[SLIP]===[μ₂]=========
```
→ Asymmetric displacement
→ Complex spatial pattern
→ Different decay rates in different directions
→ Influenced by μ along all paths

**Observation:** Only Scenario C produces realistic complexity matching real data!

#### Why Surface μ Alone Is Insufficient

If only surface μ mattered, you could have:
- Deep fault slip (30 km depth)
- Strong heterogeneity at depth
- But identical surface displacement for any subsurface μ distribution

**This is physically wrong!** Static elasticity theory shows:
- Deep heterogeneity affects surface displacement because equilibrium couples all depths
- The elliptic nature of the governing equations makes the problem non-local
- Numerical experiments confirm: changing deep μ measurably changes surface displacement
- This is why geodetic inversions must account for Earth's 3D structure

---

## Question 2: Slip Inversion Trade-offs with μ Structure

### Short Answer

The difference in inferred slip between correct and homogeneous inversions is a **compound effect** involving:
1. μ contrast across the fault (most important)
2. μ distribution in the fault zone (very important)
3. μ heterogeneity throughout the entire volume (important)

It's NOT just the fault-plane contrast—the full 3D μ field matters.

### Detailed Explanation

#### The Inverse Problem

You're solving:

```
d_obs = G(μ_true) · m_true + noise
```

Where:
- **d_obs** = observed surface displacement
- **G(μ_true)** = forward operator (Green's function) with true μ
- **m_true** = true slip distribution

**Inversion attempt 1 (correct structure):**
```
m_inv1 = argmin ||d_obs - G(μ_true)·m||²
```
→ Should recover m ≈ m_true (if well-posed)

**Inversion attempt 2 (homogeneous structure):**
```
m_inv2 = argmin ||d_obs - G(μ_hom)·m||²
```
→ Will get biased estimate because **G(μ_hom) ≠ G(μ_true)**

#### The Structure Trade-off

The key insight: **Slip and structure trade-off**

When you use wrong μ structure, the inversion compensates by adjusting the slip distribution. This is because:

```
G(μ_true) · m_true ≈ G(μ_hom) · m_inv2
```

The homogeneous Green's function is trying to produce the same surface displacement, but with wrong physics. The slip estimate must be biased to compensate.

#### Where Do Trade-offs Occur?

**1. Fault-Plane μ Contrast (Most Critical)**

Consider fault with:
- Hanging wall (left): μ₁ = 25 GPa
- Footwall (right): μ₂ = 35 GPa

**True model:** Softer hanging wall deforms more → asymmetric displacement

**Homogeneous inversion (μ_hom = 30 GPa):**
- Predicts symmetric deformation
- To match asymmetric data, must infer more slip on left side of fault
- **Bias:** Overestimates slip where μ is actually lower

**Effect:** ✓✓✓ **Largest bias** - typically 10-30% error in slip amplitude

**2. Along-Fault μ Variations (Very Important)**

Consider fault with μ varying along strike or downdip:
- Shallow fault zone: μ = 20 GPa (sediments)
- Deep fault zone: μ = 40 GPa (crystalline rock)

**True model:**
- Same slip at shallow and deep generates different surface displacement
- Shallow slip produces more displacement (softer material)

**Homogeneous inversion:**
- Assumes constant μ everywhere
- Interprets larger displacement as larger slip
- **Bias:** Overestimates shallow slip, underestimates deep slip

**Effect:** ✓✓✓ **Large bias** - can flip depth distribution

**3. Near-Fault Volume μ (Important)**

The volume immediately surrounding the fault (±10 km) strongly influences:
- How strain localizes near the fault
- Amplitude of near-field displacement
- Gradient of displacement field

**Homogeneous inversion:**
- Wrong strain partitioning in near-fault zone
- **Bias:** Can shift inferred slip location laterally or vertically

**Effect:** ✓✓ **Moderate bias** - typically affects fine-scale features

**4. Far-Field μ (Moderate Effect)**

Regional heterogeneity far from fault:
- Affects long-wavelength displacement pattern
- Controls far-field station observations
- Less critical than near-fault structure

**Homogeneous inversion:**
- Wrong regional displacement pattern
- **Bias:** Affects total moment (slip integral)

**Effect:** ✓ **Smaller bias** - typically 5-15% in total moment

#### Trade-off Patterns: Specific Examples

**Example 1: Low-velocity zone on one side**

```
Fault cross-section:
        |
  μ=30  |  μ=20   ← Low-velocity zone on right
  GPa   |  GPa
        |
    [Uniform slip = 1 m]
```

**True forward:** Right side deforms more → displacement asymmetry

**Homogeneous inversion (μ=25 GPa):**
- To match stronger deformation on right, infers more slip on right
- **Recovered slip:** 0.8 m left, 1.2 m right
- **Error:** 20% left, +20% right

**Example 2: Shallow sediments**

```
Depth profile:
  0 km ────  μ = 10 GPa (sediments)
  5 km ────  μ = 30 GPa (crystalline)
 20 km ────  μ = 40 GPa (mantle)
        |
    [Uniform slip]
```

**True forward:** Shallow slip produces 3× more surface displacement

**Homogeneous inversion (μ=30 GPa):**
- Interprets large displacement as large slip
- **Recovered slip profile:**
  - Shallow: 3× overestimated
  - Deep: underestimated or negative (unphysical!)

**Example 3: 3D wedge structure**

```
Subduction zone:
    ╱╲ Forearc wedge (low μ)
   ╱  ╲
  ╱____╲ Slab interface (slip)
```

**True structure:** Wedge deforms easily, amplifies displacement

**Homogeneous inversion:**
- No wedge amplification in forward model
- Compensates by increasing slip estimate
- **Bias:** Overestimates slip magnitude by 20-50%

#### The Mathematics of Trade-off

The inversion solves:

```
m_inv = [G^T G + λR]^(-1) G^T d_obs
```

Where:
- **G** = forward operator (depends on μ)
- **R** = regularization
- **λ** = regularization parameter

When G is wrong (G_hom instead of G_true):

```
m_hom = [G_hom^T G_hom + λR]^(-1) G_hom^T d_obs
```

But d_obs = G_true · m_true, so:

```
m_hom = [G_hom^T G_hom + λR]^(-1) G_hom^T G_true m_true
       = [stuff]^(-1) G_hom^T G_true m_true
```

**Key term:** G_hom^T G_true is the "structure trade-off matrix"

If G_hom = G_true: this becomes identity → perfect recovery

If G_hom ≠ G_true: this creates bias that depends on:
1. Where μ_hom differs from μ_true
2. The sensitivity kernel relating each slip patch to each station
3. The geometry of station coverage

#### Depth-Dependent Trade-off Kernels

Different parts of the μ field affect different aspects of slip recovery:

**Surface μ (0-2 km):**
- ✓ Affects slip amplitude scaling
- Does not strongly affect slip pattern shape
- Like a "gain" factor on displacement

**Shallow fault zone μ (2-15 km):**
- ✓✓✓ Strongly affects shallow slip distribution
- Critical for tsunami/near-field hazard
- High leverage on inversion

**Deep fault zone μ (15-40 km):**
- ✓✓ Affects deep slip and moment magnitude
- Important for long-period ground motion
- Moderate leverage

**Off-fault μ (laterally away from fault):**
- ✓ Affects slip pattern shape and smoothness
- Can create artifacts in slip distribution
- Subtle but non-negligible

#### Resolution Matrix Analysis

The resolution matrix shows how well each slip patch is recovered:

```
R = [G^T G + λR_reg]^(-1) G^T G
```

**Ideal case:** R = I (perfect resolution)

**With wrong structure:** R becomes non-diagonal
- Off-diagonal elements = cross-talk between slip patches
- Diagonal elements < 1 = damping/bias
- Pattern depends on μ errors

**Example patterns:**

| μ Error Location | Resolution Effect |
|------------------|-------------------|
| Across fault | Lateral smearing of slip |
| With depth | Vertical smearing of slip |
| Asymmetric | Biased slip location |
| Underestimated μ | Underestimated slip amplitude |
| Overestimated μ | Overestimated slip amplitude |

#### Practical Implications

**For your experiment (3D vs. homogeneous):**

You will likely observe:

1. **Slip amplitude differences**
   - Homogeneous inversion: biased amplitudes
   - Bias pattern correlates with μ anomalies
   - Where μ_true < μ_hom: slip overestimated
   - Where μ_true > μ_hom: slip underestimated

2. **Slip pattern distortion**
   - Sharp gradients in μ → artifacts in slip
   - Smooth slip becomes patchy, or vice versa
   - Asymmetry in recovered slip pattern

3. **Depth distribution errors**
   - Shallow/deep partitioning wrong
   - Peak slip depth may shift
   - Downdip extent biased

4. **Total moment bias**
   - Integrated slip × μ may be off
   - Compensating errors possible
   - Typically 10-30% error in moment

**Diagnostic:** Plot slip_hom / slip_3D vs. μ_true
- Expect anti-correlation
- Where μ is low, slip ratio > 1 (overestimated)
- Where μ is high, slip ratio < 1 (underestimated)

---

## Unified Picture: The Complete Trade-off

### Forward Problem (from previous document)

```
Given: slip m, structure μ(x,y,z)
Solve: displacement u
```

- u depends on ENTIRE μ field (volume-integrated)
- Kinematic constraint fixes geometry
- Absolute μ scaling doesn't affect u
- Relative μ contrasts DO affect u

### Inverse Problem (this document)

```
Given: displacement d, structure μ(x,y,z)
Solve: slip m
```

- If μ is correct: recover true m
- If μ is wrong: recover biased m
- Bias depends on μ errors everywhere (volume-integrated)
- But strongest dependence on near-fault μ

### The Asymmetry

**Forward:** uniform μ scaling → no change in u
**Inverse:** uniform μ scaling → changes estimated m!

**Why the asymmetry?**

In forward problem:
- Slip is prescribed (kinematic BC)
- μ determines stress, not displacement

In inverse problem:
- Displacement is given (data constraint)
- Must infer slip using assumed μ
- If assumed μ is wrong (even uniformly scaled), the slip estimate changes to compensate

**Example:**

True model: m = 1 m, μ = 30 GPa → u = 0.1 m

Inversion with μ = 15 GPa:
- Forward model: m=1 m, μ=15 GPa → u = 0.2 m (too large)
- To match u = 0.1 m, must reduce m
- **Recovered:** m = 0.5 m (50% error!)

Even though uniform μ scaling doesn't change forward displacement (when slip is prescribed), it DOES affect inverse slip (when displacement is prescribed).

---

## Summary Tables

### Question 1: Displacement Dependence on μ

| μ Location | Influence on Surface Displacement | Physical Mechanism |
|------------|-----------------------------------|-------------------|
| Surface (0-2 km) | High | Large strain concentration near surface |
| Fault zone (fault ± 5 km) | Very High | Primary deformation zone |
| Near-fault volume (5-30 km) | High | Strain partitioning through equilibrium coupling |
| Deep (> 30 km) | Moderate | Long-wavelength background deformation |
| Far-field lateral | Moderate | Regional displacement pattern |

**Answer:** Surface displacement is a **compound effect** of the entire 3D μ field, with highest sensitivity to the fault zone and near-surface regions.

### Question 2: Slip Bias Dependence on μ Errors

| μ Error Location | Effect on Inferred Slip | Typical Bias Magnitude |
|------------------|------------------------|------------------------|
| Fault-plane contrast | Slip amplitude and asymmetry | 10-30% |
| Along-fault variation | Depth/lateral distribution | 20-50% |
| Near-fault volume (< 10 km) | Slip location and fine structure | 10-20% |
| Far-field structure | Total moment and regional pattern | 5-15% |

**Answer:** Slip bias is a **compound effect** of μ errors throughout the volume, with strongest dependence on:
1. μ contrast across fault
2. μ distribution along fault
3. Near-fault volume μ
4. Regional μ structure (to lesser extent)

---

## Key Insights

### 1. Non-local Effects Dominate

Both displacement and slip inversion are fundamentally **non-local** problems:
- Every point in the μ field affects every displacement observation
- Errors in μ anywhere can bias slip estimates everywhere
- Cannot isolate effects to single interfaces or boundaries

### 2. Sensitivity Hierarchy

There IS a hierarchy of importance:
```
Fault zone μ  >  Near-fault μ  >  Regional μ  >  Far-field μ
```

But even the "least important" far-field structure contributes measurably.

### 3. Different Problems, Different Sensitivities

**Forward problem (slip → displacement):**
- Most sensitive to μ contrasts (ratios)
- Insensitive to absolute μ scaling
- Controlled by kinematic boundary conditions

**Inverse problem (displacement → slip):**
- Sensitive to both absolute and relative μ values
- Strongly dependent on assumed μ structure
- Controlled by data fit and regularization

### 4. Why This Matters for Your Research

**Goal:** Infer earthquake slip from GPS/GNSS surface displacement

**Challenge:** Earth's μ structure is uncertain

**Your experiment tests:**
- How much does μ uncertainty affect slip estimates?
- Can we invert slip reliably with imperfect structure knowledge?

**Expected result:**
- 3D inversion: more accurate slip, consistent with "true" model
- Homogeneous inversion: biased slip, but may still capture first-order pattern
- Difference map reveals sensitivity to structure

**Practical lesson:**
- Must incorporate realistic μ structure in operational inversions
- Ignoring heterogeneity → systematic errors in hazard assessment
- Joint inversion of slip + structure may be necessary for precision

---

## Recommendations for Analysis

### Diagnostic Plots to Create

1. **Slip difference map**
   ```
   Δm = m_hom - m_3D
   ```
   Shows where homogeneous assumption fails

2. **Correlation with μ anomalies**
   ```
   Plot: Δm/m_3D vs. (μ_3D - μ_hom)/μ_hom
   ```
   Should show anti-correlation (bias compensates for μ error)

3. **Depth profiles**
   ```
   Compare slip(depth) for both inversions
   ```
   Reveals depth-dependent trade-offs

4. **Moment tensor comparison**
   ```
   M₀_hom vs. M₀_3D
   ```
   Total moment may differ despite fitting same data

5. **Resolution matrices**
   ```
   R_hom vs. R_3D
   ```
   Shows how structure affects slip resolution

### Quantitative Metrics

1. **RMS slip difference:**
   ```
   RMS = sqrt(mean((m_hom - m_3D)²))
   ```

2. **Normalized bias:**
   ```
   Bias = (M₀_hom - M₀_3D) / M₀_3D
   ```

3. **Pattern correlation:**
   ```
   ρ = corr(m_hom, m_3D)
   ```
   Tests if spatial pattern is preserved

4. **Depth centroid shift:**
   ```
   Δz_center = z_hom - z_3D
   ```
   Common systematic error

### Physical Interpretation

For each difference observed:
1. Identify corresponding μ anomaly
2. Check if bias direction matches expected trade-off
3. Quantify sensitivity kernel
4. Assess implications for hazard assessment

This analysis will reveal:
- Which parts of slip distribution are robust to structure uncertainty
- Which parts require accurate μ knowledge
- Whether simplified (homogeneous) models are sufficient for operational use

---

## Connection to Previous Document

**Previous document conclusion:**
> Absolute μ scaling doesn't affect forward displacement (kinematic constraint)

**This document extends:**
> But absolute and relative μ both affect inverse slip estimates (data constraint)

The forward problem is kinematically driven (slip prescribed).
The inverse problem is data driven (displacement prescribed).

This fundamental difference explains why:
- Forward: only μ ratios matter
- Inverse: both absolute values and ratios matter

Together, these documents provide complete picture of μ-slip-displacement relationships in elastic half-space problems.

---

## Important Clarification: Static vs. Dynamic Problems

### This is a STATIC (Coseismic) Problem

**What we're modeling:**
- GPS/GNSS displacement measured days/weeks after earthquake
- Final, equilibrated displacement field
- Time-independent, steady-state solution

**Governing equations:**
- **Equilibrium:** ∇·σ = 0 (no acceleration terms)
- **Compatibility:** ε = sym(∇u)
- **Constitutive:** σ = C:ε (depends on μ)
- Together form an **elliptic PDE** (like Laplace's equation)

**NO wave propagation:**
- No ∂²u/∂t² terms
- No seismic waves
- No ray paths, refraction, or scattering in the wave sense
- Just static elastic deformation

### How Deformation "Spreads" (Not Waves!)

When we say deformation "spreads through the volume," we mean:

**Correct understanding:**
- Solving an elliptic PDE that couples all spatial points
- Like how temperature spreads in steady-state heat conduction
- Or how electric potential distributes in electrostatics
- Non-local coupling through equilibrium constraints

**Incorrect understanding:** ❌
- Seismic waves propagating through medium
- Time-dependent wave fronts
- Ray theory or wave refraction

### The Analogy: Sponge, Not Drum

**Good analogy:** Compressed sponge
- Push your finger in (fault slip)
- Surface deforms everywhere immediately
- No waves traveling through
- Deformation pattern depends on sponge's varying stiffness
- This is STATIC elasticity

**Bad analogy:** ❌ Drum being struck
- Hit the drum (earthquake)
- Waves travel across membrane
- Time-dependent oscillations
- This is DYNAMIC elasticity (NOT your problem)

### Why the Entire Volume Still Matters

Even though there are no waves:
- Elliptic PDEs have **infinite-speed influence**
- Changing μ anywhere instantly affects equilibrium solution everywhere
- Influence decays exponentially with distance
- But coupling is through global equilibrium, not local wave propagation

**Mathematical reason:**
```
∇·[μ(x,y,z)·(∇u + ∇u^T)] = 0
```

This equation couples ALL spatial points through μ(x,y,z). The solution at any point depends on μ everywhere, with exponentially decaying sensitivity.

### Static Green's Functions vs. Dynamic Green's Functions

**Static Green's functions** (your problem):
- Solution to elliptic PDE
- Time-independent
- Represent instantaneous equilibrium response
- Used for: coseismic GPS, postseismic, tectonic loading

**Dynamic Green's functions** (NOT your problem):
- Solution to wave equation (hyperbolic PDE)
- Time-dependent
- Represent wave propagation
- Used for: seismograms, strong ground motion, seismic tomography

Both are called "Green's functions" but fundamentally different physics!

### Summary

✅ **Your problem IS:**
- Static equilibrium
- Elliptic PDE
- Time-independent
- Global spatial coupling
- Like deforming a heterogeneous elastic solid

❌ **Your problem is NOT:**
- Wave propagation
- Dynamic rupture
- Time-dependent
- Ray paths or wave refraction
- Seismic waves

The 3D μ field matters for **equilibrium reasons**, not wave propagation reasons!
