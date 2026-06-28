# Understanding Why Absolute Shear Modulus (μ) Doesn't Affect Ground Displacement

**Date:** 2025-12-03
**Code:** `synth_checkslip_inv_hetmu_nicoyaCK_lock_noi2.py`
**Function:** `solveCoseismicForward` (line 348)

## The Question

Why does changing the absolute value of shear modulus μ across the fault plane not change the ground displacement in the coseismic forward modeling?

## Quick Answer

Ground displacement is **insensitive to absolute changes in μ** because:

1. **Kinematic constraint dominates** - Fault slip is prescribed as a displacement boundary condition
2. **Scaling invariance** - Uniform scaling of μ preserves the displacement field geometry
3. **Relative μ contrast matters** - Only the ratio μ₁/μ₂ affects displacement, not absolute values

---

## Detailed Explanation

### 1. Kinematic Constraint Dominates

#### What's in the Code

The fault boundary condition (lines 340-341):
```python
- ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip),
            tau('+')*n('+'))*dS(fault)
```

This implements a **displacement jump boundary condition** across the fault:
- The displacement **must** jump by `m_strike` in the strike direction
- The displacement **must** jump by `m_dip` in the dip direction

#### Physical Interpretation

This is fundamentally different from a force-driven problem:

| Problem Type | What's Prescribed | What's Solved For |
|--------------|-------------------|-------------------|
| **Kinematic** (our case) | Displacement (fault slip) | Stress field |
| **Dynamic** | Force/stress | Displacement field |

In coseismic modeling, **slip is the cause** (prescribed), and surface displacement is the **effect** that results from static elastic deformation of the medium.

#### Why Absolute μ Doesn't Matter

Once you specify "these two sides of the fault must move X meters apart," the **geometry of deformation** is already determined. The material stiffness μ only affects:
- ✓ How much **stress** builds up
- ✓ How much **energy** is stored
- ✗ NOT the displacement pattern itself

**Analogy:** Pulling a spring vs. rubber band by the same distance - both stretch the same amount (what you control), but require different forces.

---

### 2. Scaling Invariance

#### Mathematical Demonstration

The governing equations consist of:

1. **Kinematic relation:** ε = sym(∇u) — purely geometric
2. **Constitutive relation:** ε = 1/(2μ) × σ' — material dependent
3. **Equilibrium:** div(σ) + f = 0 — force balance
4. **Fault BC:** [[u]] = m — prescribed slip

#### What Happens When μ → k·μ Everywhere

**Original problem:**
- Displacement field: **u**
- Strain field: **ε** = sym(∇**u**)
- Stress field: **σ**
- Constitutive: **ε** = [1/(2μ)] **σ'**

**Scaled problem (μ → k·μ):**
- Displacement field: **u** (SAME!)
- Strain field: **ε** (SAME! Purely geometric)
- Stress field: **σ_new** = k·**σ** (scales proportionally)
- Constitutive: **ε** = [1/(2k·μ)] (k·**σ'**) ✓ Still satisfied

**Check equilibrium:**
```
div(σ_new) = div(k·σ) = k·div(σ) = k·(-f)
```
If we also scale body forces, equilibrium is preserved!

For quasi-static problems where f ≈ 0, equilibrium is automatically satisfied.

#### Physical Intuition

Imagine a rubber sheet with a cut (fault):

1. Force the two sides to slide 1 meter (kinematic BC)
2. The sheet deforms to accommodate this slip
3. The **deformation pattern** is purely geometric

If you make the rubber 10× stiffer:
- ✓ Deformation pattern stays the same (same geometry)
- ✓ Internal stress becomes 10× larger
- ✓ Surface displacement is unchanged (same geometry)

#### In the Code

At line 469:
```python
pde.solveFwd(x[hp.STATE], x)
```

This solves for state variables (σ, u, r) given slip m. The displacement **u** is determined by:
- Fault slip constraint (prescribed)
- Compatibility (smooth deformation away from fault)
- **These are geometric constraints, independent of absolute μ**

The stress **σ** simply adjusts to maintain equilibrium with that displacement field.

---

### 3. Relative μ Contrast DOES Matter

#### Why Heterogeneity Is Important

While **absolute** μ values don't matter, **relative variations** (μ₁/μ₂) DO affect displacement because they control how deformation is **partitioned** between regions.

#### Example Scenario

Fault cutting through two materials:
- Left block: μ₁ = 30 GPa (stiff)
- Right block: μ₂ = 20 GPa (soft)
- Fault slip: 1 meter

**Homogeneous case (μ₁ = μ₂ = 30 GPa):**
- Deformation spreads symmetrically
- Surface displacement is symmetric around fault

**Heterogeneous case (μ₁ = 30 GPa, μ₂ = 20 GPa):**
- Softer material deforms MORE for the same stress
- Deformation is **asymmetric**
- Surface displacement pattern shifts toward soft side

#### The Physics: Stress Continuity at Interfaces

At material boundaries, **traction must be continuous** (Newton's 3rd law):
```
σ₁·n = σ₂·n
```

But constitutive law says:
```
σ₁ = 2μ₁·ε₁
σ₂ = 2μ₂·ε₂
```

Therefore:
```
2μ₁·ε₁·n = 2μ₂·ε₂·n
ε₁/ε₂ = μ₂/μ₁
```

**Strains must be inversely proportional to stiffness!**

Since ε = ∇u, different strains mean different displacement gradients, which means **the displacement field changes**.

#### Where This Appears in Code

Heterogeneous material definition (lines 200-214):
```python
class K_2LAYER(dl.UserExpression):
    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == blockright:
            values[0] = self.k_r
        elif self.subdomains[cell.index] == blockleft:
            values[0] = self.k_l
```

And shear modulus expression (lines 260-262):
```python
def mu_expression(m):
    mu = 20*(2.+ufl.tanh(m))
    return mu
```

#### Real-World Relevance

From the code header (line 1):
> "Compare how the pattern can be resolved within a **homogeneous or prescribed heterogeneous half-space**"

The research question: Can we **detect Earth's heterogeneous structure** by inverting surface displacement data?

The heterogeneity (μ contrasts) creates detectable differences in surface displacement patterns, even though absolute μ scale doesn't matter.

---

## Understanding Equilibrium

### What Is Equilibrium?

**Equilibrium = Balance of Forces**

In solid mechanics: every piece of material must have **zero net force** (Newton's 2nd Law for static problems).

#### The Equilibrium Equation

In the code (line 308):
```python
+ ufl.inner(ufl.div(sigma), w)*ufl.dx \
```

This implements:
```
div(σ) + f = 0
```

Where:
- **σ** = stress tensor (internal forces)
- **div(σ)** = how stress changes from point to point
- **f** = body force (gravity)

#### Physical Picture

Small cube of rock:
```
        ↑ σ_top
        |
   ←----□----→
        |
        ↓ σ_bottom + gravity
```

Forces must sum to zero, otherwise the material would accelerate.

### The Three Governing Equations

Complete elasticity problem:

#### 1. Kinematic (Compatibility)
```
ε = sym(∇u)
```
Strain is the symmetric gradient of displacement. **Purely geometric.**

#### 2. Constitutive (Material Law)
```python
# Line 165-167:
A = 1./(2.*mu)*( s - nu/( 1 + nu*(dim-2) )*ufl.tr(s)*ufl.Identity(dim) )
```
Relates stress to strain: ε = AEsigma(σ, μ, ν)

Depends on **material properties** (μ, ν).

#### 3. Equilibrium (Force Balance)
```python
# Line 308:
+ ufl.inner(ufl.div(sigma), w)*ufl.dx \
```
Ensures **forces balance** everywhere: div(σ) + f = 0

### How They Work Together in Kinematic Problems

**Given:** Fault slip boundary condition (displacement jump prescribed)

**Solution process:**
1. Guess displacement field satisfying fault BC
2. Compute strain: ε = sym(∇u) [kinematic]
3. Compute stress: σ = σ(ε, μ) [constitutive]
4. Check equilibrium: div(σ) + f = 0? [equilibrium]
5. If not balanced, adjust displacement and repeat

### Why Equilibrium Matters for the μ Question

#### When μ → k·μ everywhere:

1. Displacement field **u**: Determined by kinematic constraint
2. Strain field **ε** = sym(∇**u**): Purely geometric
3. Stress-strain: σ = 2μ·ε → if μ → k·μ, then σ → k·σ
4. Check equilibrium: div(k·σ) = k·div(σ) = 0 ✓

**Conclusion:** Same displacement field **u** still satisfies equilibrium!

#### When μ varies spatially (heterogeneous):

At material interfaces:
- **Traction continuity** (equilibrium): σ₁·n = σ₂·n
- But: σ₁ = 2μ₁·ε₁ and σ₂ = 2μ₂·ε₂
- Therefore: μ₁·ε₁ = μ₂·ε₂
- So strains must differ: ε₁/ε₂ = μ₂/μ₁

**The displacement field must change** to maintain equilibrium at interfaces!

### Key Insight

Equilibrium is a **linear constraint** in stress, so uniform scaling preserves it. But material interfaces create **coupling** between regions that breaks this scaling symmetry.

---

## Summary Table

| Change | Displacement Affected? | Reason |
|--------|------------------------|--------|
| μ → k·μ everywhere | ❌ NO | Kinematic constraint; uniform scaling |
| μ → different values in different regions | ✅ YES | Changes deformation partitioning |
| Increase absolute μ but keep ratios | ❌ NO | Only relative stiffness matters |
| Change μ₁/μ₂ ratio | ✅ YES | Changes strain distribution at interfaces |

---

## Key Code Locations

| Line | Component | Purpose |
|------|-----------|---------|
| 165-167 | `AEsigma()` | Compliance matrix (stress-strain relation) |
| 260-262 | `mu_expression()` | Shear modulus expression |
| 200-214 | `K_2LAYER` | Heterogeneous material properties |
| 321-343 | `PDEVarf` | Variational formulation |
| 333 | Constitutive term | Links stress to strain via μ |
| 308 | Equilibrium term | Force balance constraint |
| 340-341 | Fault BC | Prescribed displacement jump |
| 348-496 | `solveCoseismicForward()` | Forward problem solver |
| 469 | `pde.solveFwd()` | Solves for state variables |

---

## The Big Picture

In kinematically-driven problems (prescribed fault slip):

1. **Fault slip** (prescribed) → determines displacement geometry
2. **Displacement** → determines strain (kinematic)
3. **Strain + μ** → determines stress (constitutive)
4. **Stress** → must satisfy equilibrium

**The flow is unidirectional**: slip → displacement → strain → stress

Absolute μ scaling changes stress magnitude but not the displacement geometry, because displacement is **upstream** in the causal chain.

Only when μ varies spatially do the equilibrium constraints at interfaces **feed back** to affect the displacement field.

---

## Analogies

### Mobile Balancing
```
        ┌──── 🎨 (weight W₁)
    ────┤
        └──── 🎨 (weight W₂)
```

- **Uniform scaling** (W₁ → k·W₁, W₂ → k·W₂): angles stay the same, tension scales
- **Ratio change** (W₁/W₂ changes): angles must adjust to rebalance
- Like heterogeneous μ requiring displacement adjustment

### Spring vs. Rubber Band

Pull both by the same distance:
- Displacement: same (what you control)
- Force: different (spring needs more)
- Like uniform μ scaling: same displacement, different stress

---

## Research Implications

From this code's perspective:

**Question:** Can we invert surface GNSS displacement data to recover:
1. Fault slip distribution?
2. Earth's heterogeneous structure (μ variations)?

**Insight from this analysis:**
- Absolute μ calibration is impossible from displacement data alone
- But **relative** μ contrasts ARE observable
- The inversion can recover μ₁/μ₂ ratios, not absolute values
- This is why they compare "homogeneous or prescribed heterogeneous half-space" results

**Practical consequence:**
- Must constrain absolute μ from other data (lab measurements, seismic velocities)
- Displacement data constrains relative variations around that reference value
