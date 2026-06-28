# Iterative Solver Workaround for 32-bit PETSc

## What Was Changed

Modified `testmesh_3DCK.py` to use **GMRES iterative solver** instead of **MUMPS direct solver**.

### Changes Made (2 locations):

1. **Line ~445-467** (in `solveCoseismicForward` function)
2. **Line ~790-813** (in `solveCoseismicInversion_TanhSlip` function)

**Old code:**
```python
type_solver = "mumps"
pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
```

**New code:**
```python
# TEMPORARY WORKAROUND: Use iterative solver to avoid 32-bit PETSc integer overflow
pde.solver = hp.PETScKrylovSolver(mesh.mpi_comm(), "gmres", "hypre_amg")
pde.solver.parameters["relative_tolerance"] = 1e-9
pde.solver.parameters["absolute_tolerance"] = 1e-12
pde.solver.parameters["maximum_iterations"] = 5000
pde.solver.parameters["monitor_convergence"] = False

pde.solver_fwd_inc = hp.PETScKrylovSolver(mesh.mpi_comm(), "gmres", "hypre_amg")
# ... (similar for fwd_inc and adj_inc)
```

## How to Test

### Test 1: Verify it runs without segfault on dense4

Edit `testmesh_3DCK.py` line 194 to use dense4:

```python
# Change from:
meshname = "nicoyaCK3_dense2_sm"

# To:
meshname = "nicoyaCK3_dense4_sm"  # This was crashing before!
```

Then run:
```bash
cd /home/staff/chao/SSEinv/Nicoya/codes
python testmesh_3DCK.py
```

**Expected behavior:**
- ✓ Should print warning about iterative solver
- ✓ Should NOT crash with segfault
- ✓ May take longer than MUMPS (iterative solvers are slower)
- ✓ Watch for convergence messages

### Test 2: Compare results with dense2

To verify the iterative solver gives similar results to MUMPS:

1. Run with dense2 (should work with both solvers)
2. Compare outputs between iterative and direct solver

## Performance Comparison

| Aspect | MUMPS (Direct) | GMRES (Iterative) |
|--------|----------------|-------------------|
| Memory | High (fills in during LU) | Lower |
| Speed | Fast (single solve) | Slower (many iterations) |
| 32-bit limit | Hits limit at ~30k nodes | No limit |
| Accuracy | Exact (within machine precision) | Approximate (controlled by tolerance) |

## Solver Parameters Explained

```python
pde.solver.parameters["relative_tolerance"] = 1e-9   # Stop when ||r|| < 1e-9 * ||b||
pde.solver.parameters["absolute_tolerance"] = 1e-12  # Stop when ||r|| < 1e-12
pde.solver.parameters["maximum_iterations"] = 5000   # Max iterations before giving up
pde.solver.parameters["monitor_convergence"] = False # Set True to see iteration progress
```

**Recommendations:**
- For debugging: Set `monitor_convergence = True` to watch convergence
- If solver doesn't converge, try:
  - Increase `maximum_iterations` to 10000
  - Relax tolerances to `1e-6` (faster but less accurate)
  - Try different preconditioner: `"ilu"` instead of `"hypre_amg"`

## Alternative Solvers to Try

If GMRES doesn't work well, try these alternatives:

### 1. BiCGStab (for non-symmetric problems)
```python
pde.solver = hp.PETScKrylovSolver(mesh.mpi_comm(), "bicgstab", "hypre_amg")
```

### 2. MINRES (for symmetric problems)
```python
pde.solver = hp.PETScKrylovSolver(mesh.mpi_comm(), "minres", "hypre_amg")
```

### 3. CG with better preconditioner
```python
pde.solver = hp.PETScKrylovSolver(mesh.mpi_comm(), "cg", "hypre_amg")
```

## Known Limitations

1. **Slower**: Iterative solvers take 2-10x longer than MUMPS
2. **Convergence**: May not converge for ill-conditioned problems
3. **Accuracy**: Slightly less accurate than direct solvers (but usually negligible)

## When This Workaround Fails

If the iterative solver:
- **Doesn't converge**: Problem is too ill-conditioned → need 64-bit PETSc
- **Takes too long**: >1 hour for single solve → need 64-bit PETSc with MUMPS
- **Gives wrong results**: Check tolerances and preconditioner

## Reverting to MUMPS

To revert back to MUMPS (if you rebuild PETSc with 64-bit):

```bash
# In testmesh_3DCK.py, replace iterative solver blocks with:
type_solver = "mumps"
pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
```

## Next Steps

1. **Test with dense4** - Verify no segfault
2. **Check convergence** - Monitor iteration counts
3. **Compare results** - Verify accuracy vs MUMPS on dense2
4. **If satisfactory** - Use for production runs
5. **If not satisfactory** - Follow `REBUILD_PETSC_64BIT.md` for permanent fix
