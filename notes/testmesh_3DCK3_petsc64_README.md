# testmesh_3DCK3_petsc64.py - Usage Instructions

## Purpose

This is a modified version of `testmesh_3DCK3.py` configured to use **64-bit PETSc** with **MUMPS direct solver**.

**Differences from original:**
1. ✓ Requires fenics64 environment with 64-bit PETSc
2. ✓ Uses MUMPS direct solver (fast, accurate)
3. ✓ Can handle large meshes: dense4, dense5, dense6
4. ✓ Automatically verifies 64-bit PETSc at startup
5. ✓ Will fail with clear error message if 32-bit PETSc detected

---

## How to Run

### Step 1: Activate fenics64 environment
```bash
conda activate fenics64
```

### Step 2: Set PETSc environment variables
```bash
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit
```

**Note**: These variables must be set BEFORE running the script!

### Step 3: Run the script
```bash
cd /home/staff/chao/SSEinv/Nicoya/codes
python testmesh_3DCK3_petsc64.py
```

---

## Expected Output on Startup

```
================================================================================
VERIFYING 64-BIT PETSC CONFIGURATION
================================================================================
PETSc Integer Type: <class 'numpy.int64'>
PETSc Version: (3, 20, 1)
✓ SUCCESS: Using 64-bit PETSc integers
  This script can handle large meshes (dense4, dense5, dense6)
================================================================================

################################################################################
 START of inversion
################################################################################
...
================================================================================
Using MUMPS direct solver with 64-bit PETSc
This version can handle large meshes (dense4, dense5, dense6)
================================================================================
```

---

## Troubleshooting

### Error: "32-bit PETSc detected! Need 64-bit PETSc for this script."

**Cause**: You're running with the wrong conda environment or PETSc installation.

**Solution**:
```bash
# 1. Check current environment
conda info --envs

# 2. Activate correct environment
conda activate fenics64

# 3. Set environment variables
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit

# 4. Verify
python -c "from petsc4py import PETSc; print(f'PETSc Int: {PETSc.IntType}')"
# Should show: PETSc Int: <class 'numpy.int64'>
```

### Error: "Cannot import petsc4py"

**Cause**: The fenics64 environment doesn't have petsc4py installed yet.

**Solution**: Complete the 64-bit PETSc build process first (see `REBUILD_PLAN_STEP_BY_STEP.md`)

---

## Comparison with testmesh_3DCK.py (Iterative Solver Version)

| Aspect | testmesh_3DCK.py (iterative) | testmesh_3DCK3_petsc64.py (64-bit) |
|--------|------------------------------|-------------------------------------|
| Environment | fenics (32-bit PETSc) | fenics64 (64-bit PETSc) |
| Solver | GMRES iterative | MUMPS direct |
| Speed | Slower (many iterations) | Fast (single solve) |
| Memory | Lower | Higher (but manageable) |
| Accuracy | Approximate | Exact |
| Max mesh | dense2 (~30k nodes) | dense4/5/6 (>50k nodes) |
| Status | **Works now** | **Requires 64-bit PETSc build** |

---

## When to Use Each Version

### Use testmesh_3DCK.py (iterative solver):
- ✓ Need to run RIGHT NOW without rebuild
- ✓ Using smaller meshes (dense2 or nicoyaCK3)
- ✓ Don't mind slower solve times
- ✓ Current fenics environment

### Use testmesh_3DCK3_petsc64.py (64-bit):
- ✓ Need to run large meshes (dense4, dense5, dense6)
- ✓ Want fast, accurate MUMPS solver
- ✓ After completing 64-bit PETSc rebuild
- ✓ fenics64 environment ready

---

## Quick Setup Script

Save this as `run_petsc64.sh`:

```bash
#!/bin/bash
# Quick setup script for running with 64-bit PETSc

# Activate fenics64 environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fenics64

# Set PETSc paths
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit

# Verify
echo "Checking PETSc configuration..."
python -c "from petsc4py import PETSc; print(f'PETSc Int Type: {PETSc.IntType}')"

# Run script
echo "Running testmesh_3DCK3_petsc64.py..."
python testmesh_3DCK3_petsc64.py "$@"
```

Then use:
```bash
chmod +x run_petsc64.sh
./run_petsc64.sh
```

---

## Status: Awaiting 64-bit PETSc Build

**Current Status**: This script is ready but requires 64-bit PETSc to be built first.

**Next Steps**:
1. Complete PETSc 3.20.1 build with `--with-64-bit-indices=1` (in progress)
2. Install petsc4py linked to 64-bit PETSc
3. Test verification script passes
4. Run this script with dense4/5/6 meshes

---

**Created**: 2025-11-29
**Based on**: testmesh_3DCK3.py
**Purpose**: Enable large mesh runs with 64-bit PETSc and MUMPS
