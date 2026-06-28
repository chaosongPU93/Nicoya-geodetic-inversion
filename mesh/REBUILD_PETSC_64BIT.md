# Rebuild PETSc with 64-bit Integers for FEniCS

**Problem**: Your PETSc uses 32-bit integers, causing segfaults when mesh size exceeds ~30k nodes.

**Solution**: Rebuild PETSc with `--with-64-bit-indices=1`

---

## Option 1: New Environment (Recommended - Keeps Current Setup Intact)

### Step 1: Create new conda environment
```bash
conda create -n fenics64 python=3.12 -y
conda activate fenics64
```

### Step 2: Install MPI and compilers
```bash
conda install -c conda-forge mpich -y
# OR if you prefer OpenMPI:
# conda install -c conda-forge openmpi -y
```

### Step 3: Download and build PETSc with 64-bit integers
```bash
# Create build directory
mkdir -p ~/software
cd ~/software

# Download PETSc (use version compatible with your FEniCS)
wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.12.5.tar.gz
tar -xzf petsc-3.12.5.tar.gz
cd petsc-3.12.5

# Set environment variables
export PETSC_DIR=$PWD
export PETSC_ARCH=linux-gnu-c-opt-64bit

# Configure with 64-bit indices
./configure \
  --with-64-bit-indices=1 \
  --with-debugging=0 \
  --download-mumps \
  --download-scalapack \
  --download-parmetis \
  --download-metis \
  --download-hypre \
  --with-shared-libraries=1 \
  COPTFLAGS='-O3' \
  CXXOPTFLAGS='-O3' \
  FOPTFLAGS='-O3'

# Build (takes 10-30 minutes)
make PETSC_DIR=$PWD PETSC_ARCH=linux-gnu-c-opt-64bit all

# Run tests to verify
make PETSC_DIR=$PWD PETSC_ARCH=linux-gnu-c-opt-64bit check
```

### Step 4: Install petsc4py linked to your 64-bit PETSc
```bash
# Still in ~/software/petsc-3.12.5
export PETSC_DIR=$PWD
export PETSC_ARCH=linux-gnu-c-opt-64bit

pip install --no-cache-dir --no-binary :all: petsc4py
```

### Step 5: Install FEniCS and other dependencies
```bash
conda install -c conda-forge fenics -y
pip install hippylib
```

### Step 6: Verify 64-bit integers
```bash
python -c "from petsc4py import PETSc; print(f'PETSc Int Type: {PETSc.IntType}')"
```
**Expected output**: `PETSc Int Type: <class 'numpy.int64'>` ✓

### Step 7: Add to your ~/.bashrc (for persistent environment)
```bash
# Add these lines to ~/.bashrc
echo 'export PETSC_DIR=~/software/petsc-3.12.5' >> ~/.bashrc
echo 'export PETSC_ARCH=linux-gnu-c-opt-64bit' >> ~/.bashrc
```

---

## Option 2: Quick Test with Docker (No Installation Required)

If you want to quickly test if 64-bit PETSc solves your problem:

```bash
docker run -it --rm -v $(pwd):/work dolfinx/dolfinx:stable
# Inside container, PETSc has 64-bit integers by default
python -c "from petsc4py import PETSc; print(PETSc.IntType)"
```

---

## Option 3: Rebuild in Current Environment (Not Recommended)

**WARNING**: This modifies your working `fenics` environment. Back it up first:
```bash
conda create --clone fenics --name fenics_backup
```

Then follow Steps 3-6 from Option 1, but activate your `fenics` environment instead.

---

## Testing Your New Setup

Run this test script:
```python
from petsc4py import PETSc
import numpy as np

print(f"PETSc Int Type: {PETSc.IntType}")
assert PETSc.IntType == np.int64, "ERROR: Still using 32-bit integers!"
print("✓ Successfully using 64-bit PETSc integers")

# Test matrix creation
A = PETSc.Mat().createAIJ([100000, 100000])
print(f"✓ Can create large matrices (100k × 100k)")
```

---

## Common Issues

### Issue 1: "Cannot find MPI compiler"
**Solution**:
```bash
conda install -c conda-forge mpich
```

### Issue 2: petsc4py still shows int32
**Solution**: Make sure `PETSC_DIR` and `PETSC_ARCH` are set before installing petsc4py:
```bash
echo $PETSC_DIR  # Should show path to petsc-3.12.5
echo $PETSC_ARCH # Should show linux-gnu-c-opt-64bit
pip uninstall petsc4py
pip install --no-cache-dir --no-binary :all: petsc4py
```

### Issue 3: FEniCS can't find PETSc
**Solution**: Export variables before running Python:
```bash
export PETSC_DIR=~/software/petsc-3.12.5
export PETSC_ARCH=linux-gnu-c-opt-64bit
python your_script.py
```

---

## Performance Comparison

After rebuilding, you should be able to use:

| Mesh | Nodes | Status with int32 | Status with int64 |
|------|-------|-------------------|-------------------|
| dense2 | 28,767 | ✓ Works | ✓ Works |
| dense6 | 33,595 | ✗ SEGFAULT | ✓ Works |
| dense5 | 38,214 | ✗ SEGFAULT | ✓ Works |
| dense4 | 55,736 | ✗ SEGFAULT | ✓ Works |

---

## Estimated Time

- **Download PETSc**: 2 minutes
- **Configure PETSc**: 5 minutes
- **Build PETSc**: 15-30 minutes (depends on CPU)
- **Install petsc4py**: 3 minutes
- **Total**: ~30-45 minutes

---

## Alternative: Use Iterative Solver (Temporary Workaround)

If you can't rebuild PETSc immediately, try switching to an iterative solver in your code (line 772-775):

```python
# TEMPORARY WORKAROUND - change this:
# type_solver = "mumps"
# pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

# To this:
from petsc4py import PETSc
ksp = PETSc.KSP().create(mesh.mpi_comm())
ksp.setType('gmres')  # or 'cg' for symmetric problems
ksp.getPC().setType('hypre')  # or 'ilu', 'jacobi'
ksp.setTolerances(rtol=1e-9, atol=1e-12, max_it=1000)
# Then wrap this for hippylib use
```

This uses less memory but may be slower or less accurate than MUMPS.

---

**Questions?** Check PETSc installation docs: https://petsc.org/release/install/
