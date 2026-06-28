# Step-by-Step Plan to Rebuild PETSc with 64-bit Integers

**Goal**: Create `fenics64` environment identical to `fenics` except with 64-bit PETSc

**Current Environment Analysis:**
- **Environment**: `fenics` at `/home/staff/chao/anaconda3/envs/fenics`
- **Python**: 3.12.3
- **FEniCS**: 2019.1.0
- **PETSc**: 3.20.1 (32-bit integers) ← TO BE REPLACED
- **petsc4py**: 3.20.1 (32-bit) ← TO BE REPLACED
- **SLEPc**: 3.20.1 (32-bit) ← TO BE REPLACED
- **slepc4py**: 3.20.0 (32-bit) ← TO BE REPLACED
- **MPI**: mpich 4.2.3
- **MUMPS**: 5.2.1 (OK, will work with new PETSc)
- **Total packages**: 296 packages

---

## STRATEGY: Clone → Remove PETSc → Rebuild with 64-bit → Reinstall

This approach is safest because:
1. Preserves all package versions exactly
2. Only touches PETSc-related packages
3. Keeps your working `fenics` environment intact

---

## STEP 1: Clone the current fenics environment

**Command:**
```bash
conda create --clone fenics --name fenics64
```

**Expected time**: 5-10 minutes
**Expected output**: Creates exact copy with all 296 packages

---

## STEP 2: Activate new environment and verify

**Commands:**
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fenics64
python -c "from petsc4py import PETSc; print(f'PETSc: {PETSc.Sys.getVersion()}, Int: {PETSc.IntType}')"
```

**Expected output:**
```
PETSc: (3, 20, 1), Int: <class 'numpy.int32'>
```

---

## STEP 3: Remove only PETSc-related packages

**Commands:**
```bash
conda activate fenics64
conda remove --force petsc petsc4py slepc slepc4py --yes
```

**What this does:**
- Removes: petsc, petsc4py, slepc, slepc4py (4 packages)
- Keeps: All other 292 packages unchanged
- `--force` prevents removing dependencies

**Expected time**: 1 minute

---

## STEP 4: Build PETSc 3.20.1 with 64-bit integers

**Setup:**
```bash
# Create build directory
mkdir -p ~/software/petsc_64bit
cd ~/software/petsc_64bit

# Download exact version currently used
wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.20.1.tar.gz
tar -xzf petsc-3.20.1.tar.gz
cd petsc-3.20.1
```

**Configure:**
```bash
export PETSC_DIR=$PWD
export PETSC_ARCH=linux-gnu-c-opt-64bit

./configure \
  --with-64-bit-indices=1 \
  --with-debugging=0 \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpif90 \
  --download-mumps \
  --download-scalapack \
  --download-parmetis \
  --download-metis \
  --download-hypre \
  --with-shared-libraries=1 \
  COPTFLAGS='-O3 -march=native' \
  CXXOPTFLAGS='-O3 -march=native' \
  FOPTFLAGS='-O3 -march=native'
```

**Key configuration options explained:**
- `--with-64-bit-indices=1` ← **THE CRITICAL FLAG**
- `--with-debugging=0` → Optimized build (matches current)
- `--download-mumps` → Includes MUMPS (you're using this)
- `--download-hypre` → Includes hypre (for iterative solvers)
- `--with-shared-libraries=1` → Required for Python bindings

**Expected time**: 3-5 minutes
**Expected output**: Configuration summary showing `sizeof(PetscInt) = 8`

**Build:**
```bash
make PETSC_DIR=$PWD PETSC_ARCH=linux-gnu-c-opt-64bit all
```

**Expected time**: 20-40 minutes (depends on CPU cores)

**Verify:**
```bash
make PETSC_DIR=$PWD PETSC_ARCH=linux-gnu-c-opt-64bit check
```

**Expected output**: All tests pass

---

## STEP 5: Install petsc4py 3.20.1 with 64-bit PETSc

**Activate environment:**
```bash
conda activate fenics64
```

**Set environment variables:**
```bash
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit
```

**Install petsc4py:**
```bash
pip install --no-cache-dir --no-binary :all: petsc4py==3.20.1
```

**Expected time**: 3-5 minutes
**Expected output**: Builds from source, links to 64-bit PETSc

**Verify:**
```bash
python -c "from petsc4py import PETSc; print(f'PETSc Int Type: {PETSc.IntType}')"
```

**Expected output:**
```
PETSc Int Type: <class 'numpy.int64'>  ← SUCCESS!
```

---

## STEP 6: Build SLEPc 3.20.1 with 64-bit PETSc

**Download:**
```bash
cd ~/software/petsc_64bit
wget https://slepc.upv.es/download/distrib/slepc-3.20.1.tar.gz
tar -xzf slepc-3.20.1.tar.gz
cd slepc-3.20.1
```

**Configure and build:**
```bash
export SLEPC_DIR=$PWD
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit

./configure
make SLEPC_DIR=$PWD all
make SLEPC_DIR=$PWD check
```

**Expected time**: 5-10 minutes

---

## STEP 7: Install slepc4py 3.20.0 with 64-bit SLEPc

**Install:**
```bash
conda activate fenics64
export SLEPC_DIR=~/software/petsc_64bit/slepc-3.20.1
export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
export PETSC_ARCH=linux-gnu-c-opt-64bit

pip install --no-cache-dir --no-binary :all: slepc4py==3.20.0
```

**Expected time**: 2-3 minutes

---

## STEP 8: Make environment variables persistent

**Add to ~/.bashrc:**
```bash
cat >> ~/.bashrc << 'EOF'

# PETSc 64-bit configuration for fenics64 environment
if [[ "$CONDA_DEFAULT_ENV" == "fenics64" ]]; then
    export PETSC_DIR=~/software/petsc_64bit/petsc-3.20.1
    export PETSC_ARCH=linux-gnu-c-opt-64bit
    export SLEPC_DIR=~/software/petsc_64bit/slepc-3.20.1
fi
EOF
```

**Reload:**
```bash
source ~/.bashrc
```

---

## STEP 9: Comprehensive verification

**Test script:**
```bash
conda activate fenics64
python << 'PYTEST'
import sys
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import dolfin as dl
import hippylib as hp

print("="*60)
print("VERIFICATION TEST FOR FENICS64 ENVIRONMENT")
print("="*60)

# 1. PETSc integer type
print(f"\n1. PETSc Int Type: {PETSc.IntType}")
assert PETSc.IntType == np.int64, "ERROR: Still 32-bit!"
print("   ✓ SUCCESS: Using 64-bit integers")

# 2. PETSc version
version = PETSc.Sys.getVersion()
print(f"\n2. PETSc Version: {version}")
assert version == (3, 20, 1), f"ERROR: Wrong version {version}"
print("   ✓ SUCCESS: Correct version")

# 3. SLEPc version
slepc_version = SLEPc.Sys.getVersion()
print(f"\n3. SLEPc Version: {slepc_version}")
print("   ✓ SUCCESS: SLEPc loaded")

# 4. FEniCS version
print(f"\n4. FEniCS/dolfin version: {dl.__version__}")
print("   ✓ SUCCESS: FEniCS loaded")

# 5. hippylib
print(f"\n5. hippylib loaded successfully")
print("   ✓ SUCCESS: hippylib available")

# 6. Test large matrix creation
print(f"\n6. Testing large matrix (200k x 200k)...")
A = PETSc.Mat().createAIJ([200000, 200000])
print("   ✓ SUCCESS: Can create large matrices")

# 7. MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"\n7. MPI rank: {comm.rank}, size: {comm.size}")
print("   ✓ SUCCESS: MPI working")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
PYTEST
```

**Expected output**: All tests pass with ✓

---

## STEP 10: Test with your actual mesh

**Test with dense4 (previously crashing):**
```bash
cd /home/staff/chao/SSEinv/Nicoya/codes
conda activate fenics64

# Modify testmesh_3DCK.py line 194:
# meshname = "nicoyaCK3_dense4_sm"

# Also REVERT iterative solver back to MUMPS (since we now have 64-bit)
# Lines 445-467 and 790-813: change back to:
# type_solver = "mumps"
# pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

python testmesh_3DCK.py
```

**Expected result**:
- ✓ No segfault
- ✓ Faster than iterative solver
- ✓ Accurate results

---

## COMPARISON: Old vs New

| Aspect | fenics (old) | fenics64 (new) |
|--------|--------------|----------------|
| Python | 3.12.3 | 3.12.3 ✓ |
| FEniCS | 2019.1.0 | 2019.1.0 ✓ |
| PETSc | 3.20.1 (int32) | 3.20.1 (int64) ✓ |
| petsc4py | 3.20.1 (int32) | 3.20.1 (int64) ✓ |
| SLEPc | 3.20.1 (int32) | 3.20.1 (int64) ✓ |
| slepc4py | 3.20.0 (int32) | 3.20.0 (int64) ✓ |
| All other packages | Same | Same ✓ |
| Max mesh nodes | ~30k | >100k ✓ |

---

## TOTAL TIME ESTIMATE

| Step | Time |
|------|------|
| 1. Clone environment | 10 min |
| 2. Verify | 1 min |
| 3. Remove packages | 1 min |
| 4. Build PETSc | 30 min |
| 5. Install petsc4py | 5 min |
| 6. Build SLEPc | 10 min |
| 7. Install slepc4py | 3 min |
| 8. Configure env vars | 2 min |
| 9. Verification | 2 min |
| **TOTAL** | **~1 hour** |

---

## TROUBLESHOOTING

### Issue: Configure fails with "Cannot find MPI"
**Solution:**
```bash
which mpicc  # Should show conda's mpicc
export PATH=/home/staff/chao/anaconda3/envs/fenics64/bin:$PATH
```

### Issue: petsc4py still shows int32
**Solution:**
```bash
echo $PETSC_DIR  # Must be set!
echo $PETSC_ARCH  # Must be set!
pip uninstall petsc4py
pip install --no-cache-dir --no-binary :all: petsc4py==3.20.1
```

### Issue: Import error "cannot find libpetsc.so"
**Solution:**
```bash
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
```

---

## READY TO PROCEED?

When you're ready, I will execute these steps one by one, showing you the output at each stage so you can verify everything is working correctly.

**Shall I begin with Step 1 (cloning the environment)?**
