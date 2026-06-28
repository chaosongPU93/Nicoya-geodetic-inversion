#!/usr/bin/env python3
"""
Test script to verify mesh and velocity model projection for nicoyaCK3_dense

This script:
1. Loads the new dense FEniCS mesh (nicoyaCK3_dense.xml)
2. Reads 3D velocity and 1D density models
3. Projects them to the FEM mesh using ut.process_velocity_models_v3
4. Saves velocity, density, and shear modulus models to XDMF files for inspection

gmsh -3 nicoyaCK3_dense.geo -format msh2 -optimize_netgen -algo del3d -smooth 10 2>&1 | tee log_mesh
gmsh -3 nicoyaCK3_dense3.geo -format msh2 -optimize_netgen -algo del3d -smooth 20 2>&1 | tee log_mesh3
gmsh -3 nicoyaCK3_dense4.geo -format msh2 -optimize_netgen -algo del3d -smooth 20 2>&1 | tee log_mesh4
gmsh -3 nicoyaCK3_dense5.geo -format msh2 -optimize_netgen -algo del3d -smooth 20 2>&1 | tee log_mesh5

dolfin-convert nicoyaCK3_dense.msh nicoyaCK3_dense.xml

Adapted from slip_inv_hetmu3D_nicoyaCK_locking_both.py
"""

# %%
# Limit threads
import sys, os
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['VECLIB_MAXIMUM_THREADS'] = '10'
os.environ['NUMEXPR_NUM_THREADS'] = '10'

# %%
# Import libraries
import dolfin as dl
import pandas as pd
import numpy as np
import utils as ut

# Mute FFC and UFL warnings
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

print("="*80)
print("Testing mesh and velocity model projection")
print("="*80)

# %%
# Define coordinate transformation parameters (must match mesh generation)
lon0, lat0 = -84, 7     # Centroid for coordinate transformation
rot = 45                 # Rotation angle in degrees, positive is CCW
x0, y0 = 130e3, 350e3   # Offset for x and y coordinates, in m

# %%
# Read 3D velocity model from DeShon et al. 2006
print("\n1. Reading 3D velocity model...")
veldir = "/home/staff/chao/SSEinv/Nicoya/DeShon_2006GJI/"
vel3dfile = "DeShon2006_3Dmodel.csv"
vel3d = pd.read_csv(veldir + vel3dfile, sep=",")

# Convert to relative locations in meters, rotate, and offset to align with mesh
x_rot, y_rot = ut.LL2ckmd(vel3d['lon'], vel3d['lat'], lon0, lat0, rot)
vel3d['x'], vel3d['y'] = x_rot - x0, y_rot - y0
vel3d['z'] = vel3d['z'] * -1 * 1e3  # Convert depth to negative z (meters)
vel3d = vel3d[(vel3d['z'] <= 0)].reset_index(drop=True)  # Only keep data below ground
print(f"   3D model shape: {vel3d.shape}")
print(f"   3D model columns: {vel3d.columns.tolist()}")

# %%
# Read reference 1D velocity model
print("\n2. Reading 1D velocity model...")
vel1dfile = "DeShon2006_1Dmodel.csv"
vel1d = pd.read_csv(veldir + vel1dfile, sep=r'\s+', skiprows=1,
                    names=['z', 'vp', 'vs', 'vp_vs_ratio'])
vel1d['z'] = vel1d['z'] * -1 * 1e3  # Convert km to m (negative depth)
vel1d = vel1d[(vel1d['z'] <= 0)].reset_index(drop=True)
print(f"   1D velocity model shape: {vel1d.shape}")
print(vel1d.head())

# %%
# Read 1D density model
print("\n3. Reading 1D density model...")
den1dfile = "Density_1Dmodel.csv"
den1d = pd.read_csv(veldir + den1dfile, sep=r'\s+', skiprows=1,
                    names=['z', 'den'])
den1d['z'] = den1d['z'] * -1 * 1e3  # Convert km to m (negative depth)
den1d = den1d[(den1d['z'] <= 0)].reset_index(drop=True)
den1d['den'] = den1d['den'] * 1e3  # Convert g/cm^3 to kg/m^3
print(f"   1D density model shape: {den1d.shape}")
print(den1d.head())

# %%
# Load the dense FEniCS mesh
print("\n4. Loading FEniCS mesh...")

# select mesh name
# meshname = "nicoyaCK3_dense2"   #probably okay to use, inspected with TWB  
# meshname = "nicoyaCK3_dense4"   #latest model, denser, v4
# meshname = "nicoyaCK3_dense5"   #latest model, same geometry as v4, but slightly coarser
meshname = "nicoyaCK3_dense6"   #latest model, same geometry as v6, even coarser

meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"

mesh = dl.Mesh(meshpath + meshname + '.xml')
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')

print(f"   Mesh: {meshname}")
print(f"   Number of vertices: {mesh.num_vertices()}")
print(f"   Number of cells: {mesh.num_cells()}")
print(f"   Mesh dimension: {mesh.topology().dim()}")

# %%
# Create function space for material properties
print("\n5. Creating function space...")
CG_mu = dl.FunctionSpace(mesh, "CG", 1)
print(f"   Function space DoFs: {CG_mu.dim()}")

# %%
# Process velocity models and project to mesh
print("\n6. Processing velocity models and projecting to mesh...")

vs_func, den_func, mu_func, _ = ut.process_velocity_models(
    vel3d, vel1d, den1d, mesh, verbose=True
)

# ============================================================================
# OPTION A: Use v2 with wider transition (100-150 km)
# ============================================================================
# print("   Using ut.process_velocity_models_v2 with wide transition...")
# vs_func, den_func, mu_func, _ = ut.process_velocity_models_v2(
#     vel3d, vel1d, den1d, mesh,
#     transition_width=100e3,  # Wider transition to reduce visible boundary
#     verbose=True
# )

# ============================================================================
# OPTION B: Use v3 with depth-averaged background (progressive windowing)
# ============================================================================
# print("   Using ut.process_velocity_models_v3 (depth-averaged background)...")
# vs_func, den_func, mu_func, _ = ut.process_velocity_models_v3(
#     vel3d, vel1d, den1d, mesh,
#     transition_width=100e3,  # Transition zone width
#     verbose=True
# )

# ============================================================================
# OPTION C: Use v4 (Enhanced v1 with linear interpolation)
# ============================================================================
# print("   Using ut.process_velocity_models_v4 (enhanced v1)...")
# vs_func, den_func, mu_func, _ = ut.process_velocity_models_v4(
#     vel3d, vel1d, den1d, mesh,
#     use_rbf_sparse=False,      # Default: nearest-neighbor (like v1)
#     transition_width=0.0,       # Default: sharp boundary (like v1)
#     verbose=True
# )

# ============================================================================
# OPTION C-RBF: Use v4 with RBF for sparse points
# ============================================================================
# print("   Using ut.process_velocity_models_v4 with RBF...")
# vs_func, den_func, mu_func, _ = ut.process_velocity_models_v4(
#     vel3d, vel1d, den1d, mesh,
#     use_rbf_sparse=True,        # Use RBF for sparse points
#     transition_width=0.0,        # Sharp boundary
#     verbose=True
# )

# # ============================================================================
# # OPTION C-TRANSITION: Use v4 with narrow transition zone
# # ============================================================================
# print("   Using ut.process_velocity_models_v4 with transition...")
# vs_func, den_func, mu_func, _ = ut.process_velocity_models_v4(
#     vel3d, vel1d, den1d, mesh,
#     use_rbf_sparse=False,       # Nearest-neighbor
#     transition_width=5e3,        # 5 km transition zone
#     verbose=True
# )

print(f"   Vs range: {vs_func.vector().min():.2f} - {vs_func.vector().max():.2f} m/s")
print(f"   Density range: {den_func.vector().min():.2f} - {den_func.vector().max():.2f} kg/m^3")
print(f"   Shear modulus range: {mu_func.vector().min():.2f} - {mu_func.vector().max():.2f} GPa")

# %%
# Apply contrast factor (optional, as done in the reference code)
print("\n7. Applying contrast factor...")
contrast_factor = 1.0  # Amplification factor
# contrast_factor = 4.0  # Amplification factor
mu_ref = np.mean(mu_func.vector()[:])
min_mu = 5.0  # Minimum physical value (GPa)

mtrue_mu_fun = dl.Function(CG_mu)
new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)

print(f"   Reference shear modulus: {mu_ref:.2f} GPa")
print(f"   Contrast factor: {contrast_factor}")
print(f"   Adjusted shear modulus range: {mtrue_mu_fun.vector().min():.2f} - {mtrue_mu_fun.vector().max():.2f} GPa")

# %%
# Save results to XDMF files
print("\n8. Saving results to XDMF files...")
resultpath = "/home/staff/chao/SSEinv/Nicoya/rst_locking/"
os.makedirs(resultpath, exist_ok=True)

mu_str = f"_DeShon3D_ref_{round(contrast_factor)}"

# Save shear modulus
print("   Saving shear modulus...")
m_mu_true = dl.project(mtrue_mu_fun, CG_mu)
mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str + '.xdmf')
m_mu_true.rename('shear modulus', 'shear modulus')
mu_id.write(m_mu_true)
print(f"   Saved to: mu_true_{meshname}{mu_str}.xdmf")

# Save Vs
print("   Saving shear velocity...")
vs_true = dl.project(vs_func, CG_mu)
vs_id = dl.XDMFFile(resultpath + 'vs_true_' + meshname + mu_str + '.xdmf')
vs_true.rename('shear velocity Vs', 'shear velocity Vs')
vs_id.write(vs_true)
print(f"   Saved to: vs_true_{meshname}{mu_str}.xdmf")

# Save density
print("   Saving density...")
den_true = dl.project(den_func, CG_mu)
den_id = dl.XDMFFile(resultpath + 'pho_true_' + meshname + mu_str + '.xdmf')
den_true.rename('density', 'density')
den_id.write(den_true)
print(f"   Saved to: pho_true_{meshname}{mu_str}.xdmf")

# %%
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Mesh: {meshname}")
print(f"  Vertices: {mesh.num_vertices()}")
print(f"  Cells: {mesh.num_cells()}")
print(f"\nMaterial properties (after contrast factor {contrast_factor}):")
print(f"  Vs:       {vs_func.vector().min():.2f} - {vs_func.vector().max():.2f} m/s")
print(f"  Density:  {den_func.vector().min():.2f} - {den_func.vector().max():.2f} kg/m^3")
print(f"  Mu:       {mtrue_mu_fun.vector().min():.2f} - {mtrue_mu_fun.vector().max():.2f} GPa")
print(f"\nResults saved to: {resultpath}")
print("="*80)
print("Test completed successfully!")
print("="*80)
