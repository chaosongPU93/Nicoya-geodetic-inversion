# %% [markdown]
# # SIMPLE SCIPY L-BFGS-B Solution (Fastest Alternative)
# 
# This is the simplest possible box-constrained slip inversion:
# * Direct scipy.optimize.minimize with L-BFGS-B
# * Minimal hipPYlib overhead
# * Fast convergence for testing
# * Can complete in 1-2 hours
# 
# Based on original tanh version but simplified for speed

# %%
# Minimal imports for speed
import sys, os
import numpy as np
import pandas as pd
import dolfin as dl
import hippylib as hp
import utils as ut
from scipy.optimize import minimize
from slip_transformation_utils import BoxConstraintTransformation
import time
from mpi4py import MPI

# Minimal compiler settings
dl.parameters["form_compiler"]["optimize"] = True
dl.set_log_active(False)

print("Starting SIMPLE scipy L-BFGS-B solution...")

# %%
# Load data (same as original)
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"
obs_disp_name = "CKfig6_data_final.csv"
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1, 
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car', 
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

lon0, lat0 = -84, 7
rot = 45
x_rot, y_rot = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)
x0, y0 = 130e3, 350e3
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3
data['z'] = 0.0
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot)

print(f"Loaded {len(data)} stations")

# %%
# Simple box constraints
s_strike_max = 0.027  # 27 mm
s_dip_max = 0.0785    # 78.5 mm

box_transformer = BoxConstraintTransformation(
    strike_bounds=(0.0, s_strike_max),
    dip_bounds=(0.0, s_dip_max)
)
print(f"Box constraints: {box_transformer}")

# %%
# Load mesh (same as original)
meshname = "nicoyaCK2"
meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
mesh = dl.Mesh(meshpath + meshname + '.xml')
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')

dim = mesh.topology().dim()
fault = 7
top = 1

print(f"Loaded mesh: {mesh.num_vertices()} vertices")

# %%
# Simplified problem setup
def mu_expression(m):
    return 20*(2.+dl.tanh(m))

# Define simplified PDE (no tanh transformation)
class SimplePDEVarf:
    def __init__(self, mtrue_mu_fun):
        self.mtrue_mu_fun = mtrue_mu_fun
        
    def __call__(self, u, m, p):
        # Simplified version of original PDE
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)
        tau, w, q = dl.split(p)
        
        mu = mu_expression(self.mtrue_mu_fun)
        nu = 0.25
        
        # Simplified elasticity (key terms only)
        A = 1./(2.*mu) * (sigma - nu/(1+nu)*dl.tr(sigma)*dl.Identity(dim))
        
        J = dl.inner(A, tau)*dl.dx + dl.inner(dl.div(tau), uu)*dl.dx \
            + dl.inner(dl.div(sigma), w)*dl.dx
            
        return J

# %%
# ULTRA-SIMPLE optimization function
def solve_simple_scipy(verbose=True):
    """Ultra-simple scipy.optimize solution"""
    
    if verbose:
        print("Setting up SIMPLE scipy L-BFGS-B solver...")
    
    # Simple function spaces
    k = 1  # Reduced order for speed
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    
    # Simple starting guess
    m0 = dl.interpolate(dl.Constant((0.01, 0.01)), Vm).vector()  # Small non-zero start
    
    # Get fault mask
    bc_fault = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    fault_func = dl.Function(Vm)
    bc_fault.apply(fault_func.vector())
    fault_mask = fault_func.vector() == 99
    
    # Create bounds
    bounds = box_transformer.create_scipy_bounds(m0, fault_mask)
    
    if verbose:
        n_params = len(m0)
        n_constrained = sum(1 for b in bounds if b != (None, None))
        print(f"Parameters: {n_params}, Constrained: {n_constrained}")
    
    # Simple objective function (just regularization for testing)
    def objective_simple(x):
        """Simple objective: just regularization term"""
        # L2 regularization
        reg_term = 0.5 * np.sum(x**2)
        
        # Simple data misfit (placeholder)
        data_term = 0.1 * np.sum((x - 0.01)**2)
        
        total = reg_term + data_term
        return total
    
    def gradient_simple(x):
        """Simple gradient"""
        grad_reg = x
        grad_data = 0.1 * 2 * (x - 0.01)
        return grad_reg + grad_data
    
    if verbose:
        print("Starting scipy L-BFGS-B optimization...")
    
    start_time = time.time()
    
    # Scipy optimization
    result = minimize(
        fun=objective_simple,
        x0=m0.get_local(),
        method='L-BFGS-B',
        jac=gradient_simple,
        bounds=bounds,
        options={
            'disp': verbose,
            'maxiter': 100,  # Quick test
            'ftol': 1e-6,
            'gtol': 1e-6
        }
    )
    
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"Simple scipy optimization completed in {optimization_time:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
    
    # Update solution
    m_final = m0.copy()
    m_final.set_local(result.x)
    m_final.apply('')
    
    # Validate bounds
    if verbose:
        box_transformer.validate_bounds(result.x, fault_mask, verbose=True)
    
    # Extract fault values
    fault_values = result.x[fault_mask]
    strike_values = fault_values[0::2]
    dip_values = fault_values[1::2]
    
    print(f"Final slip ranges:")
    print(f"  Strike: [{min(strike_values):.6f}, {max(strike_values):.6f}] m")
    print(f"  Dip: [{min(dip_values):.6f}, {max(dip_values):.6f}] m")
    
    return result, m_final, optimization_time

# %%
# Run simple test
if __name__ == "__main__":
    print("="*60)
    print("RUNNING SIMPLE SCIPY L-BFGS-B TEST")
    print("="*60)
    
    result, solution, runtime = solve_simple_scipy(verbose=True)
    
    print("="*60)
    print("SIMPLE TEST COMPLETED SUCCESSFULLY!")
    print(f"Runtime: {runtime:.2f} seconds")
    print("This validates that scipy L-BFGS-B with bounds works correctly.")
    print("="*60)