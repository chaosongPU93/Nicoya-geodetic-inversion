# %% [markdown]
# # FULL SCIPY L-BFGS-B Solution with Real Physics
# 
# This combines the ultra-fast scipy L-BFGS-B solver (proven to work) 
# with the full physics from your original code:
# * Real PDE formulation with elasticity
# * Proper GNSS data misfit  
# * BiLaplacian regularization
# * Box constraints for slip bounds
# * All the physics, none of the TAO complexity
# 
# Based on successful simple test + original physics

# %%
# Thread setup for maximum performance
import sys, os
import multiprocessing
n_cores = 5
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores) 
os.environ['MKL_NUM_THREADS'] = str(n_cores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)

import dolfin as dl
import ufl
import numpy as np
import pandas as pd
import utils as ut
import hippylib as hp
from pointwiseStateObs_weights import PointwiseStateObservation
from slip_transformation_utils import BoxConstraintTransformation
from scipy.optimize import minimize
from mpi4py import MPI
import time

# Optimized compiler settings
dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
dl.set_log_active(False)

print("Starting FULL SCIPY solution with real physics...")

# %%
# Define data directory and load GNSS data (same as original)
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"
obs_disp_name = "CKfig6_data_final.csv"

data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1, \
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car', \
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

# Coordinate transformation (same as original)
lon0, lat0 = -84, 7
rot = 45
x_rot, y_rot  = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)
x0, y0 = 130e3, 350e3
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3
data['z'] = 0.0
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot)

print(f"Loaded {len(data)} GNSS stations")

# Mesh will be loaded in the solver function

# %%
# Physics definitions (same as original)
def AEsigma(s, mu, nu):
    A = 1./(2.*mu)*( s - nu/( 1 + nu*(dim-2) )*ufl.tr(s)*ufl.Identity(dim) )
    return A

def asym(s):
    if dim == 2:
        as_ = s[1,0] - s[0,1]
    elif dim == 3:
        as_ = ufl.as_vector( [ s[1,2] - s[2,1], s[2,0] - s[0,2], s[0,1] - s[1,0] ] )
    return as_

def dir_strike(n):
    z_dir = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    strike_dir = n_cross_z / ufl.sqrt( ufl.dot(n_cross_z, n_cross_z ) )
    return strike_dir

def dir_dip(n):
    dip_dir =  ufl.cross( dir_strike(n), n )
    return dip_dir

def mu_expression(m):
    mu = 20*(2.+ufl.tanh(m)) 
    return mu

# %%
# Material property class (same as original)
class K_2LAYER(dl.UserExpression):
    def __init__(self, subdomains, k_r, k_l, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_r = k_r
        self.k_l = k_l

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == blockright:
            values[0] = self.k_r 
        elif self.subdomains[cell.index] == blockleft:
            values[0] = self.k_l
    
    def value_shape(self):
        return ()

# %%
# Global variables (same style as reference code)
nu = 0.25
GPa2Pa = 1e9

# Will be defined after mesh loading
dim = None
n = None
ds = None
dS = None
f = None

# Boundary definitions (same as original)
top = 1
bottom = 2
west = 3
east = 4
north = 5
south = 6
fault = 7
blockleft = 8
blockright = 9

# %%
# PDE Variational Formulation (exact same style as reference)
class PDEVarf_ScipyBoxConstrained:
    """
    Real physics PDE with box constraints enforced by scipy
    
    Uses the exact same style as your reference code
    """
    def __init__(self, mtrue_mu_fun):
        self.mtrue_mu_fun = mtrue_mu_fun
        
    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)  # Direct physical slip (no tanh)
        tau, w, q = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        # Real physics - exact same as reference
        mu = mu_expression(self.mtrue_mu_fun)

        J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
            + ufl.inner(ufl.div(tau), uu)*ufl.dx \
            + ufl.inner(asym(tau), r)*ufl.dx \
            + ufl.inner(ufl.div(sigma), w)*ufl.dx \
            + ufl.inner(asym(sigma), q)*ufl.dx \
            + ufl.inner(f, w)*dl.dx \
            - ufl.inner(u0, tau*n)*ds(bottom) \
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip), 
                       tau('+')*n('+'))*dS(fault)

        return J

# %%
# Scipy-hipPYlib Interface Class
class ScipyHippylibInterface:
    """
    Interface between scipy.optimize and hipPYlib
    
    Handles the objective function and gradient computation
    using the full physics from your original code
    """
    def __init__(self, model, Vh, d_obs, idx_d, verbose=True):
        self.model = model
        self.Vh = Vh
        self.d_obs = d_obs
        self.idx_d = idx_d
        self.verbose = verbose
        
        # Pre-allocate vectors for efficiency
        self.u = model.generate_vector(hp.STATE)
        self.p = model.generate_vector(hp.ADJOINT)
        self.mg = model.generate_vector(hp.PARAMETER)
        
        # Performance tracking
        self.eval_count = 0
        self.total_solve_time = 0.0
        
    def objective_and_gradient(self, m_array):
        """Compute objective function and gradient for scipy"""
        start_time = time.time()
        self.eval_count += 1
        
        try:
            # Convert numpy array to hipPYlib vector
            m = self.model.generate_vector(hp.PARAMETER)
            m.set_local(m_array)
            m.apply('')
            
            # Create state vector
            x = [self.u, m, self.p]
            
            # Solve forward and adjoint problems
            self.model.solveFwd(self.u, x)
            self.model.solveAdj(self.p, x)
            
            # Compute total cost
            total_cost, reg_cost, misfit_cost = self.model.cost(x)
            
            # Compute gradient
            self.model.evalGradientParameter(x, self.mg)
            gradient = self.mg.get_local()
            
            solve_time = time.time() - start_time
            self.total_solve_time += solve_time
            
            # Progress reporting
            if self.verbose and self.eval_count % 5 == 0:
                avg_time = self.total_solve_time / self.eval_count
                print(f"Eval {self.eval_count}: cost={total_cost:.4e} "
                      f"(reg={reg_cost:.2e}, misfit={misfit_cost:.2e}), "
                      f"avg_time={avg_time:.2f}s")
            
            return total_cost, gradient
            
        except Exception as e:
            print(f"Error in objective evaluation: {e}")
            raise

# %%
# Main solver function
def solve_scipy_full_physics(verbose=True):
    """
    Full physics slip inversion using scipy L-BFGS-B
    
    Uses all the real physics from your original code:
    - Full elasticity PDE
    - Real GNSS data misfit
    - BiLaplacian regularization  
    - Box constraints via scipy
    """
    
    if verbose:
        print("="*60)
        print("FULL SCIPY SOLUTION WITH REAL PHYSICS")
        print("="*60)
    
    # ===== LOAD MESH (same as original) =====
    meshname = "nicoyaCK2"
    meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
    mesh = dl.Mesh(meshpath + meshname + '.xml')
    boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
    subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')

    # Initialize global variables (same style as reference)
    global dim, n, f, ds, dS
    dim = mesh.topology().dim()
    n = dl.FacetNormal(mesh)
    f = dl.Constant((0., 0., 0.))

    ds = dl.Measure("ds")(domain=mesh, subdomain_data=boundaries)
    dS = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

    if verbose:
        print(f"Loaded mesh: {mesh.num_vertices()} vertices")
    
    # ===== SETUP FUNCTION SPACES (same as original) =====
    k = 2
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]
    
    ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
    
    if verbose:
        print(f"DoFs: STATE={ndofs[0]}, PARAMETER={ndofs[1]}, ADJOINT={ndofs[2]}")
    
    # ===== BOUNDARY CONDITIONS (same as original) =====
    zero_tensor = dl.Expression(( ("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.") ), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    
    # ===== STARTING GUESS =====
    m0_s_expr = dl.Constant((0., 0.))
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()
    
    # ===== MATERIAL PROPERTIES (same as original) =====
    # Using homogeneous structure (as per your updated reference)
    mu_b = 0   # 40 GPa
    mu_l = 0   # 40 GPa  
    mu_u = 0   # 40 GPa
    
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu_expr_hom = K_2LAYER(subdomains, mu_u, mu_l, degree=5)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_hom, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    
    # ===== PDE SETUP =====
    pde_varf = PDEVarf_ScipyBoxConstrained(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    
    # Solvers
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    
    # ===== OBSERVATION SETUP (same as original) =====
    ntargets = data.shape[0]
    targets_x = np.array(data['x'])*1e3
    targets_y = np.array(data['y'])*1e3
    targets_z = np.array(data['z'])*1e3
    targets = np.zeros([ntargets, dim])
    targets[:,0] = targets_x; targets[:,1] = targets_y; targets[:,2] = targets_z
    
    # Data weights (updated to match reference)
    f_h, f_v = 1, 1
    
    indicator_vec = dl.interpolate( dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE] ).vector()
    weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
    
    obs_weights = np.zeros(targets.shape[0]*15,)
    weight_x, weight_y = ut.compute_rotated_weights((data['vx_std_Car']**2).to_numpy(), \
                                                 (data['vy_std_Car']**2).to_numpy(), rot)
    obs_weights[9::15] = weight_x / (f_h**2)
    obs_weights[10::15] = weight_y / (f_h**2)
    obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)
    
    weights.set_local(obs_weights)
    weights.apply('')
    
    # Misfit function
    misfit = PointwiseStateObservation( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
    misfit.noise_variance = 1.
    
    # Set observed data
    tmp = np.zeros(len(misfit.d),)
    tmp[9::15] = np.array(data['vx_Car'])
    tmp[10::15] = np.array(data['vy_Car'])
    tmp[11::15] = np.array(data['vz_Car'])
    misfit.d.set_local(tmp)
    misfit.d.apply('')
    
    idx_d = list(np.nonzero(misfit.weight)[0])
    d_obs = misfit.d[idx_d]
    
    # ===== REGULARIZATION (same as original) =====
    rho_s = 1e9
    gamma_val_H1 = 1e3  # Updated to match reference
    delta_val_L2 = gamma_val_H1 / rho_s
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )
    
    # ===== CONSTRUCT MODEL =====
    model = hp.Model(pde, reg, misfit)
    
    if verbose:
        print(f"Model setup complete:")
        print(f"  Observations: {len(d_obs)}")
        print(f"  Regularization: gamma={gamma_val_H1:.1e}, delta={delta_val_L2:.1e}")
    
    # ===== BOX CONSTRAINTS =====
    s_strike_max = 0.027
    s_dip_max = 0.0785
    
    box_transformer = BoxConstraintTransformation(
        strike_bounds=(0.0, s_strike_max),
        dip_bounds=(0.0, s_dip_max)
    )
    
    # Get fault mask
    bc_fault = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    fault_func = dl.Function(Vm)
    bc_fault.apply(fault_func.vector())
    fault_mask = fault_func.vector() == 99
    
    # Create bounds
    bounds = box_transformer.create_scipy_bounds(m0_s, fault_mask)
    
    if verbose:
        n_constrained = sum(1 for b in bounds if b != (None, None))
        print(f"Box constraints: {box_transformer}")
        print(f"  {n_constrained} parameters constrained")
    
    # ===== SCIPY OPTIMIZATION =====
    interface = ScipyHippylibInterface(model, Vh, d_obs, idx_d, verbose=verbose)
    
    if verbose:
        print("Starting scipy L-BFGS-B optimization with REAL PHYSICS...")
    
    start_time = time.time()
    
    result = minimize(
        fun=interface.objective_and_gradient,
        x0=m0_s.get_local(),
        method='L-BFGS-B',
        jac=True,  # We provide gradient
        bounds=bounds,
        options={
            'disp': verbose,
            'maxiter': 1000,
            'ftol': 1e-9,
            'gtol': 1e-6
        }
    )
    
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Total evaluations: {interface.eval_count}")
        print(f"Average solve time: {interface.total_solve_time/interface.eval_count:.3f}s")
    
    # ===== VALIDATE SOLUTION =====
    m_final = m0_s.copy()
    m_final.set_local(result.x)
    m_final.apply('')
    
    # Validate bounds
    if verbose:
        print("\nValidating solution bounds...")
    box_transformer.validate_bounds(result.x, fault_mask, verbose=verbose)
    
    # Extract slip values
    fault_values = result.x[fault_mask]
    strike_values = fault_values[0::2]
    dip_values = fault_values[1::2]
    
    if verbose:
        print(f"\nFinal slip ranges:")
        print(f"  Strike: [{min(strike_values):.6f}, {max(strike_values):.6f}] m")
        print(f"  Dip: [{min(dip_values):.6f}, {max(dip_values):.6f}] m")
    
    # ===== COMPUTE FINAL MISFIT =====
    # Solve forward with final solution to get predicted data
    x_final = [interface.u, m_final, interface.p]
    model.solveFwd(interface.u, x_final)
    
    misfit.B.mult(x_final[hp.STATE], misfit.Bu)
    d_cal = misfit.Bu[idx_d]
    
    data_misfit = np.linalg.norm(d_cal - d_obs, 2)
    
    if verbose:
        print(f"\nFinal data misfit: {data_misfit:.6e}")
        total_cost, reg_cost, misfit_cost = model.cost(x_final)
        print(f"Final costs: total={total_cost:.4e}, reg={reg_cost:.4e}, misfit={misfit_cost:.4e}")
    
    return {
        'solution': m_final,
        'optimization_time': optimization_time,
        'result': result,
        'strike_slip': strike_values,
        'dip_slip': dip_values,
        'data_misfit': data_misfit,
        'interface': interface
    }

# %%
# Run full scipy optimization
if __name__ == "__main__":
    print("="*60)
    print("RUNNING FULL SCIPY L-BFGS-B WITH REAL PHYSICS")
    print("="*60)
    
    results = solve_scipy_full_physics(verbose=True)
    
    print("="*60)
    print("FULL SCIPY OPTIMIZATION COMPLETED!")
    print(f"Runtime: {results['optimization_time']:.2f} seconds")
    print(f"Success: {results['result'].success}")
    print(f"Data misfit: {results['data_misfit']:.6e}")
    print("This combines scipy speed with full physics!")
    print("="*60)