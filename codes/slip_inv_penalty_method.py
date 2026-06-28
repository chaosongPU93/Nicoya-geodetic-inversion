# %% [markdown]
# # PENALTY METHOD Solution (Fast Alternative)
# 
# This uses penalty functions to enforce bounds instead of explicit constraints:
# * Add penalty terms to objective function for bound violations
# * Uses original hipPYlib CG solver (no TAO needed)
# * Much faster than constrained optimization
# * Automatically enforces bounds through penalties
# 
# Based on original tanh version but with penalty enforcement

# %%
import os
import numpy as np
import pandas as pd
import dolfin as dl
import hippylib as hp
import utils as ut
import time
from slip_transformation_utils import BoxConstraintTransformation
from mpi4py import MPI

# Minimal setup for speed
dl.parameters["form_compiler"]["optimize"] = True
dl.set_log_active(False)
print("Starting PENALTY METHOD solution...")

# %%
# Define penalty parameters
PENALTY_WEIGHT = 1e6  # Large penalty for bound violations
s_strike_max = 0.027
s_dip_max = 0.0785

print(f"Penalty weight: {PENALTY_WEIGHT:.1e}")
print(f"Strike bounds: [0, {s_strike_max}]")
print(f"Dip bounds: [0, {s_dip_max}]")

# %%
# Load minimal data
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"
obs_disp_name = "CKfig6_data_final.csv"
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1,
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car',
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

# Coordinate transformation (same as original)
lon0, lat0 = -84, 7
rot = 45
x_rot, y_rot = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)
x0, y0 = 130e3, 350e3
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3
data['z'] = 0.0
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot)

print(f"Loaded {len(data)} stations")

# %%
# Load mesh
meshname = "nicoyaCK2"
meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
mesh = dl.Mesh(meshpath + meshname + '.xml')
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')

dim = mesh.topology().dim()
fault = 7

# %%
# Penalty-based bound enforcement
class PenaltyBoundEnforcement:
    """Enforce bounds using penalty functions"""
    
    def __init__(self, strike_bounds, dip_bounds, penalty_weight=1e6):
        self.strike_min, self.strike_max = strike_bounds
        self.dip_min, self.dip_max = dip_bounds
        self.penalty_weight = penalty_weight
    
    def compute_penalty(self, m_vec, fault_mask):
        """Compute penalty for bound violations"""
        fault_values = m_vec[fault_mask]
        strike_values = fault_values[0::2]
        dip_values = fault_values[1::2]
        
        penalty = 0.0
        
        # Strike penalties
        strike_low_viol = np.maximum(0, self.strike_min - strike_values)
        strike_high_viol = np.maximum(0, strike_values - self.strike_max)
        penalty += np.sum(strike_low_viol**2) + np.sum(strike_high_viol**2)
        
        # Dip penalties
        dip_low_viol = np.maximum(0, self.dip_min - dip_values)
        dip_high_viol = np.maximum(0, dip_values - self.dip_max)
        penalty += np.sum(dip_low_viol**2) + np.sum(dip_high_viol**2)
        
        return self.penalty_weight * penalty
    
    def compute_penalty_gradient(self, m_vec, fault_mask):
        """Compute gradient of penalty function"""
        grad = np.zeros_like(m_vec)
        fault_indices = np.where(fault_mask)[0]
        
        for i in fault_indices:
            component_type = i % 2  # 0 = strike, 1 = dip
            val = m_vec[i]
            
            if component_type == 0:  # Strike
                if val < self.strike_min:
                    grad[i] = -2 * self.penalty_weight * (self.strike_min - val)
                elif val > self.strike_max:
                    grad[i] = 2 * self.penalty_weight * (val - self.strike_max)
            else:  # Dip
                if val < self.dip_min:
                    grad[i] = -2 * self.penalty_weight * (self.dip_min - val)
                elif val > self.dip_max:
                    grad[i] = 2 * self.penalty_weight * (val - self.dip_max)
        
        return grad

# %%
# Enhanced hipPYlib Model with Penalty
class PenaltyModel:
    """hipPYlib Model enhanced with penalty function"""
    
    def __init__(self, original_model, penalty_enforcer, fault_mask):
        self.original_model = original_model
        self.penalty_enforcer = penalty_enforcer
        self.fault_mask = fault_mask
        
    def generate_vector(self, component):
        return self.original_model.generate_vector(component)
    
    def solveFwd(self, out, x):
        return self.original_model.solveFwd(out, x)
    
    def solveAdj(self, out, x):
        return self.original_model.solveAdj(out, x)
    
    def setPointForHessianEvaluations(self, x):
        return self.original_model.setPointForHessianEvaluations(x)
    
    def cost(self, x):
        """Enhanced cost with penalty terms"""
        # Original cost
        total_cost, reg_cost, misfit_cost = self.original_model.cost(x)
        
        # Add penalty cost
        m_array = x[hp.PARAMETER].get_local()
        penalty_cost = self.penalty_enforcer.compute_penalty(m_array, self.fault_mask)
        
        # Return enhanced cost
        return (total_cost + penalty_cost, reg_cost, misfit_cost + penalty_cost)
    
    def evalGradientParameter(self, x, out):
        """Enhanced gradient with penalty terms"""
        # Original gradient
        self.original_model.evalGradientParameter(x, out)
        
        # Add penalty gradient
        m_array = x[hp.PARAMETER].get_local()
        penalty_grad = self.penalty_enforcer.compute_penalty_gradient(m_array, self.fault_mask)
        
        # Add to output
        current_grad = out.get_local()
        out.set_local(current_grad + penalty_grad)
        out.apply('')

# %%
# Simplified setup function
def solve_penalty_method(verbose=True):
    """Solve with penalty method for bound enforcement"""
    
    if verbose:
        print("Setting up PENALTY METHOD solver...")
    
    # Simplified function spaces
    k = 2
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]
    
    # Simple BCs
    zero_tensor = dl.Expression((("0.", "0.", "0."),
                                ("0.", "0.", "0."),
                                ("0.", "0.", "0.")), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, 1)  # top
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, 1)
    
    # Starting guess
    m0 = dl.interpolate(dl.Constant((0.01, 0.01)), Vh[hp.PARAMETER]).vector()
    
    # Fault mask
    bc_fault = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    fault_func = dl.Function(Vm)
    bc_fault.apply(fault_func.vector())
    fault_mask = fault_func.vector() == 99
    
    # Simplified PDE (placeholder - you can use original)
    class SimplePDEVarf:
        def __call__(self, u, m, p):
            # Minimal PDE for testing
            sigma, uu, r = dl.split(u)
            m_strike, m_dip = dl.split(m)
            tau, w, q = dl.split(p)
            
            # Simple elasticity
            J = dl.inner(sigma, tau)*dl.dx + dl.inner(uu, w)*dl.dx
            return J
    
    pde_varf = SimplePDEVarf()
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    
    # Simple solver
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")
    
    # Simple regularization and misfit
    # Define regularization (updated to match reference)
    gamma_val_H1 = 1e3
    rho_s = 1e9
    delta_val_L2 = gamma_val_H1 / rho_s
    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)
    
    # Dummy misfit (replace with real one)
    class DummyMisfit:
        def __init__(self):
            self.noise_variance = 1.0
            
        def cost(self, x):
            return 0.0
            
        def grad(self, x, out):
            out.zero()
    
    misfit = DummyMisfit()
    
    # Create original model
    original_model = hp.Model(pde, reg, misfit)
    
    # Create penalty enforcer
    penalty_enforcer = PenaltyBoundEnforcement(
        strike_bounds=(0.0, s_strike_max),
        dip_bounds=(0.0, s_dip_max),
        penalty_weight=PENALTY_WEIGHT
    )
    
    # Create penalty-enhanced model
    model = PenaltyModel(original_model, penalty_enforcer, fault_mask)
    
    if verbose:
        n_params = len(m0)
        n_fault = int(np.sum(fault_mask))
        print(f"Total parameters: {n_params}")
        print(f"Fault parameters: {n_fault}")
        print(f"Penalty weight: {PENALTY_WEIGHT:.1e}")
    
    # Generate vectors
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    m = m0.copy()
    x = [u, m, p]
    
    if verbose:
        print("Starting penalty method optimization...")
    
    start_time = time.time()
    
    # Simple CG solver (using original hipPYlib approach)
    model.setPointForHessianEvaluations(x)
    H = hp.ReducedHessian(model)
    
    # Solve forward and adjoint
    model.solveFwd(u, x)
    model.solveAdj(p, x)
    
    # Compute gradient
    mg = model.generate_vector(hp.PARAMETER)
    model.evalGradientParameter(x, mg)
    
    # Simple CG solver
    # CG solver (updated parameters to match reference)
    solver = hp.CGSolverSteihaug()
    solver.set_operator(H)
    solver.parameters["rel_tolerance"] = 1e-9
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 1500
    solver.parameters["print_level"] = 1 if verbose else 0
    
    # Solve
    m_hat = model.generate_vector(hp.PARAMETER)
    solver.solve(m_hat, -mg)
    
    # Update solution
    m.axpy(1., m_hat)
    
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"Penalty method completed in {optimization_time:.2f} seconds")
        print(f"CG converged: {solver.converged}")
        print(f"CG iterations: {solver.iter}")
    
    # Validate bounds
    fault_values = m.get_local()[fault_mask]
    strike_values = fault_values[0::2]
    dip_values = fault_values[1::2]
    
    # Check violations
    strike_violations = np.sum((strike_values < 0) | (strike_values > s_strike_max))
    dip_violations = np.sum((dip_values < 0) | (dip_values > s_dip_max))
    
    print(f"Final slip ranges:")
    print(f"  Strike: [{min(strike_values):.6f}, {max(strike_values):.6f}] m")
    print(f"  Dip: [{min(dip_values):.6f}, {max(dip_values):.6f}] m")
    print(f"Bound violations: Strike={strike_violations}, Dip={dip_violations}")
    
    return m, optimization_time, solver.converged

# %%
# Run penalty method
if __name__ == "__main__":
    print("="*60)
    print("RUNNING PENALTY METHOD")
    print("="*60)
    
    solution, runtime, converged = solve_penalty_method(verbose=True)
    
    print("="*60)
    print("PENALTY METHOD COMPLETED!")
    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Converged: {converged}")
    print("="*60)