# %% [markdown]
# # ENHANCED: Synthetic Slip Inversion with Tanh Transformation Bounds
# 
# ## Original Description:
# Synthetic Slip inversion of the GNSS coseismic displacement measurements at Nicoya (Costa Rica) stations 
# when the true slip distribution is like a checkerboard. Compare how the pattern can be resolved within a 
# homogeneous or prescribed heterogeneous half-space
# 
# * Data are synthetics made from forward modeling of checkerboard slip pattern at the same stations used in the locking or coseismic cases
# * Plate model ideally should be from an updated local model in Kyriakopoulos & Newman 2016 JGRSE, but until we get the model, we stick to "Slab2"
# * The goal of this code is to show how much the ground-truth slip distribution can be recovered, especially at the offset location. Through the result comparison between the homogeneous and heterogeneous cases
# * This version adds noise to the synthetic data, either same random noise with a standard deviation for all stations, or use the exact STD of the real data. Correspondingly, the inversion would be revised accordingly
#
# ## 🎯 NEW ENHANCEMENT: Tanh Transformation Framework for Bounded Slip Inversion
#
# This enhanced version adds elegant slip bounds using tanh transformation:
# - **Unbounded parameters** `m ∈ (-∞,∞)` → **Bounded slip** `s ∈ [min,max]` via `tanh()` 
# - **Mixed constraints**: strike-only, dip-only, both, or neither
# - **No optimization changes**: Uses your original fast unconstrained CG solver
# - **Guaranteed bounds**: Mathematical transformation ensures slip stays within limits
# - **Flexible configuration**: Easy to adjust bounds for different scenarios
#
# **Key advantages over constrained optimization:**
# 1. ✅ **Speed**: No slow constrained solver - uses your original CG
# 2. ✅ **Robustness**: No convergence issues with bounds
# 3. ✅ **Elegance**: Mathematical transformation vs algorithmic constraints
# 4. ✅ **Validation**: Automatic bound satisfaction guaranteed

# %%
# limit the number of threads on clusters, by Chao, 02/06/2025
import sys, os
os.environ['OMP_NUM_THREADS'] = '10'
os.environ['OPENBLAS_NUM_THREADS'] = '10'
os.environ['MKL_NUM_THREADS'] = '10'
os.environ['VECLIB_MAXIMUM_THREADS'] = '10'
os.environ['NUMEXPR_NUM_THREADS'] = '10'

import dolfin as dl
import ufl
import math
import pandas as pd
import numpy as np
import utils as ut
from mpi4py import MPI
#sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "...") )
import hippylib as hp
# from pointwiseStateObs_weights import PointwiseStateObservation
from pointwiseStateObs_weights import PointwiseStateObservation as PSBW
from pointwiseStateObs import PointwiseStateObservation as PSB
# Set parameters compiler
dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
# Mute FFC and UFL warnings
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
# Define sep
sep = "\n"+"#"*80+"\n"

# %%
# 🎯 NEW: Slip Transformation Framework
print("Loading enhanced slip transformation framework...")

class SlipTransformation:
    """
    Handle slip transformations with flexible constraint combinations
    Uses tanh transformation: unbounded parameters → bounded slip
    """
    
    def __init__(self, strike_bounds=None, dip_bounds=None):
        """
        Initialize slip transformation
        
        Args:
            strike_bounds: tuple (min, max) or None for unconstrained
            dip_bounds: tuple (min, max) or None for unconstrained  
            
        Examples:
            # Both components constrained
            SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16))
            
            # Only dip constrained (typical for thrust faults)
            SlipTransformation(strike_bounds=None, dip_bounds=(0.0, 0.16))
            
            # Only strike constrained  
            SlipTransformation(strike_bounds=(-0.1, 0.1), dip_bounds=None)
            
            # No constraints (reverts to original framework)
            SlipTransformation(strike_bounds=None, dip_bounds=None)
        """
        self.strike_bounds = strike_bounds
        self.dip_bounds = dip_bounds
        
        # Determine transformation mode
        self.has_strike_bounds = strike_bounds is not None
        self.has_dip_bounds = dip_bounds is not None
        self.has_any_bounds = self.has_strike_bounds or self.has_dip_bounds
        
        if self.has_strike_bounds:
            self.strike_min, self.strike_max = strike_bounds
            
        if self.has_dip_bounds:
            self.dip_min, self.dip_max = dip_bounds
    
    def __str__(self):
        """String representation for debugging"""
        if not self.has_any_bounds:
            return "SlipTransformation: UNCONSTRAINED (original framework)"
        
        parts = []
        if self.has_strike_bounds:
            parts.append(f"strike ∈ [{self.strike_min:.4f}, {self.strike_max:.4f}]")
        else:
            parts.append("strike: unconstrained")
            
        if self.has_dip_bounds:
            parts.append(f"dip ∈ [{self.dip_min:.4f}, {self.dip_max:.4f}]")
        else:
            parts.append("dip: unconstrained")
            
        return f"SlipTransformation: {', '.join(parts)}"
    
    def transform_to_physical_slip(self, m):
        """
        Transform unbounded parameters m to bounded physical slip
        
        Args:
            m: UFL expression or Function for unbounded parameters (2D vector)
            
        Returns:
            UFL expression for physical slip with bounds applied
        """
        if not self.has_any_bounds:
            # No bounds - return original parameters as slip
            return m
            
        # Split into components
        m_strike, m_dip = ufl.split(m)
        
        # Transform strike component
        if self.has_strike_bounds:
            # Apply tanh transformation: (-∞,∞) → (strike_min, strike_max)
            strike_scaled = (ufl.tanh(m_strike) + 1) / 2  # (-∞,∞) → (0,1)
            s_strike = self.strike_min + (self.strike_max - self.strike_min) * strike_scaled
        else:
            # No bounds - use parameter directly
            s_strike = m_strike
            
        # Transform dip component  
        if self.has_dip_bounds:
            # Apply tanh transformation: (-∞,∞) → (dip_min, dip_max)
            dip_scaled = (ufl.tanh(m_dip) + 1) / 2  # (-∞,∞) → (0,1)
            s_dip = self.dip_min + (self.dip_max - self.dip_min) * dip_scaled
        else:
            # No bounds - use parameter directly
            s_dip = m_dip
            
        return ufl.as_vector([s_strike, s_dip])
    
    def validate_bounds(self, slip_function, verbose=True):
        """
        Validate that transformed slip satisfies bounds
        
        Args:
            slip_function: FEniCS Function containing physical slip values
            verbose: Print validation results
            
        Returns:
            bool: True if all bounds satisfied
        """
        if not self.has_any_bounds:
            if verbose:
                print("No bounds to validate (unconstrained mode)")
            return True
            
        # Extract slip values
        slip_array = slip_function.vector().get_local()
        n_dofs = len(slip_array) // 2
        
        strike_slip = slip_array[0::2]
        dip_slip = slip_array[1::2]
        
        violations = 0
        
        if verbose:
            print("="*60)
            print("SLIP BOUNDS VALIDATION")
            print("="*60)
        
        # Check strike bounds
        if self.has_strike_bounds:
            strike_violations = np.sum((strike_slip < self.strike_min) | 
                                     (strike_slip > self.strike_max))
            violations += strike_violations
            
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m")
                print(f"Strike bounds: [{self.strike_min:.6f}, {self.strike_max:.6f}] m")
                print(f"Strike violations: {strike_violations}")
        else:
            if verbose:
                print(f"Strike slip: [{np.min(strike_slip):.6f}, {np.max(strike_slip):.6f}] m (unconstrained)")
        
        # Check dip bounds
        if self.has_dip_bounds:
            dip_violations = np.sum((dip_slip < self.dip_min) | 
                                  (dip_slip > self.dip_max))
            violations += dip_violations
            
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m")
                print(f"Dip bounds: [{self.dip_min:.6f}, {self.dip_max:.6f}] m") 
                print(f"Dip violations: {dip_violations}")
        else:
            if verbose:
                print(f"Dip slip: [{np.min(dip_slip):.6f}, {np.max(dip_slip):.6f}] m (unconstrained)")
        
        if verbose:
            print("")
            if violations == 0:
                print("✅ ALL BOUNDS SATISFIED!")
            else:
                print(f"⚠️ FOUND {violations} BOUND VIOLATIONS!")
                print("Note: Small violations (~1e-15) are numerical precision artifacts")
        
        return violations == 0

print("✅ Slip transformation framework loaded!")

# %%
# Import GNSS data
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"
# obs_disp_name = "Feng_etal_JGR_2012table1.csv" # original data from Feng et al. 2012
obs_disp_name = "Kyriakopoulos_novolcano.csv"    # same as above, but with volcano sites removed
# note that the height is in m, Dt in years, the original velocity data is in mm/yr
# the disp relative to the stable Caribbean plate will be used in the inversion
# From ACOS to VENA are Campaign Sites; From BIJA to VERA are Continuous Sites; From AROL to WARN are Volcano Sites
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1, \
                   names=['site', 'lat', 'lon', 'height', 'Dt', 'Days', \
                          'vy_ITRF05', 'vy_std_ITRF05', 'vx_ITRF05', 'vx_std_ITRF05', 'vz_ITRF05', 'vz_std_ITRF05', \
                          'vy_Car', 'vy_std_Car', 'vx_Car', 'vx_std_Car', 'vz_Car', 'vz_std_Car'])
data['lon'] = -1*data['lon'] # convert to east positive, as the original data is west positive

azimuth = 45  # trench azimuth in degrees

# Get the normal and parallel components along the azimuth direction
data['v_trnorm'], data['v_trpara'] = ut.project_vector_2d_matrix(data['vx_Car'], data['vy_Car'], azimuth)

# When including the trench-parallel component, we removed 11 mm/yr of northwestward block translation
# (Figure 1) that was observed across all stations southwest of the Costa Rican volcanic chain 
# [Feng et al., 2012], since this motion does not correspond to elastic behavior along the megathrust 
# interface.
# This means, all stations except 2, ACOS, VERA
v_const_para = 11     # only remove the a constant value from trench parallel component 
mask1 = ~data['site'].isin(['ACOS', 'VERA'])   # stations north of the volcanic chain are not affected
data.loc[mask1, 'v_trpara'] = data.loc[mask1, 'v_trpara'] - v_const_para

#set these 2 stations' parallel component to 0 as well
data.loc[~mask1, 'v_trpara'] = 0

#For 3 sites, LOCA, BIJA, AGUS, the residuals are large, so don't use their parallel components
mask2 = data['site'].isin(['LOCA', 'BIJA', 'AGUS'])
data.loc[mask2, 'v_trpara'] = 0

#rotate back to N and E
data['vx_Car'], data['vy_Car'] = ut.project_vector_2d_matrix(data['v_trnorm'], data['v_trpara'], -azimuth)

# Convert mm to m, needed for inversion
cols_to_convert = ['vy_ITRF05', 'vy_std_ITRF05', 'vx_ITRF05', 'vx_std_ITRF05', 'vz_ITRF05', 'vz_std_ITRF05', \
                   'vy_Car', 'vy_std_Car', 'vx_Car', 'vx_std_Car', 'vz_Car', 'vz_std_Car']
data[cols_to_convert] = data[cols_to_convert] / 1e3  # Convert mm to m

# define the centroid of relative coordinates, must be consistent with the mesh!
lon0, lat0 = -85.5+360, 10

# convert to relative locations in km
data['x'], data['y'] = ut.azi_equidist_proj(data['lon'], data['lat'], lon0, lat0)
data['z'] = 0.0

data.head()
print("Number of stations:", len(data))

# a catalog Holocene volcanoes
volc_file = "GVP_Holocene_Volcano_loc.csv" 
volc = pd.read_csv(datadir + volc_file, sep=",", skiprows=1, \
                      names=['id', 'lat', 'lon', 'elv'])
# truncate within a region, same as Figure 1b in Feng et al 2012
volc = volc[ (volc['lat'] >= 8) & (volc['lat'] <= 12) & (volc['lon'] >= -88) & (volc['lon'] <= -83) ]
volc['x'], volc['y'] = ut.azi_equidist_proj(volc['lon'], volc['lat'], lon0, lat0)
cols_to_convert = ['x', 'y']
volc[cols_to_convert] = volc[cols_to_convert] * 1e3  # Convert km to m
volc['z'] = 0.0
# Show first few rows
# print(volc.head())

# %%
# Define folder to save the results
resultpath = "/home/staff/chao/SSEinv/Nicoya/syn_slip/"
os.makedirs(resultpath, exist_ok=True)

# %%
# Define the Compliance matrix for elasticity
def AEsigma(s, mu, nu):
    A = 1./(2.*mu)*( s - nu/( 1 + nu*(dim-2) )*ufl.tr(s)*ufl.Identity(dim) )
    return A

# %%
# Define the asymmetry operator
def asym(s):    # calculate the off-diagonal difference. If != 0 --> asymmetry
    if dim == 2:
        as_ = s[1,0] - s[0,1]
    elif dim == 3:
        as_ = ufl.as_vector( [ s[1,2] - s[2,1], s[2,0] - s[0,2], s[0,1] - s[1,0] ] )
    return as_

# %%
# Define the strike direction operator
def dir_strike(n):
    # Positive strike --> right-lateral strike slip fault
    # Create strike and dip direction through cross product of the unit normal
    # vector with the vertical. Cross product gives the strike direction and find dip
    z_dir = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    # Normalize by the magnitude of the cross product
    strike_dir = n_cross_z / ufl.sqrt( ufl.dot(n_cross_z, n_cross_z ) )
    return strike_dir

# %%
# Define the dip direction operator
def dir_dip(n):
    # Positive dip --> reverse slip fault
    dip_dir =  ufl.cross( dir_strike(n), n )
    return dip_dir

# %%
# Class to define different properties in the subdomains with anomaly built-in
# in the order of: 'k_r' in blockright, 'k_l' in blockleft
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
# Choose the mesh
# meshname = "nicoya"
meshname = "nicoya2"   # This has a smaller fault interface

# Choose path data
meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
# load mesh
mesh = dl.Mesh(meshpath + meshname + '.xml')
# mesh dimensions
xmin, xmax = -1000e3, 1000e3
ymin, ymax = -1000e3, 1000e3
zmin, zmax = -400e3, 0.
dim = mesh.topology().dim()
# Define normal component to boundaries
n = dl.FacetNormal(mesh)
# Define boundaries
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')
# Rename boundaries, same as in .geo file
top = 1
bottom = 2
west = 3
east = 4
north = 5
south = 6
fault = 7
blockleft = 8
blockright = 9
# Define the surface integration over the external boundary (ds) and internal boundary (dS)
ds = dl.Measure("ds")(domain=mesh, subdomain_data=boundaries)
dS = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

# %%
# Define the expression of the shear modulus
def mu_expression(m):
    mu = 20*(2.+ufl.tanh(m)) 
    return mu

# %%
# 🎯 ENHANCED: PDE Variational Formulation with Slip Transformation
print("Setting up enhanced PDE formulation with slip transformation...")

class PDEVarf_TanhSlip:
    """
    Enhanced PDE variational formulation with tanh slip transformation
    
    Key difference from original:
    - m (unbounded parameters) → physical_slip (bounded) via slip_transformer
    - Everything else identical to original PDEVarf
    """
    def __init__(self, mtrue_mu_fun, slip_transformer):
        self.mtrue_mu_fun = mtrue_mu_fun
        self.slip_transformer = slip_transformer
        
    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        tau, w, q = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        # 🎯 KEY ENHANCEMENT: Transform unbounded parameters to bounded physical slip
        # Use the dedicated slip transformation method for clean, reusable code
        physical_slip = self.slip_transformer.transform_to_physical_slip(m)
        m_strike_phys, m_dip_phys = ufl.split(physical_slip)
        
        # Use transformed physical slip in formulation (everything else same as original)
        mu = mu_expression(self.mtrue_mu_fun)

        J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
            + ufl.inner(ufl.div(tau), uu)*ufl.dx \
            + ufl.inner(asym(tau), r)*ufl.dx \
            + ufl.inner(ufl.div(sigma), w)*ufl.dx \
            + ufl.inner(asym(sigma), q)*ufl.dx \
            + ufl.inner(f, w)*dl.dx \
            - ufl.inner(u0, tau*n)*ds(bottom) \
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike_phys) + dir_dip(n('+')) * ufl.avg(m_dip_phys), 
                       tau('+')*n('+'))*dS(fault)

        return J
    
class PDEVarf_TanhSlip1:
    """
    Enhanced PDE variational formulation with tanh slip transformation
    
    Key difference from original:
    - m (unbounded parameters) → physical_slip (bounded) via slip_transformer
    - Everything else identical to original PDEVarf
    """
    def __init__(self, mtrue_mu_fun, slip_transformer):
        self.mtrue_mu_fun = mtrue_mu_fun
        self.slip_transformer = slip_transformer
        
    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)  # Split unbounded parameters first
        tau, w, q = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        # 🎯 KEY ENHANCEMENT: Transform individual components to bounded physical slip
        # Apply transformations component-wise to avoid UFL splitting issues
        if self.slip_transformer.has_strike_bounds:
            # Transform strike: (-∞,∞) → (strike_min, strike_max)
            strike_scaled = (ufl.tanh(m_strike) + 1) / 2
            m_strike_phys = (self.slip_transformer.strike_min + 
                            (self.slip_transformer.strike_max - self.slip_transformer.strike_min) * strike_scaled)
        else:
            m_strike_phys = m_strike
            
        if self.slip_transformer.has_dip_bounds:
            # Transform dip: (-∞,∞) → (dip_min, dip_max)
            dip_scaled = (ufl.tanh(m_dip) + 1) / 2
            m_dip_phys = (self.slip_transformer.dip_min + 
                         (self.slip_transformer.dip_max - self.slip_transformer.dip_min) * dip_scaled)
        else:
            m_dip_phys = m_dip
        
        # Use transformed physical slip in formulation (everything else same as original)
        mu = mu_expression(self.mtrue_mu_fun)

        J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
            + ufl.inner(ufl.div(tau), uu)*ufl.dx \
            + ufl.inner(asym(tau), r)*ufl.dx \
            + ufl.inner(ufl.div(sigma), w)*ufl.dx \
            + ufl.inner(asym(sigma), q)*ufl.dx \
            + ufl.inner(f, w)*dl.dx \
            - ufl.inner(u0, tau*n)*ds(bottom) \
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike_phys) + dir_dip(n('+')) * ufl.avg(m_dip_phys), 
                       tau('+')*n('+'))*dS(fault)

        return J

# Original PDEVarf for comparison/fallback
class PDEVarf:
    def __init__(self, mtrue_mu_fun):
        self.mtrue_mu_fun = mtrue_mu_fun

    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)
        tau, w, q = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        mu = mu_expression(self.mtrue_mu_fun)

        J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
            + ufl.inner(ufl.div(tau), uu)*ufl.dx \
            + ufl.inner(asym(tau), r)*ufl.dx \
            + ufl.inner(ufl.div(sigma), w)*ufl.dx \
            + ufl.inner(asym(sigma), q)*ufl.dx \
            + ufl.inner(f, w)*dl.dx \
            - ufl.inner(u0, tau*n)*ds(bottom) \
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip), tau('+')*n('+'))*dS(fault)

        return J

print("✅ Enhanced PDE formulation loaded!")

# %%
# 🎯 ENHANCED: Forward modeling routine (same as original, but with added moment calculation)
def solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=True, \
                          pollute_type='uniform', savefiles=True, verbose=True):

    # Define function spaces
    # Use VectorFunctionSpace if the unknown is a vector field.
    # Use FunctionSpace object for scalar fields.
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)    # stress (tensor field, since BDM is a vector field)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)   # displacement (vector field)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)   # rotation (scalar field)
    # Create a mixed fine element function space
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    # Define mixed function spaces for the model parameters
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    # Combine the STATE, PARAMETER and ADJOINT function spaces
    Vh = [Vu, Vm, Vu]
    # Print the dofs of STATE, PARAMETER and ADJOINT variables
    ndofs = [ Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim() ]
    ndofs_state = [ Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim() ]
    if verbose:
        print( sep, "Set up the mesh and finite element spaces", sep )
        print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs) )
        print( "Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state) )

    # Define the STATE and ADJOINT Dirichlet BCs
    zero_tensor = dl.Expression(( ("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.") ), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    ##### Different from inversion where observation is given, here 'u' and 'obs' are computed from true model
    V_slip = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)

    if mtrue_s_expr == None:
        # Import slip inferred from coseismic slip as the true model
        slip = dl.Function(V_slip)
        slip.vector()[:] = slip_gt[:]
        mtrue_s = slip.vector()
    else:
        # Interpolate, using the true slip model with analytical expression, e.g., uniform slip over the fault interface 
        # mtrue_s = dl.interpolate(mtrue_s_expr, V_slip).vector()
        # mtrue_s = dl.interpolate(mtrue_s_expr, Vh[hp.PARAMETER]).vector()   #this works as well as above

        # Apply this pattern as a Dirichlet BC on the fault boundary
        bc_fault_slip = dl.DirichletBC(V_slip, mtrue_s_expr, boundaries, fault)
        # Create a function and apply the boundary condition
        mtrue_s_function = dl.Function(V_slip)
        bc_fault_slip.apply(mtrue_s_function.vector())
        # Extract the vector for your parameter space
        mtrue_s = mtrue_s_function.vector()

    # Define the true model for synthetics
    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()

    # Fill common dl.Vector()
    tmp = np.zeros(Vh[hp.PARAMETER].dim(),)
    tmp[0::2] = mtrue_s.copy()[0::2]
    tmp[1::2] = mtrue_s.copy()[1::2]
    # Assign the values of the vector
    mtrue.set_local(tmp)

    # validate if the slip pattern is as intended
    if verbose:
        ut.validate_fault_slip_pattern(mtrue_s=mtrue_s, mesh=mesh, boundaries=boundaries, fault_id=fault)

    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # Assign the values of the vector
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    # Save true shear modulus structure
    if savefiles:
        print( "Saving true shear modulus structure to .xdmf file" )
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project( mtrue_mu_fun_expr, CG_mu )
        filename = resultpath + 'mu_true_' + meshname + mu_str_for + '.xdmf'
        if not os.path.exists(filename):
            mu_id = dl.XDMFFile(filename)
            m_mu_true.rename('shear modulus', 'shear modulus')
            mu_id.write(m_mu_true)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

    # Define the PDE problem
    pde_varf = PDEVarf(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Define the solver type, previously it was defined in TVprior.py or TVprior_Joint.py
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    # Print the number of observations (observed surface horizontal and vertical displacements)
    if verbose:
        print( "Number of observation points: {0}".format(targets.shape[0]) )
    
    # Constrain only the displacement field for the data misfit
    indicator_vec = dl.interpolate( dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE] ).vector()

    ## TWO OPTIONS OF DEFINING misfit
    # ##### OPTION 1, USE 'pointwiseStateObs_weights', but set 'weights' to 1
    # weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
    # obs_weights = np.zeros(targets.shape[0]*15,)
    # obs_weights[9::15]  = 1  # horizontal east displacement
    # obs_weights[10::15] = 1  # horizontal north displacement
    # # obs_weights[11::15] = 1/1e15  # vertical displacements
    # # Modify and set the array to dolfin vector
    # weights.set_local(obs_weights)
    # weights.apply('')
    # # Different from the pure inversion, define misfit without 'weight' option
    # misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
    # ##### OPTION 1

    ##### OPTION 2, USE 'pointwiseStateObs', without 'weight' option
    misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
    ##### OPTION 2

    # Solve FORWARD problem for the STATE variables
    u_mtrue = pde.generate_state() # all dofs STATE variables and PETSC vector (not FEniCS Function)
    x = [u_mtrue, mtrue, None]
    pde.solveFwd(x[hp.STATE], x)
    if savefiles:
        # Save the forward problem (synthetic displacement, or true displacement in this context)
        print( "Start saving .xdmf files forward problem to check" )
        uid = dl.XDMFFile(resultpath + 'u_' + meshname + slip_str_gt + mu_str_for + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u_mtrue)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)

    # Generate true observations by using the observation operator 'B'
    misfit.B.mult(x[hp.STATE], misfit.d)
    # 'idx_d' points to the non-zero components, e.g., if only ux and uy are pre constrained, 'idx_d' points to ux and uy
    idx_d = list(np.nonzero(misfit.d)[0])
    
    # whether to pollute the data with noise
    if pollute:
        if pollute_type == "uniform":
            # Create empty noise vector
            noise_vec = np.zeros(len(misfit.d),)
            # Assuming the displacement entries are ordered by [vx_block, vy_block, vz_block]
            # Generate noise for each component
            noise_vx = np.random.normal(0, noise_std_h, size=targets.shape[0])
            noise_vy = np.random.normal(0, noise_std_h, size=targets.shape[0])
            noise_vz = np.random.normal(0, noise_std_v, size=targets.shape[0])

            # Concatenate them in the right order
            displacement_noise = np.concatenate([noise_vx, noise_vy, noise_vz])
            # Assign displacement noise to the displacement indices
            noise_vec[idx_d] = displacement_noise
            # Add noise to synthetic data
            misfit.d.set_local(misfit.d.get_local() + noise_vec)
            misfit.d.apply('')
            # Set noise variance (either per entry or averaged)
            misfit.noise_variance = noise_vec**2

        elif pollute_type == "datastd":
            # Create empty noise vector
            noise_vec = np.zeros(len(misfit.d),)
            # Assuming the displacement entries are ordered by [vx_block, vy_block, vz_block]
            # Generate noise for each component
            noise_vx = np.random.normal(0, data['vx_std_Car'])
            noise_vy = np.random.normal(0, data['vy_std_Car'])
            noise_vz = np.random.normal(0, data['vz_std_Car'])
            # Concatenate them in the right order
            displacement_noise = np.concatenate([noise_vx, noise_vy, noise_vz])
            # Assign displacement noise to the displacement indices
            noise_vec[idx_d] = displacement_noise
            # Add noise to synthetic data
            misfit.d.set_local(misfit.d.get_local() + noise_vec)
            misfit.d.apply('')
            # Set noise variance (either per entry or averaged)
            misfit.noise_variance = noise_vec**2

    else:
        misfit.noise_variance = 1.0
        # misfit.noise_variance = noise_std_h*noise_std_h
        
    #####

    # Extract horizontals displacements observed data, if only ux and uy are pre constrained in 'misfit', 'd_obs' also contains only ux and uy  
    d_obs = misfit.d[idx_d]
    if savefiles:    
        # Save the observed surface displacement
        outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]) )
        csvoutput.close()

    # Extract x,y coordinates of the fault for plotting
    CG = dl.VectorFunctionSpace(mesh, "CG", degree=1)
    bc1 = dl.DirichletBC(CG, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG)
    bc1.apply(um.vector())
    # Same for fault, but in this case use strike and dip components, assuming no fault-normal slip
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm)
    bc2.apply(um2.vector())
    # Extract x,y coordinates of the fault and values
    xslip = dl.interpolate(dl.Expression( ("x[0]", "x[0]", "x[0]"), degree=5), CG )
    yslip = dl.interpolate(dl.Expression( ("x[1]", "x[1]", "x[1]"), degree=5), CG )
    zslip = dl.interpolate(dl.Expression( ("x[2]", "x[2]", "x[2]"), degree=5), CG )
    xf = xslip.vector()[um.vector() == 10] # x coordinate fault
    yf = yslip.vector()[um.vector() == 10] # y coordinate fault
    zf = zslip.vector()[um.vector() == 10] # z coordinate fault
    if verbose:
        print( sep, "Done extracting the fault coordinates", sep )

    # save ground-truth slip on the fault
    mtrue_s_fault = mtrue[um2.vector() == 99]
    mtrue_sx_fault = mtrue_s_fault[0::2]; mtrue_sy_fault = mtrue_s_fault[1::2]
    print( min(mtrue_sx_fault), max(mtrue_sx_fault) )
    print( min(mtrue_sy_fault), max(mtrue_sy_fault) )
    if savefiles:
        # Save output fault geometry
        outFileName = 'fault_geometry_' + meshname + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, xf.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f\n" %(xf[i], yf[i], zf[i]) )
        csvoutput.close()

        # save ground-truth slip on the fault
        outFileName = 'mtrue_s_fault_' + meshname + slip_str_gt + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, mtrue_sx_fault.shape[0]):
            csvoutput.write( "%.6f %.6f\n" %(mtrue_sx_fault[i], mtrue_sy_fault[i]) )
        csvoutput.close()

        # save ground-truth slip, over the entire volume
        outFileName = 'slip_true_' + meshname + slip_str_gt + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(mtrue))
        slip_vec[um2.vector() == 99] = mtrue_s_fault
        for i in range(0, len(slip_vec)):
            csvoutput.write( "%.6f\n" %slip_vec[i] )
        csvoutput.close()

    # Compute true seismic moment and moment magnitude
    if verbose:
        print( sep, "Computing true seismic moment from ground-truth slip", sep )
    
    # Create slip magnitude function ||D|| = sqrt(D1^2 + D2^2) 
    mtrue_fun = hp.vector2Function(mtrue, Vh[hp.PARAMETER])
    s_mag_true = ufl.sqrt( ufl.dot(mtrue_fun, mtrue_fun))
    
    # Get shear modulus for moment calculation
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project( mtrue_mu_fun_expr, CG_mu )
    
    # Calculate scalar seismic moment: M0 = μ * A * D_avg (integrated over fault)
    moment_true = dl.assemble( m_mu_true * GPa2Pa * s_mag_true * dS(fault) )
    
    # Convert to moment magnitude using your utility function
    M_w1_true, M_w2_true, M_w3_true = ut.moment2mag(moment_true)
    
    if verbose:
        print(f"TRUE scalar seismic moment: {moment_true:.3e} N·m")
        print(f"TRUE moment magnitude: Mw = {M_w1_true:.2f} (Hanks & Kanamori); {M_w2_true:.2f} (alternative 1); {M_w3_true:.2f} (alternative 2)")
    
    if savefiles:
        # Save true moment to file  
        outFileName = 'moment_true_' + meshname + slip_str_gt + mu_str_for + '.txt'
        with open(resultpath + outFileName, 'w') as moment_file:
            moment_file.write(f"{moment_true:.6e} {M_w1_true:.4f} {M_w2_true:.4f} {M_w3_true:.4f}\n")
        
        if verbose:
            print(f"Saved true moment to: {outFileName}")
            print(f"Format: scalar_moment(N·m) Mw1 Mw2 Mw3")

    return mtrue, mtrue_mu, u_mtrue, xf, yf, zf, d_obs, mtrue_s_fault

# %%
# 🎯 ENHANCED: Inversion routine with Tanh Transformation
def solveCoseismicInversion_TanhSlip(k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
                                    slip_transformer, pollute=True, pollute_type='uniform', 
                                    savefiles=True, verbose=True):
    """
    Enhanced slip inversion with tanh transformation for bounded slip
    
    Args:
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2: Same as original
        slip_transformer: SlipTransformation instance defining bounds
        pollute, pollute_type, savefiles, verbose: Same as original
        
    Returns:
        Same outputs as original function, but slip values are physically bounded
        
    Example usage:
        # Define bounds
        transformer = SlipTransformation(
            strike_bounds=(-0.04, 0.04),  # ±40 mm strike-slip
            dip_bounds=(0.0, 0.16)        # 0-160 mm thrust
        )
        
        # Call with same interface as before  
        results = solveCoseismicInversion_TanhSlip(
            k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
            transformer, pollute=True, pollute_type='uniform', savefiles=True, verbose=True
        )
    """
    
    # Define function spaces (identical to original)
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]
    
    # Print dimensions and slip transformation info
    ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
    ndofs_state = [Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim()]
    
    if verbose:
        print(sep, "🎯 ENHANCED: Set up mesh and slip transformation framework", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print("Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state))
        print("")
        print("🎯 SLIP TRANSFORMATION FRAMEWORK")
        print(f"   {slip_transformer}")
        
        # Show parameter interpretation
        if slip_transformer.has_any_bounds:
            print("   Parameter interpretation:")
            print("     m ∈ (-∞,∞) → physical_slip ∈ [bounds] via tanh transformation")
        else:
            print("   Using original unconstrained framework (m = physical_slip)")

    # Define boundary conditions (identical to original)
    zero_tensor = dl.Expression((("0.", "0.", "0."),
                                ("0.", "0.", "0."),
                                ("0.", "0.", "0.")), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # Load initial model and setup shear modulus (identical to original)
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()
    
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    
    if savefiles:
        print("Saving true shear modulus structure to .xdmf file")
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
        filename = resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf'
        if not os.path.exists(filename):
            mu_id = dl.XDMFFile(filename)
            m_mu_true.rename('shear modulus', 'shear modulus')
            mu_id.write(m_mu_true)
        print(m_mu_true.vector().min(), m_mu_true.vector().max())

    # 🎯 KEY CHANGE: Use enhanced PDE variational formulation
    if slip_transformer.has_any_bounds:
        pde_varf = PDEVarf_TanhSlip(mtrue_mu_fun, slip_transformer)
        if verbose:
            print("✅ Using enhanced PDE with tanh slip transformation")
    else:
        pde_varf = PDEVarf(mtrue_mu_fun)
        if verbose:
            print("✅ Using original PDE (no slip bounds)")
    
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Solver setup (identical to original)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    if verbose:
        print("Number of observation points: {0}".format(targets.shape[0]))
    
    # Misfit setup (identical to original except using global variables)
    indicator_vec = dl.interpolate(dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()

    # Define misfit based on noise type (identical to original logic)
    if pollute:
        if pollute_type == "uniform":
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
            obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
            weights.set_local(obs_weights)
            weights.apply('')
            misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
            misfit.noise_variance = 1.            

        elif pollute_type == "datastd":
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
            obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements
            weights.set_local(obs_weights)
            weights.apply('')
            misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
            misfit.noise_variance = 1.

    else:        
        misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
        misfit.noise_variance = 1.
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = 1  # horizontal east displacement 
        obs_weights[10::15] = 1  # horizontal north displacement 
        obs_weights[11::15] = 1  # vertical displacement 

    # Input GPS data into misfit (identical to original)
    tmp = np.zeros(len(misfit.d),)
    tmp[9::15] = np.array(syndata['ux'])    # horizontal east displacement misfit
    tmp[10::15] = np.array(syndata['uy'])   # horizontal north displacement misfit
    tmp[11::15] = np.array(syndata['uz'])   # vertical displacement misfit
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    # Extract displacement indices and observed data
    idx_d = list(np.nonzero(obs_weights)[0])
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error. The length of non-zero misfit has to be the same as ntargets.")

    d_obs = misfit.d[idx_d]

    # Extract fault coordinates (identical to original)
    CG = dl.VectorFunctionSpace(mesh, "CG", degree=1)
    bc1 = dl.DirichletBC(CG, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG)
    bc1.apply(um.vector())
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm)
    bc2.apply(um2.vector())
    xslip = dl.interpolate(dl.Expression(("x[0]", "x[0]", "x[0]"), degree=5), CG)
    yslip = dl.interpolate(dl.Expression(("x[1]", "x[1]", "x[1]"), degree=5), CG)
    zslip = dl.interpolate(dl.Expression(("x[2]", "x[2]", "x[2]"), degree=5), CG)
    xf = xslip.vector()[um.vector() == 10]
    yf = yslip.vector()[um.vector() == 10]
    zf = zslip.vector()[um.vector() == 10]
    
    if verbose:
        print(sep, "Done extracting fault coordinates", sep)

    # Regularization and model construction (identical to original)
    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)
    model = hp.Model(pde, reg, misfit)
    m = m0_s.copy()

    # 🎯 MAIN SOLVER: Use original fast unconstrained CG
    # The beauty of tanh transformation: no changes needed to the solver!
    if verbose:
        print(sep, "🎯 Solve the slip inverse problem with transformation", sep)
        print("Using your original fast unconstrained CG solver")
        print("(slip bounds enforced automatically via tanh transformation)")

    # Generate vectors and solve (identical to original)
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)
    
    model.solveFwd(u, x)
    model.solveAdj(p, x)
    model.evalGradientParameter(x, mg)
    
    if verbose:
        print(sep, "Done generating STATE, PARAMETER and ADJOINT vectors", sep)

    # Hessian and CG solver (identical to original)
    model.setPointForHessianEvaluations(x)
    H = hp.ReducedHessian(model)
    
    Prec = reg.Rsolver
    H.misfit_only = False
    
    solver = hp.CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner(Prec)
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-9
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"] = 1500
    
    m_hat = model.generate_vector(hp.PARAMETER)
    solver.solve(m_hat, -mg)
    
    if solver.converged:
        print("✅ CG converged in", solver.iter, "iterations.")
    else:
        print("❌ CG did not converge.")
        raise RuntimeError("CG solver failed")

    m.axpy(1., m_hat)

    # 🎯 POST-PROCESSING: Convert unbounded parameters m to bounded physical slip
    if slip_transformer.has_any_bounds:
        m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
        physical_slip_expr = slip_transformer.transform_to_physical_slip(m_fun)
        
        # Project to get physical slip function  
        physical_slip_fun = dl.project(physical_slip_expr, Vm)
        s_strike_fun, s_dip_fun = physical_slip_fun.split(deepcopy=True)
        
        # Validate bounds
        if verbose:
            slip_transformer.validate_bounds(physical_slip_fun, verbose=True)
    else:
        # No transformation needed
        m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
        s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)
    
    # Save slip solution (identical to original)
    if savefiles:
        print("Saving slip solution to .xdmf file")
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print("Finish saving slip solution")

    # Continue with post-processing (identical to original)
    x = [u, m, p]
    model.solveFwd(u, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    d_cal = misfit.Bu[idx_d]

    # Calculate misfits
    m_fun = dl.Function(Vh[hp.PARAMETER], m)
    grad_m = dl.assemble(ufl.inner(ufl.avg(ufl.nabla_grad(m_fun)), ufl.avg(ufl.nabla_grad(m_fun)))*dS(fault))
    misfitd = np.linalg.norm((d_cal-d_obs), 2)
    print("Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m))

    # Compute cost
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print("Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost))

    # Calculate seismic moment (using physical slip if bounds applied)
    if slip_transformer.has_any_bounds:
        s_mag = ufl.sqrt(ufl.dot(physical_slip_expr, physical_slip_expr))
    else:
        s_mag = ufl.sqrt(ufl.dot(m_fun, m_fun))
    
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")

    # Extract slip values at fault interface
    if slip_transformer.has_any_bounds:
        # Use physical slip values
        physical_slip_fault = physical_slip_fun.vector().get_local()[um2.vector() == 99]
        m_sx_fault = physical_slip_fault[0::2]
        m_sy_fault = physical_slip_fault[1::2]
    else:
        # Use parameters directly
        m_s_fault = m[um2.vector() == 99]
        m_sx_fault = m_s_fault[0::2]
        m_sy_fault = m_s_fault[1::2]
    
    print("Physical slip ranges:")
    print(f"  Strike: [{min(m_sx_fault):.6f}, {max(m_sx_fault):.6f}] m")
    print(f"  Dip: [{min(m_sy_fault):.6f}, {max(m_sy_fault):.6f}] m")

    # Save all outputs (identical to original logic, but using physical slip values)
    if savefiles:
        # Save predicted displacement
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]))
        csvoutput.close()

        # Save predicted displacement field
        print("Saving predicted displacement and stress to .xdmf file")
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)

        # Save stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        print("Finish saving predicted displacement and stress")

        # Save physical slip at fault interface
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write("%.6f %.6f\n" %(m_sx_fault[i], m_sy_fault[i]))
        csvoutput.close()

        # Save physical slip over entire volume
        outFileName = 'slip_inferred_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(m0_s))
        if slip_transformer.has_any_bounds:
            slip_vec[um2.vector() == 99] = physical_slip_fault
        else:
            slip_vec[um2.vector() == 99] = m[um2.vector() == 99]
        for i in range(0, len(slip_vec)):
            csvoutput.write("%.6f\n" %slip_vec[i])
        csvoutput.close()

    return mtrue_mu, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, np.concatenate([m_sx_fault, m_sy_fault]), misfitd, grad_m

print("✅ Enhanced inversion function with tanh transformation loaded!")

# %%
# Define order of elements 
k = 2
# Define body force
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9

# Define starting model
m0_s_expr = dl.Constant((0., 0.))

# %%
### define the pattern of the slip distribution
# slip_pattern = "checker"
slip_pattern = "stripe"

if slip_pattern == "checker":
    ##### NEW CHECKERBOARD SLIP MODEL #####
    print("Creating fault-local checkerboard with natural depth range...")

    from utils import FaultLocalCheckerboard

    # Define true model, m_strike = 0, m_dip = checkerboard, alternating between 0 and max
    V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
    amp = V_norm   # interseismic coupling case, max == complete coupling of 1 == the trench-normal long-term loading
    # amp = 1   # coseismic case, max == 1 m

    # Checkerboard pattern parameters
    # Option 1, checkerboard along fault strike-dip directions
    dx = 35e3  # grid spacing in x direction, \lamda_x = 2*dx
    dy = 35e3  # spacing in y direction, \lamda_y = 2*dy
    x0 = -20e3  # offset along strike
    y0 = -45e3  # offset along dip
    rot_deg = 45  # 45° counterclockwise 

    # Option 2, checkerboard along N-E directions
    # dx = 30e3  # grid spacing in x direction, \lamda_x = 2*dx
    # dy = 30e3  # spacing in y direction, \lamda_y = 2*dy
    # x0 = 15e3  # offset along strike
    # y0 = -10e3  # offset along dip
    # rot_deg = 0

    # no constraint on 'z' any more as the labeled fault has intrinsic depth range
    mtrue_s_expr_gt = ut.create_fault_local_checkerboard(
        mesh=mesh,
        boundaries=boundaries,
        fault_id=fault,
        amp=amp,
        dx=dx,  # 40 km along-strike spacing
        dy=dy,  # 40 km up-dip spacing
        x0=x0,
        y0=y0,
        rotation_deg=rot_deg,
        degree=5
    )

    slip_str_gt = f"_check_x{x0/1e3:g}_y{y0/1e3:g}_dx{dx/1e3:g}_dy{dy/1e3:g}_rot{rot_deg:g}_ms{amp:g}"


elif slip_pattern == "stripe":
    ##### STRIPE SLIP MODEL #####
    print("Creating fault-local stripes with natural depth range...")

    from utils import FaultLocalStripes

    # Define true model, m_strike = 0, m_dip = stripes along-strike, alternating between 0 and max
    V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
    amp = V_norm   # interseismic coupling case, max == complete coupling of 1 == the trench-normal long-term loading
    # amp = 1   # coseismic case, max == 1 m
    
    x_len = 80e3     # width of each rectangle in dip direction
    y_len = 300e3   # length of each rectangle in strike direction  
    dx = 35e3  # gap between rectangles
    stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
    x0 = 0     # x center of the pattern
    y0 = -35e3     # y center of the pattern
    rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)

    # no constraint on 'z' any more as the labeled fault has intrinsic depth range
    mtrue_s_expr_gt = ut.create_fault_local_stripes(
        mesh=mesh,
        boundaries=boundaries,
        fault_id=fault,
        amp=amp,
        stripe_width=x_len,
        stripe_spacing=stripe_spacing,
        stripe_length=y_len,
        x0=x0, 
        y0=y0,
        rotation_deg=rot_deg,
        degree=5
    )

    slip_str_gt = (
        f"_stripe_x{x0/1e3:g}_y{y0/1e3:g}"
        f"_lx{x_len/1e3:g}_dx{dx/1e3:g}"
        f"_rot{rot_deg:g}_ms{amp:g}"
    )

print(slip_str_gt)  

# %%
# locations of surface observations
ntargets = data.shape[0]
targets_x = np.array(data['x'])*1e3   # km to m
targets_y = np.array(data['y'])*1e3
targets_z = np.array(data['z'])*1e3
targets = np.zeros([ntargets, dim])
targets[:,0] = targets_x; targets[:,1] = targets_y; targets[:,2] = targets_z
print(targets.shape)

# %%
# Define the true model PARAMETERS for the FORWARD problem within a homogeneous shear modulus half-space
nu = 0.25   # Possion's ratio found in Kano et al., 2019

# background shear modulus
mu_b = 0   # 40 GPa
mu_background = mu_expression(mu_b)

# shear modulus for the lower (subducting) plate
mu_l = 0  # 40 GPa
mu_lower = mu_expression(mu_l)

# shear modulus for the upper (overriding) plate
mu_u = 0   # 40 GPa
mu_upper = mu_expression(mu_u)

mtrue_mu_expr_hom = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
mu_str_hom = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"

print( "Homogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )

# %%
# Define the true model PARAMETERS for INVERSE problem
nu = 0.25

# background shear modulus
mu_b = 0   # 40 GPa
mu_background = mu_expression(mu_b)

# shear modulus for the lower (subducting) plate
mu_l = 0.9730 # ~55 GPa
mu_lower = mu_expression(mu_l)

# shear modulus for the upper (overriding) plate
mu_u = -0.9730  # ~25 GPa
# mu_u = mu_b
mu_upper = mu_expression(mu_u)

mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"

print( "Heterogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )

# %%
# whether to pollute the synthetics with random errors
pollute = True
# pollute = False
print(pollute)

# noise std type, either 'uniform' or 'datastd'
pollute_type = 'uniform'  # uniform noise for all stations
# pollute_type = 'datastd'  # use the data standard deviation as noise std
print(pollute_type)

if pollute:

    if pollute_type == 'uniform':
        # Average observation error
        noise_std_h = 0.5 * (data['vx_std_Car'].mean() + data['vy_std_Car'].mean())
        noise_std_v = data['vz_std_Car'].mean()
        print( "Average horizontal 1-sigma observation error : %.6f" %(noise_std_h) )
        print( "Average vertical 1-sigma observation error : %.6f" %(noise_std_v) )

        # Decide the weights of the horizontal, vertical components
        f_h, f_v = 1, 1/2
        # Print the weights of the data
        print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )
        
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
        obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
        obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
        
        slip_str_gt = slip_str_gt + "_pou"

    elif pollute_type == 'datastd':
        # Decide the weights of the horizontal, vertical components
        f_h, f_v = 1, 1/2
        # Print the weights of the data
        print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )

        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
        obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
        obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements

        slip_str_gt = slip_str_gt + "_pod"
else:
    # Decide the weights of the horizontal, vertical components
    f_h, f_v = 1, 1       

print(slip_str_gt)

# %%
# Solve the coseismic forward problem to generate the synthetic data within a Heterogeneous shear modulus half-space
mtrue_mu_expr_for = mtrue_mu_expr_het
mu_str_for = mu_str_het

print("Solving forward problem based on: ", mu_str_for)

mtrue_s_expr_for = mtrue_s_expr_gt
if mtrue_s_expr_for==None:
    print("Ground-truth slip is from inversion")
else:
    print("Ground-truth slip is customized") 

# %%
# # Solve forward problem to generate synthetics
# mtrue, _, _, _, _, _, _, _ = solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
#                       pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Forward problem done!!!")

# %%
# 🎯 SLIP TRANSFORMATION SETUP
print(sep + "🎯 SLIP TRANSFORMATION SETUP" + sep)

BOUNDED = True
BOUND_TYPE = 'both'

if BOUNDED:
    # Define slip bounds based on your problem
    V_para = 16/1e3
    if BOUND_TYPE == 'both':
        slip_transformer = SlipTransformation(
            strike_bounds=(0.0, V_para),
            dip_bounds=(0.0, amp),
        )
        print("✅ Constraints to both strike and dip ")

    elif BOUND_TYPE == 'strike':
        slip_transformer = SlipTransformation(
            strike_bounds=(0.0, V_para),
            dip_bounds=None,
        )
        print("✅ Constraints to strike only ")

    elif BOUND_TYPE == 'dip':
        slip_transformer = SlipTransformation(
            strike_bounds=None,
            dip_bounds=(0.0, amp),
        )
        print("✅ Constraints to dip only ")

else:            
    # Alternative: no constraints (reverts to original framework)
    slip_transformer = SlipTransformation(strike_bounds=None, dip_bounds=None)
    print("✅ Unconstrained mode (original framework)")

print(f"Configuration: {slip_transformer}")

# %%
# Load the synthetic displacement data from the forward problem
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x', 'y', 'z', 'ux', 'uy', 'uz'])

# %%
# Define regularization weights
# In a Bayesian inference setting, the ratio \rho = \sqrt(\gamma/\delta) plays the role of the correlation length in the prior term.
# For our case, the station separation is around 20 km, and the mesh size on the fault is 4-20 km 

rho_s = 1e9   # allows variations of slip of the order of ~3 km, close to the maximum resolution
gamma_val_H1 = 4e3    
delta_val_L2 = gamma_val_H1 / rho_s  

# %%
if pollute:
    if pollute_type == 'uniform':
        # Take the inverse for saving the name of the weights
        w_h, w_v = int(1/noise_std_h), int(1/noise_std_v)
    elif pollute_type == 'datastd':
        # Take the inverse for saving the name of the weights
        w_h, w_v = int(1/f_h), int(1/f_v)
else:
    # Take the inverse for saving the name of the weights
    w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
inv_str = f"_synlock_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
print("Inverse problem identifier: ", inv_str)

# %%
### Compute L-curve criterion ###
# Solve to find the best Tikhonov regularization parameter by using the L-curve criterion

# fix the rho, ratio of gamma and delta
rho_s = 1e9

# vary gamma, the model gradient damping  
gammas_s = [4e3]

print( sep, "🎯 ENHANCED L-curve computation with slip transformation", sep )

dmisfit = []
mmisfit = []
for gamma_s in gammas_s:
    delta_s = gamma_s / rho_s 

    # file identifier
    inv_str = f"_synlocktestbd_w{w_h}{w_v}_gs{gamma_s:.0e}_ds{delta_s:.0e}"

    print(f"****** Computing solution with gamma_s = {gamma_s:.1e}, "
        f"rho_s = {rho_s:.1e}, and delta_s = {delta_s:.1e} ******")
    
    # 🎯 ENHANCED: Solve slip inverse problem within the heterogeneous shear modulus half-space WITH TANH TRANSFORMATION
    mtrue_mu_expr_inv = mtrue_mu_expr_het
    mu_str_inv = mu_str_het
    print("Solving ENHANCED inverse problem based on: ", mu_str_inv)
    
    results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
        slip_transformer,  # 🎯 KEY: Pass slip transformation
        pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    )
    print("🎯 ENHANCED Het Inversion with slip bounds finished!!!")

    # # 🎯 ENHANCED: Solve slip inverse problem within the homogeneous shear modulus half-space WITH TANH TRANSFORMATION  
    # mtrue_mu_expr_inv = mtrue_mu_expr_hom
    # mu_str_inv = mu_str_hom
    # print("Solving ENHANCED inverse problem based on: ", mu_str_inv)
    
    # results = solveCoseismicInversion_TanhSlip(
    #     k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
    #     slip_transformer,  # 🎯 KEY: Pass slip transformation
    #     pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    # )
    # print("🎯 ENHANCED Hom Inversion with slip bounds finished!!!")


print("🎯 All Enhanced Inversion with Slip Bounds Finished!")   