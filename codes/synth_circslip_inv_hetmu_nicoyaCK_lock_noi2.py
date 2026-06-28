# %% [markdown]
# # Synthetic Slip inversion of the GNSS coseismic displacement measurements of at Nicoya (Costa Rica) stations when the true slip distribution is like a checkerboard. Compare how the pattern can be resolved within a homogeneous or prescribed heterogeneous half-space
# 
# * Data are synthetics made from forward modeling of checkerboard slip pattern at the same stations used in the locking or coseismic cases
# * Plate model ideally should be from an updated local model in Kyriakopoulos & Newman 2016 JGRSE, but until we get the model, we stick to "Slab2"
# 
# * The goal of this code is to show how much the ground-truth slip distribution can be recovered, especially at the offset location. Through the result comparison between the homogeneous and heterogeneous cases
# 
# * This version adds noise to the synthetic data, either same random noise with a standard deviation for all stations, or use the exact STD of the real data. Correspondingly, the inversion would be revised accordingly

# %%
# limit the number of threads on clusters, by Chao, 02/06/2025
import sys, os
os.environ['OMP_NUM_THREADS'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
os.environ['VECLIB_MAXIMUM_THREADS'] = '5'
os.environ['NUMEXPR_NUM_THREADS'] = '5'

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
print(sep, "START of inversion", sep)

# %%
# NEW: Import Slip Transformation Utility
from slip_transformation_utils import SlipTransformation
print("SlipTransformation imported successfully!")

# Test the import
test_transformer = SlipTransformation(strike_bounds=(-0.04, 0.04), dip_bounds=(0.0, 0.16))
print(f"Test transformer: {test_transformer}")

# %%
# Define data directory
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"

# Import GNSS data, originally from Feng et al. 2012, but no volcano sites, both trench-parallel and normal components
obs_disp_name = "CKfig6_data_final.csv"   # the EXACT data file for figure 6 in Kyriakopoulos et al. (2016)

# the processed data has the unit of m/yr that was converted from mm/yr
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1, \
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car', \
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

# define the centroid of relative coordinates, must be consistent with the mesh!
lon0, lat0 = -84, 7     # from Christos's email
# print(lon0, lat0)

# convert to relative locations in meters, and then rotate
rot = 45  # rotation angle in degrees, positive is CCW
x_rot, y_rot  = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)

# offset in x and y direction, the same as being done to the mesh in 'Kyriakopoulos2016JGR/convert_exodus_to_msh.ipynb'
x0, y0 = 130e3, 350e3  # offset for x and y coordinates, in m
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3   # offset to match the mesh coordinates
data['z'] = 0.0

# print(data[['lon', 'lat', 'x', 'y']].head())
print("Number of stations:", len(data))

# a catalog Holocene volcanoes
volc_file = "GVP_Holocene_Volcano_loc.csv" 
volc = pd.read_csv(datadir + volc_file, sep=",", skiprows=1, \
                      names=['id', 'lat', 'lon', 'elv'])
# truncate within a region, same as Figure 1b in Feng et al 2012
volc = volc[ (volc['lat'] >= 8) & (volc['lat'] <= 12) & (volc['lon'] >= -88) & (volc['lon'] <= -83) ]
# convert to relative locations in meters, and then rotate, then offset
x_rot, y_rot  = ut.LL2ckmd(volc['lon'], volc['lat'], lon0, lat0, rot)
volc['x'], volc['y'] = x_rot - x0, y_rot - y0   # offset to match the mesh coordinates
volc['z'] = 0.0
# Show first few rows
# print(volc.head())

# %%
# CREATE DENSE OBSERVATION GRID IN LAT/LON THEN CONVERT TO MESH COORDINATES
print(sep, "Creating dense observation grid", sep)

# Define regular lat/lon grid covering study area
# Based on your image, approximate coverage around Costa Rica/Nicaragua region
# region=[-87, -84, 8.5, 11.5]    # suitable region for chopping the plate interface grid file 
# region=[-86.75, -84.4, 8.75, 11.25]    # suitable region for chopping the plate interface grid file 
# region=[-88, -83, 7.5, 12.5] 
region=[-88, -83, 7.4, 12.6] 
# region=[-87.5, -83.5, 8, 12] 
# region=[-88, -83, 6, 14]    # suitable region for chopping the plate interface grid file 

lon_min, lon_max = region[0], region[1]  # degrees longitude
lat_min, lat_max = region[2], region[3]    # degrees latitude

# Grid resolution - adjust as needed for your visualization requirements
grid_spacing_deg = 0.01  # ~2 km spacing at this latitude
# grid_spacing_deg = 0.1  # ~20 km spacing at this latitude

# Create regular lat/lon meshgrid
lon_grid = np.arange(lon_min, lon_max + grid_spacing_deg, grid_spacing_deg)
lat_grid = np.arange(lat_min, lat_max + grid_spacing_deg, grid_spacing_deg)
LON_GRID, LAT_GRID = np.meshgrid(lon_grid, lat_grid)

# Flatten for processing
lon_2d = LON_GRID.flatten()
lat_2d = LAT_GRID.flatten()

# Convert 2D grid to relative coordinates
x_rot_2d, y_rot_2d = ut.LL2ckmd(lon_2d, lat_2d, lon0, lat0, rot)
x_2d = (x_rot_2d - x0) / 1e3  # convert to km
y_2d = (y_rot_2d - y0) / 1e3

# Define depth levels (0 to -80 km with 10 km increment)
# depth_levels = np.arange(0, -80 - 10, -10)  # [0, -10, -20, ..., -80]
depth_levels = [0]
print(f"Depth levels: {depth_levels} km")

# Replicate the 2D grid for each depth level
n_2d = len(x_2d)
n_depths = len(depth_levels)
n_total = n_2d * n_depths

lon_dense = np.tile(lon_2d, n_depths)
lat_dense = np.tile(lat_2d, n_depths)
x_dense = np.tile(x_2d, n_depths)
y_dense = np.tile(y_2d, n_depths)
z_dense = np.repeat(depth_levels, n_2d)  # repeat each depth n_2d times

print(f"Dense grid: {len(lon_grid)} x {len(lat_grid)} x {n_depths} depths = {n_total} points")

# Create dense grid dataframe
dense_data = pd.DataFrame({
    'lon': lon_dense,
    'lat': lat_dense,
    'x': x_dense,
    'y': y_dense,
    'z': z_dense
})

print(f"Dense grid coordinate ranges:")
print(f"  x: [{x_dense.min():.1f}, {x_dense.max():.1f}] km")
print(f"  y: [{y_dense.min():.1f}, {y_dense.max():.1f}] km")
print(f"  z: [{z_dense.min():.1f}, {z_dense.max():.1f}] km")

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
# meshname = "nicoya2"   # This has a smaller fault interface
# meshname = "nicoyaCK"   # local interface model from C. Kyriakopoulos_etal2015JGRSE
# meshname = "nicoyaCK2"   # same as above but 5-km mesh size on fault
# meshname = "nicoyaCK3"   # fault zone extended to the whole subduction zone
# meshname = "nicoyaCK4"   # same as CK2, but connecting the trench now

meshname = "nicoyaCKden_sm"   # based on nicoyaCK3 or 4, but denser mesh size, and smaller fault zone
# meshname = "nicoyaCKden_all"   # based on nicoyaCK3 or 4, but denser mesh size, and all subduction interface

print(meshname)

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
# ENHANCED: PDE Variational Formulation with Slip Transformation
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


# %%
# The linear coseismic inversion problem
# Define the weak formulation of the FORWARD problem
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
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip), 
                        tau('+')*n('+'))*dS(fault)

        return J

# %%
# Create a routine that solves the forward problem from a given slip to the ground displacement based on ANY given elastic structure
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

        # Set random seed at the beginning for reproducibility
        # np.random.seed(68)  # You can use any integer value
        # Modern approach (NumPy >= 1.17)
        # rng = np.random.default_rng(seed=4)

        #same seed for diff structures under same pattern & pattern_option, but different when pattern or option changes
        if slip_pattern == "checker":
            seed_num = (pattern_option-1)*5+1        
        elif slip_pattern == "stripe":
            seed_num = (pattern_option-1)*5+2
        elif slip_pattern == "circle":
            seed_num = (pattern_option-1)*5+3            
        rng = np.random.default_rng(seed=seed_num)
            
        #Create empty noise vector
        noise_vec = np.zeros(len(misfit.d),)

        if pollute_type == "uniform":
            # # Pollute true observations with random noise and generate synthetic data
            # hp.Random().uniform_perturb(noise_std_h, misfit.d)
            # misfit.noise_variance = noise_std_h*noise_std_h

            # # Modify misift.d to remove the values for other variables except for displacement
            # tmp = np.zeros(len(misfit.d),)
            # tmp[idx_d] = misfit.d[idx_d].copy()                 # horizontal and vertical displacement misfit         
            # # Assign the values of the new vector to misfit2.d
            # misfit.d.set_local(tmp)
            # misfit.d.apply('')

            #Generate noise for each component
            noise_vx = rng.normal(0, noise_std_h, size=targets.shape[0])
            noise_vy = rng.normal(0, noise_std_h, size=targets.shape[0])
            noise_vz = rng.normal(0, noise_std_v, size=targets.shape[0])

        elif pollute_type == "datastd":
            #Generate noise for each component
            noise_vx = rng.normal(0, data['vx_std_Car'])
            noise_vy = rng.normal(0, data['vy_std_Car'])
            noise_vz = rng.normal(0, data['vz_std_Car'])

        #Create displacement noise in interleaved format
        displacement_noise = np.zeros(3 * targets.shape[0])
        displacement_noise[0::3] = noise_vx  # Every 3rd element starting from 0
        displacement_noise[1::3] = noise_vy  # Every 3rd element starting from 1  
        displacement_noise[2::3] = noise_vz  # Every 3rd element starting from 2
        #Assign displacement noise to the displacement indices
        noise_vec[idx_d] = displacement_noise
        #Add noise to synthetic data
        misfit.d.set_local(misfit.d.get_local() + noise_vec)
        misfit.d.apply('')
        #Set noise variance (either per entry or averaged)
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

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2)
    mtrue_fun = dl.Function(Vh[hp.PARAMETER], mtrue)
    s_mag_true = ufl.sqrt( ufl.dot(mtrue_fun, mtrue_fun))
    # calculate the total moment on the fault 
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)  # presumably returns a UFL expression
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment_true = dl.assemble( m_mu_true * GPa2Pa * s_mag_true * dS(fault) )        
    print(f"True scalar seismic moment: {moment_true:.3e} N·m")
    M_w1_true, M_w2_true, M_w3_true = ut.moment2mag(moment_true)
    print(f"True moment magnitude: {M_w1_true:.2f}; {M_w2_true:.2f}; {M_w3_true:.2f}")
    # calculate seismic potency, independent of the assumed elastic properties
    potency_true = dl.assemble(s_mag_true * dS(fault))
    print(f"True seismic potency: {potency_true:.3e} m^3")
    if savefiles:
        # Save true moment to file  
        outFileName = 'moment_true_' + meshname + slip_str_gt + mu_str_for + '.txt'
        with open(resultpath + outFileName, 'w') as moment_file:
            moment_file.write(f"{moment_true:.6e} {M_w3_true:.4f} {potency_true:.6e}\n")

    return mtrue, mtrue_mu, u_mtrue, xf, yf, zf, d_obs, mtrue_s_fault

# %%
def computeGridDisplacements(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=False, \
                          pollute_type='uniform', savefiles=True, verbose=True):
    """
    Compute surface displacements at given target points using specified slip and shear modulus
    
    Args:
        targets: numpy array of observation points [N x 3] (x, y, z in meters)
        mtrue_mu_expr: shear modulus expression
        mu_str: string identifier for shear modulus model
        slip_identifier: string identifier for slip model
        savefiles: whether to save results
        verbose: print progress info
    """
    
    # Define function spaces
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]
    
    # Define boundary conditions
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

    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # Assign the values of the vector
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    
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
    
    ##### OPTION 2, USE 'pointwiseStateObs', without 'weight' option
    misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
    ##### OPTION 2    
    
    # Solve FORWARD problem for the STATE variables
    u_mtrue = pde.generate_state() # all dofs STATE variables and PETSC vector (not FEniCS Function)
    x = [u_mtrue, mtrue, None]
    pde.solveFwd(x[hp.STATE], x)
    # if savefiles:
    #     # Save the forward problem (synthetic displacement, or true displacement in this context)
    #     print( "Start saving .xdmf files forward problem to check" )
    #     uid = dl.XDMFFile(resultpath + 'u_' + meshname + slip_str_gt + mu_str_for + '.xdmf')
    #     u_save = dl.Function(Vh[hp.STATE].sub(1), u_mtrue)
    #     u_save.rename('displacement', 'displacement')
    #     uid.write(u_save)

    # Generate true observations by using the observation operator 'B'
    misfit.B.mult(x[hp.STATE], misfit.d)
    # 'idx_d' points to the non-zero components, e.g., if only ux and uy are pre constrained, 'idx_d' points to ux and uy
    idx_d = list(np.nonzero(misfit.d)[0])

    # whether to pollute the data with noise
    if pollute:

        # Set random seed at the beginning for reproducibility
        # np.random.seed(68)  # You can use any integer value
        # Modern approach (NumPy >= 1.17)
        # rng = np.random.default_rng(seed=4)

        #same seed for diff structures under same pattern & pattern_option, but different when pattern or option changes
        if slip_pattern == "checker":
            seed_num = (pattern_option-1)*5+1        
        elif slip_pattern == "stripe":
            seed_num = (pattern_option-1)*5+2
        elif slip_pattern == "circle":
            seed_num = (pattern_option-1)*5+3            
        rng = np.random.default_rng(seed=seed_num)
            
        #Create empty noise vector
        noise_vec = np.zeros(len(misfit.d),)

        if pollute_type == "uniform":
            # # Pollute true observations with random noise and generate synthetic data
            # hp.Random().uniform_perturb(noise_std_h, misfit.d)
            # misfit.noise_variance = noise_std_h*noise_std_h

            # # Modify misift.d to remove the values for other variables except for displacement
            # tmp = np.zeros(len(misfit.d),)
            # tmp[idx_d] = misfit.d[idx_d].copy()                 # horizontal and vertical displacement misfit         
            # # Assign the values of the new vector to misfit2.d
            # misfit.d.set_local(tmp)
            # misfit.d.apply('')

            #Generate noise for each component
            noise_vx = rng.normal(0, noise_std_h, size=targets.shape[0])
            noise_vy = rng.normal(0, noise_std_h, size=targets.shape[0])
            noise_vz = rng.normal(0, noise_std_v, size=targets.shape[0])

        elif pollute_type == "datastd":
            #Generate noise for each component
            noise_vx = rng.normal(0, data['vx_std_Car'])
            noise_vy = rng.normal(0, data['vy_std_Car'])
            noise_vz = rng.normal(0, data['vz_std_Car'])

        #Create displacement noise in interleaved format
        displacement_noise = np.zeros(3 * targets.shape[0])
        displacement_noise[0::3] = noise_vx  # Every 3rd element starting from 0
        displacement_noise[1::3] = noise_vy  # Every 3rd element starting from 1  
        displacement_noise[2::3] = noise_vz  # Every 3rd element starting from 2
        #Assign displacement noise to the displacement indices
        noise_vec[idx_d] = displacement_noise
        #Add noise to synthetic data
        misfit.d.set_local(misfit.d.get_local() + noise_vec)
        misfit.d.apply('')
        #Set noise variance (either per entry or averaged)
        misfit.noise_variance = noise_vec**2

    else:
        misfit.noise_variance = 1.0
        # misfit.noise_variance = noise_std_h*noise_std_h
        
    #####

    # Extract horizontals displacements observed data, if only ux and uy are pre constrained in 'misfit', 'd_obs' also contains only ux and uy  
    d_obs = misfit.d[idx_d]

    if verbose:
        print(f"Computed {len(d_obs)} displacement values")
        
    if savefiles:
        # Save surface displacements in same format as GPS stations
        outFileName = 'd_obs_grid' + meshname + slip_str_gt + mu_str_for + str(grid_spacing_deg) + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(
                targets[i,0], targets[i,1], targets[i,2], 
                d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]) )
        csvoutput.close()

    return d_obs

# %%
# Create a routine that solves the joint deterministic adjoint-based inversion
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
        print(sep, "ENHANCED: Set up mesh and slip transformation framework", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print("Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state))
        print("")
        print("SLIP TRANSFORMATION FRAMEWORK")
        print(f"   {slip_transformer}")
        
        # Show parameter interpretation
        if slip_transformer.has_any_bounds:
            print("   Parameter interpretation:")
            print("     m ∈ (-∞,∞) → physical_slip ∈ [bounds] via tanh transformation")
        else:
            print("   Using original unconstrained framework (m = physical_slip)")

    # Define the STATE and ADJOINT Dirichlet BCs
    zero_tensor = dl.Expression(( ("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.") ), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # Load the initial starting model of mu and s
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()
    
    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # Assign the values of the vector
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    # Save true shear modulus structure
    if savefiles:
        print( "Saving true shear modulus structure to .xdmf file" )
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project( mtrue_mu_fun_expr, CG_mu )
        filename = resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf'
        mu_id = dl.XDMFFile(filename)
        m_mu_true.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

    # KEY CHANGE: Use enhanced PDE variational formulation
    if slip_transformer.has_any_bounds:
        pde_varf = PDEVarf_TanhSlip(mtrue_mu_fun, slip_transformer)
        if verbose:
            print("Using enhanced PDE with tanh slip transformation")
    else:
        pde_varf = PDEVarf(mtrue_mu_fun)
        if verbose:
            print("Using original PDE (no slip bounds)")

    # Define the PDE problem
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

    ### Define the misfit based on whether the noise was added, and how it was added
    if pollute:

        ##### USE 'pointwiseStateObs_weights'
        weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
        obs_weights = np.zeros(targets.shape[0]*15,)

        if pollute_type == "uniform":
            # ##### USE 'pointwiseStateObs', without 'weight' option
            # misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
            # misfit.noise_variance = noise_std_h*noise_std_h
            # ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
            # obs_weights = np.zeros(targets.shape[0]*15,)
            # obs_weights[9::15]  = 1  # horizontal east displacement 
            # obs_weights[10::15] = 1  # horizontal north displacement 
            # obs_weights[11::15] = 1  # vectical displacement 

            obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements

        elif pollute_type == "datastd":
            obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements

        # Modify and set the array to dolfin vector
        weights.set_local(obs_weights)
        weights.apply('')
        # Different from the pure inversion, define misfit without 'weight' option
        misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
        misfit.noise_variance = 1.

    else:        
        ##### USE 'pointwiseStateObs', without 'weight' option
        misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
        misfit.noise_variance = 1.
        ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = 1  # horizontal east displacement 
        obs_weights[10::15] = 1  # horizontal north displacement 
        obs_weights[11::15] = 1  # vectical displacement 

    # Imput GPS data into misift.d.
    # Replace the synthetic data with the recorded GPS data
    tmp = np.zeros(len(misfit.d),)
    # Horizontal and vertical displacement misfit
    tmp[9::15] = np.array(syndata['ux'])    # horizontal east displacement misfit
    tmp[10::15] = np.array(syndata['uy'])   # horizontal north displacement misfit
    tmp[11::15] = np.array(syndata['uz'])   # vertical displacement misfit
    # Assign the values of the new vector to misfit.d
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
    idx_d = list(np.nonzero(obs_weights)[0]) # misfit = 2*ntargets (2 displacement components, since uz=0)
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error. The length of non-zero misfit has to be the same as ntargets.")

    # Extract horizontals displacements observed data, if only ux and uy are pre constrained in 'misfit', 'd_obs' also contains only ux and uy  
    d_obs = misfit.d[idx_d]

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

    # Define the regularization,
    # Below was used in the pure slip inversion 
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )

    ### CONSTRUCT MODEL (LAGRANGIAN FORMALISM) ###
    # Construct the "Model" --> objective function
    model = hp.Model(pde, reg, misfit)
    m = m0_s.copy()

    # MAIN SOLVER: Use original fast unconstrained CG
    # The beauty of tanh transformation: no changes needed to the solver!
    if verbose:
        print(sep, "Solve the slip inverse problem with transformation", sep)
        print("Using your original fast unconstrained CG solver")
        print("(slip bounds enforced automatically via tanh transformation)")

    # Generate STATE, PARAMETER and ADJOINT vectors
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)
    # Solve the FORWARD problem to find the STATE variables
    model.solveFwd(u, x)
    # Solve the ADJOINT problem to find the ADJOINT variables
    model.solveAdj(p, x)
    # Calculate the GRADIENT
    model.evalGradientParameter(x, mg)
    if verbose:
        print( sep, "Done generating STATE, PARAMETER and ADJOINT vectors", sep )

    ### HESSIAN ###
    # Solve the inverse problem with CG with some preconditioner to reduce the number of CG iterations
    model.setPointForHessianEvaluations(x) #gauss_newton_approx=False)
    # Define the Hessian operator 'H'. Since the inverse problem is LINEAR, the
    # Hessian operator 'H' is independent of the model parameter 'm'
    H = hp.ReducedHessian(model)

    ### PRECONDITIONED CONJUGATE GRADIENT (CG) METHOD ###
    # Use the regularization as a preconditioner for the CG algorithm
    Prec = reg.Rsolver
    # Consider all the Hessian to solve the inverse problem
    H.misfit_only = False
    # Solve the linear system: 'A x = b' using preconditioned conjugate gradient CG
    # and the Steihaug stopping criterion (to avoid negative curvature)
    solver = hp.CGSolverSteihaug()
    # Set the operator 'A'
    solver.set_operator(H)
    # Set the preconditioner R, such that:
    # (1) R is symmetric and positive definite;
    # (2) R is such that 'R\{hat}r = r', where 'r' is the residual, can be solved efficiently;
    # (3) R is an approximation of A^{−1} in the sense that: || I - R^{1} A || < 1
    # Set preconditioner: either low-rank of the Hessian or regularization
    solver.set_preconditioner(Prec)
    # Set parameters for the solver
    solver.parameters["print_level"] = 1
    solver.parameters["rel_tolerance"] = 1e-9 
    solver.parameters["abs_tolerance"] = 1e-12 
    solver.parameters["max_iter"]      = 1500
    # Solve 'H\{hat}m = -g' with preconditioned CG
    m_hat = model.generate_vector(hp.PARAMETER)
    solver.solve(m_hat, -mg)
    # Print when CG converges and how many CG iterations it takes
    if solver.converged:
        print( "CG converged in ", solver.iter, " iterations." )
    else:
        print( "CG did not converged." )
        raise RuntimeError("CG solver failed")


    # Solve the FORWARD problem to compute the "predicted data" with the inverted
    # model parameter 'm' (= slip along the fault).
    # Set the solution m = m0 + \{hat}m
    m.axpy(1., m_hat) # m = m + 1*\{hat}m

    # POST-PROCESSING: Convert unbounded parameters m to bounded physical slip
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

    # Save inversion results (coseismic slip)
    if savefiles:
        print( "Saving slip solution to .xdmf file" )
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print( "Finish saving slip solution" )

    # Solve the forward problem to compute the calculated STATE variables
    # Generate STATE and ADJOINT vectors
    #u = model.generate_vector(hp.STATE)
    #p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]   # u and p have been initialized before, so no need to do it again
    # Solve the forward problem to compute the calculated STATE variables
    model.solveFwd(u, x)
    # Use the observational operator to extract the surface displacement: d_cal = Bu
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    # Calculate the norm of the gradient of the solution for L-curve criterion
    m_fun = dl.Function(Vh[hp.PARAMETER], m)
    grad_m = dl.assemble( ufl.inner( ufl.avg(ufl.nabla_grad(m_fun)), ufl.avg(ufl.nabla_grad(m_fun)) )*dS(fault) )
    misfitd = np.linalg.norm((d_cal-d_obs), 2)
    print( "Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m) )

    # Compute the cost functional to plot misfit
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2)
    if slip_transformer.has_any_bounds:
        s_mag = ufl.sqrt(ufl.dot(physical_slip_expr, physical_slip_expr))
    else:
        s_mag = ufl.sqrt(ufl.dot(m_fun, m_fun))

    # Calculate seismic moment (using physical slip if bounds applied)
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    # calculate seismic potency, independent of the assumed elastic properties
    potency = dl.assemble(s_mag * dS(fault))
    print(f"Seismic potency: {potency:.3e} m^3")
    if savefiles:
        # Save true moment to file  
        outFileName = 'moment_' + meshname +  slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        with open(resultpath + outFileName, 'w') as moment_file:
            moment_file.write(f"{moment:.6e} {M_w3:.4f} {potency:.6e}\n")

    # Extract values from the model PARAMETER at the fault interface
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

    if savefiles:
        # Save the predicted surface displacement
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]) )
        csvoutput.close()

        # Save perdicted displacement field
        print( "Saving predicted displacement and stress to .xdmf file" )
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)
        # Stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        print( "Finish saving predicted displacement and stress" )

        # Save inferred slip values at the fault interface
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write( "%.6f %.6f\n" %(m_sx_fault[i], m_sy_fault[i]) )
        csvoutput.close()

        # Save inferred slip, over the entire volume
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
slip_pattern = "circle"

### circular pattern options
pattern_option = 1  

if slip_pattern == "circle":
    ##### CIRCULAR STEP SLIP MODEL #####
    print("Creating fault-local circular step pattern with natural depth range...")

    # from utils import create_fault_local_circular_step

    # Define true model, circular patch in dip-slip
    amp = 1   # assuming a coseismic slip of 1 m, m

    ### Circular pattern parameters
    if pattern_option == 1:
        # Single medium-size circular patch, roughly at coseismic centroid
        radius = 25e3    # radius of circular region
        lon0_slip, lat0_slip = -85.5, 10    # center of the pattern
        x0_tmp, y0_tmp = ut.LL2ckmd(lon0_slip, lat0_slip, lon0, lat0, rot)
        x0_slip = np.round((x0_tmp - x0) / 1e3)   # x center in km (integer)
        y0_slip = np.round((y0_tmp - y0) / 1e3)   # y center in km (integer)
        rot_deg = 0.0    # rotation angle in degrees

    else:
        raise ValueError(f"Invalid pattern_option {pattern_option} for circular pattern")

    print(f"Circular step pattern option {pattern_option}:")
    print(f"  Center: ({x0_slip:.1f}, {y0_slip:.1f}) km")  # Use x0_slip, y0_slip directly
    print(f"  Radius: {radius/1000:.1f} km")
    print(f"  Amplitude: {amp:.1f} m")
    print(f"  Rotation: {rot_deg:.1f} degrees")

    # Create the circular step slip pattern
    mtrue_s_expr_gt = ut.create_fault_local_circular_step(
        mesh=mesh,
        boundaries=boundaries,
        fault_id=fault,
        amp=amp,
        x0=x0_slip * 1e3,  # Convert to meters here
        y0=y0_slip * 1e3,  # Convert to meters here
        radius=radius,
        rotation_deg=rot_deg,
        strike_slip=False,
        degree=1
    )

    slip_str_gt = (
        f"_circ_x{x0_slip:g}_y{y0_slip:g}"  # Use km values directly
        f"_r{radius/1e3:g}_ms{amp:g}"
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

# # Dense grid
# ntargets_dense = len(dense_data)
# targets_grid = np.zeros([ntargets_dense, 3])
# targets_grid[:,0] = np.array(dense_data['x'])*1e3   # km to m
# targets_grid[:,1] = np.array(dense_data['y'])*1e3
# targets_grid[:,2] = np.array(dense_data['z'])*1e3
# print(targets_grid.shape)

# # Save dense grid coordinates with lat/lon for plotting
# coord_file = f"dense_grid_coordinates_{meshname}_{grid_spacing_deg}.txt"
# # if not os.path.exists(resultpath + coord_file):
# coord_output = open(resultpath + coord_file, 'w+')
# # coord_output.write("# lon lat x_km y_km z_km\n")
# for i in range(0, targets_grid.shape[0]):
#     coord_output.write("%.6f %.6f %.6f %.6f %.6f\n" %(
#         dense_data['lon'].iloc[i], dense_data['lat'].iloc[i],
#         dense_data['x'].iloc[i], dense_data['y'].iloc[i], dense_data['z'].iloc[i]
#     ))
# coord_output.close()

# %%
# Define the true model PARAMETERS for the FORWARD problem within a homogeneous shear modulus half-space
nu = 0.25   # Possion's ratio found in Kano et al., 2019
# mu = 40e9 / GPa2Pa    # shear modulus (rigidity) found in Kano et al., 2019 
# E = 2*mu*(1+nu) / GPa2Pa # Lame parameter

# background shear modulus
mu_b = 0   # 40 GPa
mu_background = mu_expression(mu_b)

# shear modulus for the lower (subducting) plate
mu_l = 0  # 40 GPa
mu_lower = mu_expression(mu_l)

# shear modulus for the upper (overriding) plate
mu_u = 0   # 40 GPa
mu_upper = mu_expression(mu_u)

# # shear modulus for volcanoes
# mu_v = 0  # 40 GPa
# mu_volcano = mu_expression(mu_v)

mtrue_mu_expr_hom = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
mu_str_hom = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"
# mu_str_hom = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}v{round(mu_expression(mu_v))}"

print( "Homogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )
# print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f and volcano mu = %.1f" %(mu_upper, mu_lower, mu_volcano) )

# %%
# Define the true model PARAMETERS for INVERSE problem
nu = 0.25
# mu = 40e9 / GPa2Pa    # shear modulus (rigidity) found in Kano et al., 2019 
# E = 2*mu*(1+nu) / GPa2Pa # Lame parameter

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

# # shear modulus for volcanoes
# mu_v = -0.9730  # ~25 GPa
# mu_volcano = mu_expression(mu_v) 

mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"
# mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}v{round(mu_expression(mu_v))}"

print( "Heterogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )
# print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f and volcano mu = %.1f" %(mu_upper, mu_lower, mu_volcano) )


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
        # noise_var_h = noise_std_h**2
        # print(noise_var_h)

        # Decide the weights of the horizontal, vertical components
        # f_h, f_v = 1, 1/2
        f_h, f_v = 1, 1
        # Print the weights of the data
        print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )
        
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
        obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
        obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
        
        slip_str_gt = slip_str_gt + "_pou"

    elif pollute_type == 'datastd':
        # Decide the weights of the horizontal, vertical components
        # f_h, f_v = 1, 1/2
        f_h, f_v = 1, 1
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
# mtrue_s_expr_for = None
mtrue_s_expr_for = mtrue_s_expr_gt
if mtrue_s_expr_for==None:
    print("Ground-truth slip is from inversion")
else:
    print("Ground-truth slip is customized")  

# %%
# Solve the coseismic forward problem to generate the synthetic data within a Homogeneous shear modulus half-space
mtrue_mu_expr_for = mtrue_mu_expr_hom
mu_str_for = mu_str_hom

# # Solve the coseismic forward problem to generate the synthetic data within a Heterogeneous shear modulus half-space
# mtrue_mu_expr_for = mtrue_mu_expr_het
# mu_str_for = mu_str_het

print("Solving forward problem based on: ", mu_str_for)

# %%
# Solve forward problem to generate synthetics
solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
                      pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
print("Forward problem done!!!")

# # Solve forward problem to generate synthetics at a dense grid
# computeGridDisplacements(k, targets_grid, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
#                       pollute=False, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Dense grid forward problem done!!!")

# %% [markdown]
# # Use the synthetic displacement for inversion within any $\mu$ structure model

# %%
# Load the synethetic displacement data from the forward problem
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x', 'y', 'z', 'ux', 'uy', 'uz'])

# %%
# SLIP TRANSFORMATION SETUP
print(sep + "SLIP TRANSFORMATION SETUP" + sep)

# BOUNDED = True
BOUNDED = False

BOUND_TYPE = 'both'
# BOUND_TYPE = 'dip'

if BOUNDED:
    # Define slip bounds based on your problem
    V_para = 16/1e3
    if BOUND_TYPE == 'both':
        slip_transformer = SlipTransformation(
            strike_bounds=(-2e-3, 2e-3),    # just a little mm. although the truth is 0
            dip_bounds=(0.0, amp),
        )
        print("Constraints to both strike and dip ")

    elif BOUND_TYPE == 'strike':
        slip_transformer = SlipTransformation(
            strike_bounds=(0.0, V_para),
            dip_bounds=None,
        )
        print("Constraints to strike only ")

    elif BOUND_TYPE == 'dip':
        slip_transformer = SlipTransformation(
            strike_bounds=None,
            dip_bounds=(0.0, amp),
        )
        print("Constraints to dip only ")

else:            
    # Alternative: no constraints (reverts to original framework)
    slip_transformer = SlipTransformation(strike_bounds=None, dip_bounds=None)
    print("Unconstrained mode (original framework)")

print(f"Configuration: {slip_transformer}")

# %%
# Define regularization weights
# In a Bayesian inference setting, the ratio \rho = \sqrt(\gamma/\delta) plays the role of the correlation length in the prior term.
# For our case, the station separation is around 20 km, and the mesh size on the fault is 4-20 km 

if not BOUNDED:
    if meshname == "nicoyaCK3":    # large fault extending to the whole subduction zone
        # gamma_val_H1 = 1e3  
        # delta_val_L2 = 1e-4
        # delta_val_L2 = 1e-5
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 2.5e-5
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 5e-5
        gamma_val_H1 = 5e3    # best so far
        delta_val_L2 = 5e-6
        # gamma_val_H1 = 7.5e3  
        # delta_val_L2 = 7.5e-6

else:
    if meshname == "nicoyaCK4":     # small fault connecting trench
        rho_s = 1e9   # allows variations of slip of the order of ~30 km 

        # rho_s = 1e8   # allows variations of slip of the order of ~3 km, close to the maximum resolution
        # gamma_val_H1 = 5e1  
        gamma_val_H1 = 1e2    # best so far
        # gamma_val_H1 = 4e2
        # gamma_val_H1 = 1e3    
        # gamma_val_H1 = 1e4    
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

# # file identifier
# if BOUNDED:
#     inv_str = f"_synlockbd_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
# else:
#     inv_str = f"_synlock_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
# print("Inverse problem identifier: ", inv_str)

# # %%
# # Solve slip inverse problem within the homogeneous shear modulus half-space
# mtrue_mu_expr_inv = mtrue_mu_expr_hom
# mu_str_inv = mu_str_hom
# print("Solving inverse problem based on: ", mu_str_inv)
# results = solveCoseismicInversion_TanhSlip(
#         k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
#         slip_transformer,  # KEY: Pass slip transformation
#         pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
#     )
# print("Hom Inversion finished!!!")


# %%
### Compute L-curve criterion ###
# Solve to find the best Tikhonov regularization parameter by using the L-curve criterion

import gc  # for explicit memory management

# fix the rho, ratio of gamma and delta
rho_s = 1e9

# vary gamma, the model gradient damping  
if not BOUNDED:
    gammas_s = [2e2, 3e2, 4e2]

print( sep, "Hom L-curve computation with slip transformation", sep )

dmisfit = []
mmisfit = []
for gamma_s in gammas_s:
    delta_s = gamma_s / rho_s 

    # file identifier
    if BOUNDED:
        inv_str = f"_syncoseisbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    else:
        inv_str = f"_syncoseis_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    print("Inverse problem identifier: ", inv_str)

    print(f"****** Computing solution with gamma_s = {gamma_s:.1e}, "
        f"rho_s = {rho_s:.1e}, and delta_s = {delta_s:.1e} ******")

    # Solve slip inverse problem within the homogeneous shear modulus half-space WITH TANH TRANSFORMATION  
    mtrue_mu_expr_inv = mtrue_mu_expr_hom
    mu_str_inv = mu_str_hom
    print("Solving inverse problem based on: ", mu_str_inv)
    results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
        slip_transformer,  # KEY: Pass slip transformation
        pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    )

    # ===== EXTRACT ONLY WHAT'S NEEDED =====
    # Extract scalars immediately (avoid keeping large FEniCS objects)
    misfit_d = float(results[-2])   # misfitd (scalar)
    misfit_m = float(results[-1])   # grad_m norm or model misfit (scalar)
    dmisfit.append(misfit_d)
    mmisfit.append(misfit_m)

    # ===== AGGRESSIVE MEMORY CLEANUP =====
    # Delete the results tuple containing 11 other large objects
    del results

    # Force Python garbage collector
    gc.collect()

    # Optional: Monitor memory
    try:
        import psutil
        mem_gb = psutil.Process().memory_info().rss / 1024**3
        print(f"  Memory after cleanup: {mem_gb:.2f} GB")
    except ImportError:
        pass
    # ======================================

# save the data misfit and model misfit, in order for later plotting of L-curve to find the best weights
if BOUNDED:
    outFileName = f"Lcurvesyncoseisbd_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_inv}.txt"
else:
    outFileName = f"Lcurvesyncoseis_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_inv}.txt"
csvoutput = open(resultpath + outFileName, 'w+') # make new and overwrite
# csvoutput = open(resultpath + outFileName, 'a') # append to existing file
# csvoutput = open(resultpath + outFileName + '_extra', 'w+') # make new and overwrite
# csvoutput = open(resultpath + outFileName + '_extra', 'a') # append to existing file
for i in range(0, len(dmisfit)):
    csvoutput.write( "%.6e %.6e\n" %(dmisfit[i], mmisfit[i]) )
csvoutput.close()

print("All Hom Inversion with Slip Bounds Finished!")   