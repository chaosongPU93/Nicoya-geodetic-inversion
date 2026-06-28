# %% [markdown]
# # Synthetic recovery test — Az-constrained SCALAR amplitude inversion (3D forward)
#   Nicoya dense mesh (nicoyaden_sm), even top boundary, stripe pattern.
#   Forward μ: DeShon 3D velocity model (CG2 elements).  Inversion L-curve: Het(3D) + Hom.
#
# Based on synth_stripeslip_az_inv_hetmu3D_nicoya_lock_noi.py (reference Az TanhSlip 3D)
# and synth_stripeslip_az_scalaramp_inv_hetmu3D_uneven_nicoyaCK_lock_noi.py (AzSlip 3D).
#
# KEY DIFFERENCE — inversion uses solveCoseismicInversion_AzSlip (scalar amplitude):
#   - True slip:   Az-directed (FaultLocalStripesAz), same as reference
#   - Forward:     3D μ (DeShon, CG2); call commented out (files from reference _noi3D.py)
#   - Inversion:   SCALAR amplitude a(x) ∈ [0, amp_max] via tanh; direction fixed to Az
#                  Run with both 3D Het mu and Hom mu (_passes) to quantify mu-model effect
#   - Assessment:  amplitude recovery at each fault vertex (dual-ratio, 7-column output)
#
# SlipTransformation / PDEVarf_TanhSlip / solveCoseismicInversion_TanhSlip removed.
# compute_az_cg1_functions / PDEVarf_AzSlip / solveCoseismicInversion_AzSlip added.
# solveCoseismicInversion_AzSlip accepts CG_mu_deg; isinstance check for seismic moment.
# PDEVarf_AzSlip uses mu = self.mtrue_mu_fun directly (already in GPa, for 3D or hom).
#
# * Adds noise to synthetic data (uniform or data-std); inversion weighted accordingly.

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
import h5py
import utils as ut
from scipy.interpolate import griddata, interp1d
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
print(sep, "START of Az scalar-amplitude synth recovery test — 3D forward (nicoyaden_sm)", sep)

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
lon0, lat0 = -85.5+360, 10
# print(lon0, lat0)

# convert to relative locations in km
data['x'], data['y'] = ut.azi_equidist_proj(data['lon'], data['lat'], lon0, lat0)
data['z'] = 0.0

# print(data[['lon', 'lat', 'x', 'y']].head())
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
# read in the 3D velocity model (DeShon 2006), using nicoyaden_sm coordinate system
veldir = "/home/staff/chao/SSEinv/Nicoya/DeShon_2006GJI/"
vel3d = pd.read_csv(veldir + "DeShon2006_3Dmodel.csv", sep=",")
# convert to relative locations in km using the same projection as nicoyaden_sm
vel3d['x'], vel3d['y'] = ut.azi_equidist_proj(vel3d['lon'], vel3d['lat'], lon0, lat0)
cols_to_convert = ['x', 'y', 'z']
vel3d[cols_to_convert] = vel3d[cols_to_convert] * 1e3  # Convert km to m
vel3d['z'] = vel3d['z'] * -1  # negative depth means downward
vel3d = vel3d[(vel3d['z'] <= 0)].reset_index(drop=True)

# read in the reference 1D velocity model
vel1d = pd.read_csv(veldir + "DeShon2006_1Dmodel.csv", sep=r'\s+', skiprows=1,
                    names=['z', 'vp', 'vs', 'vp_vs_ratio'])
vel1d['z'] = vel1d['z'] * -1 * 1e3  # negative depth means downward, km to m
vel1d = vel1d[(vel1d['z'] <= 0)].reset_index(drop=True)

# read in the 1D density model
den1d = pd.read_csv(veldir + "Density_1Dmodel.csv", sep=r'\s+', skiprows=1,
                    names=['z', 'den'])
den1d['z'] = den1d['z'] * -1 * 1e3  # negative depth means downward, km to m
den1d = den1d[(den1d['z'] <= 0)].reset_index(drop=True)
den1d['den'] = den1d['den'] * 1e3   # g/cm^3 → kg/m^3

contrast_factor = 2.5  # amplification factor relative to 1D depth-layered values

# %%
# Helper function to save CG2 function data with ALL DOF points
# Standard XDMFFile.write() only exports values at mesh vertices, losing edge midpoint DOFs for CG2
def save_function_with_dofs(function, filename_base, attr_name, cg_degree=1):
    """Save FEniCS function; for CG2+ also writes _dofs.h5 with all DOF points."""
    xdmf_file = dl.XDMFFile(filename_base + '.xdmf')
    function.rename(attr_name, attr_name)
    xdmf_file.write(function)
    if cg_degree > 1:
        V = function.function_space()
        dof_coords = V.tabulate_dof_coordinates()
        dof_values = function.vector()[:]
        n_dofs = len(dof_coords)
        n_vertices = V.mesh().num_vertices()
        print(f"  CG{cg_degree} function has {n_dofs} DOFs (vs {n_vertices} mesh vertices)")
        h5_filename = filename_base + '_dofs.h5'
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('coordinates', data=dof_coords)
            f.create_dataset('values', data=dof_values)
            f.attrs['cg_degree'] = cg_degree
            f.attrs['attr_name'] = attr_name
            f.attrs['n_dofs'] = n_dofs
            f.attrs['n_vertices'] = n_vertices
        print(f"  Saved all DOF data to: {h5_filename}")

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
x_2d, y_2d = ut.azi_equidist_proj(lon_2d, lat_2d, lon0, lat0)

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
# meshname = "nicoya3"   # the same as above but 5-km mesh size on fault, connecting the trench now

meshname = "nicoyaden_sm"   # based on nicoya3 or 4, but denser mesh size, and smaller fault zone
# meshname = "nicoyaden_all"   # based on nicoya3 or 4, but denser mesh size, and all subduction interface

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
# PDE variational formulation — 2-component slip, used by forward functions
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

        #same seed for same pattern_option and diff structures, but different from the previous version
        if slip_pattern == "checker":
            seed_num = (pattern_option-1)*5+1        
        elif slip_pattern == "stripe":
            seed_num = (pattern_option-1)*5+2
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
        # outFileName = 'd_obs_grid' + meshname + slip_str_gt + mu_str_for + '.txt'
        outFileName = 'd_obs_grid' + meshname + slip_str_gt + mu_str_for + str(grid_spacing_deg) + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(
                targets[i,0], targets[i,1], targets[i,2], 
                d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]) )
        csvoutput.close()

    return d_obs

# %%
# ============================================================================
# NEW: Az-constrained scalar amplitude inversion functions
# ============================================================================

def compute_az_cg1_functions(mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg=0.0):
    """
    Project per-facet Az coefficients (c_strike, c_dip) to CG1 vertex functions
    by averaging over adjacent fault facets.
    """
    az_coeffs = ut._compute_az_per_facet_coeffs(
        mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg)
    CG1 = dl.FunctionSpace(mesh, "CG", 1)
    c_strike_fun = dl.Function(CG1)
    c_dip_fun    = dl.Function(CG1)
    v2d = dl.vertex_to_dof_map(CG1)
    vertex_cs = {}; vertex_cd = {}
    for facet in dl.facets(mesh):
        if boundaries[facet] != fault_id:
            continue
        fi = facet.index()
        if fi not in az_coeffs:
            continue
        c_s, c_d = az_coeffs[fi]
        for vertex in dl.vertices(facet):
            vi = vertex.index()
            vertex_cs.setdefault(vi, []).append(c_s)
            vertex_cd.setdefault(vi, []).append(c_d)
    cs_arr = c_strike_fun.vector().get_local()
    cd_arr = c_dip_fun.vector().get_local()
    for vi in vertex_cs:
        dof = v2d[vi]
        cs_arr[dof] = np.mean(vertex_cs[vi])
        cd_arr[dof] = np.mean(vertex_cd[vi])
    c_strike_fun.vector().set_local(cs_arr); c_strike_fun.vector().apply('')
    c_dip_fun.vector().set_local(cd_arr);   c_dip_fun.vector().apply('')
    n_fault_verts = len(vertex_cs)
    cs_fault = cs_arr[cs_arr != 0.0]; cd_fault = cd_arr[cd_arr != 0.0]
    mag = np.sqrt(cs_fault**2 + cd_fault**2)
    print(f"  Projected to {n_fault_verts} fault vertices")
    print(f"  c_strike: [{cs_fault.min():.4f}, {cs_fault.max():.4f}]")
    print(f"  c_dip:    [{cd_fault.min():.4f}, {cd_fault.max():.4f}]")
    print(f"  ||c||:    [{mag.min():.4f}, {mag.max():.4f}]  (should be ≈ 1.0)")
    return c_strike_fun, c_dip_fun


class PDEVarf_AzSlip:
    """
    PDE with scalar amplitude parameter and fixed Az slip direction.
    m_phys = amp_max * (tanh(m) + 1) / 2  ∈ [0, amp_max]
    slip = m_phys * (c_strike_fun, c_dip_fun)
    """
    def __init__(self, mtrue_mu_fun, c_strike_fun, c_dip_fun, amp_max):
        self.mtrue_mu_fun = mtrue_mu_fun
        self.c_strike_fun = c_strike_fun
        self.c_dip_fun    = c_dip_fun
        self.amp_max      = amp_max

    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        tau, w, q    = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))
        m_phys = self.amp_max * (ufl.tanh(m) + 1) / 2
        mu = mu_expression(self.mtrue_mu_fun)
        J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
          + ufl.inner(ufl.div(tau), uu)*ufl.dx \
          + ufl.inner(asym(tau), r)*ufl.dx \
          + ufl.inner(ufl.div(sigma), w)*ufl.dx \
          + ufl.inner(asym(sigma), q)*ufl.dx \
          + ufl.inner(f, w)*dl.dx \
          - ufl.inner(u0, tau*n)*ds(bottom) \
          - ufl.inner(dir_strike(n('+')) * ufl.avg(m_phys * self.c_strike_fun)
                    + dir_dip(n('+'))   * ufl.avg(m_phys * self.c_dip_fun),
                     tau('+')*n('+'))*dS(fault)
        return J


def solveCoseismicInversion_AzSlip(k, targets, m0_amp_expr, mtrue_mu_expr_inv,
                                   gamma_val_H1, delta_val_L2,
                                   c_strike_fun, c_dip_fun, amp_max,
                                   CG_mu_deg=1,
                                   pollute=True, pollute_type='uniform',
                                   savefiles=True, verbose=True):
    """
    Synth recovery version of Az-constrained scalar amplitude inversion.
    Reads synthetic data from global `syndata` (loaded from d_obs_* file).
    Recovery metric saved to: slip_recovery_<meshname><...>.txt
      columns: xf_m  yf_m  zf_m  true_amp_m  recovered_amp_m  recovery_global  recovery_local

    mtrue_mu_expr_inv = None triggers 3D velocity model build via process_velocity_models_hull.
    CG_mu_deg = 2 for 3D model; 1 for K_2LAYER/homogeneous.
    """
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)

    # KEY: scalar (not 2-component) parameter space
    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vu, Vm_amp, Vu]

    ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
    ndofs_state = [Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim()]
    if verbose:
        print(sep, "Az-constrained scalar amplitude inversion (synth recovery)", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print("Number of STATE/ADJOINT dofs: STRESS={0}, DISPL={1}, ROT={2}".format(*ndofs_state))
        print(f"Amplitude bound: [0, {amp_max*1e3:.1f} mm/yr]")

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    m0_amp = dl.interpolate(m0_amp_expr, Vh[hp.PARAMETER]).vector()

    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    if mtrue_mu_expr_inv is None:
        # 3D velocity model (CG2) — build from process_velocity_models_hull
        vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull(
            vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)
        mtrue_mu_fun = ut.scale_shear_modulus_by_1d(
            mu_func, vel1d, den1d, contrast_factor, min_mu=None, verbose=False)
        mtrue_mu_param_fun = None  # no tanh parameter for 3D case
    else:
        # K_2LAYER or homogeneous: tanh-parameterized → convert to physical GPa expression
        mtrue_mu_param = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
        mtrue_mu_param_fun = hp.vector2Function(mtrue_mu_param, CG_mu)
        mtrue_mu_fun = mu_expression(mtrue_mu_param_fun)  # UFL expression in GPa

    if savefiles:
        print("Saving true shear modulus to .xdmf file")
        if isinstance(mtrue_mu_fun, dl.Function):
            m_mu_true_save = mtrue_mu_fun
        else:
            m_mu_true_save = dl.Function(CG_mu)
            m_values = mtrue_mu_param_fun.vector()[:]
            m_mu_true_save.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
        save_function_with_dofs(m_mu_true_save, resultpath + 'mu_true_' + meshname + mu_str_inv,
                                'shear modulus', cg_degree=CG_mu_deg)

    pde_varf = PDEVarf_AzSlip(mtrue_mu_fun, c_strike_fun, c_dip_fun, amp_max)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    if verbose:
        print("Number of observation points: {0}".format(targets.shape[0]))

    indicator_vec = dl.interpolate(
        dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()

    if pollute:
        weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
        obs_weights = np.zeros(targets.shape[0]*15)
        if pollute_type == 'uniform':
            obs_weights[9::15]  = 1./noise_std_h**2 * 1./f_h**2
            obs_weights[10::15] = 1./noise_std_h**2 * 1./f_h**2
            obs_weights[11::15] = 1./noise_std_v**2 * 1./f_v**2
        elif pollute_type == 'datastd':
            obs_weights[9::15]  = (1./data['vx_std_Car']**2).to_numpy() * 1./f_h**2
            obs_weights[10::15] = (1./data['vy_std_Car']**2).to_numpy() * 1./f_h**2
            obs_weights[11::15] = (1./data['vz_std_Car']**2).to_numpy() * 1./f_v**2
        weights.set_local(obs_weights); weights.apply('')
        misfit = PSBW(Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec)
        misfit.noise_variance = 1.
    else:
        misfit = PSB(Vh[hp.STATE], targets, indicator_vec=indicator_vec)
        misfit.noise_variance = 1.
        obs_weights = np.zeros(targets.shape[0]*15)
        obs_weights[9::15] = 1; obs_weights[10::15] = 1; obs_weights[11::15] = 1

    # Load synthetic data (from solveCoseismicForward output)
    tmp = np.zeros(len(misfit.d))
    tmp[9::15]  = np.array(syndata['ux'])
    tmp[10::15] = np.array(syndata['uy'])
    tmp[11::15] = np.array(syndata['uz'])
    misfit.d.set_local(tmp); misfit.d.apply('')

    idx_d = list(np.nonzero(obs_weights)[0])
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error: non-zero misfit length mismatch.")
    d_obs = misfit.d[idx_d]

    # Fault mask for scalar amplitude
    bc_amp = dl.DirichletBC(Vm_amp, 99.0, boundaries, fault)
    um_amp = dl.Function(Vm_amp); bc_amp.apply(um_amp.vector())

    # Fault coordinates (scalar CG1, consistent with Vm_amp)
    fault_mask_arr = um_amp.vector().get_local() == 99.0
    xslip = dl.interpolate(dl.Expression("x[0]", degree=1), Vm_amp)
    yslip = dl.interpolate(dl.Expression("x[1]", degree=1), Vm_amp)
    zslip = dl.interpolate(dl.Expression("x[2]", degree=1), Vm_amp)
    xf = xslip.vector().get_local()[fault_mask_arr]
    yf = yslip.vector().get_local()[fault_mask_arr]
    zf = zslip.vector().get_local()[fault_mask_arr]

    if verbose:
        print(sep, "Done extracting fault coordinates", sep)

    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)
    model = hp.Model(pde, reg, misfit)
    m = m0_amp.copy()

    if verbose:
        print(sep, "Solve the Az-constrained scalar amplitude inverse problem", sep)

    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)
    model.solveFwd(u, x)
    model.solveAdj(p, x)
    model.evalGradientParameter(x, mg)
    if verbose:
        print(sep, "Done generating STATE, PARAMETER and ADJOINT vectors", sep)

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
        print("CG converged in", solver.iter, "iterations.")
    else:
        print("CG did not converge.")
        raise RuntimeError("CG solver failed")

    m.axpy(1., m_hat)

    # Post-processing: recover physical amplitude and decompose into (s_strike, s_dip)
    m_fun = hp.vector2Function(m, Vm_amp)
    m_phys_fun = dl.project(amp_max * (ufl.tanh(m_fun) + 1) / 2, Vm_amp)
    s_strike_fun = dl.project(m_phys_fun * c_strike_fun, Vm_amp)
    s_dip_fun    = dl.project(m_phys_fun * c_dip_fun,    Vm_amp)

    if savefiles:
        print("Saving amplitude and slip components to .xdmf files")
        amp_id = dl.XDMFFile(resultpath + 'slip_amp_' + meshname + slip_str_gt
                             + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        m_phys_fun.rename('slip amplitude', 'slip amplitude'); amp_id.write(m_phys_fun)

        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + slip_str_gt
                                  + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip'); s_strike_id.write(s_strike_fun)

        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + slip_str_gt
                               + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip'); s_dip_id.write(s_dip_fun)
        print("Finish saving slip solution")

    # Forward solve with inverted m to get predicted data
    x = [u, m, p]
    model.solveFwd(u, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    d_cal = misfit.Bu[idx_d]

    # L-curve quantities
    m_fun2 = dl.Function(Vm_amp, m)
    grad_m = dl.assemble(ufl.inner(ufl.avg(ufl.nabla_grad(m_fun2)),
                                   ufl.avg(ufl.nabla_grad(m_fun2)))*dS(fault))
    misfitd = np.linalg.norm((d_cal - d_obs), 2)
    print("Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m))
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print("Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(
        total_cost, reg_cost, misfit_cost))

    # Seismic moment (using recovered amplitude × ||c|| ≈ 1)
    s_mag = m_phys_fun   # ||slip|| ≈ m_phys since ||c|| ≈ 1
    # mtrue_mu_fun is already in GPa (Function for 3D, UFL expr for K_2LAYER)
    if isinstance(mtrue_mu_fun, dl.Function):
        m_mu_true = mtrue_mu_fun
    else:
        m_mu_true = dl.Function(CG_mu)
        m_values = mtrue_mu_param_fun.vector()[:]
        m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    potency = dl.assemble(s_mag * dS(fault))
    print(f"Seismic potency: {potency:.3e} m^3")

    if savefiles:
        with open(resultpath + 'moment_' + meshname + slip_str_gt
                  + mu_str_for + inv_str + mu_str_inv + '.txt', 'w') as fout:
            fout.write(f"{moment:.6e} {M_w3:.4f} {potency:.6e}\n")

    # Extract recovered amplitude at fault vertices (scalar mask)
    m_amp_arr = m_phys_fun.vector().get_local()
    m_amp_fault = m_amp_arr[um_amp.vector().get_local() == 99.0]

    # Extract derived (s_strike, s_dip) at fault
    cs_arr = c_strike_fun.vector().get_local()
    cd_arr = c_dip_fun.vector().get_local()
    cs_fault = cs_arr[um_amp.vector().get_local() == 99.0]
    cd_fault = cd_arr[um_amp.vector().get_local() == 99.0]
    m_sx_fault = m_amp_fault * cs_fault
    m_sy_fault = m_amp_fault * cd_fault

    print("Physical slip ranges:")
    print(f"  Amplitude: [{m_amp_fault.min():.6f}, {m_amp_fault.max():.6f}] m")
    print(f"  Strike:    [{m_sx_fault.min():.6f}, {m_sx_fault.max():.6f}] m")
    print(f"  Dip:       [{m_sy_fault.min():.6f}, {m_sy_fault.max():.6f}] m")

    if savefiles:
        # Predicted displacements
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]))
        csvoutput.close()

        # Predicted displacement and stress fields
        print("Saving predicted displacement and stress to .xdmf file")
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt
                         + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement'); uid.write(u_save)
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt
                         + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress'); sid.write(sigma_save)
        print("Finish saving predicted displacement and stress")

        # m_s_fault (2-column, backward compatible with plotting notebooks)
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(m_sx_fault.shape[0]):
            csvoutput.write("%.6f %.6f\n" % (m_sx_fault[i], m_sy_fault[i]))
        csvoutput.close()

    return mtrue_mu, xf, yf, zf, m, u, m_phys_fun, s_strike_fun, s_dip_fun, \
           d_obs, d_cal, m_amp_fault, m_sx_fault, m_sy_fault, misfitd, grad_m


# %%
# ============================================================================
# COMMON PARAMETERS
# ============================================================================
k = 2
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9

# Scalar initial guess (tanh param space: 0 → m_phys = amp_max/2 initially)
m0_amp_expr = dl.Constant(0.)

nu = 0.25

# %%
# Shear modulus structures
def mu_expr_fn(m):
    return 20*(2. + ufl.tanh(m))

mu_b = 0; mu_background = mu_expr_fn(mu_b)

mu_l_hom = 0; mu_lower_hom = mu_expr_fn(mu_l_hom)
mu_u_hom = 0; mu_upper_hom = mu_expr_fn(mu_u_hom)
mtrue_mu_expr_hom = K_2LAYER(subdomains, mu_u_hom, mu_l_hom, degree=5)
mu_str_hom = f"_mul{round(mu_expr_fn(mu_l_hom))}u{round(mu_expr_fn(mu_u_hom))}"
print("Homogeneous structure:")
print(f"  mu_upper = {mu_upper_hom:.1f}, mu_lower = {mu_lower_hom:.1f}")

# 3D heterogeneous model — uses DeShon velocity model inside solveCoseismicInversion_AzSlip
mtrue_mu_expr_het = None   # sentinel: triggers process_velocity_models_hull inside solver
# String: _hull indicates process_velocity_models_hull method; _CG2 suffix added after CG_mu_deg is set
mu_str_het = f"_DeShon3D_ref1D_{round(contrast_factor)}_hull"
print("3D heterogeneous structure:")
print(f"  mu_str_het = {mu_str_het} (contrast_factor = {contrast_factor})")

# %%
# Az configuration — nicoyaden_sm mesh, no rotation
# slip_azimuth_deg  = 45.0   # CW from North; N45E = Cocos-Caribbean trench-normal convergence
# slip_azimuth_deg  = 26.0   # CW from North; N26E = roughly oblique convergence
azimuth_deg       = 33.5   # CW from North; N33.5E, a little oblique convergence
mesh_rotation_deg = 0.0    # nicoyaden_sm mesh is not rotated

# Plate convergence velocity decomposition (back-slip convention)
V_norm       = 78.5   # mm/yr, trench-normal component (Cocos-Caribbean)
V_para       = 27     # mm/yr, trench-parallel component
V_para_const = 11     # mm/yr, correction for N33.5E azimuth
if azimuth_deg == 45.0:
    amp = V_norm                                                   # 78.5 mm/yr, purely trench-normal
elif azimuth_deg == 26.0:
    amp = round(np.sqrt(V_norm**2 + V_para**2))                   # ~83 mm/yr, oblique
elif azimuth_deg == 33.5:
    amp = round(np.sqrt(V_norm**2 + (V_para - V_para_const)**2))  # ~80 mm/yr
amp     = amp / 1e3   # convert mm/yr → m/yr
amp_max = amp

print(sep + "Az-constrained slip setup" + sep)
print(f"  Plate convergence azimuth: {azimuth_deg:.1f}° CW from North")
print(f"  Amplitude bound: [0, {amp_max*1e3:.1f} mm/yr]")

print("Computing Az coefficient CG1 functions ...")
c_strike_fun, c_dip_fun = compute_az_cg1_functions(
    mesh, boundaries, fault, azimuth_deg, mesh_rotation_deg)

# %%
# ============================================================================
# SLIP PRESCRIPTION  (Az-directed, stripe pattern, nicoyaden_sm parameters)
# ============================================================================
# slip_pattern = "checker"
slip_pattern = "stripe"

### stripe pattern options
pattern_option = 1  # 1: 2-stripe, shallow-deep; 2: 1-stripe, intermediate
# pattern_option = 2

slip_azimuth_deg = azimuth_deg   # must match azimuth_deg above
# amp and amp_max already computed from plate velocity decomposition above

if slip_pattern == "checker":
    print("Creating fault-local checkerboard (Az)...")
    if pattern_option == 1:
        dx = 35e3; dy = 35e3; x0_p = -20e3; y0_p = -60e3; pattern_rot_deg = 45
    elif pattern_option == 2:
        dx = 30e3; dy = 30e3; x0_p =  20e3; y0_p = -15e3; pattern_rot_deg = 0
    mtrue_s_expr_gt = ut.create_fault_local_checkerboard_az(
        mesh=mesh, boundaries=boundaries, fault_id=fault,
        amp=amp, dx=dx, dy=dy, x0=x0_p, y0=y0_p,
        rotation_deg=pattern_rot_deg, azimuth_deg=slip_azimuth_deg,
        mesh_rotation_deg=mesh_rotation_deg, degree=5)
    slip_str_gt = (f"_check_x{x0_p/1e3:g}_y{y0_p/1e3:g}_dx{dx/1e3:g}_dy{dy/1e3:g}"
                   f"_rot{pattern_rot_deg:g}_ms{amp:g}_az{slip_azimuth_deg:g}")

elif slip_pattern == "stripe":
    print("Creating fault-local stripes (Az)...")
    if pattern_option == 1:
        # Stripe pattern, 2-stripe, shallow-deep  (nicoyaden_sm params: x_len=90 km, y0=-40 km)
        x_len = 90e3; y_len = 300e3; dx = 35e3
        stripe_spacing = x_len + dx; x0_p = 0; y0_p = -40e3; pattern_rot_deg = 0.0
    elif pattern_option == 2:
        x_len = 40e3; y_len = 300e3; dx = 100e3
        stripe_spacing = x_len + dx; x0_p = 0; y0_p = 22.5e3; pattern_rot_deg = 0.0
    mtrue_s_expr_gt = ut.create_fault_local_stripes_az(
        mesh=mesh, boundaries=boundaries, fault_id=fault,
        amp=amp, stripe_width=x_len, stripe_spacing=stripe_spacing, stripe_length=y_len,
        x0=x0_p, y0=y0_p, rotation_deg=pattern_rot_deg,
        azimuth_deg=slip_azimuth_deg, mesh_rotation_deg=mesh_rotation_deg, degree=5)
    slip_str_gt = (f"_stripe_x{x0_p/1e3:g}_y{y0_p/1e3:g}"
                   f"_lx{x_len/1e3:g}_dx{dx/1e3:g}"
                   f"_rot{pattern_rot_deg:g}_ms{amp:g}_az{slip_azimuth_deg:g}")

print(slip_str_gt)

# %%
# Station targets
ntargets = data.shape[0]
targets = np.zeros([ntargets, dim])
targets[:,0] = np.array(data['x'])*1e3
targets[:,1] = np.array(data['y'])*1e3
targets[:,2] = np.array(data['z'])*1e3
print(targets.shape)

# Dense grid targets
ntargets_dense = len(dense_data)
targets_grid = np.zeros([ntargets_dense, dim])
targets_grid[:,0] = np.array(dense_data['x'])*1e3
targets_grid[:,1] = np.array(dense_data['y'])*1e3
targets_grid[:,2] = np.array(dense_data['z'])*1e3
print(targets_grid.shape)

# Save dense grid coordinates
coord_file = f"dense_grid_coordinates_{meshname}_{grid_spacing_deg}.txt"
coord_output = open(resultpath + coord_file, 'w+')
for i in range(targets_grid.shape[0]):
    coord_output.write("%.6f %.6f %.6f %.6f %.6f\n" % (
        dense_data['lon'].iloc[i], dense_data['lat'].iloc[i],
        dense_data['x'].iloc[i], dense_data['y'].iloc[i], dense_data['z'].iloc[i]))
coord_output.close()

# %%
# Forward shear modulus (3D DeShon model, CG2)
mtrue_mu_expr_for = mtrue_mu_expr_het   # = None → 3D model inside forward solver
mu_str_for = mu_str_het
# CG degree: CG2 for 3D model (None), CG1 for K_2LAYER/homogeneous
if mtrue_mu_expr_for is None:
    CG_mu_deg = 2
    mu_str_for = mu_str_for + f"_CG{CG_mu_deg}"
else:
    CG_mu_deg = 1
print(f"Solving forward problem based on: {mu_str_for}, CG_mu_deg = {CG_mu_deg}")

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
        print("Average horizontal 1-sigma observation error: %.6f" % noise_std_h)
        print("Average vertical 1-sigma observation error: %.6f" % noise_std_v)
        f_h, f_v = 1, 1
        print("Data weight horizontal / vertical: %.2f / %.2f" % (f_h, f_v))
        slip_str_gt = slip_str_gt + "_pou"
    elif pollute_type == 'datastd':
        f_h, f_v = 1, 1
        print("Data weight horizontal / vertical: %.2f / %.2f" % (f_h, f_v))
        slip_str_gt = slip_str_gt + "_pod"
else:
    f_h, f_v = 1, 1

print(slip_str_gt)

# %%
# True slip expression for forward solve
mtrue_s_expr_for = mtrue_s_expr_gt

# %%
# Forward solve — SKIPPED: output files already generated by reference script
# (synth_stripeslip_az_inv_hetmu3D_nicoya_lock_noi.py with same parameters)
# Uncomment if forward files are missing or parameters have changed.
#
# solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for,
#                       pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Forward problem done!!!")
#
# computeGridDisplacements(k, targets_grid, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for,
#                          pollute=False, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Dense grid forward problem done!!!")

# %%
# Load synthetic data for inversion
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x','y','z','ux','uy','uz'])

# %%
# Regularization parameters
rho_s = 1e9
if pollute:
    if pollute_type == 'uniform':
        w_h, w_v = int(1/noise_std_h), int(1/noise_std_v)
    elif pollute_type == 'datastd':
        w_h, w_v = int(1/f_h), int(1/f_v)
else:
    w_h, w_v = int(1/f_h), int(1/f_v)

# %%
import gc

# ============================================================
# Mode switch: set RUN_LCURVE = False for single test run,
#              set RUN_LCURVE = True  to run the full L-curve
# ============================================================
RUN_LCURVE = True

# ============================================================
# SINGLE TEST RUN — 3D Het μ, large damping, Az scalar amplitude
# ============================================================
if not RUN_LCURVE:
    gamma_s_test = 1e3
    delta_s_test = gamma_s_test / rho_s
    inv_str    = f"_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s_test:.1e}_ds{delta_s_test:.1e}"
    mu_str_inv = mu_str_het + f"_CG{CG_mu_deg}"
    print(sep, "SINGLE TEST RUN — Az scalar amplitude inversion (3D Het μ)", sep)
    print("Inverse problem identifier:", inv_str)
    print(f"gamma_s = {gamma_s_test:.1e}, delta_s = {delta_s_test:.1e}, rho_s = {rho_s:.1e}")
    print("Solving inverse problem based on:", mu_str_inv)

    results_test = solveCoseismicInversion_AzSlip(
        k, targets, m0_amp_expr, mtrue_mu_expr_het,
        gamma_s_test, delta_s_test,
        c_strike_fun, c_dip_fun, amp_max,
        CG_mu_deg=CG_mu_deg,
        pollute=pollute, pollute_type=pollute_type,
        savefiles=True, verbose=True)

    # Unpack and compute amplitude recovery ratio (dual: global + local)
    _, xf, yf, zf, _, _, m_phys_fun_t, _, _, d_obs_t, d_cal_t, \
        m_amp_fault_t, m_sx_fault_t, m_sy_fault_t, misfitd_t, grad_m_t = results_test

    # True amplitude at fault vertices (from saved file)
    true_slip_file = resultpath + 'mtrue_s_fault_' + meshname + slip_str_gt + '.txt'
    true_s = np.loadtxt(true_slip_file)  # shape (N, 2): col0=s_strike, col1=s_dip
    true_amp_arr = np.sqrt(true_s[:,0]**2 + true_s[:,1]**2)

    # Global recovery: divide by plate convergence amplitude (no NaN, ∈ [0,1])
    recovery = m_amp_fault_t / amp

    # Local recovery: divide by true amplitude at each patch (NaN on creeping patches)
    eps = 0.01 * amp  # 1% of full amplitude threshold
    valid = true_amp_arr > eps
    recovery_local = np.where(valid, m_amp_fault_t / np.where(valid, true_amp_arr, 1.0), np.nan)

    print(f"  Amplitude recovery (global): [{recovery.min():.3f}, {recovery.max():.3f}]  (1.0 = fully locked)")
    print(f"  Amplitude recovery (local, stripe only): [{np.nanmin(recovery_local):.3f}, {np.nanmax(recovery_local):.3f}]")

    outFileName = ('slip_recovery_' + meshname + slip_str_gt
                   + mu_str_for + inv_str + mu_str_inv + '.txt')
    csvoutput = open(resultpath + outFileName, 'w+')
    csvoutput.write("# xf_m  yf_m  zf_m  true_amp_m  recovered_amp_m  recovery_ratio  recovery_local\n")
    for i in range(len(xf)):
        csvoutput.write("%.3f %.3f %.3f %.6f %.6f %.6f %.6f\n" % (
            xf[i], yf[i], zf[i], true_amp_arr[i], m_amp_fault_t[i],
            recovery[i], recovery_local[i]))
    csvoutput.close()
    print("Recovery file saved:", outFileName)
    del results_test; gc.collect()
    print("Single test run complete.")

# %%
# ============================================================
# L-curve: Het μ (3D) + Hom μ, Az scalar amplitude  (_passes)
# ============================================================
if RUN_LCURVE:
    gammas_s = [1e1, 4e2, 6e2, 8e2]

    outFileName_het = f"Lcurvesynscaamp_azbd_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_het}_CG{CG_mu_deg}.txt"
    outFileName_hom = f"Lcurvesynscaamp_azbd_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_hom}.txt"

    # Passes: (label, gammas, outFileName, mtrue_mu_expr, mu_str_inv_val, cg_deg)
    _passes = [
        ("Het L-curve", gammas_s, outFileName_het, mtrue_mu_expr_het, mu_str_het + f"_CG{CG_mu_deg}", CG_mu_deg),
        ("Hom L-curve", gammas_s, outFileName_hom, mtrue_mu_expr_hom, mu_str_hom,                    1),
    ]

    for _label, _gammas, _outFile, _mtrue_mu_expr, _mu_str_inv, _cg_deg in _passes:
        print(sep, f"{_label} — Az scalar amplitude inversion", sep)
        csvoutput = open(resultpath + _outFile, 'a')
        for gamma_s in _gammas:
            delta_s    = gamma_s / rho_s
            inv_str    = f"_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
            mu_str_inv = _mu_str_inv
            print("Inverse problem identifier:", inv_str)
            print(f"****** gamma_s = {gamma_s:.1e}, delta_s = {delta_s:.1e} ******")
            results = solveCoseismicInversion_AzSlip(
                k, targets, m0_amp_expr, _mtrue_mu_expr,
                gamma_s, delta_s, c_strike_fun, c_dip_fun, amp_max,
                CG_mu_deg=_cg_deg,
                pollute=pollute, pollute_type=pollute_type,
                savefiles=True, verbose=True)
            misfitd = float(results[-2]); grad_m = float(results[-1])
            csvoutput.write("%.6e %.6e %.1e %.0e\n" % (misfitd, grad_m, gamma_s, rho_s))
            csvoutput.flush()
            del results; gc.collect()
            try:
                import psutil
                print(f"  Memory: {psutil.Process().memory_info().rss/1024**3:.2f} GB")
            except ImportError:
                pass
        csvoutput.close()
        print(f"{_label} finished!")
    print("All Het(3D)+Hom Az scalar amplitude inversions finished!")
