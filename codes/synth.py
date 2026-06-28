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
# meshname = "nicoya2"   # This has a smaller fault interface
meshname = "nicoya3"   # the same as above but 5-km mesh size on fault
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
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip), tau('+')*n('+'))*dS(fault)

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
            # # Pollute true observations with random noise and generate synthetic data
            # hp.Random().uniform_perturb(noise_std_h, misfit.d)
            # misfit.noise_variance = noise_std_h*noise_std_h

            # # Modify misift.d to remove the values for other variables except for displacement
            # tmp = np.zeros(len(misfit.d),)
            # tmp[idx_d] = misfit.d[idx_d].copy()                 # horizontal and vertical displacement misfit         
            # # Assign the values of the new vector to misfit2.d
            # misfit.d.set_local(tmp)
            # misfit.d.apply('')
            
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
        if not os.path.exists(filename):
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
        if pollute_type == "uniform":
            # ##### USE 'pointwiseStateObs', without 'weight' option
            # misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
            # misfit.noise_variance = noise_std_h*noise_std_h
            # ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
            # obs_weights = np.zeros(targets.shape[0]*15,)
            # obs_weights[9::15]  = 1  # horizontal east displacement 
            # obs_weights[10::15] = 1  # horizontal north displacement 
            # obs_weights[11::15] = 1  # vectical displacement 

            ##### USE 'pointwiseStateObs_weights'
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
            obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
            # Modify and set the array to dolfin vector
            weights.set_local(obs_weights)
            weights.apply('')
            # Different from the pure inversion, define misfit without 'weight' option
            misfit = PSBW( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
            misfit.noise_variance = 1.            

        elif pollute_type == "datastd":
            ##### USE 'pointwiseStateObs_weights'
            weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
            obs_weights = np.zeros(targets.shape[0]*15,)
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
# ##### Problematic!!! OLD CHECKERBOARD SLIP MODEL #####
# # Define true model, m_strike = 0, m_dip = checkerboard, alternating between 0 and max
# V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
# amp = V_norm   # interseismic coupling case, max == complete coupling of 1 == the trench-normal long-term loading
# # amp = 1   # coseismic case, max == 1 m
# dx = 40e3  # grid spacing in x direction, \lamda_x = 2*dx
# dy = 40e3  # spacing in y direction, \lamda_y = 2*dy
# x0 = -20e3
# y0 = -20e3
# omega_x = np.pi/dx  # \omega = 2*pi/ \lamda_x
# omega_y = np.pi/dy
# # sin or cos checkerboard pattern in x and y, 0.5*(sin+1) to make the variation between 0 and 1
# mtrue_s_expr_1 = dl.Expression('amp * 0.5*(sin(omega_x*(x[0]-x0))+1.0) * 0.5*(sin(omega_y*(x[1]-y0))+1.0)', 
#                                amp=amp, omega_x=omega_x, omega_y=omega_y, x0=x0, y0=y0, degree=5 )
# ##### CHECKERBOARD SLIP MODEL #####

# # Only apply depth and constraints
# mtrue_s_expr_2 = dl.Expression(
#     '((x[2] >= zmin) && (x[2] <= zmax)) ? 1.0 : 0.0',
#     zmin=-60e3, zmax=0,
#     degree=1
# )

# # Final true model expression (m_strike = 0, m_dip = checkerboard inside the specified region)
# mtrue_s_expr_gt = dl.Expression(('0.', 'mask1 * mask2'), 
#                              mask1=mtrue_s_expr_1, mask2=mtrue_s_expr_2, degree=5)

# slip_str_gt = f"_check_x{x0/1e3:g}_y{y0/1e3:g}_dx{dx/1e3:g}_dy{dy/1e3:g}_ms{amp:g}"
# print(slip_str_gt)  

# %%
### define the pattern of the slip distribution
slip_pattern = "checker"
# slip_pattern = "stripe"

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

    # # Option 2, checkerboard along N-E directions
    # dx = 30e3  # grid spacing in x direction, \lamda_x = 2*dx
    # dy = 30e3  # spacing in y direction, \lamda_y = 2*dy
    # x0 = 20e3  # offset along strike
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
    
    # # Stripe pattern parameters
    # x_len = 35e3     # width of each rectangle in dip direction
    # y_len = 300e3   # length of each rectangle in strike direction  
    # dx = 25e3  # gap between rectangles
    # stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
    # x0 = 0     # x center of the pattern
    # y0 = 0     # y center of the pattern
    # rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)
    # # rot_deg = 45.0

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
# # Solve the coseismic forward problem to generate the synthetic data within a Homogeneous shear modulus half-space
# mtrue_mu_expr_for = mtrue_mu_expr_hom
# mu_str_for = mu_str_hom

# Solve the coseismic forward problem to generate the synthetic data within a Heterogeneous shear modulus half-space
mtrue_mu_expr_for = mtrue_mu_expr_het
mu_str_for = mu_str_het

print("Solving forward problem based on: ", mu_str_for)

# mtrue_s_expr_for = None
mtrue_s_expr_for = mtrue_s_expr_gt
if mtrue_s_expr_for==None:
    print("Ground-truth slip is from inversion")
else:
    print("Ground-truth slip is customized")   

# %%
# # Solve forward problem to generate synthetics
# solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
#                       pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Forward problem done!!!")

# %% [markdown]
# # Use the synthetic displacement for inversion within any $\mu$ structure model

# %%
# Load the synethetic displacement data from the forward problem
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x', 'y', 'z', 'ux', 'uy', 'uz'])

# %%
# SLIP TRANSFORMATION SETUP
print(sep + "SLIP TRANSFORMATION SETUP" + sep)

BOUNDED = True
BOUND_TYPE = 'both'
# BOUND_TYPE = 'dip'

if BOUNDED:
    # Define slip bounds based on your problem
    V_para = 16/1e3
    if BOUND_TYPE == 'both':
        slip_transformer = SlipTransformation(
            strike_bounds=(-2e-3, 2e-3),
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
# rho_s = 1e9   # allows variations of slip of the order of ~30 km 

rho_s = 1e9   # allows variations of slip of the order of ~3 km, close to the maximum resolution

# gamma_val_H1 = 1e1
# gamma_val_H1 = 2e1    
gamma_val_H1 = 4e1    
# gamma_val_H1 = 5e1  
# gamma_val_H1 = 1e2  
# gamma_val_H1 = 2e2
# gamma_val_H1 = 4e2
# gamma_val_H1 = 5e2
# gamma_val_H1 = 1e3    
# gamma_val_H1 = 1e4    
delta_val_L2 = gamma_val_H1 / rho_s  

# %%
if pollute:
    if pollute_type == 'uniform':
        # Take the inverse for saving the name of the weights
        # w_h, w_v = int(1/noise_std_h), int(1/noise_std_h)
        w_h, w_v = int(1/noise_std_h), int(1/noise_std_v)
    elif pollute_type == 'datastd':
        # Take the inverse for saving the name of the weights
        w_h, w_v = int(1/f_h), int(1/f_v)
else:
    # Take the inverse for saving the name of the weights
    w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
if BOUNDED:
    inv_str = f"_synlockbd_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
else:
    inv_str = f"_synlock_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
print("Inverse problem identifier: ", inv_str)

# %%
# Solve slip inverse problem within the heterogeneous shear modulus half-space
mtrue_mu_expr_inv = mtrue_mu_expr_het
mu_str_inv = mu_str_het
print("Solving inverse problem based on: ", mu_str_inv)
results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
        slip_transformer,  # KEY: Pass slip transformation
        pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    )
print("Het Inversion finished!!!")


# %%
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
# import numpy as np
# import dolfin as dl
# from scipy import ndimage
# from sklearn.cluster import DBSCAN
# import scipy.stats

# def identify_and_assess_individual_stripes(mtrue_s, minferred_s, mesh, boundaries, fault_id, 
#                                          stripe_threshold=0.5, min_stripe_size=10):
#     """
#     Identify individual stripes and assess their recovery separately
    
#     Parameters:
#     -----------
#     mtrue_s, minferred_s : dolfin.Vector
#         True and inferred slip vectors
#     mesh, boundaries, fault_id : dolfin objects
#         Mesh information
#     stripe_threshold : float
#         Threshold for identifying active stripe regions (fraction of max amplitude)
#     min_stripe_size : int
#         Minimum number of nodes to consider as a valid stripe
#     """
    
#     # Extract fault interface values and coordinates
#     bc_fault = dl.DirichletBC(dl.VectorFunctionSpace(mesh, "CG", 1, dim=2), (99, 99), boundaries, fault_id)
#     um_fault = dl.Function(dl.VectorFunctionSpace(mesh, "CG", 1, dim=2))
#     bc_fault.apply(um_fault.vector())
    
#     fault_mask = (um_fault.vector().get_local() == 99)
#     fault_indices = np.where(fault_mask)[0]
    
#     # Get fault coordinates
#     fault_coords = []
#     Vm = dl.VectorFunctionSpace(mesh, "CG", 1, dim=2)
#     dof_coords = Vm.tabulate_dof_coordinates()
    
#     for idx in fault_indices[1::2]:  # Take every other index (dip component)
#         coord_idx = idx // 2  # Convert to coordinate index
#         if coord_idx < len(dof_coords):
#             fault_coords.append(dof_coords[coord_idx])
    
#     fault_coords = np.array(fault_coords)
    
#     # Extract dip slip components
#     mtrue_fault = mtrue_s.get_local()[fault_mask]
#     minferred_fault = minferred_s.get_local()[fault_mask]
    
#     mtrue_dip = mtrue_fault[1::2]
#     minferred_dip = minferred_fault[1::2]
    
#     # Identify individual stripes using connected component analysis
#     stripes_info = identify_stripes_spatial(mtrue_dip, fault_coords, stripe_threshold, min_stripe_size)
    
#     print(f"\n=== INDIVIDUAL STRIPE IDENTIFICATION ===")
#     print(f"Total fault nodes: {len(mtrue_dip)}")
#     print(f"Identified {len(stripes_info['stripes'])} individual stripes")
    
#     # Assess each stripe individually
#     stripe_assessments = {}
    
#     for i, stripe_data in enumerate(stripes_info['stripes']):
#         stripe_id = f"stripe_{i+1}"
#         print(f"\n--- {stripe_id.upper()} ---")
        
#         stripe_indices = stripe_data['indices']
#         stripe_coords = stripe_data['coordinates']
        
#         # Extract stripe values
#         stripe_true = mtrue_dip[stripe_indices]
#         stripe_inferred = minferred_dip[stripe_indices]
        
#         # Individual stripe assessment
#         assessment = assess_individual_stripe(
#             stripe_true, stripe_inferred, stripe_coords, stripe_id
#         )
        
#         stripe_assessments[stripe_id] = assessment
        
#         # Print key metrics
#         print(f"  Nodes: {len(stripe_indices)}")
#         print(f"  Depth range: {stripe_coords[:, 2].min()/1000:.1f} to {stripe_coords[:, 2].max()/1000:.1f} km")
#         print(f"  Recovery score: {assessment['recovery_metrics']['recovery_score']:.3f}")
#         print(f"  Amplitude recovery: {assessment['amplitude_analysis']['amplitude_recovery_percentage']:.1f}%")
#         print(f"  Spatial coverage: {assessment['spatial_analysis']['coverage_percentage']:.1f}%")
#         print(f"  Pattern integrity: {assessment['pattern_integrity']['integrity_score']:.3f}")
    
#     # Global assessment for comparison
#     global_assessment = comprehensive_pattern_recovery_global(mtrue_dip, minferred_dip)
    
#     # Summary comparison
#     print(f"\n=== STRIPE vs GLOBAL COMPARISON ===")
#     individual_scores = [stripe_assessments[s]['recovery_metrics']['recovery_score'] 
#                         for s in stripe_assessments.keys()]
    
#     if individual_scores:
#         # Filter out nan values for statistics
#         valid_scores = [score for score in individual_scores if not np.isnan(score)]
        
#         print(f"Global recovery score: {global_assessment['recovery_score']:.3f}")
#         if valid_scores:
#             print(f"Average stripe score: {np.mean(valid_scores):.3f}")
#             print(f"Best stripe score: {np.max(valid_scores):.3f}")
#             print(f"Worst stripe score: {np.min(valid_scores):.3f}")
#         else:
#             print("Average stripe score: N/A (all nan)")
#             print("Best stripe score: N/A")
#             print("Worst stripe score: N/A")
    
#     return {
#         'stripe_assessments': stripe_assessments,
#         'global_assessment': global_assessment,
#         'stripes_info': stripes_info,
#         'summary_stats': {
#             'num_stripes': len(stripes_info['stripes']),
#             'average_stripe_score': np.mean(individual_scores) if individual_scores else 0,
#             'global_score': global_assessment['recovery_score']
#         }
#     }

# def identify_stripes_from_true_model(mtrue_dip, fault_coords, threshold=0.5, min_size=10):
#     """
#     Identify individual stripes based on TRUE model using spatial connectivity
#     """
#     # Create binary mask for active regions in TRUE model
#     max_amp = np.max(mtrue_dip)
#     active_mask = mtrue_dip > (threshold * max_amp)
#     active_indices = np.where(active_mask)[0]
    
#     print(f"Active region identification:")
#     print(f"  Max amplitude: {max_amp:.6f}")
#     print(f"  Threshold ({threshold*100:.0f}%): {threshold * max_amp:.6f}")
#     print(f"  Active nodes: {len(active_indices)} / {len(mtrue_dip)}")
    
#     if len(active_indices) < min_size:
#         print(f"  WARNING: Too few active nodes ({len(active_indices)} < {min_size})")
#         return {'stripes': [], 'method': 'insufficient_active_nodes'}
    
#     # For stripe patterns, use depth-based separation primarily
#     active_coords = fault_coords[active_indices]
#     active_depths = active_coords[:, 2]
    
#     # Use hierarchical clustering on depth to separate stripes
#     from scipy.cluster.hierarchy import fcluster, linkage
#     from scipy.spatial.distance import pdist
    
#     # If only one coordinate, can't cluster
#     if len(active_coords) < 2:
#         return {'stripes': [], 'method': 'insufficient_data_for_clustering'}
    
#     # Primary clustering on depth (z-coordinate)
#     depths_reshaped = active_depths.reshape(-1, 1)
    
#     try:
#         # Use depth-based clustering with gap detection
#         if len(np.unique(active_depths)) > 1:
#             # Linkage clustering on depth
#             linkage_matrix = linkage(depths_reshaped, method='ward')
            
#             # Estimate number of clusters based on depth gaps
#             sorted_depths = np.sort(active_depths)
#             depth_diffs = np.diff(sorted_depths)
            
#             # Look for large gaps (> 2 * median gap)
#             if len(depth_diffs) > 0:
#                 median_gap = np.median(depth_diffs)
#                 large_gaps = depth_diffs > 2 * median_gap
#                 estimated_clusters = np.sum(large_gaps) + 1
#                 estimated_clusters = max(1, min(estimated_clusters, len(active_indices) // min_size))
#             else:
#                 estimated_clusters = 1
            
#             print(f"  Estimated clusters from depth analysis: {estimated_clusters}")
            
#             # Apply clustering
#             cluster_labels = fcluster(linkage_matrix, estimated_clusters, criterion='maxclust') - 1
#         else:
#             # All points at same depth - single cluster
#             cluster_labels = np.zeros(len(active_indices))
#             estimated_clusters = 1
            
#     except Exception as e:
#         print(f"  Clustering failed: {e}")
#         # Fallback: single cluster
#         cluster_labels = np.zeros(len(active_indices))
#         estimated_clusters = 1
    
#     # Build stripe information
#     stripes = []
#     for cluster_id in range(estimated_clusters):
#         cluster_mask = cluster_labels == cluster_id
#         cluster_indices = active_indices[cluster_mask]
        
#         if len(cluster_indices) >= min_size:
#             cluster_coords = fault_coords[cluster_indices]
            
#             stripe_info = {
#                 'label': int(cluster_id),
#                 'indices': cluster_indices,
#                 'coordinates': cluster_coords,
#                 'size': len(cluster_indices),
#                 'depth_range': (cluster_coords[:, 2].min(), cluster_coords[:, 2].max()),
#                 'centroid': np.mean(cluster_coords, axis=0),
#                 'depth_centroid': np.mean(cluster_coords[:, 2])
#             }
#             stripes.append(stripe_info)
            
#             print(f"  Cluster {cluster_id}: {len(cluster_indices)} nodes, "
#                   f"depth {cluster_coords[:, 2].min()/1000:.1f} to {cluster_coords[:, 2].max()/1000:.1f} km")
    
#     # Sort stripes by depth (shallowest first)
#     stripes.sort(key=lambda x: x['depth_centroid'], reverse=True)
    
#     # Renumber stripes after sorting
#     for i, stripe in enumerate(stripes):
#         stripe['label'] = i
    
#     return {
#         'stripes': stripes,
#         'method': 'depth_based_hierarchical',
#         'parameters': {
#             'threshold': threshold, 
#             'min_size': min_size,
#             'estimated_clusters': estimated_clusters
#         }
#     }

# def assess_individual_stripe(stripe_true, stripe_inferred, stripe_coords, stripe_id):
#     """
#     Comprehensive assessment of individual stripe recovery
#     """
#     assessment = {}
    
#     # 1. Basic recovery metrics
#     # Handle correlation calculation safely
#     if len(stripe_true) > 1 and np.std(stripe_true) > 1e-10 and np.std(stripe_inferred) > 1e-10:
#         correlation = np.corrcoef(stripe_true, stripe_inferred)[0, 1]
#         if np.isnan(correlation):
#             correlation = 0.0
#     else:
#         correlation = 0.0
    
#     rmse = np.sqrt(np.mean((stripe_true - stripe_inferred)**2))
    
#     # Handle NRMSE calculation safely
#     data_range = stripe_true.max() - stripe_true.min()
#     if data_range > 1e-10:
#         nrmse = rmse / data_range
#     else:
#         nrmse = 0.0 if rmse < 1e-10 else 1.0  # If no variation in true data
    
#     # Calculate recovery score safely
#     recovery_score = correlation * (1 - min(nrmse, 1)) if not np.isnan(correlation) else 0.0
    
#     assessment['recovery_metrics'] = {
#         'correlation': float(correlation),
#         'rmse': float(rmse),
#         'nrmse': float(nrmse),
#         'recovery_score': float(recovery_score)
#     }
    
#     # 2. Amplitude analysis
#     true_max = np.max(stripe_true)
#     inferred_max = np.max(stripe_inferred)
#     true_mean = np.mean(stripe_true)
#     inferred_mean = np.mean(stripe_inferred)
    
#     # Percentage of stripe where inferred amplitude exceeds half the true amplitude
#     half_true_threshold = 0.5 * true_max
#     recovery_nodes = np.sum(stripe_inferred > half_true_threshold)
#     amplitude_recovery_percentage = 100 * recovery_nodes / len(stripe_true)
    
#     assessment['amplitude_analysis'] = {
#         'true_max': float(true_max),
#         'inferred_max': float(inferred_max),
#         'true_mean': float(true_mean),
#         'inferred_mean': float(inferred_mean),
#         'amplitude_ratio': float(inferred_max / true_max) if true_max > 0 else 0,
#         'mean_ratio': float(inferred_mean / true_mean) if true_mean > 0 else 0,
#         'amplitude_recovery_percentage': float(amplitude_recovery_percentage),
#         'nodes_above_half_amplitude': int(recovery_nodes)
#     }
    
#     # 3. Spatial coverage analysis (amplitude-based)
#     # Define active region: where true slip > 10% of maximum
#     active_threshold = 0.1 * true_max
#     true_active_mask = stripe_true > active_threshold
#     true_active_nodes = np.sum(true_active_mask)
    
#     # Within active region, where does inferred slip exceed 50% of true maximum?
#     recovery_threshold = 0.5 * true_max
#     if true_active_nodes > 0:
#         recovered_mask = (true_active_mask) & (stripe_inferred > recovery_threshold)
#         recovered_nodes = np.sum(recovered_mask)
#         coverage_percentage = 100 * recovered_nodes / true_active_nodes
#     else:
#         recovered_nodes = 0
#         coverage_percentage = 0
    
#     assessment['spatial_analysis'] = {
#         'total_nodes': len(stripe_true),
#         'true_active_nodes': int(true_active_nodes),
#         'recovered_nodes': int(recovered_nodes),
#         'coverage_percentage': float(coverage_percentage),
#         'coverage_method': 'amplitude_based',
#         'active_threshold_pct': 10.0,
#         'recovery_threshold_pct': 50.0,
#         'depth_range_km': float((stripe_coords[:, 2].max() - stripe_coords[:, 2].min()) / 1000),
#         'centroid_depth_km': float(np.mean(stripe_coords[:, 2]) / 1000)
#     }
    
#     # 4. Pattern integrity (how well the shape is preserved)
#     # Normalize both patterns to [0,1] for shape comparison
#     if true_max > 1e-10 and inferred_max > 1e-10:
#         stripe_true_norm = stripe_true / true_max
#         stripe_inferred_norm = stripe_inferred / inferred_max
        
#         # Safe shape correlation calculation
#         if len(stripe_true) > 1 and np.std(stripe_true_norm) > 1e-10 and np.std(stripe_inferred_norm) > 1e-10:
#             shape_correlation = np.corrcoef(stripe_true_norm, stripe_inferred_norm)[0, 1]
#             if np.isnan(shape_correlation):
#                 shape_correlation = 0.0
#         else:
#             shape_correlation = 0.0
#     else:
#         shape_correlation = 0.0
    
#     # Pattern sharpness preservation - safe gradient calculation
#     if len(stripe_true) > 2:
#         true_gradient = np.gradient(stripe_true)
#         inferred_gradient = np.gradient(stripe_inferred)
        
#         if np.std(true_gradient) > 1e-10 and np.std(inferred_gradient) > 1e-10:
#             gradient_correlation = np.corrcoef(true_gradient, inferred_gradient)[0, 1]
#             if np.isnan(gradient_correlation):
#                 gradient_correlation = 0.0
#         else:
#             gradient_correlation = 0.0
#     else:
#         gradient_correlation = 0.0
    
#     assessment['pattern_integrity'] = {
#         'shape_correlation': float(shape_correlation),
#         'gradient_correlation': float(gradient_correlation),
#         'integrity_score': float((shape_correlation + gradient_correlation) / 2)
#     }
    
#     # 5. Statistical distribution comparison - safe calculations
#     true_std = np.std(stripe_true)
#     inferred_std = np.std(stripe_inferred)
#     std_ratio = inferred_std / true_std if true_std > 1e-10 else 0.0
    
#     # Safe skewness calculation
#     if len(stripe_true) > 2:
#         try:
#             # Only calculate if there's sufficient variation
#             if true_std > 1e-10 and inferred_std > 1e-10:
#                 true_skew = scipy.stats.skew(stripe_true)
#                 inferred_skew = scipy.stats.skew(stripe_inferred)
#                 skewness_difference = abs(true_skew - inferred_skew) if not (np.isnan(true_skew) or np.isnan(inferred_skew)) else 0.0
#             else:
#                 skewness_difference = 0.0
#         except:
#             skewness_difference = 0.0
#     else:
#         skewness_difference = 0.0
    
#     assessment['distribution_analysis'] = {
#         'true_std': float(true_std),
#         'inferred_std': float(inferred_std),
#         'std_ratio': float(std_ratio),
#         'skewness_difference': float(skewness_difference)
#     }
    
#     return assessment

# def comprehensive_pattern_recovery_global(mtrue_dip, minferred_dip):
#     """Global assessment for comparison - safe calculations"""
#     if len(mtrue_dip) > 1 and np.std(mtrue_dip) > 1e-10 and np.std(minferred_dip) > 1e-10:
#         correlation = np.corrcoef(mtrue_dip, minferred_dip)[0, 1]
#         if np.isnan(correlation):
#             correlation = 0.0
#     else:
#         correlation = 0.0
        
#     data_range = mtrue_dip.max() - mtrue_dip.min()
#     rmse = np.sqrt(np.mean((mtrue_dip - minferred_dip)**2))
    
#     if data_range > 1e-10:
#         nrmse = rmse / data_range
#     else:
#         nrmse = 0.0 if rmse < 1e-10 else 1.0
        
#     recovery_score = correlation * (1 - min(nrmse, 1)) if not np.isnan(correlation) else 0.0
    
#     return {
#         'recovery_score': float(recovery_score),
#         'correlation': float(correlation),
#         'nrmse': float(nrmse)
#     }


# %%
# def analyze_stripes_correctly(mtrue_s, minferred_s, mesh, boundaries, fault_id):
#     """
#     Correct approach - no spatial assumptions about indices
#     """
#     # Use your exact working method
#     Vm = dl.VectorFunctionSpace(mesh, "CG", 1, dim=2)
#     bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault_id)
#     um2 = dl.Function(Vm)
#     bc2.apply(um2.vector())
    
#     # Extract fault slip - EXACTLY like your validation
#     mtrue_s_fault = mtrue_s.get_local()[um2.vector().get_local() == 99]
#     minferred_s_fault = minferred_s.get_local()[um2.vector().get_local() == 99]
    
#     mtrue_sy_fault = mtrue_s_fault[1::2]  # Dip slip
#     minferred_sy_fault = minferred_s_fault[1::2]  # Dip slip
    
#     print(f"Fault analysis:")
#     print(f"  Total fault nodes: {len(mtrue_sy_fault)}")
#     print(f"  Non-zero nodes: {np.count_nonzero(mtrue_sy_fault)}")
#     print(f"  Zero nodes: {len(mtrue_sy_fault) - np.count_nonzero(mtrue_sy_fault)}")
#     print(f"  True dip range: {mtrue_sy_fault.min():.6f} to {mtrue_sy_fault.max():.6f}")
    
#     # Find active nodes (non-zero or above threshold)
#     threshold = 0.1 * mtrue_sy_fault.max()  # Lower threshold
#     active_mask = mtrue_sy_fault > threshold
#     active_indices = np.where(active_mask)[0]
    
#     print(f"\nActive region analysis:")
#     print(f"  Threshold (10% max): {threshold:.6f}")
#     print(f"  Active nodes: {len(active_indices)} / {len(mtrue_sy_fault)}")
    
#     # PROPER way: need spatial coordinates to separate stripes
#     # But since we can't get coordinates easily, let's use a different approach
#     # Look at the pattern of non-zero values to infer stripe structure
    
#     # Method: Find continuous segments of active nodes
#     if len(active_indices) > 0:
#         # Find gaps in the sequence of active indices
#         index_diffs = np.diff(active_indices)
#         large_gaps = np.where(index_diffs > 50)[0]  # Gaps of >50 indices
        
#         print(f"Gap analysis in index space:")
#         print(f"  Index differences: min={index_diffs.min()}, max={index_diffs.max()}, median={np.median(index_diffs):.1f}")
#         print(f"  Large gaps (>50): {len(large_gaps)} found")
        
#         if len(large_gaps) > 0:
#             print(f"  Gap positions: {large_gaps}")
            
#             # Split at gaps
#             split_points = [0] + (large_gaps + 1).tolist() + [len(active_indices)]
            
#             for i in range(len(split_points) - 1):
#                 start_idx = split_points[i]
#                 end_idx = split_points[i + 1]
                
#                 segment_indices = active_indices[start_idx:end_idx]
                
#                 if len(segment_indices) > 20:  # Only analyze significant segments
#                     true_vals = mtrue_sy_fault[segment_indices]
#                     inferred_vals = minferred_sy_fault[segment_indices]
                    
#                     # Correlation calculation
#                     if np.std(true_vals) > 1e-10 and np.std(inferred_vals) > 1e-10:
#                         correlation = np.corrcoef(true_vals, inferred_vals)[0, 1]
#                         if np.isnan(correlation):
#                             correlation = 0.0
#                     else:
#                         correlation = 0.0
                    
#                     # Amplitude recovery
#                     recovery_threshold = 0.5 * true_vals.max()
#                     recovery_count = np.sum(inferred_vals > recovery_threshold)
#                     recovery_pct = 100 * recovery_count / len(true_vals)
                    
#                     print(f"\n--- SEGMENT {i+1} ---")
#                     print(f"  Node indices: {segment_indices.min()} to {segment_indices.max()}")
#                     print(f"  Actual node count: {len(segment_indices)}")  # This should match
#                     print(f"  Index range size: {segment_indices.max() - segment_indices.min() + 1}")
#                     print(f"  True max: {true_vals.max():.6f}")
#                     print(f"  Inferred max: {inferred_vals.max():.6f}")
#                     print(f"  Correlation: {correlation:.3f}")
#                     print(f"  Amplitude recovery: {recovery_pct:.1f}%")
        
#         else:
#             print("  No significant gaps found - likely single connected region")
    
#     else:
#         print("  No active nodes found!")

# # Test the corrected version:
# analyze_stripes_correctly(mtrue, m, mesh, boundaries, fault)

# %%
# def analyze_stripes_with_correct_coordinates(mtrue_s, minferred_s, mesh, boundaries, fault_id, 
#                                            x_len=80e3, y_len=300e3, dx=35e3, 
#                                            x0=0, y0=-35e3, rot_deg=0.0):
#     """
#     Analyze stripes using your proven coordinate extraction method
#     """
#     print("\n" + "="*60)
#     print("STRIPE ANALYSIS WITH CORRECT COORDINATE EXTRACTION")
#     print("="*60)
    
#     # Step 1: Extract fault slip using your working method
#     Vm = dl.VectorFunctionSpace(mesh, "CG", 1, dim=2)
#     bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault_id)
#     um2 = dl.Function(Vm)
#     bc2.apply(um2.vector())
    
#     mtrue_s_fault = mtrue_s.get_local()[um2.vector().get_local() == 99]
#     minferred_s_fault = minferred_s.get_local()[um2.vector().get_local() == 99]
    
#     mtrue_dip = mtrue_s_fault[1::2]
#     minferred_dip = minferred_s_fault[1::2]
    
#     print(f"Extracted fault slip:")
#     print(f"  Total fault nodes: {len(mtrue_dip)}")
#     print(f"  Non-zero nodes: {np.count_nonzero(mtrue_dip)}")
    
#     # Step 2: Extract fault coordinates using YOUR proven method
#     CG = dl.VectorFunctionSpace(mesh, "CG", degree=1)
#     bc1 = dl.DirichletBC(CG, (10, 10, 10), boundaries, fault_id)
#     um = dl.Function(CG)
#     bc1.apply(um.vector())
    
#     # Create coordinate interpolation functions
#     xslip = dl.interpolate(dl.Expression(("x[0]", "x[0]", "x[0]"), degree=5), CG)
#     yslip = dl.interpolate(dl.Expression(("x[1]", "x[1]", "x[1]"), degree=5), CG)
#     zslip = dl.interpolate(dl.Expression(("x[2]", "x[2]", "x[2]"), degree=5), CG)
    
#     # Extract fault coordinates
#     xf = xslip.vector().get_local()[um.vector().get_local() == 10]
#     yf = yslip.vector().get_local()[um.vector().get_local() == 10] 
#     zf = zslip.vector().get_local()[um.vector().get_local() == 10]
    
#     print(f"Extracted fault coordinates:")
#     print(f"  Coordinate points: {len(xf)}")
#     print(f"  X range: {xf.min()/1000:.1f} to {xf.max()/1000:.1f} km")
#     print(f"  Y range: {yf.min()/1000:.1f} to {yf.max()/1000:.1f} km")
#     print(f"  Z range: {zf.min()/1000:.1f} to {zf.max()/1000:.1f} km")
    
#     # Verify array lengths match
#     print(f"Array length check:")
#     print(f"  Slip values: {len(mtrue_dip)}")
#     print(f"  Coordinates: {len(xf)}")
    
#     if len(xf) != len(mtrue_dip):
#         print(f"  WARNING: Array length mismatch!")
#         # Try to handle mismatch by truncating
#         min_len = min(len(xf), len(mtrue_dip))
#         xf = xf[:min_len]
#         yf = yf[:min_len]
#         zf = zf[:min_len]
#         mtrue_dip = mtrue_dip[:min_len]
#         minferred_dip = minferred_dip[:min_len]
#         print(f"  Truncated to {min_len} points")
    
#     # Step 3: Compute fault geometry (same as before)
#     fault_normals = []
#     fault_centers = []
    
#     for facet in dl.facets(mesh):
#         if boundaries[facet] == fault_id:
#             normal = facet.normal()
#             fault_normals.append([normal.x(), normal.y(), normal.z()])
#             center = facet.midpoint()
#             fault_centers.append([center.x(), center.y(), center.z()])
    
#     fault_normals = np.array(fault_normals)
#     avg_normal = np.mean(fault_normals, axis=0)
#     avg_normal = avg_normal / np.linalg.norm(avg_normal)
    
#     horizontal_normal = np.array([avg_normal[0], avg_normal[1], 0])
#     horizontal_normal_mag = np.linalg.norm(horizontal_normal)
    
#     if horizontal_normal_mag > 1e-10:
#         strike_vector = np.array([-avg_normal[1], avg_normal[0], 0])
#         strike_vector = strike_vector / np.linalg.norm(strike_vector)
#     else:
#         strike_vector = np.array([1, 0, 0])
    
#     dip_vector = np.cross(strike_vector, avg_normal)
#     dip_vector = dip_vector / np.linalg.norm(dip_vector)
    
#     if dip_vector[2] > 0:
#         dip_vector = -dip_vector
    
#     print(f"Fault geometry:")
#     print(f"  Strike vector: [{strike_vector[0]:.3f}, {strike_vector[1]:.3f}, {strike_vector[2]:.3f}]")
#     print(f"  Dip vector: [{dip_vector[0]:.3f}, {dip_vector[1]:.3f}, {dip_vector[2]:.3f}]")
    
#     # Step 4: Apply stripe classification to each node
#     stripe_width = x_len
#     stripe_spacing = x_len + dx
#     stripe_length = y_len
    
#     rotation_rad = np.deg2rad(rot_deg)
#     cos_rot = np.cos(rotation_rad)
#     sin_rot = np.sin(rotation_rad)
    
#     print(f"\nStripe classification:")
#     print(f"  Stripe width: {stripe_width/1000:.0f} km")
#     print(f"  Gap width: {dx/1000:.0f} km")
#     print(f"  Stripe spacing: {stripe_spacing/1000:.0f} km")
#     print(f"  Stripe length: {stripe_length/1000:.0f} km")
    
#     stripe_assignments = []
    
#     for i in range(len(xf)):
#         x_coord = [xf[i], yf[i], zf[i]]
        
#         # Transform to fault-local coordinates
#         s = np.dot(x_coord, strike_vector)
#         d = np.dot(x_coord, dip_vector)
        
#         # Apply rotation
#         s_centered = s - x0
#         d_centered = d - y0
#         s_rot = cos_rot * s_centered - sin_rot * d_centered
#         d_rot = sin_rot * s_centered + cos_rot * d_centered
        
#         # Apply stripe pattern logic
#         d_mod = abs(d_rot) % stripe_spacing
#         stripe_half_width = stripe_width / 2.0
#         in_stripe_d = (d_mod <= stripe_half_width) or (d_mod >= (stripe_spacing - stripe_half_width))
        
#         stripe_half_length = stripe_length / 2.0
#         in_stripe_s = abs(s_rot) <= stripe_half_length
        
#         is_in_stripe = in_stripe_d and in_stripe_s
        
#         # Determine stripe number
#         stripe_id = None
#         if is_in_stripe:
#             # Determine which stripe based on position in dip direction
#             if d_mod <= stripe_half_width:
#                 stripe_id = "stripe_1"
#             else:
#                 stripe_id = "stripe_2"
        
#         stripe_assignments.append({
#             'index': i,
#             'coordinates': x_coord,
#             'fault_local_s': s,
#             'fault_local_d': d,
#             'rotated_d': d_rot,
#             'in_stripe': is_in_stripe,
#             'stripe_id': stripe_id
#         })
    
#     # Group by stripe
#     stripe_groups = {'stripe_1': [], 'stripe_2': [], 'gap': []}
    
#     for assignment in stripe_assignments:
#         if assignment['stripe_id']:
#             stripe_groups[assignment['stripe_id']].append(assignment)
#         else:
#             stripe_groups['gap'].append(assignment)
    
#     print(f"Classification results:")
#     print(f"  Stripe 1 nodes: {len(stripe_groups['stripe_1'])}")
#     print(f"  Stripe 2 nodes: {len(stripe_groups['stripe_2'])}")
#     print(f"  Gap nodes: {len(stripe_groups['gap'])}")
    
#     # Step 5: Analyze each stripe
#     for stripe_name in ['stripe_1', 'stripe_2']:
#         stripe_data = stripe_groups[stripe_name]
        
#         if len(stripe_data) > 10:
#             indices = [item['index'] for item in stripe_data]
#             coords = np.array([item['coordinates'] for item in stripe_data])
            
#             true_vals = mtrue_dip[indices]
#             inferred_vals = minferred_dip[indices]
            
#             # Calculate correlation safely
#             correlation = 0.0
#             if len(true_vals) > 1 and np.std(true_vals) > 1e-10 and np.std(inferred_vals) > 1e-10:
#                 correlation = np.corrcoef(true_vals, inferred_vals)[0, 1]
#                 if np.isnan(correlation):
#                     correlation = 0.0
            
#             # Amplitude recovery metrics
#             true_max = np.max(true_vals)
#             recovery_threshold = 0.5 * true_max
#             recovery_count = np.sum(inferred_vals > recovery_threshold)
#             amplitude_recovery = 100 * recovery_count / len(true_vals)
            
#             # Spatial coverage (improved definition)
#             active_threshold = 0.1 * true_max
#             true_active_mask = true_vals > active_threshold
#             true_active_count = np.sum(true_active_mask)
#             recovered_count = np.sum(true_active_mask & (inferred_vals > recovery_threshold))
#             spatial_coverage = 100 * recovered_count / true_active_count if true_active_count > 0 else 0
            
#             print(f"\n--- {stripe_name.upper().replace('_', ' ')} ---")
#             print(f"  Nodes: {len(indices)}")
#             print(f"  Depth range: {coords[:, 2].min()/1000:.1f} to {coords[:, 2].max()/1000:.1f} km")
#             print(f"  Strike extent: {coords[:, 0].min()/1000:.1f} to {coords[:, 0].max()/1000:.1f} km")
#             print(f"  Dip extent: {coords[:, 1].min()/1000:.1f} to {coords[:, 1].max()/1000:.1f} km")
#             print(f"  True slip range: {true_vals.min():.6f} to {true_vals.max():.6f}")
#             print(f"  Inferred slip range: {inferred_vals.min():.6f} to {inferred_vals.max():.6f}")
#             print(f"  Correlation: {correlation:.3f}")
#             print(f"  Amplitude recovery: {amplitude_recovery:.1f}%")
#             print(f"  Spatial coverage: {spatial_coverage:.1f}%")
#         else:
#             print(f"\n--- {stripe_name.upper().replace('_', ' ')} ---")
#             print(f"  Too few nodes ({len(stripe_data)}) for analysis")

# # Run with your exact parameters:
# analyze_stripes_with_correct_coordinates(
#     mtrue, m, mesh, boundaries, fault,
#     x_len=80e3, y_len=300e3, dx=35e3, x0=0, y0=-35e3, rot_deg=0.0
# )

# %%
# # Usage example:
# print("Analyzing individual stripe recovery...")
# stripe_analysis = identify_and_assess_individual_stripes(
#     mtrue_s=mtrue,
#     minferred_s=m, 
#     mesh=mesh,
#     boundaries=boundaries,
#     fault_id=fault,
#     stripe_threshold=0.5,  # 30% of max amplitude to define active regions
#     min_stripe_size=15     # Minimum nodes per stripe
# )

# # Access individual stripe results
# for stripe_id, assessment in stripe_analysis['stripe_assessments'].items():
#     print(f"\n{stripe_id} detailed metrics:")
#     print(f"  Amplitude recovery: {assessment['amplitude_analysis']['amplitude_recovery_percentage']:.1f}%")
#     print(f"  Shape preservation: {assessment['pattern_integrity']['shape_correlation']:.3f}")
#     print(f"  Depth range: {assessment['spatial_analysis']['depth_range_km']:.1f} km")


