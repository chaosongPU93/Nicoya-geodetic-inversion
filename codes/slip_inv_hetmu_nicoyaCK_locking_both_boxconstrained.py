# %% [markdown]
# # Back-slip inversion with DIRECT BOX CONSTRAINTS (like MATLAB lsqlin)
# 
# This version implements direct box-constrained optimization to eliminate the 
# smoothing effects of tanh transformation and reproduce sharp slip patterns.
# 
# Key differences from the tanh version:
# * Uses scipy.optimize.minimize with L-BFGS-B method and explicit bounds
# * No tanh parameterization - slip parameters are physical slip directly
# * Can achieve exact boundary values (0, 0.0785 m)
# * Better reproduces localized, sharp slip distributions
# 
# Based on: slip_inv_hetmu_nicoyaCK_locking_both.py

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
import hippylib as hp
from pointwiseStateObs_weights import PointwiseStateObservation
from scipy.optimize import minimize
import time

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
# Import Box Constraint Utility
from slip_transformation_utils import BoxConstraintTransformation
print("BoxConstraintTransformation imported successfully!")

# Test the import
test_transformer = BoxConstraintTransformation(strike_bounds=(-0.027, 0.027), dip_bounds=(0.0, 0.0785))
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

# convert to relative locations in meters, and then rotate
rot = 45  # rotation angle in degrees, positive is CCW
x_rot, y_rot  = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)

# offset in x and y direction, the same as being done to the mesh in 'Kyriakopoulos2016JGR/convert_exodus_to_msh.ipynb'
x0, y0 = 130e3, 350e3  # offset for x and y coordinates, in m
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3   # offset to match the mesh coordinates
data['z'] = 0.0

# To aligned with mesh, must rotate data and error as well; no need for synthetics generated self-consistently
# Rotate displacement as well, data error would be dealt with later in defining weights
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot) 

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

# %%
# Define folder to save the results
resultpath = "/home/staff/chao/SSEinv/Nicoya/rst_locking/"
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
# in the order of: 'k_r' in blockright, 'k_l' in blockleft, 'k_a' in lowvelocitylayer
class K_LVL(dl.UserExpression):
    def __init__(self, subdomains, k_r, k_l, k_a, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_r = k_r
        self.k_l = k_l
        self.k_a = k_a

    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == blockright:
            values[0] = self.k_r 
        elif self.subdomains[cell.index] == blockleft:
            values[0] = self.k_l
        elif self.subdomains[cell.index] == lowvelocitylayer:
            values[0] = self.k_a
    
    def value_shape(self):
        return ()

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
meshname = "nicoyaCK2"   # same as above but 5-km mesh size on fault
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
# BOX CONSTRAINED PDE Variational Formulation (NO TANH TRANSFORMATION)
class PDEVarf_BoxConstrained:
    """
    Direct box-constrained PDE variational formulation
    
    Key difference: m parameters are PHYSICAL SLIP directly (no transformation)
    Bounds are enforced by the optimization solver, not by parameterization
    """
    def __init__(self, mtrue_mu_fun):
        self.mtrue_mu_fun = mtrue_mu_fun
        
    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)  # These are physical slip values directly
        tau, w, q = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        # NO TRANSFORMATION: m_strike and m_dip are physical slip
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
# Box-Constrained Inversion Function with scipy.optimize
def solveCoseismicInversion_BoxConstrained(k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2, \
                            box_transformer, savefiles=True, verbose=True):
    """
    Box-constrained slip inversion using scipy.optimize (like MATLAB lsqlin)
    
    Args:
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2: Same as original
        box_transformer: BoxConstraintTransformation instance defining bounds
        savefiles, verbose: Same as original
        
    Returns:
        Same outputs as original function, but slip values are directly bounded
    """

    # Define function spaces
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
        print(sep, "BOX-CONSTRAINED: Set up mesh and direct constraint framework", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print("Number of STATE and ADJOINT dofs: STRESS={0}, DISPLACEMENT={1}, ROTATION={2}".format(*ndofs_state))
        print("")
        print("BOX CONSTRAINT FRAMEWORK")
        print(f"   {box_transformer}")
        
        if box_transformer.has_any_bounds:
            print("   Parameter interpretation:")
            print("     m = physical_slip (DIRECT, no transformation)")
            print("     Bounds enforced by optimization solver")
        else:
            print("   Using unconstrained framework")

    # Define the STATE and ADJOINT Dirichlet BCs
    zero_tensor = dl.Expression(( ("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.") ), degree=0)
    bc = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # Interpolate starting guess
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()

    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # Assign the values of the vector
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()

    # Step 4: Convert to Function for later use
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    # Save true shear modulus structure
    if savefiles:
        print( "Saving true shear modulus structure to .xdmf file" )
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project( mtrue_mu_fun_expr, CG_mu )
        mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf')
        m_mu_true.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

    # Use box-constrained PDE variational formulation
    pde_varf = PDEVarf_BoxConstrained(mtrue_mu_fun)
    if verbose:
        print("Using box-constrained PDE (no slip transformation)")

    # Define the PDE problem
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    # Define the solver type
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    # Print the number of observations
    if verbose:
        print( "Number of observation points: {0}".format(targets.shape[0]) )
    
    # Constrain only the displacement field for the data misfit
    indicator_vec = dl.interpolate( dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE] ).vector()

    # Create the weight vector
    weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)

    # Use the standard deviation of observational errors to construct the weights
    obs_weights = np.zeros(targets.shape[0]*15,)
    
    # Considering the coordinate rotation, location error changes, so are the weights 
    weight_x, weight_y = ut.compute_rotated_weights((data['vx_std_Car']**2).to_numpy(), \
                                                 (data['vy_std_Car']**2).to_numpy(), rot)
    # Apply the computed weights
    obs_weights[9::15] = weight_x / (f_h**2)   # x displacement weights
    obs_weights[10::15] = weight_y / (f_h**2)  # y displacement weights
    obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements
    # Modify and set the array to dolfin vector
    weights.set_local(obs_weights)
    weights.apply('')

    # Define the misfit function
    misfit = PointwiseStateObservation( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
    # Impose misift.noise_variance = 1 since we modified the single data noise variables
    misfit.noise_variance = 1.

    # Generate observations by using the observation operator 'B'
    if len(misfit.weight) % 15 != 0:
        print("Mismatch data with targets sizes. Check that targets are in the domain.")

    # Infer the index position of all non-zero entries in the misfit,  
    idx_d = list(np.nonzero(misfit.weight)[0]) # misfit = 2*ntargets (2 displacement components, since uz=0)
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error. The length of non-zero misfit has to be the same as ntargets.")

    # Imput GPS data into misift.d.
    # Replace the synthetic data with the recorded GPS data
    tmp = np.zeros(len(misfit.d),)
    # Horizontal and vertical displacement misfit
    tmp[9::15] = np.array(data['vx_Car'])    # horizontal east displacement misfit
    tmp[10::15] = np.array(data['vy_Car'])   # horizontal north displacement misfit
    tmp[11::15] = np.array(data['vz_Car'])   # vertical displacement misfit
    # Assign the values of the new vector to misfit.d
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    # Extract horizontals and vertical displacements observed data
    d_obs = misfit.d[idx_d]
    if savefiles:
        # Save the observed surface displacement
        outFileName = 'd_obs_' + meshname + inv_str + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]) )
        csvoutput.close()

    # Extract x,y coordinates of the fault and values
    CG_s = dl.VectorFunctionSpace(mesh, "CG", 1)    # if not specified, the 'dim' will be 3, same as the mesh.dim
    bc1 = dl.DirichletBC(CG_s, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG_s)
    bc1.apply(um.vector())
    # Same for fault, but in this case use strike and dip components, assuming no fault-normal slip
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm)
    bc2.apply(um2.vector())
    # Extract x,y,z coordinates of the fault and values
    xslip = dl.interpolate(dl.Expression( ("x[0]", "x[0]", "x[0]"), degree=5), CG_s )
    yslip = dl.interpolate(dl.Expression( ("x[1]", "x[1]", "x[1]"), degree=5), CG_s )
    zslip = dl.interpolate(dl.Expression( ("x[2]", "x[2]", "x[2]"), degree=5), CG_s )
    xf = xslip.vector()[um.vector() == 10] # x coordinate fault
    yf = yslip.vector()[um.vector() == 10] # y coordinate fault
    zf = zslip.vector()[um.vector() == 10] # z coordinate fault
    if verbose:
        print( sep, "Done extracting the fault coordinates", sep )

    # Define the regularization
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )

    ### CONSTRUCT MODEL (LAGRANGIAN FORMALISM) ###
    model = hp.Model(pde, reg, misfit)
    
    # Generate initial vectors
    m = m0_s.copy()
    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    
    # Get fault interface mask for bounds
    fault_mask = um2.vector() == 99
    
    # Create bounds using box transformer
    if box_transformer.has_any_bounds:
        bounds = box_transformer.create_scipy_bounds(m, fault_mask)
        if verbose:
            print(f"Created {len(bounds)} parameter bounds")
            n_constrained = sum(1 for b in bounds if b != (None, None))
            print(f"  {n_constrained} parameters are constrained")
            print(f"  {len(bounds) - n_constrained} parameters are unconstrained")
    else:
        bounds = None
        if verbose:
            print("No bounds applied (unconstrained optimization)")
    
    # Define objective function for scipy.optimize
    def objective_and_gradient(m_array):
        """Compute objective function and gradient for scipy.optimize"""
        # Update parameter vector
        m.set_local(m_array)
        m.apply('')
        
        # Update state variables
        x[hp.PARAMETER] = m
        
        # Solve forward and adjoint problems
        model.solveFwd(u, x)
        model.solveAdj(p, x)
        
        # Compute cost
        cost = model.cost(x)[0]  # total cost
        
        # Compute gradient
        mg = model.generate_vector(hp.PARAMETER)
        model.evalGradientParameter(x, mg)
        gradient = mg.get_local()
        
        return cost, gradient
    
    if verbose:
        print(sep, "Solve the slip inverse problem with box constraints", sep)
        print("Using scipy.optimize.minimize with L-BFGS-B method")
        print("(Direct constraint enforcement like MATLAB lsqlin)")
    
    # BOX-CONSTRAINED OPTIMIZATION with scipy
    start_time = time.time()
    
    result = minimize(
        fun=objective_and_gradient,
        x0=m.get_local(),
        method='L-BFGS-B',
        jac=True,  # We provide gradient
        bounds=bounds,
        options={
            'disp': verbose,
            'maxiter': 1000,
            'ftol': 1e-12,
            'gtol': 1e-9
        }
    )
    
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
    
    # Update solution
    m.set_local(result.x)
    m.apply('')
    x[hp.PARAMETER] = m
    
    # Validate bounds
    if box_transformer.has_any_bounds:
        if verbose:
            print("\nValidating final solution bounds...")
        box_transformer.validate_bounds(m.get_local(), fault_mask, verbose=verbose)
    
    # Create slip functions (m is already physical slip - no transformation needed)
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
    s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)
    
    # Save inversion results
    if savefiles:
        print( "Saving slip solution to .xdmf file" )
        s_id = dl.XDMFFile(resultpath + 'slip_' + meshname + inv_str + mu_str_inv + '.xdmf')
        m_fun.rename('coseismic slip', 'coseismic slip')
        s_id.write(m_fun)
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + inv_str + mu_str_inv + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + inv_str + mu_str_inv + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print( "Finish saving slip solution" )

    # Solve the forward problem to compute the calculated STATE variables
    model.solveFwd(u, x)
    # Use the observational operator to extract the surface displacement: d_cal = Bu
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    if savefiles:
        # Save the predicted surface displacement
        outFileName = 'd_cal_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(targets[i,0], targets[i,1], targets[i,2], d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]) )
        csvoutput.close()

    # Calculate the norm of the gradient of the solution for L-curve criterion
    m_fun = dl.Function(Vh[hp.PARAMETER], m)
    grad_m = dl.assemble( ufl.inner( ufl.avg(ufl.nabla_grad(m_fun)), ufl.avg(ufl.nabla_grad(m_fun)) )*dS(fault) )
    misfitd = np.linalg.norm((d_cal-d_obs), 2)
    print( "Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m) )

    # Compute the cost functional to plot misfit
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print( "Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(total_cost, reg_cost, misfit_cost) )

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2) (m is already physical slip)
    s_mag = ufl.sqrt(ufl.dot(m_fun, m_fun))

    # Calculate seismic moment
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    if savefiles:
        # Save true moment to file  
        outFileName = 'moment_' + meshname + inv_str + mu_str_inv + '.txt'
        with open(resultpath + outFileName, 'w') as moment_file:
            moment_file.write(f"{moment:.6e} {M_w1:.4f} {M_w2:.4f} {M_w3:.4f}\n")

    if savefiles:
        # Save predicted displacement field
        print( "Saving predicted displacement and stress to .xdmf file" )
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)
        # Stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + inv_str + mu_str_inv + '.xdmf')
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        print( "Finish saving predicted displacement and stress" )

        # Save output fault geometry and slip
        outFileName = 'fault_geometry_' + meshname + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, xf.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f\n" %(xf[i], yf[i], zf[i]) )
        csvoutput.close()

    # Extract values from the model PARAMETER at the fault interface (m is physical slip directly)
    m_s_fault = m.get_local()[fault_mask]
    m_sx_fault = m_s_fault[0::2]
    m_sy_fault = m_s_fault[1::2]
    
    print("Physical slip ranges:")
    print(f"  Strike: [{min(m_sx_fault):.6f}, {max(m_sx_fault):.6f}] m")
    print(f"  Dip: [{min(m_sy_fault):.6f}, {max(m_sy_fault):.6f}] m")

    if savefiles:
        # Save inferred slip at the fault interface
        outFileName = 'm_s_fault_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write( "%.6f %.6f\n" %(m_sx_fault[i], m_sy_fault[i]) )
        csvoutput.close()

        # Save inferred slip, over the entire volume
        outFileName = 'slip_inferred_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(m0_s))
        slip_vec[fault_mask] = m_s_fault
        for i in range(0, len(slip_vec)):
            csvoutput.write("%.6f\n" %slip_vec[i])
        csvoutput.close()

    return mtrue_mu, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, np.concatenate([m_sx_fault, m_sy_fault]), misfitd, grad_m 

# %% [markdown]
# ## DEFINE COMMON PARAMETERS

# %%
# Define order of elements 
k = 2
# Define body force
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9

# Define starting model
m0_s_expr = dl.Constant((0., 0.))

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
nu = 0.25   # shear modulus found in Kano et al., 2019

# background shear modulus
mu_b = 0   # 40 GPa
mu_background = mu_expression(mu_b)

# shear modulus for the lower (subducting) plate
mu_l = 0.9730 # ~55 GPa
mu_lower = mu_expression(mu_l)

# shear modulus for the upper (overriding) plate
mu_u = -0.9730  # ~25 GPa
mu_upper = mu_expression(mu_u)

mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"

print( "Heterogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )

# %%
# locations of surface observations
ntargets = data.shape[0]
targets_x = np.array(data['x'])*1e3   # km to m
targets_y = np.array(data['y'])*1e3
targets_z = np.array(data['z'])*1e3
targets = np.zeros([ntargets, dim])
targets[:,0] = targets_x; targets[:,1] = targets_y; targets[:,2] = targets_z

# %%
# Decide the weights of the horizontal, vertical components
f_h, f_v = 1, 1/2
# Print the weights of the data
print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )

obs_weights = np.zeros(targets.shape[0]*15,)

# %%
# BOX CONSTRAINT SETUP
print(sep + "BOX CONSTRAINT SETUP" + sep)

BOUNDED = True
BOUND_TYPE = 'both'

if BOUNDED:
    # Define slip bounds based on your problem (using same values as quote)
    s_strike_max = 0.027  # 27 mm max along-strike slip
    s_dip_max = 0.0785    # 78.5 mm max dip slip (as in quote)
    
    if BOUND_TYPE == 'both':
        box_transformer = BoxConstraintTransformation(
            strike_bounds=(0.0, s_strike_max),  # Positivity constraints
            dip_bounds=(0.0, s_dip_max),        # Positivity constraints + max value
        )
        print("Constraints to both strike and dip (positivity + max values)")

    elif BOUND_TYPE == 'strike':
        box_transformer = BoxConstraintTransformation(
            strike_bounds=(0.0, s_strike_max),
            dip_bounds=None,
        )
        print("Constraints to strike only")

    elif BOUND_TYPE == 'dip':
        box_transformer = BoxConstraintTransformation(
            strike_bounds=None,
            dip_bounds=(0.0, s_dip_max),
        )
        print("Constraints to dip only")

else:            
    # Alternative: no constraints
    box_transformer = BoxConstraintTransformation(strike_bounds=None, dip_bounds=None)
    print("Unconstrained mode")

print(f"Configuration: {box_transformer}")

# %%
# Define regularization weights
rho_s = 1e9   # allows variations of slip of the order of ~3 km, close to the maximum resolution
gamma_val_H1 = 1e3  
delta_val_L2 = gamma_val_H1 / rho_s  

# %%
# Take the inverse for saving the name of the weights
w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
if BOUNDED:
    inv_str = f"_lockbothboxbd_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
else:
    inv_str = f"_lockboth_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"

# %%
### Compute L-curve criterion ###
# Solve to find the best Tikhonov regularization parameter by using the L-curve criterion

# fix the rho, ratio of gamma and delta
rho_s = 1e9

# vary gamma, the model gradient damping  
gammas_s = [5e1]

print( sep, "L-curve computation with box constraints", sep )

dmisfit = []
mmisfit = []
for gamma_s in gammas_s:
    delta_s = gamma_s / rho_s 

    # file identifier
    if BOUNDED:
        inv_str = f"_lockbothboxbd_w{w_h}{w_v}_gs{gamma_s:.0e}_ds{delta_s:.0e}"
    else:
        inv_str = f"_lockboth_w{w_h}{w_v}_gs{gamma_s:.0e}_ds{delta_s:.0e}"
    print("Inverse problem identifier: ", inv_str)

    print(f"****** Computing solution with gamma_s = {gamma_s:.1e}, "
        f"rho_s = {rho_s:.1e}, and delta_s = {delta_s:.1e} ******")

    # Solve slip inverse problem within the heterogeneous shear modulus half-space WITH BOX CONSTRAINTS
    mtrue_mu_expr_inv = mtrue_mu_expr_het
    mu_str_inv = mu_str_het
    print("Solving inverse problem based on: ", mu_str_inv)
    results = solveCoseismicInversion_BoxConstrained(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
        box_transformer,  # KEY: Pass box constraint transformation
        savefiles=True, verbose=True
    )
    print("Het Inversion with box constraints finished!!!")

print("All Inversion with Box Constraints Finished!")   