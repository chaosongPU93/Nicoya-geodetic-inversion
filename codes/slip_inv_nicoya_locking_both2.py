# %% [markdown]
# # Back-slip inversion of the GNSS displacement measurements of interseismic locking (coupling) at Nicoya (Costa Rica) within a homogeneous half-space
# 
# * Data (station location info and disp. observations) is stored under "Nicoya/data/Feng_etal_JGR_2012table1.csv"
# * Plate model comes from *Slab2.0*, and trench is from PB2003; they are processed by "Nicoya/plateinterface/plt_slab.ipynb"
# * The processed plate and trench are then compiled by "Nicoya/plateinterface/build_gmsh_geo.ipynb" to generate a *Gmsh* .geo file. Further mannual operations were used to finalized the geometry.
# * The resulted "nicoya.geo" is used to generate the mesh file by *Gmsh* using ``gmsh -3 nicoya.geo -format msh2 -optimize_netgen -smooth 3`` 
# * The resulted mesh file 'nicoya.msh' is then converted to *FEniCS* format inside the conda environment using ``dolfin-convert nicoya.msh nicoya.xml``. The commands generates the following files under "Nicoya/mesh/":
# 
#     * nicoya.xml: Contains the mesh data.
#     * nicoya_facet_region.xml: Contains boundary tags.
#     * nicoya_physical_region.xml: Contains subdomain tags.    
# 
# * Upon having all of the above, now we are ready to carry out the inversion using *FEniCS* and *hIPPYlib*    
# 
# 
# * The bulk part of this code is similar to "Kano_etal2019SciRep/slip_inv_3D_GNSS_Kano2019.ipynb", but we need to modify the code so that it will take in the vertical component as well as the horizontal ones. 
# 
# 
# * This version of the code would take in both the trench-normal and trench-parallel velocities, and invert for slip rate in terms of 2 model parameters "s_strike", "s_dip" as normally done 
# 
# * However, the interseismic coupling ratio is still computed as s_dip/V_norm, where s_dip is back normal slip rate, V_norm is the trench-normal plate convergence rate.

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
from pointwiseStateObs_weights import PointwiseStateObservation
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

# define the centroid of relative coordinates
lon0, lat0 = -85.5+360, 10

# convert to relative locations in km
data['x'], data['y'] = ut.azi_equidist_proj(data['lon'], data['lat'], lon0, lat0)
data['z'] = 0.0

data.head()
print("Number of stations:", len(data))

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
# Choose the mesh
meshname = "nicoya"
# meshname = "nicoya2"   # This has a smaller fault interface
print(meshname)

# Choose path data
meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
# load mesh
mesh = dl.Mesh(meshpath + meshname + '.xml')
# mesh dimensions
xmin, xmax = -1000e3, 1000e3
ymin, ymax = -1000e3, 1000e3
zmin, zmax = -200e3, 0.
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
# The linear coseismic inversion problem
# Define the weak formulation of the FORWARD problem
def pde_varf(u, m, p):
    # Split the STATE and ADJOINT variables. Use dl.split() and not
    # .split(deepcopy=True), since the latter breaks FEniCS symbolic differentiation
    sigma, uu, r = dl.split(u)
    # Split the slip to be strike/dip components.
    m_strike, m_dip = dl.split(m)
    tau, w, q = dl.split(p)
    u0 = dl.Constant((0., 0., 0.))

    # Define the weak formulation of the forward problem, note the change to the last term
    J = ufl.inner( AEsigma(sigma, mu, nu), tau )*ufl.dx \
        + ufl.inner( ufl.div(tau), uu )*ufl.dx \
        + ufl.inner( asym(tau), r )*ufl.dx \
        + ufl.inner( ufl.div(sigma), w )*ufl.dx + ufl.inner( asym(sigma), q )*ufl.dx \
        + ufl.inner( f, w )*dl.dx \
        - ufl.inner( u0, tau*n )*ds(bottom) \
        - ufl.inner( dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip), tau('+')*n('+') )*dS(fault)
    return J

# %%
# Create a routine that solves the slip inversion problem, compare with 'CoseismicSlip_DeterministicInversion.ipynb'
def solveCoseismicInversion(k, targets, m0_s_expr, gamma_val_H1, delta_val_L2, savefiles=True, verbose=True):

    # Define function spaces
    # Use VectorFunctionSpace if the unknown is a vector field.
    # Use FunctionSpace object for scalar fields.
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)    # stress (tensor field, since BDM is a vector field)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)   # displacement (vector field)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)   # rotation (scalar field)
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
    # Interpolate starting guess
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()

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

    # Create the weight vector to impose different 1 / standard deviation of observation error for
    # horizontal components, which are at indices 9*, 10*, and 11* 
    weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)

    # Use the standard deviation of observational errors to construct the weights
    obs_weights = np.zeros(targets.shape[0]*15,)
    obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
    obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
    obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements

    # obs_weights[9::15]  = 1/(noise_std_h*noise_std_h) * 1/(f_h**2)  # horizontal east displacement
    # obs_weights[10::15] = 1/(noise_std_h*noise_std_h) * 1/(f_h**2)  # horizontal north displacement
    # obs_weights[11::15] = 1/(noise_std_v*noise_std_v) * 1/(f_v**2)  # vertical displacements

    # Modify and set the array to dolfin vector
    weights.set_local(obs_weights)
    weights.apply('')
    misfit = PointwiseStateObservation( Vh[hp.STATE], targets, weight=weights, indicator_vec=indicator_vec )
    # Impose misift.noise_variance = 1 since we modified the single data noise variables
    misfit.noise_variance = 1.
    # Define the regularization, NOT total variation any more
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )

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

    ### CONSTRUCT MODEL (LAGRANGIAN FORMALISM) ###
    # Construct the "Model" --> objective function
    model = hp.Model(pde, reg, misfit)
    ### CHECK the Gradient and Hessian with FINITE DIFFERENCE (FD) ###
    m = m0_s.copy()
    
    if verbose:
        print( sep, "Solve the deterministic coseismic slip inverse problem", sep )
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
        raise

    # Solve the FORWARD problem to compute the "predicted data" with the inverted
    # model parameter 'm' (= slip along the fault).
    # Set the solution m = m0 + \{hat}m
    m.axpy(1., m_hat) # m = m + 1*\{hat}m

    # Extract inverse solutions
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
    s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)
    if savefiles:
        # Save inversion results (coseismic slip)
        print( "Saving slip solution to .xdmf file" )
        s_id = dl.XDMFFile(resultpath + 'slip_' + meshname + inv_str + '.xdmf')
        m_fun.rename('coseismic slip', 'coseismic slip')
        s_id.write(m_fun)
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + inv_str + '.xdmf')
        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id.write(s_strike_fun)
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + inv_str + '.xdmf')
        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id.write(s_dip_fun)
        print( "Finish saving slip solution" )

    # Solve the forward problem to compute the calculated STATE variables
    # Generate STATE and ADJOINT vectors
    #u = model.generate_vector(hp.STATE)
    #p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    # Solve the forward problem to compute the calculated STATE variables
    model.solveFwd(u, x)
    # Use the observational operator to extract the surface displacement: d_cal = Bu
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    if savefiles:
        # Save the predicted surface displacement
        outFileName = 'd_cal_' + meshname + inv_str + '.txt'
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

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2)
    s_mag = ufl.sqrt( ufl.dot(m_fun, m_fun))
    # calculate the total moment on the fault 
    moment = dl.assemble( mu * GPa2Pa * s_mag * dS(fault) )        
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")

    if savefiles:
        # Save perdicted displacement field
        print( "Saving predicted displacement and stress to .xdmf file" )
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + inv_str + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)
        # Stress
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + inv_str + '.xdmf')
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

    # Extract values from the model PARAMETER at the fault interface
    m_s_fault = m[um2.vector() == 99]
    m_sx_fault = m_s_fault[0::2]; m_sy_fault = m_s_fault[1::2]
    print( min(m_sx_fault), max(m_sx_fault) )
    print( min(m_sy_fault), max(m_sy_fault) )
    if savefiles:
        outFileName = 'm_s_fault_' + meshname + inv_str + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(0, m_sx_fault.shape[0]):
            csvoutput.write( "%.6f %.6f\n" %(m_sx_fault[i], m_sy_fault[i]) )
        csvoutput.close()

        # Save inferred slip, whole volume
        outFileName = 'slip_inferred_' + meshname + inv_str + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        slip_vec = np.zeros(len(m0_s))
        slip_vec[um2.vector() == 99] = m_s_fault
        for i in range(0, len(slip_vec)):
            csvoutput.write( "%.6f\n" %slip_vec[i] )
        csvoutput.close()

    return xf, yf, zf, m, u, m_fun, d_obs, d_cal, m_s_fault, misfitd, grad_m    

# %% [markdown]
# ## DEFINE COMMON PARAMETERS
# 

# %%
# Define order of elements 
k = 2
# Define body force
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9

# %%
# Define the true model PARAMETERS
nu = 0.25   # Poisson's ratio found in Kano et al., 2019
mu = 40e9 / GPa2Pa    # shear modulus (rigidity) found in Kano et al., 2019 
E = 2*mu*(1+nu) / GPa2Pa # Lame parameter
# Define starting model
m0_s_expr = dl.Constant((0., 0.))

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

# noise_std_h = 2e-3      # assuming a 2 mm noise in horizontal displacements
# noise_std_v = 1e-2      # 5 times larger noise in vertical displacements, 10 mm
# noise_std_h = 1e-3      # assuming a 1 mm noise in horizontal displacements
# noise_std_v = 3e-3      # 3 times larger noise in vertical displacements, 3 mm

# obs_weights = np.zeros(targets.shape[0]*15,)
# obs_weights[9::15]  = 1/(noise_std_h*noise_std_h) * 1/(f_h**2)  # horizontal east displacement
# obs_weights[10::15] = 1/(noise_std_h*noise_std_h) * 1/(f_h**2)  # horizontal north displacement
# obs_weights[11::15] = 1/(noise_std_v*noise_std_v) * 1/(f_v**2)  # vertical displacements

obs_weights = np.zeros(targets.shape[0]*15,)
obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements

# %%
# Define regularization weights
# In a Bayesian inference setting, the ratio \rho = \sqrt(\gamma/\delta) plays the role of the correlation length in the prior term.
# For our case, the station separation is around 20 km, and the mesh size on the fault is 4-20 km 
# rho_s = 1e9   # allows variations of slip of the order of ~30 km 

rho_s = 1e9   # allows variations of slip of the order of ~3 km, close to the maximum resolution
gamma_val_H1 = 1e3
# gamma_val_H1 = 1e4  
# gamma_val_H1 = 1e2 
# gamma_val_H1 = 4e2   
delta_val_L2 = gamma_val_H1 / rho_s  

# %%
# Take the inverse for saving the name of the weights
w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
inv_str = f"_lockingboth_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"


# %%
# Solve coseismic slip inverse problem
solveCoseismicInversion(k, targets, m0_s_expr, gamma_val_H1, delta_val_L2, savefiles=True, verbose=True)

print("Finished!!!")

# %%
# ### Compute L-curve criterion ###
# # Solve to find the best regularization parameter by using the L-curve criterion
# # rho_s = rho_s
# gammas = np.logspace(-2,5,8) 
# dmisfit = []
# mmisfit = []

# print( sep, "L-curve computation", sep )

# for gamma in gammas:
#     delta = gamma / rho_s
#     print('****** Computing solution with gamma = ', gamma, 'rho = ', rho_s, 'and delta = ', delta, '******')
#     _, _, _, _, _, _, _, _, _, misfit_d, misfit_m = solveCoseismicInversion(k, targets, m0_s_expr, gamma, delta, savefiles=False, verbose=True)
#     dmisfit.append(misfit_d)
#     mmisfit.append(misfit_m)

# # save the data misfit and model misfit, in order for later plotting of L-curve to find the best weights
# outFileName = 'Lcurve' + meshname + '.txt'
# csvoutput = open(resultpath + outFileName, 'w+')
# for i in range(0, len(dmisfit)):
#     csvoutput.write( "%.6f %.6f\n" %(dmisfit[i], mmisfit[i]) )
# csvoutput.close()

# print("Finished! Done saving data and model misfits for L-curve.")    


