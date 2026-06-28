# %% [markdown]
# # Back-slip inversion of the GNSS displacement measurements of interseismic locking (coupling) at Nicoya (Costa Rica) within a prescribed heterogeneous half-space
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
# 
# * The difference from "Nicoya/codes/slip_inv_nicoya_locking_both.ipynb" is that the elastic structure of the half-space can be customized and then be injected into the FE space. Again, if the structure around the source region is indeed heterogeneous, then the inversion under a prescribed, reasonable structure should better fit the observation than the homogeneous case. 
# 
# * Meanwhile, it is expected that the homogeneous solution from "slip_inv_nicoya_locking_both.ipynb" should be nearly identical to that of this code when setting the structure flag to homogeneous.
# 
# * Use the slab geometry from slab2.0, and use the 3D shear modulus model from DeShon et al. (2006)
# 

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
from scipy.interpolate import griddata, interp1d
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
# print(data.tail())

# define the centroid of relative coordinates
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
# read in the 3D velocity model
veldir = "/home/staff/chao/SSEinv/Nicoya/DeShon_2006GJI/"
vel3dfile = "DeShon2006_3Dmodel.csv"
vel3d = pd.read_csv(veldir + vel3dfile, sep=",")
# convert to relative locations in km in my own coordinate system
vel3d['x'], vel3d['y'] = ut.azi_equidist_proj(vel3d['lon'], vel3d['lat'], lon0, lat0)
cols_to_convert = ['x', 'y', 'z']
vel3d[cols_to_convert] = vel3d[cols_to_convert] * 1e3  # Convert km to m
vel3d['z'] = vel3d['z'] * -1  # negative depth means downward
vel3d = vel3d[(vel3d['z'] <= 0)].reset_index(drop=True)  # ignore everything above the ground
# print(vel3d.shape)
# print(vel3d.head())

# read in the reference 1D velocity model with same depth layers
vel1dfile = "DeShon2006_1Dmodel.csv"
vel1d = pd.read_csv(veldir + vel1dfile, sep=r'\s+',skiprows=1, \
                 names=['z', 'vp', 'vs', 'vp_vs_ratio'])
vel1d['z'] = vel1d['z'] * -1 * 1e3 # negative depth means downward, Convert km to m
vel1d = vel1d[(vel1d['z'] <= 0)].reset_index(drop=True)  # ignore everything above the ground
# print(vel1d)

# read a made-up 1D velocity model of density, ref. DeShon2004_1Dmodel, but with same depth layers as 3d models
den1dfile = "Density_1Dmodel.csv"
den1d = pd.read_csv(veldir + den1dfile, sep=r'\s+',skiprows=1, \
                 names=['z', 'den'])
den1d['z'] = den1d['z'] * -1 * 1e3 # negative depth means downward, Convert km to m
den1d = den1d[(den1d['z'] <= 0)].reset_index(drop=True)  # ignore everything above the ground
den1d['den'] = den1d['den'] * 1e3 # Convert g/cm^3 to kg/m^3
# print(den1d.head())
# print(den1d)

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
# meshname = "nicoya"
# meshname = "nicoya2"   # This has a smaller fault interface
# meshname = "nicoya3"   # the same as above but 5-km mesh size on fault
meshname = "nicoya4"   # extended fault area
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
        # mu = mu_expression(self.mtrue_mu_fun)
        mu = self.mtrue_mu_fun  # no need to convert, already as real modulus in GPa

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

        # mu = mu_expression(self.mtrue_mu_fun)
        mu = self.mtrue_mu_fun  # no need to convert, already as real modulus in GPa

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
# Create a routine that solves the joint deterministic adjoint-based inversion
# 🎯 ENHANCED: Inversion routine with Tanh Transformation
def solveCoseismicInversion_TanhSlip(k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2, \
                                     slip_transformer, savefiles=True, verbose=True):

    """
    Enhanced slip inversion with tanh transformation for bounded slip
    
    Args:
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2: Same as original
        slip_transformer: SlipTransformation instance defining bounds
        savefiles, verbose: Same as original
        
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
            transformer, savefiles=True, verbose=True
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

    # Interpolate starting guess
    m0_s = dl.interpolate(m0_s_expr, Vh[hp.PARAMETER]).vector()

    # shear modulus
    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    # If mtrue_mu_expr_inv is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_inv == None:
        vs_func, den_func, mu_func, _ = ut.process_velocity_models(vel3d, vel1d, den1d, mesh, verbose=False)
        # mtrue_mu_fun = mu_func  # Contains shear modulus in GPa
        # contrast_factor = 4.0
        # mtrue_mu_fun.vector()[:] *= contrast_factor

        # Amplify the contrast wrt a reference value
        # mu_ref = mu_expression(0)   # which gives 40 GPa
        mu_ref = np.mean(mu_func.vector()[:])
        # contrast_factor = 4.0
        min_mu = 5.0  # Minimum physical value (GPa)
        mtrue_mu_fun = dl.Function(CG_mu)
        new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
        mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)

    else:
        # Assign the values of the vector
        mtrue_mu_expr = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()

        # for i in range(volc.shape[0]):
        #     volc_x = volc['x'].iloc[i]
        #     volc_y = volc['y'].iloc[i]
        #     volc_z = -30e3  # center depth in meters
        #     radius = 30e3

        #     # Conditional expression: assign mu_v inside the sphere, 0 outside
        #     mtrue_mu_volc_expr = dl.Expression(
        #         'pow(x[0] - x0, 2) + pow(x[1] - y0, 2) + pow(x[2] - z0, 2) < pow(r, 2) ? 1.0 : 0.0',
        #         x0=volc_x, y0=volc_y, z0=volc_z, r=radius,
        #         degree=2
        #     )

        #     mtrue_mu_volc = dl.interpolate(mtrue_mu_volc_expr, CG_mu)
        #     idx_volc = np.where(mtrue_mu_volc.vector().get_local() > 0.5)[0]

        #     # Overwrite values with mu_v inside the volcano sphere
        #     mtrue_mu_expr[idx_volc] = mu_v

        # Step 4: Convert to Function for later use
        mtrue_mu_expr_fun = hp.vector2Function(mtrue_mu_expr, CG_mu)
        mtrue_mu_fun = mu_expression(mtrue_mu_expr_fun)


    # Save true shear modulus structure
    if savefiles:
        print( "Saving true shear modulus structure to .xdmf file" )
        m_mu_true = dl.project( mtrue_mu_fun, CG_mu )   #didn't break anything, but redundant, already in CG_mu space  
        mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf')
        m_mu_true.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

        # Save true Vs and density structures if using 3D & 1D velocity models
        if mtrue_mu_expr_inv == None:
            print( "Saving true Vs structure to .xdmf file" )
            vs_true = dl.project( vs_func, CG_mu )  #didn't break anything, but redundant, already in CG_mu space  
            vs_id = dl.XDMFFile(resultpath + 'vs_true_' + meshname + mu_str_inv + '.xdmf')
            vs_true.rename('shear velocity Vs', 'shear velocity Vs')
            vs_id.write(vs_true)
            print( vs_true.vector().min(), vs_true.vector().max() )
            
            print( "Saving true density structure to .xdmf file" )
            den_true = dl.project( den_func, CG_mu )    #didn't break anything, but redundant, already in CG_mu space  
            den_id = dl.XDMFFile(resultpath + 'pho_true_' + meshname + mu_str_inv + '.xdmf')
            den_true.rename('density', 'density')
            den_id.write(den_true)
            print( den_true.vector().min(), den_true.vector().max() )

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

    # Create the weight vector to impose different 1 / standard deviation of observation error for
    # horizontal components, which are at indices 9*, 10*, and 11* 
    weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)

    # Use the standard deviation of observational errors to construct the weights
    obs_weights = np.zeros(targets.shape[0]*15,)
    obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
    obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
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

    # Define the regularization,
    # Below was used in the joint inversion
    # reg = hp.SqrtPrecisionPDE_Prior( Vh[hp.PARAMETER], reg_handler )
    # Below was used in the pure slip inversion 
    reg = hp.BiLaplacianPrior( Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False )

    ### CONSTRUCT MODEL (LAGRANGIAN FORMALISM) ###
    # Construct the "Model" --> objective function
    model = hp.Model(pde, reg, misfit)
    ### CHECK the Gradient and Hessian with FINITE DIFFERENCE (FD) ###
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
        
        # Validate bounds (using fault interface mask for consistency)
        if verbose:
            # # use 'fault_mask' for the fault interface only
            # fault_mask = um2.vector() == 99
            # slip_transformer.validate_bounds(physical_slip_fun, fault_mask=fault_mask, verbose=True)
            
            # OR, validate the whole mesh volume
            slip_transformer.validate_bounds(physical_slip_fun, verbose=True)

    else:
        # No transformation needed
        m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])
        s_strike_fun, s_dip_fun = m_fun.split(deepcopy=True)

    # Save slip solution (identical to original)
    if savefiles:
        # Save inversion results (coseismic slip)
        print( "Saving slip solution to .xdmf file" )
        s_id = dl.XDMFFile(resultpath + 'slip_' + meshname + inv_str + mu_str_inv + '.xdmf')
        if slip_transformer.has_any_bounds:
            physical_slip_fun.rename('coseismic slip', 'coseismic slip')
            s_id.write(physical_slip_fun)
        else:
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

    # Compute slip magnitude ||D|| = sqrt(D1^2 + D2^2)
    if slip_transformer.has_any_bounds:
        s_mag = ufl.sqrt(ufl.dot(physical_slip_expr, physical_slip_expr))
    else:
        s_mag = ufl.sqrt(ufl.dot(m_fun, m_fun))

    # Calculate seismic moment (using physical slip if bounds applied)
    m_mu_true = dl.project(mtrue_mu_fun, CG_mu) #didn't break anything, but redundant, already in CG_mu space  
    moment = dl.assemble(m_mu_true * GPa2Pa * s_mag * dS(fault))        
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    # calculate seismic potency, independent of the assumed elastic properties
    potency = dl.assemble(s_mag * dS(fault))
    print(f"Seismic potency: {potency:.3e} m^3")
    if savefiles:
        # Save true moment to file  
        outFileName = 'moment_' + meshname + inv_str + mu_str_inv + '.txt'
        with open(resultpath + outFileName, 'w') as moment_file:
            moment_file.write(f"{moment:.6e} {M_w3:.4f} {potency:.6e}\n")

    if savefiles:
        # Save perdicted displacement field
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
        if slip_transformer.has_any_bounds:
            slip_vec[um2.vector() == 99] = physical_slip_fault
        else:
            slip_vec[um2.vector() == 99] = m[um2.vector() == 99]
        for i in range(0, len(slip_vec)):
            csvoutput.write("%.6f\n" %slip_vec[i])
        csvoutput.close()
    
    return mtrue_mu_fun, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, np.concatenate([m_sx_fault, m_sy_fault]), misfitd, grad_m 

# %% [markdown]
# ## DEFINE COMMON PARAMETERS
# 

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
# # Define the true model PARAMETERS for INVERSE problem
# nu = 0.25   # shear modulus found in Kano et al., 2019
# # mu = 40e9 / GPa2Pa    # shear modulus (rigidity) found in Kano et al., 2019 
# # E = 2*mu*(1+nu) / GPa2Pa # Lame parameter

# # background shear modulus
# mu_b = 0   # 40 GPa
# mu_background = mu_expression(mu_b)

# # shear modulus for the lower (subducting) plate
# mu_l = 0.9730 # ~55 GPa
# mu_lower = mu_expression(mu_l)

# # shear modulus for the upper (overriding) plate
# # mu_u = -0.9730  # ~25 GPa
# mu_u = mu_b
# mu_upper = mu_expression(mu_u)

# # shear modulus for volcanoes
# mu_v = -0.9730  # ~25 GPa
# mu_volcano = mu_expression(mu_v) 

# mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
# # mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"
# mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}v{round(mu_expression(mu_v))}"

# print( "Heterogeneous structure:")
# # print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )
# print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f and volcano mu = %.1f" %(mu_upper, mu_lower, mu_volcano) )


# %%
# Define the true model PARAMETERS for INVERSE problem
# use the 3D & 1D velocity models to compute shear modulus, call 'process_velocity_models' inside solvers 
# vs_func, den_func, mu_func, _ = process_velocity_models(vel3d, vel1d, den1d, mesh, verbose=False)

nu = 0.25   
mtrue_mu_expr_het = None
contrast_factor = 4.0  # amplification factor 
# mu_str_het = "_DeShon3D"
# mu_str_het = f"_DeShon3D_{round(contrast_factor)}"
mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}"
print( "Heterogeneous structure:")
print( "Converted from 3D & 1D velocity models to shear modulus, mu_str_het = ", mu_str_het)


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
# f_h, f_v = 1, 1/2
f_h, f_v = 1, 1     # same as in coseismic case
# Print the weights of the data
print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )

obs_weights = np.zeros(targets.shape[0]*15,)
# obs_weights[9::15]  = (1. / data['vx_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
# obs_weights[10::15] = (1. / data['vy_std_Car']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
# obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)  # vertical displacements

# %%
# SLIP TRANSFORMATION SETUP
print(sep + "SLIP TRANSFORMATION SETUP" + sep)

# BOUNDED = True
BOUNDED = False

BOUND_TYPE = 'both'
# BOUND_TYPE = 'dip'

if BOUNDED:
    # Define slip bounds based on your problem
    V_para = 27.0 / 1e3     # the max trench-parallel long-term loading of 27 mm
    v_const_para = 11.0 / 1e3     # only remove the a constant value from trench parallel component 
    s_strike_max = V_para - v_const_para   # remove non-elastic motion
    V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
    s_dip_max = V_norm
    if BOUND_TYPE == 'both':
        slip_transformer = SlipTransformation(
            # strike_bounds=(-s_strike_max, s_strike_max),
            # strike_bounds=(0.0, s_strike_max),
            strike_bounds=(-V_para, V_para),
            dip_bounds=(0.0, s_dip_max),
        )
        print("Constraints to both strike and dip ")

    elif BOUND_TYPE == 'strike':
        slip_transformer = SlipTransformation(
            strike_bounds=(0.0, s_strike_max),
            dip_bounds=None,
        )
        print("Constraints to strike only ")

    elif BOUND_TYPE == 'dip':
        slip_transformer = SlipTransformation(
            strike_bounds=None,
            dip_bounds=(0.0, s_dip_max),
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

# rho_s = 1e8   # allows variations of slip of the order of ~10 km, close to the maximum resolution
# gamma_val_H1 = 1e2  
# gamma_val_H1 = 4e2  
# gamma_val_H1 = 1e3  
# gamma_val_H1 = 5e1  
# delta_val_L2 = gamma_val_H1 / rho_s  

if not BOUNDED:
    # preferred damping for unconstrained inversion
    if meshname == "nicoya3":   
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 1e-5
        gamma_val_H1 = 1e3  
        delta_val_L2 = 5e-6
        # delta_val_L2 = 1e-6

    elif meshname == "nicoya4":   # extended fault
        gamma_val_H1 = 1e3  
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 1e-5
        # delta_val_L2 = 1e-6
        # delta_val_L2 = 5e-6
        delta_val_L2 = 4e-6

# %%
# Take the inverse for saving the name of the weights
w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
if BOUNDED:
    inv_str = f"_lockbothbd_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
else:
    inv_str = f"_lockboth_w{w_h}{w_v}_gs{gamma_val_H1:.0e}_ds{delta_val_L2:.0e}"
# print("Inverse problem identifier: ", inv_str)

# %%
# Solve slip inverse problem within the amplified 3D heterogeneous shear modulus half-space
mtrue_mu_expr_inv = mtrue_mu_expr_het
mu_str_inv = mu_str_het
# mu_v = -0.9730  # ~25 GPa
print("Solving inverse problem based on: ", mu_str_inv)
results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
        slip_transformer,  # KEY: Pass slip transformation
        savefiles=True, verbose=True
    )
print("Amplified 3D Het Inversion finished!!!")


# %%
# Solve slip inverse problem within the original 3D heterogeneous shear modulus half-space
contrast_factor = 1.0  # amplification factor 
# mu_str_het = "_DeShon3D"
# mu_str_het = f"_DeShon3D_{round(contrast_factor)}"
mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}"
print( "Heterogeneous structure:")
print( "Converted from 3D & 1D velocity models to shear modulus, mu_str_het = ", mu_str_het)

mu_str_inv = mu_str_het
# mu_v = -0.9730  # ~25 GPa
print("Solving inverse problem based on: ", mu_str_inv)
results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
        slip_transformer,  # KEY: Pass slip transformation
        savefiles=True, verbose=True
    )
print("Original 3D Het Inversion finished!!!")