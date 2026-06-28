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
# * Use the slab geometry from Kyriakopoulos et al. (2015), and use the slab/wedge contrast shear modulus model.
# 

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

# coseismic data file
obs_disp_name = "Protti_etal_2014_tableS1.csv"

# note that the height is in m, duration and dates are in yr, and the displacements and errors are already in m
# From BAGA to VENA are Campaign Sites; From BIJA to VERA are Continuous Sites; From AROL to WARN are Volcano Sites
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1, \
                   names=['site', 'lon', 'lat', 'elv', 'ux', 'uy', 'uz', \
                          'ux_std', 'uy_std', 'uz_std', 'date0', 'date1', 'duration'])
# print(data.tail())

# define the centroid of relative coordinates, must be consistent with the mesh!
lon0, lat0 = -84, 7     # from Christos's email
# print(lon0, lat0)

# convert to relative locations in meters, and then rotate
rot = 45  # rotation angle in degrees, positive is CCW
x_rot, y_rot  = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)

# offset in x and y direction, the same as being done to the mesh in 'Kyriakopoulos2016JGR/convert_exodus_to_msh.ipynb'
x0, y0 = 130e3, 350e3  # offset for x and y coordinates, in m
# offset and convert from m to km
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3   # offset to match the mesh coordinates
data['z'] = 0.0

# To aligned with mesh, must rotate data and error as well; no need for synthetics generated self-consistently
# Rotate displacement as well, data error would be dealt with later in defining weights
data['ux'], data['uy'] = ut.rot_xy(data['ux'], data['uy'], rot) 
# data['vx_std_Car'], data['vy_std_Car'] = ut.rot_xyerror(data['vx_std_Car'], data['vy_std_Car'], rot) 
# print(data[['x', 'y', 'vx_Car', 'vy_Car']].head())
# print(data.tail())

# data['a'], data['b'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot) 
# data['c'], data['d'] = ut.project_vector_2d_matrix(data['vx_Car'], data['vy_Car'], azimuth)
# data['e'], data['f'] = ut.rot_xy(data['vx_std_Car'], data['vy_std_Car'], rot) 
# print(data[['a', 'b', 'c', 'd', 'e', 'f']].head())

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
# read in the 3D velocity model
veldir = "/home/staff/chao/SSEinv/Nicoya/DeShon_2006GJI/"
vel3dfile = "DeShon2006_3Dmodel.csv"
vel3d = pd.read_csv(veldir + vel3dfile, sep=",")
# convert to relative locations in meters, and then rotate, then offset, align with the local coordinate system of the mesh
x_rot, y_rot = ut.LL2ckmd(vel3d['lon'], vel3d['lat'], lon0, lat0, rot)
vel3d['x'], vel3d['y'] = x_rot - x0, y_rot - y0   # offset to match the mesh coordinates
vel3d['z'] = vel3d['z'] * -1 * 1e3  # negative depth means downward
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
# Helper function to save CG2 (or higher order) function data with ALL DOF points
# Standard XDMFFile.write() only exports values at mesh vertices, losing edge midpoint DOFs for CG2
import h5py

def save_function_with_dofs(function, filename_base, attr_name, cg_degree=1):
    """
    Save a FEniCS function with all DOF coordinates and values.

    For CG1: Uses standard XDMF export (DOFs = vertices)
    For CG2+: Saves additional _dofs.h5 file with all DOF points for visualization

    Parameters:
    -----------
    function : dolfin.Function
        The FEniCS function to save
    filename_base : str
        Base filename without extension (e.g., 'mu_true_nicoya')
    attr_name : str
        Attribute name for the data (e.g., 'shear modulus')
    cg_degree : int
        CG element degree (1, 2, etc.)

    Returns:
    --------
    None
    """
    # Always save standard XDMF for backward compatibility with ParaView
    xdmf_file = dl.XDMFFile(filename_base + '.xdmf')
    function.rename(attr_name, attr_name)
    xdmf_file.write(function)

    # For CG2+, also save all DOF coordinates and values to a separate HDF5 file
    if cg_degree > 1:
        V = function.function_space()
        dof_coords = V.tabulate_dof_coordinates()
        dof_values = function.vector()[:]

        n_dofs = len(dof_coords)
        n_vertices = V.mesh().num_vertices()
        print(f"  CG{cg_degree} function has {n_dofs} DOFs (vs {n_vertices} mesh vertices)")

        # Save to HDF5 with DOF coordinates and values
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
# Define folder to save the results
resultpath = "/home/staff/chao/SSEinv/Nicoya/rst_coseis/"
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
# meshname = "nicoyaCK"   # local interface model from C. Kyriakopoulos_etal2015JGRSE
# meshname = "nicoyaCK2"   # same as above but 5-km mesh size on fault
# meshname = "nicoyaCK3"   # fault zone extended to the whole subduction zone
# meshname = "nicoyaCK4"   # same as CK2, but connecting the trench now

# Meshes with even top boundary at 0 depth
# meshname = "nicoyaCKden_sm"   # based on nicoyaCK3 or 4, but denser mesh size, and smaller fault zone
# meshname = "nicoyaCKden_all"   # based on nicoyaCK3 or 4, but denser mesh size, and all subduction interface

# Mesh with uneven top boundary, left at mean trench depth ~7 km, right at 0 km
# meshname = "nicoyaCKden_une_sm"   # uneven top boundary, smaller fault zone, counterpart to 'nicoyaCKden_sm'
meshname = "nicoyaCKden_une_all"   # uneven top boundary, all subduction interface, counterpart to 'nicoyaCKden_all'

# Flag to indicate if using uneven mesh (will be set automatically based on meshname)
use_uneven_mesh = "une" in meshname

print(meshname)
print(f"Using uneven mesh: {use_uneven_mesh}")

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
    # CG_mu_deg = 2  # Use higher-order CG element for mu interpolation, 1 was used previously
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    # If mtrue_mu_expr_inv is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_inv == None:

        # Build the 1D depth-layered model for the whole mesh
        # Same return signature as process_velocity_models_hull
        vs_func, den_func, mu_func, CG_mu = ut.build_1d_shear_modulus(
            vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, method='step', verbose=True)

        # Use mu_func directly — no scaling needed
        mtrue_mu_fun = mu_func

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
        # For heterogeneous case (already a Function), just assign
        # For homogeneous case (UFL expression from mu_expression), compute values directly
        # to avoid expensive dl.project() which causes out-of-memory with CG2
        if isinstance(mtrue_mu_fun, dl.Function):
            m_mu_true = mtrue_mu_fun
        else:
            # mtrue_mu_expr_fun is already in CG_mu space, compute mu = 20*(2+tanh(m)) directly
            m_mu_true = dl.Function(CG_mu)
            m_values = mtrue_mu_expr_fun.vector()[:]
            m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
        filename_base = resultpath + 'mu_true_' + meshname + mu_str_inv
        save_function_with_dofs(m_mu_true, filename_base, 'shear modulus', cg_degree=CG_mu_deg)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

        # Save true Vs and density structures if using 3D & 1D velocity models
        if mtrue_mu_expr_inv == None:
            print( "Saving true Vs structure to .xdmf file" )
            # vs_true = dl.project( vs_func, CG_mu )  #didn't break anything, but redundant, already in CG_mu space
            vs_true = vs_func
            filename_base = resultpath + 'vs_true_' + meshname + mu_str_inv
            save_function_with_dofs(vs_true, filename_base, 'shear velocity Vs', cg_degree=CG_mu_deg)
            print( vs_true.vector().min(), vs_true.vector().max() )

            print( "Saving true density structure to .xdmf file" )
            # den_true = dl.project( den_func, CG_mu )    #didn't break anything, but redundant, already in CG_mu space
            den_true = den_func
            filename_base = resultpath + 'pho_true_' + meshname + mu_str_inv
            save_function_with_dofs(den_true, filename_base, 'density', cg_degree=CG_mu_deg)
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
    # obs_weights[9::15]  = (1. / data['ux_std']**2).to_numpy() * 1/(f_h**2)  # horizontal east displacement 
    # obs_weights[10::15] = (1. / data['uy_std']**2).to_numpy() * 1/(f_h**2)  # horizontal north displacement 
    
    # Considering the coordinate rotation, location error changes, so are the weights 
    weight_x, weight_y = ut.compute_rotated_weights((data['ux_std']**2).to_numpy(), \
                                                 (data['uy_std']**2).to_numpy(), rot)
    obs_weights[9::15] = weight_x  * 1/(f_h**2)   # x displacement weights
    obs_weights[10::15] = weight_y * 1/(f_h**2)  # y displacement weights
    obs_weights[11::15] = (1. / data['uz_std']**2).to_numpy() * 1/(f_v**2)  # vertical displacements
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
    tmp[9::15] = np.array(data['ux'])    # horizontal east displacement misfit
    tmp[10::15] = np.array(data['uy'])   # horizontal north displacement misfit
    tmp[11::15] = np.array(data['uz'])   # vertical displacement misfit
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
    
    # Save inversion results (coseismic slip)
    if savefiles:
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
    # m_mu_true = dl.project(mtrue_mu_fun, CG_mu) #didn't break anything, but redundant, already in CG_mu space  
    if isinstance(mtrue_mu_fun, dl.Function):
        m_mu_true = mtrue_mu_fun
    else:
        # mtrue_mu_expr_fun is already in CG_mu space, compute mu = 20*(2+tanh(m)) directly
        m_mu_true = dl.Function(CG_mu)
        m_values = mtrue_mu_expr_fun.vector()[:]
        m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
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
# mu_u = -0.9730  # ~25 GPa
# # mu_u = mu_b
# mu_upper = mu_expression(mu_u)

# # # shear modulus for volcanoes
# # mu_v = -0.9730  # ~25 GPa
# # mu_volcano = mu_expression(mu_v) 

# mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u, mu_l, degree=5)  #in the order of: 'k_r' in blockright, 'k_l' in blockleft
# mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}"
# # mu_str_het  = f"_mul{round(mu_expression(mu_l))}u{round(mu_expression(mu_u))}v{round(mu_expression(mu_v))}"

# print( "Heterogeneous structure:")
# print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )
# # print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f and volcano mu = %.1f" %(mu_upper, mu_lower, mu_volcano) )

# %%
# Define the true model PARAMETERS for INVERSE problem
# use the 3D & 1D velocity models to compute shear modulus, call 'process_velocity_models' inside solvers 
# vs_func, den_func, mu_func, _ = process_velocity_models(vel3d, vel1d, den1d, mesh, verbose=False)

nu = 0.25   
mtrue_mu_expr_het = None
# contrast_factor = 1.0  # amplification factor 
# contrast_factor = 4.0  # amplification factor, too extreme, needs clipping, and not adopted since 03/05/2026
# contrast_factor = 2.5  # amplification factor, more reasonable, and adopted since 03/05/2026

mu_str_het = "_DeShon1Dref"  # the 1D, depth-layered reference (starting) model used by DeShon et al., no scaling
# mu_str_het = "_DeShon3D"
# mu_str_het = f"_DeShon3D_{round(contrast_factor)}"
# mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}"

# String: _hull indicates process_velocity_models_hull method
# mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}_hull"   # ref global mean
# mu_str_het = f"_DeShon3D_ref1D_{round(contrast_factor)}_hull"   # ref 1D value at each depth layer

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
print(targets.shape)

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
            strike_bounds=(0.0, s_strike_max),
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
# if BOUNDED:
#     # gamma_val_H1 = 1e2  
#     # gamma_val_H1 = 4e2  
#     gamma_val_H1 = 5e1  
# else:
#     gamma_val_H1 = 1e3  

# delta_val_L2 = gamma_val_H1 / rho_s  

# preferred damping for the unconstrained inversion 
if not BOUNDED:
    if meshname == "nicoyaCK3":   # fault zone extended to the whole subduction zone
        gamma_val_H1 = 1e2  
        # delta_val_L2 = 2.5e-6
        delta_val_L2 = 5e-6
        gamma_val_H1 = 2.5e2  
        rho_s = 2e7
        delta_val_L2 = gamma_val_H1 / rho_s 
    elif meshname == "nicoyaCK4":   # smaller fault
        gamma_val_H1 = 1e2  
        delta_val_L2 = 5e-6

# %%
# Take the inverse for saving the name of the weights
w_h, w_v = int(1/f_h), int(1/f_v)

# # ---- single-run mode (kept for reference; switch to L-curve loop below) ----
# rho_s = 2e7   # allows variations of slip of the order of ~4.5 km, close to the maximum resolution
# gamma_val_H1 = 6e2
# delta_val_L2 = gamma_val_H1 / rho_s
#
# if BOUNDED:
#     inv_str = f"_coseisbd_w{w_h}{w_v}_gs{gamma_val_H1:.1e}_ds{delta_val_L2:.1e}"
# else:
#     inv_str = f"_coseis_w{w_h}{w_v}_gs{gamma_val_H1:.1e}_ds{delta_val_L2:.1e}"
# print("Inverse problem identifier: ", inv_str)
#
# mtrue_mu_expr_inv = mtrue_mu_expr_het
# mu_str_inv = mu_str_het
# # 1D model is piecewise constant — CG1 is sufficient (no edge-midpoint DOFs needed)
# CG_mu_deg = 1
# print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")
#
# results = solveCoseismicInversion_TanhSlip(
#         k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
#         slip_transformer,  # KEY: Pass slip transformation
#         savefiles=True, verbose=True
#     )
# print("Original 1Dref Het Inversion finished!!!")


# %%
### Compute L-curve criterion ###
# Solve to find the best Tikhonov regularization parameter by using the L-curve criterion

import gc  # for explicit memory management

# fix the rho, ratio of gamma and delta
# rho_s = 1e9   # allows variations of slip of the order of ~30 km
# rho_s = 1e8   # allows variations of slip of the order of ~10 km, close to the maximum resolution
# rho_s = 2.5e8   # allows variations of slip of the order of ~15 km, close to the maximum resolution
rho_s = 2e7   # allows variations of slip of the order of ~4.5 km, close to the maximum resolution

# vary gamma, the model gradient damping
if not BOUNDED:
    # gammas_s = [6e2]    # previous single-run value
    # gammas_s = [4e1, 8e1, 1e2, 2e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2, 2e3, 1e4]   # full list

    gammas_s = [2.5e2, 3e2]   # 1D: both γ values are new (not in existing L-curve), so append is fine

print( sep, "1Dref Het L-curve computation with slip transformation", sep )

dmisfit = []
mmisfit = []
for gamma_s in gammas_s:
    delta_s = gamma_s / rho_s

    # file identifier
    if BOUNDED:
        inv_str = f"_coseisbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    else:
        inv_str = f"_coseis_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    print("Inverse problem identifier: ", inv_str)

    print(f"****** Computing solution with gamma_s = {gamma_s:.1e}, "
        f"delta_s = {delta_s:.1e}, and rho_s = {rho_s:.1e} ******")

    # Solve slip inverse problem within the 1D-depth-layered heterogeneous shear modulus half-space
    mtrue_mu_expr_inv = mtrue_mu_expr_het
    mu_str_inv = mu_str_het
    # 1D model is piecewise constant — CG1 is sufficient (no edge-midpoint DOFs needed)
    CG_mu_deg = 1
    print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")

    results = solveCoseismicInversion_TanhSlip(
            k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
            slip_transformer,  # KEY: Pass slip transformation
            savefiles=True, verbose=True
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
    outFileName = f"Lcurvecoseisbd_rs{rho_s:.0e}_{meshname}_{mu_str_inv}.txt"
else:
    outFileName = f"Lcurvecoseis_rs{rho_s:.0e}_{meshname}_{mu_str_inv}.txt"
# csvoutput = open(resultpath + outFileName, 'w+') # make new and overwrite
csvoutput = open(resultpath + outFileName, 'a') # append to existing file
for i in range(0, len(dmisfit)):
    csvoutput.write( "%.6e %.6e %.1e %.0e\n" %(dmisfit[i], mmisfit[i], gammas_s[i], rho_s) )
csvoutput.close()

print("All 1Dref Het Inversion Finished!")

