# %% [markdown]
# # Back-slip inversion with direction constrained to plate motion azimuth (3D mu)
#
# Based on slip_inv_hetmu_nicoyaCK_locking_both_az.py + 3D bits from
# slip_inv_hetmu3D_nicoyaCK_locking_both.py.
#
# 3D shear modulus from DeShon et al. (2006) 3D + 1D velocity models,
# processed via ut.process_velocity_models_hull (convex hull, no extrapolation
# artifacts) and amplified by ut.scale_shear_modulus_by_1d (relative to 1D
# depth-layered reference, contrast_factor=2.5).
#
# KEY DIFFERENCE — Option A rake constraint:
#   - Parameter space: scalar CG1 amplitude a(x) instead of 2-component (s_strike, s_dip)
#   - Slip direction: FIXED per-facet to plate convergence azimuth via _compute_az_per_facet_coeffs
#     slip_vec(x) = a(x) × (c_strike(x), c_dip(x))   [direction fixed, amplitude free]
#   - Amplitude tanh bound: a ∈ [0, amp_max] (= [0, V_plate])
#   - BiLaplacianPrior applied to the scalar amplitude field
#
# This matches the block-model philosophy: only the scalar coupling (amplitude) is estimated;
# the back-slip direction is prescribed from the plate Euler pole kinematics.

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
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "...") )
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
sep = "\n" + "#"*80 + "\n"
print(sep, "START of Az-constrained inversion", sep)

# %%
# Define data directory
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"

# Import GNSS data, originally from Feng et al. 2012, but no volcano sites, both trench-parallel and normal components
obs_disp_name = "CKfig6_data_final.csv"   # the EXACT data file for figure 6 in Kyriakopoulos et al. (2016)

# the processed data has the unit of m/yr that was converted from mm/yr
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1,
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car',
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

# define the centroid of relative coordinates, must be consistent with the mesh!
lon0, lat0 = -84, 7     # from Christos's email

# convert to relative locations in meters, and then rotate
rot = 45  # rotation angle in degrees, positive is CCW
x_rot, y_rot = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)

# offset in x and y direction, the same as being done to the mesh in 'Kyriakopoulos2016JGR/convert_exodus_to_msh.ipynb'
x0, y0 = 130e3, 350e3  # offset for x and y coordinates, in m
# offset and convert from m to km
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3   # offset to match the mesh coordinates
data['z'] = 0.0

# To align with mesh, must rotate data and error as well; no need for synthetics generated self-consistently
# Rotate displacement as well; data error would be dealt with later in defining weights
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot)
# data['vx_std_Car'], data['vy_std_Car'] = ut.rot_xyerror(data['vx_std_Car'], data['vy_std_Car'], rot)
print("Number of stations:", len(data))

# a catalog Holocene volcanoes
volc_file = "GVP_Holocene_Volcano_loc.csv"
volc = pd.read_csv(datadir + volc_file, sep=",", skiprows=1,
                   names=['id', 'lat', 'lon', 'elv'])
# truncate within a region, same as Figure 1b in Feng et al. 2012
volc = volc[(volc['lat'] >= 8) & (volc['lat'] <= 12) &
            (volc['lon'] >= -88) & (volc['lon'] <= -83)]
# convert to relative locations in meters, and then rotate, then offset
x_rot, y_rot = ut.LL2ckmd(volc['lon'], volc['lat'], lon0, lat0, rot)
volc['x'], volc['y'] = x_rot - x0, y_rot - y0   # offset to match the mesh coordinates
volc['z'] = 0.0


# %%
# read in the 3D velocity model (DeShon et al. 2006 GJI)
veldir = "/home/staff/chao/SSEinv/Nicoya/DeShon_2006GJI/"
vel3dfile = "DeShon2006_3Dmodel.csv"
vel3d = pd.read_csv(veldir + vel3dfile, sep=",")
# convert lon/lat to mesh frame (rotated, offset)
x_rot, y_rot = ut.LL2ckmd(vel3d['lon'], vel3d['lat'], lon0, lat0, rot)
vel3d['x'], vel3d['y'] = x_rot - x0, y_rot - y0
vel3d['z'] = vel3d['z'] * -1 * 1e3
vel3d = vel3d[(vel3d['z'] <= 0)].reset_index(drop=True)

# reference 1D velocity model
vel1dfile = "DeShon2006_1Dmodel.csv"
vel1d = pd.read_csv(veldir + vel1dfile, sep=r'\s+', skiprows=1,
                    names=['z', 'vp', 'vs', 'vp_vs_ratio'])
vel1d['z'] = vel1d['z'] * -1 * 1e3
vel1d = vel1d[(vel1d['z'] <= 0)].reset_index(drop=True)

# 1D density model (DeShon-style), same depth layers as the 3D model
den1dfile = "Density_1Dmodel.csv"
den1d = pd.read_csv(veldir + den1dfile, sep=r'\s+', skiprows=1,
                    names=['z', 'den'])
den1d['z'] = den1d['z'] * -1 * 1e3
den1d = den1d[(den1d['z'] <= 0)].reset_index(drop=True)
den1d['den'] = den1d['den'] * 1e3   # g/cm^3 -> kg/m^3

# %%
# Helper to save CG2 function with all DOFs (vertices + edge midpoints).
# Standard XDMFFile.write only exports vertex values, losing CG2 edge midpoints.
import h5py


def save_function_with_dofs(function, filename_base, attr_name, cg_degree=1):
    """Save FEniCS function. CG1: standard XDMF only. CG2+: also save _dofs.h5."""
    xdmf_file = dl.XDMFFile(filename_base + '.xdmf')
    function.rename(attr_name, attr_name)
    xdmf_file.write(function)
    xdmf_file.close()
    if cg_degree > 1:
        V = function.function_space()
        dof_coords = V.tabulate_dof_coordinates()
        dof_values = function.vector()[:]
        n_dofs = len(dof_coords); n_vertices = V.mesh().num_vertices()
        print(f"  CG{cg_degree} function has {n_dofs} DOFs (vs {n_vertices} mesh vertices)")
        with h5py.File(filename_base + '_dofs.h5', 'w') as f:
            f.create_dataset('coordinates', data=dof_coords)
            f.create_dataset('values', data=dof_values)
            f.attrs['cg_degree'] = cg_degree
            f.attrs['attr_name'] = attr_name
            f.attrs['n_dofs'] = n_dofs
            f.attrs['n_vertices'] = n_vertices
        print(f"  Saved all DOF data to: {filename_base}_dofs.h5")


# %%
# Define folder to save the results
resultpath = "/home/staff/chao/SSEinv/Nicoya/rst_locking/"
os.makedirs(resultpath, exist_ok=True)

# %%
# Define the Compliance matrix for elasticity
def AEsigma(s, mu, nu):
    A = 1./(2.*mu) * (s - nu/(1 + nu*(dim-2)) * ufl.tr(s) * ufl.Identity(dim))
    return A

# %%
# Define the asymmetry operator
def asym(s):    # calculate the off-diagonal difference. If != 0 --> asymmetry
    if dim == 2:
        as_ = s[1,0] - s[0,1]
    elif dim == 3:
        as_ = ufl.as_vector([s[1,2]-s[2,1], s[2,0]-s[0,2], s[0,1]-s[1,0]])
    return as_

# %%
# Define the strike direction operator
def dir_strike(n):
    # Positive strike --> right-lateral strike slip fault
    # Create strike and dip direction through cross product of the unit normal vector
    # with the vertical. Cross product gives the strike direction and find dip.
    z_dir = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    # Normalize by the magnitude of the cross product
    return n_cross_z / ufl.sqrt(ufl.dot(n_cross_z, n_cross_z))

# %%
# Define the dip direction operator
def dir_dip(n):
    # Positive dip --> reverse slip fault
    return ufl.cross(dir_strike(n), n)

# %%
# Class supporting a low-velocity-layer subdomain (3D model option; unused for
# the standard meshes but kept for completeness)
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
#
# Meshes with even top boundary at 0 depth
# meshname = "nicoyaCKden_sm"   # based on nicoyaCK3 or 4, but denser mesh size, and smaller fault zone
# meshname = "nicoyaCKden_all"  # based on nicoyaCK3 or 4, but denser mesh size, and all subduction interface
#
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
top = 1; bottom = 2; west = 3; east = 4; north = 5; south = 6
fault = 7; blockleft = 8; blockright = 9
# Define the surface integration over the external boundary (ds) and internal boundary (dS)
ds = dl.Measure("ds")(domain=mesh, subdomain_data=boundaries)
dS = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

# %%
# Define the expression of the shear modulus
def mu_expression(m):
    return 20*(2. + ufl.tanh(m))

# %%
# ============================================================
# NEW: Azimuth configuration
# ============================================================
azimuth_deg      = 33.5   # geographic azimuth of plate CONVERGENCE (CW from North): N33.5E for Cocos-Caribbean at Nicoya
mesh_rotation_deg = rot   # CCW rotation used to build the Nicoya mesh (matches `rot` defined above)
V_plate = 80.0 / 1e3      # full plate-convergence vector magnitude [m/yr] (Cocos-Caribbean at Nicoya, N33.5°E)
                          # NOTE: this is the FULL plate-motion vector magnitude, not the trench-normal
                          # component (= 78.5 mm/yr for Nicoya). The two are confused in some legacy
                          # scripts where V_norm was used to mean the full vector.
amp_max = V_plate         # max scalar amplitude (upper bound for tanh transform)

# if taking the trench-perpendicular azimuth instead
# azimuth_deg = 45
# V_dip = 78.5 / 1e3      # trench-normal component of Cocos-Caribbean motion [m/yr]
# amp_max = V_dip         # max scalar amplitude (upper bound for tanh transform)

print(sep + "Az-constrained slip setup" + sep)
print(f"  Plate convergence azimuth: {azimuth_deg:.1f}° CW from North")
print(f"  Mesh rotation: {mesh_rotation_deg:.1f}° CCW")
print(f"  Amplitude bound: [0, {amp_max*1e3:.1f} mm/yr]")


# %%
# ============================================================
# NEW: Project per-facet Az coefficients to CG1 vertex functions
# ============================================================
def compute_az_cg1_functions(mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg=45.0):
    """
    Compute per-facet (c_strike, c_dip) coefficients for the prescribed azimuth
    and project them to CG1 vertex functions by averaging over adjacent fault facets.

    Returns
    -------
    c_strike_fun, c_dip_fun : dl.Function (CG1)
        Nonzero only at fault DOFs. Satisfy sqrt(c_strike²+c_dip²) ≈ 1 at fault vertices.
    """
    # Per-facet coefficients: {facet_index -> (c_strike, c_dip)}
    # c_strike, c_dip encode the in-plane back-slip direction (already negated for back-slip)
    az_coeffs = ut._compute_az_per_facet_coeffs(
        mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg)

    CG1 = dl.FunctionSpace(mesh, "CG", 1)
    c_strike_fun = dl.Function(CG1)
    c_dip_fun    = dl.Function(CG1)

    v2d = dl.vertex_to_dof_map(CG1)

    # Accumulate per-facet values at vertices (each fault vertex may touch 1-2 fault facets)
    vertex_cs = {}  # vertex_index -> list of c_strike values from adjacent fault facets
    vertex_cd = {}
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

    # Assign averaged values to CG1 arrays
    cs_arr = c_strike_fun.vector().get_local()
    cd_arr = c_dip_fun.vector().get_local()
    for vi in vertex_cs:
        dof = v2d[vi]
        cs_arr[dof] = np.mean(vertex_cs[vi])
        cd_arr[dof] = np.mean(vertex_cd[vi])
    c_strike_fun.vector().set_local(cs_arr)
    c_dip_fun.vector().set_local(cd_arr)
    c_strike_fun.vector().apply('')
    c_dip_fun.vector().apply('')

    n_fault_verts = len(vertex_cs)
    cs_fault = cs_arr[cs_arr != 0.0]
    cd_fault = cd_arr[cd_arr != 0.0]
    mag = np.sqrt(cs_fault**2 + cd_fault**2)
    print(f"  Projected to {n_fault_verts} fault vertices")
    print(f"  c_strike: [{cs_fault.min():.4f}, {cs_fault.max():.4f}]")
    print(f"  c_dip:    [{cd_fault.min():.4f}, {cd_fault.max():.4f}]")
    print(f"  ||c||:    [{mag.min():.4f}, {mag.max():.4f}]  (should be ≈ 1.0)")

    return c_strike_fun, c_dip_fun


# Compute once after mesh loading; reused for all inversions
print("Computing Az coefficient CG1 functions ...")
c_strike_fun, c_dip_fun = compute_az_cg1_functions(
    mesh, boundaries, fault, azimuth_deg, mesh_rotation_deg)


# %%
# ============================================================
# NEW: PDE variational formulation — Az-constrained slip
# ============================================================
class PDEVarf_AzSlip:
    """
    PDE with SCALAR amplitude parameter and fixed slip direction.

    Parameter m: scalar CG1 field. Two modes via use_tanh_amp:
        use_tanh_amp=True  (bounded, default):
            m_phys(x) = amp_max * (tanh(m(x)) + 1) / 2   ∈ [0, amp_max]
            (m=0 → m_phys = amp_max/2 — midpoint; unresolved regions collapse here)
        use_tanh_amp=False (unbounded):
            m_phys(x) = m(x)                              ∈ (-∞, +∞)
            (m=0 → m_phys = 0 — unresolved regions collapse to zero; amp_max unused
             in the math but kept around as a sanity reference)

    Slip vector (in fault-local strike/dip basis):
        slip_strike(x) = m_phys(x) * c_strike_fun(x)
        slip_dip(x)    = m_phys(x) * c_dip_fun(x)

    where c_strike_fun, c_dip_fun are fixed CG1 coefficient functions encoding the
    plate-convergence azimuth direction.  sqrt(c_strike²+c_dip²) ≈ 1, so the physical
    slip magnitude equals m_phys everywhere on the fault.
    """
    def __init__(self, mtrue_mu_fun, c_strike_fun, c_dip_fun, amp_max,
                 use_tanh_amp=True):
        self.mtrue_mu_fun = mtrue_mu_fun
        self.c_strike_fun = c_strike_fun
        self.c_dip_fun    = c_dip_fun
        self.amp_max      = amp_max
        self.use_tanh_amp = use_tanh_amp

    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        tau, w, q    = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))

        if self.use_tanh_amp:
            # Scalar amplitude: (-∞,∞) → [0, amp_max]
            m_phys = self.amp_max * (ufl.tanh(m) + 1) / 2
        else:
            # Unbounded linear: m_phys = m. amp_max not used in the math.
            m_phys = m

        # mtrue_mu_fun is ALREADY in GPa (UFL expr for K_2LAYER/hom, dl.Function for 3D).
        # No mu_expression wrapping here — that conversion is handled in solveCoseismicInversion_AzSlip.
        mu = self.mtrue_mu_fun

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


# %%
# ============================================================
# NEW: Inversion routine — Az-constrained (scalar amplitude)
# ============================================================
def solveCoseismicInversion_AzSlip(k, targets, m0_amp_expr, mtrue_mu_expr_inv,
                                   gamma_val_H1, delta_val_L2,
                                   c_strike_fun, c_dip_fun, amp_max,
                                   CG_mu_deg=1,
                                   use_tanh_amp=True,
                                   savefiles=True, verbose=True):
    """
    Slip inversion with direction constrained to plate motion azimuth (Option A).

    Parameter space: scalar CG1 amplitude a(x).
    Physical slip  = a(x) × (c_strike(x), c_dip(x)).
    Amplitude transform (set by use_tanh_amp):
        True  (default): a ∈ [0, amp_max] via tanh   — m=0 maps to amp_max/2
        False:           a = m, unbounded            — m=0 maps to 0

    Returns same tuple shape as solveCoseismicInversion_TanhSlip for compatibility.
    """

    # --- Function spaces ---
    # Use VectorFunctionSpace if the unknown is a vector field; FunctionSpace for scalars.
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)    # stress (tensor field; BDM is a vector field)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)   # displacement (vector field)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)   # rotation (scalar field, treated as vector)
    # Create a mixed fine element function space
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)

    # KEY CHANGE: scalar parameter space (CG1) — was 2-component (s_strike, s_dip) in reference
    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)
    # Combine the STATE, PARAMETER and ADJOINT function spaces
    Vh = [Vu, Vm_amp, Vu]

    # Print the dofs of STATE, PARAMETER and ADJOINT variables
    ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
    ndofs_state = [Vu.sub(0).dim(), Vu.sub(1).dim(), Vu.sub(2).dim()]

    if verbose:
        print(sep, "Az-constrained slip inversion (scalar amplitude parameter)", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print("Number of STATE/ADJOINT dofs: STRESS={0}, DISPL={1}, ROT={2}".format(*ndofs_state))
        if use_tanh_amp:
            print(f"Amplitude bound (tanh): [0, {amp_max*1e3:.1f} mm/yr]")
        else:
            print(f"Amplitude transform: linear / unbounded (amp_max={amp_max*1e3:.1f} mm/yr kept as reference scale)")

    # --- Boundary conditions ---
    # Define the STATE and ADJOINT Dirichlet BCs (zero traction on the top free surface)
    zero_tensor = dl.Expression((("0.", "0.", "0."),
                                  ("0.", "0.", "0."),
                                  ("0.", "0.", "0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # --- Initial guess: scalar 0 ---
    # Interpolate starting guess for the parameter.
    # tanh mode: m=0 → m_phys = amp_max/2 (midpoint)
    # linear mode: m=0 → m_phys = 0
    m0_amp = dl.interpolate(m0_amp_expr, Vh[hp.PARAMETER]).vector()

    # shear modulus
    # CG_mu_deg = 2  # Use higher-order CG element for mu interpolation, 1 was used previously
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    # If mtrue_mu_expr_inv is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_inv == None:
        # NEW, using convex hull to avoid extrapolation artifacts
        vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull(
            vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=True)

        ### new way of amplication, relative to 1D depth-layered values, no minimum clipping any more
        ### tested at 2x or 2.5 times both result in non-negative values
        mtrue_mu_fun = ut.scale_shear_modulus_by_1d(
            mu_func, vel1d, den1d, contrast_factor, min_mu=None, verbose=True)

    else:
        # Assign the values of the vector
        mtrue_mu_expr = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()

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
            vs_true = vs_func
            filename_base = resultpath + 'vs_true_' + meshname + mu_str_inv
            save_function_with_dofs(vs_true, filename_base, 'shear velocity Vs', cg_degree=CG_mu_deg)
            print( vs_true.vector().min(), vs_true.vector().max() )

            print( "Saving true density structure to .xdmf file" )
            den_true = den_func
            filename_base = resultpath + 'pho_true_' + meshname + mu_str_inv
            save_function_with_dofs(den_true, filename_base, 'density', cg_degree=CG_mu_deg)
            print( den_true.vector().min(), den_true.vector().max() )

    # --- PDE ---
    # KEY CHANGE: use the Az-constrained PDE form (slip = m_phys × (c_strike, c_dip))
    pde_varf = PDEVarf_AzSlip(mtrue_mu_fun, c_strike_fun, c_dip_fun, amp_max,
                              use_tanh_amp=use_tanh_amp)
    # Define the PDE problem
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    # Define the solver type
    type_solver = "mumps"
    pde.solver         = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    # --- Observation operator ---
    # Print the number of observations (observed surface horizontal and vertical displacements)
    if verbose:
        print("Number of observation points: {0}".format(targets.shape[0]))

    # Constrain only the displacement field for the data misfit (indices 9*, 10*, 11*)
    indicator_vec = dl.interpolate(
        dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()
    # Create the weight vector to impose 1 / standard deviation² of observation error
    weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
    obs_weights = np.zeros(targets.shape[0]*15)
    # Considering the coordinate rotation, location error changes, so are the weights
    weight_x, weight_y = ut.compute_rotated_weights(
        (data['vx_std_Car']**2).to_numpy(), (data['vy_std_Car']**2).to_numpy(), rot)
    obs_weights[9::15]  = weight_x  * 1/(f_h**2)   # x displacement weights
    obs_weights[10::15] = weight_y  * 1/(f_h**2)   # y displacement weights
    obs_weights[11::15] = (1. / data['vz_std_Car']**2).to_numpy() * 1/(f_v**2)   # vertical
    # Modify and set the array to dolfin vector
    weights.set_local(obs_weights)
    weights.apply('')

    # Define the misfit function
    misfit = PointwiseStateObservation(Vh[hp.STATE], targets, weight=weights,
                                       indicator_vec=indicator_vec)
    # Impose noise_variance = 1 since we already absorbed σ into the weights
    misfit.noise_variance = 1.

    # Generate observations by using the observation operator 'B'
    if len(misfit.weight) % 15 != 0:
        print("Mismatch data with targets sizes. Check that targets are in the domain.")
    # Infer the index position of all non-zero entries in the misfit
    idx_d = list(np.nonzero(misfit.weight)[0])
    if len(idx_d) / 3 != targets.shape[0]:
        print("Error: non-zero misfit length mismatch.")

    # Imput GPS data into misfit.d (replace any synthetic placeholder)
    tmp = np.zeros(len(misfit.d))
    tmp[9::15]  = np.array(data['vx_Car'])    # horizontal east displacement
    tmp[10::15] = np.array(data['vy_Car'])    # horizontal north displacement
    tmp[11::15] = np.array(data['vz_Car'])    # vertical displacement
    misfit.d.set_local(tmp)
    misfit.d.apply('')

    # Extract horizontals and vertical observed data
    d_obs = misfit.d[idx_d]
    if savefiles:
        # Save the observed surface displacement
        outFileName = 'd_obs_' + meshname + inv_str + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]))
        csvoutput.close()

    # --- Fault coordinate extraction (same as reference) ---
    # Extract x,y,z coordinates of the fault and values
    CG_s = dl.VectorFunctionSpace(mesh, "CG", 1)   # dim=3 for 3D coordinates
    bc1 = dl.DirichletBC(CG_s, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG_s)
    bc1.apply(um.vector())

    # Scalar amplitude fault mask (replaces the 2-component (99,99) mask in the reference)
    bc_amp = dl.DirichletBC(Vm_amp, 99.0, boundaries, fault)
    um_amp = dl.Function(Vm_amp)
    bc_amp.apply(um_amp.vector())

    xslip = dl.interpolate(dl.Expression(("x[0]", "x[0]", "x[0]"), degree=5), CG_s)
    yslip = dl.interpolate(dl.Expression(("x[1]", "x[1]", "x[1]"), degree=5), CG_s)
    zslip = dl.interpolate(dl.Expression(("x[2]", "x[2]", "x[2]"), degree=5), CG_s)
    xf = xslip.vector()[um.vector() == 10]   # x coordinate fault
    yf = yslip.vector()[um.vector() == 10]   # y coordinate fault
    zf = zslip.vector()[um.vector() == 10]   # z coordinate fault
    if verbose:
        print(sep, "Done extracting the fault coordinates", sep)

    # --- Regularization on scalar amplitude field ---
    # BiLaplacian prior on the scalar amplitude (a single CG1 field, not a 2-component vector)
    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)

    # --- Model (Lagrangian formalism) ---
    # Construct the "Model" --> objective function combining the PDE, prior, and misfit
    model = hp.Model(pde, reg, misfit)
    m = m0_amp.copy()

    # --- Solve ---
    # The beauty of tanh transformation: no changes needed to the unconstrained CG solver
    if verbose:
        print(sep, "Solve the Az-constrained slip inverse problem", sep)
    # Generate STATE, PARAMETER and ADJOINT vectors
    u  = model.generate_vector(hp.STATE)
    p  = model.generate_vector(hp.ADJOINT)
    x  = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)
    # Solve the FORWARD problem to find the STATE variables
    model.solveFwd(u, x)
    # Solve the ADJOINT problem to find the ADJOINT variables
    model.solveAdj(p, x)
    # Calculate the GRADIENT
    model.evalGradientParameter(x, mg)
    if verbose:
        print(sep, "Done generating STATE, PARAMETER and ADJOINT vectors", sep)

    # --- Hessian + preconditioned CG solver ---
    # Solve the inverse problem with CG with regularization-based preconditioning
    model.setPointForHessianEvaluations(x)
    # Define the reduced Hessian; since the inverse problem is LINEAR it is independent of m
    H = hp.ReducedHessian(model)
    # Use the regularization as a preconditioner for the CG algorithm
    Prec = reg.Rsolver
    # Consider all the Hessian (misfit + regularization)
    H.misfit_only = False
    # Steihaug stopping criterion to avoid negative curvature
    solver = hp.CGSolverSteihaug()
    solver.set_operator(H)
    solver.set_preconditioner(Prec)
    solver.parameters["print_level"]   = 1
    solver.parameters["rel_tolerance"] = 1e-9
    solver.parameters["abs_tolerance"] = 1e-12
    solver.parameters["max_iter"]      = 1500
    m_hat = model.generate_vector(hp.PARAMETER)
    solver.solve(m_hat, -mg)
    if solver.converged:
        print("CG converged in", solver.iter, "iterations.")
    else:
        print("CG did not converge.")
        raise RuntimeError("CG solver failed")

    m.axpy(1., m_hat)

    # --- Post-processing: scalar m → physical amplitude → (s_strike, s_dip) ---
    m_fun = hp.vector2Function(m, Vh[hp.PARAMETER])

    # Physical amplitude: match the transform used in PDEVarf_AzSlip
    if use_tanh_amp:
        m_phys_expr = amp_max * (ufl.tanh(m_fun) + 1) / 2   # [0, amp_max]
    else:
        m_phys_expr = m_fun                                  # unbounded linear
    m_phys_fun  = dl.project(m_phys_expr, Vm_amp)

    # Reconstruct strike and dip components (scalar functions)
    s_strike_fun = dl.project(m_phys_fun * c_strike_fun, Vm_amp)
    s_dip_fun    = dl.project(m_phys_fun * c_dip_fun,    Vm_amp)

    if savefiles:
        print("Saving amplitude and slip components to .xdmf files")
        m_phys_fun.rename('slip amplitude', 'slip amplitude')
        amp_id = dl.XDMFFile(resultpath + 'slip_amp_' + meshname + inv_str + mu_str_inv + '.xdmf')
        amp_id.write(m_phys_fun)
        amp_id.close()

        s_strike_fun.rename('strike slip', 'strike slip')
        s_strike_id = dl.XDMFFile(resultpath + 's_strike_' + meshname + inv_str + mu_str_inv + '.xdmf')
        s_strike_id.write(s_strike_fun)
        s_strike_id.close()

        s_dip_fun.rename('dip slip', 'dip slip')
        s_dip_id = dl.XDMFFile(resultpath + 's_dip_' + meshname + inv_str + mu_str_inv + '.xdmf')
        s_dip_id.write(s_dip_fun)
        s_dip_id.close()
        print("Finish saving slip solution")

    # --- Forward solve with inverted parameters ---
    # Solve the forward problem to compute the calculated STATE variables
    x = [u, m, p]
    model.solveFwd(u, x)
    # Use the observation operator 'B' to extract the surface displacement: d_cal = Bu
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    # Extract horizontal/vertical displacement predicted observations
    d_cal = misfit.Bu[idx_d]

    if savefiles:
        # Save the predicted surface displacement
        outFileName = 'd_cal_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]))
        csvoutput.close()

    # --- L-curve metrics ---
    # Calculate the gradient norm of the parameter for the L-curve criterion
    m_fun = dl.Function(Vh[hp.PARAMETER], m)   # refresh after axpy
    grad_m = dl.assemble(ufl.inner(
        ufl.avg(ufl.nabla_grad(m_fun)),
        ufl.avg(ufl.nabla_grad(m_fun))) * dS(fault))
    misfitd = np.linalg.norm((d_cal - d_obs), 2)
    print("Data misfit {0:.6e}; Model misfit {1:.6e};".format(misfitd, grad_m))

    # Compute the cost functional to plot misfit
    total_cost, reg_cost, misfit_cost = model.cost(x)
    print("Total cost {0:5g}; Reg Cost {1:5g}; Misfit {2:5g}".format(
        total_cost, reg_cost, misfit_cost))

    # --- Moment and potency ---
    # Physical slip magnitude: ||slip|| = m_phys * ||c||  (||c|| can be < 1 at undulations).
    # Match consistfwd convention: include c_mag_fun_inv factor (vs. m_phys alone, which would
    # over-count moment in the curved-fault regions where ||c|| dips below 1).
    c_mag_arr_inv = np.sqrt(c_strike_fun.vector().get_local()**2
                            + c_dip_fun.vector().get_local()**2)
    c_mag_fun_inv = dl.Function(Vm_amp)
    c_mag_fun_inv.vector().set_local(c_mag_arr_inv); c_mag_fun_inv.vector().apply('')

    # m_mu_true is already in GPa (Function for 3D; UFL expr for hom/het).
    # For CG1 it's safe to project; for CG2 avoid dl.project (OOM): use direct numpy.
    if isinstance(mtrue_mu_fun, dl.Function):
        m_mu_true = mtrue_mu_fun
    else:
        m_mu_true = dl.Function(CG_mu)
        m_values = mtrue_mu_expr_fun.vector()[:]
        m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
    moment = dl.assemble(m_mu_true * GPa2Pa * m_phys_fun * c_mag_fun_inv * dS(fault))
    print(f"Scalar seismic moment: {moment:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment)
    print(f"Moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    # Seismic potency, independent of the assumed elastic properties
    potency = dl.assemble(m_phys_fun * c_mag_fun_inv * dS(fault))
    print(f"Seismic potency: {potency:.3e} m^3")

    # Dip-slip only moment (for comparison with reference 2-component inversion outputs)
    s_dip_abs_expr = ufl.conditional(s_dip_fun >= 0, s_dip_fun, -s_dip_fun)
    moment_dip  = dl.assemble(m_mu_true * GPa2Pa * s_dip_abs_expr * dS(fault))
    potency_dip = dl.assemble(s_dip_abs_expr * dS(fault))
    M_w1_dip, M_w2_dip, M_w3_dip = ut.moment2mag(moment_dip)
    print(f"Scalar seismic moment (dip-slip only): {moment_dip:.3e} N·m")
    print(f"Moment magnitude (dip-slip only): {M_w1_dip:.2f}; {M_w2_dip:.2f}; {M_w3_dip:.2f}")
    print(f"Seismic potency (dip-slip only): {potency_dip:.3e} m^3")

    if savefiles:
        with open(resultpath + 'moment_' + meshname + inv_str + mu_str_inv + '.txt', 'w') as fout:
            fout.write(f"{moment:.6e} {M_w3:.4f} {potency:.6e} \n")
        with open(resultpath + 'moment_dip_' + meshname + inv_str + mu_str_inv + '.txt', 'w') as fout:
            fout.write(f"{moment:.6e} {M_w3:.4f} {potency:.6e} "
                       f"{moment_dip:.6e} {M_w3_dip:.4f} {potency_dip:.6e} \n")

    # --- Predicted displacement field ---
    if savefiles:
        print("Saving predicted displacement and stress to .xdmf file")
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + inv_str + mu_str_inv + '.xdmf')
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid.write(u_save)
        uid.close()
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + inv_str + mu_str_inv + '.xdmf')
        sigma_non  = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid.write(sigma_save)
        sid.close()

        outFileName = 'fault_geometry_' + meshname + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(xf.shape[0]):
            csvoutput.write("%.6f %.6f %.6f\n" % (xf[i], yf[i], zf[i]))
        csvoutput.close()
        print("Finish saving predicted displacement and stress")

    # --- Fault slip extraction ---
    # Extract physical slip values at the fault interface (vertices where um_amp == 99.0)
    # Recover (s_strike, s_dip) for back-compatibility with downstream plotting:
    #   s_strike = m_phys * c_strike,   s_dip = m_phys * c_dip
    fault_mask   = um_amp.vector()[:] == 99.0
    m_amp_fault  = m_phys_fun.vector().get_local()[fault_mask]
    cs_fault     = c_strike_fun.vector().get_local()[fault_mask]
    cd_fault     = c_dip_fun.vector().get_local()[fault_mask]
    m_sx_fault   = m_amp_fault * cs_fault
    m_sy_fault   = m_amp_fault * cd_fault

    print("Physical slip ranges:")
    print(f"  Amplitude: [{m_amp_fault.min():.6f}, {m_amp_fault.max():.6f}] m")
    print(f"  Strike:    [{m_sx_fault.min():.6f}, {m_sx_fault.max():.6f}] m")
    print(f"  Dip:       [{m_sy_fault.min():.6f}, {m_sy_fault.max():.6f}] m")

    if savefiles:
        # m_s_fault: same 2-column format as reference for plotting compatibility
        outFileName = 'm_s_fault_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(m_sx_fault.shape[0]):
            csvoutput.write("%.6f %.6f\n" % (m_sx_fault[i], m_sy_fault[i]))
        csvoutput.close()

        # Full-volume amplitude
        outFileName = 'slip_inferred_' + meshname + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        amp_full = np.zeros(len(m0_amp))
        amp_full[fault_mask] = m_amp_fault
        for i in range(len(amp_full)):
            csvoutput.write("%.6f\n" % amp_full[i])
        csvoutput.close()

        # ── Per-fault-vertex local strike / dip basis ──
        # Mesh-only — depends on neither γ nor μ, so the same file is overwritten
        # identically each L-curve iteration (idempotent; safe to re-run).
        # Rows are aligned with m_s_fault: row i corresponds to (m_sx_fault[i], m_sy_fault[i]),
        # so the notebook can zip the two files to recover the 3D slip vector:
        #     slip_xyz = m_strike * strike_hat + m_dip * dip_hat
        # The az-constrained inversion uses a scalar parameter space (Vm_amp), so build a
        # temporary 2-component Vm here for compatibility with compute_fault_basis_per_vertex.
        Vm_basis = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
        basis_coords, strike_vec, dip_vec = ut.compute_fault_basis_per_vertex(
            mesh, boundaries, Vm_basis, fault, verbose=True)
        if basis_coords.shape[0] != m_sx_fault.shape[0]:
            print(f"Warning: fault_basis row count {basis_coords.shape[0]} "
                  f"!= m_s_fault row count {m_sx_fault.shape[0]} — alignment broken")
        with open(resultpath + 'fault_basis_' + meshname + '.txt', 'w+') as fout:
            fout.write("# x_m  y_m  z_m  "
                       "strike_x  strike_y  strike_z  "
                       "dip_x  dip_y  dip_z\n")
            for i in range(basis_coords.shape[0]):
                fout.write(("%.6f %.6f %.6f  "
                            "%.8f %.8f %.8f  "
                            "%.8f %.8f %.8f\n") %
                           (basis_coords[i, 0], basis_coords[i, 1], basis_coords[i, 2],
                            strike_vec[i, 0], strike_vec[i, 1], strike_vec[i, 2],
                            dip_vec[i, 0],    dip_vec[i, 1],    dip_vec[i, 2]))

    return (mtrue_mu_fun, xf, yf, zf, m, u,
            s_strike_fun, s_dip_fun,
            d_obs, d_cal,
            np.concatenate([m_sx_fault, m_sy_fault]),
            misfitd, grad_m)


# %% [markdown]
# ## DEFINE COMMON PARAMETERS

# %%
# Define order of elements
k = 2
# Define body force
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9

# KEY CHANGE: scalar initial guess (was dl.Constant((0., 0.)) for the 2-component reference)
# In tanh mode m=0 → m_phys = amp_max/2 (midpoint); in linear mode m=0 → m_phys = 0.
m0_amp_expr = dl.Constant(0.)

# %%
# Define the true model PARAMETERS for the FORWARD problem within a homogeneous shear modulus half-space
nu = 0.25   # Poisson's ratio found in Kano et al., 2019

# background shear modulus
mu_b = 0   # 40 GPa
mu_background = mu_expression(mu_b)

# Homogeneous structure (mu_upper = mu_lower; both blocks the same value via K_2LAYER)
mu_l_hom = 0;   mu_lower_hom = mu_expression(mu_l_hom)   # 40 GPa
mu_u_hom = 0;   mu_upper_hom = mu_expression(mu_u_hom)   # 40 GPa
mtrue_mu_expr_hom = K_2LAYER(subdomains, mu_u_hom, mu_l_hom, degree=5)
mu_str_hom = f"_mul{round(mu_expression(mu_l_hom))}u{round(mu_expression(mu_u_hom))}"
print("Homogeneous structure:")
print(f"  mu_upper = {mu_upper_hom:.1f}, mu_lower = {mu_lower_hom:.1f}")

# # %%
# # Heterogeneous structure (slab/wedge contrast: stiff slab below, soft wedge above)
# mu_l_het = 0.9730;    mu_lower_het = mu_expression(mu_l_het)   # ~55 GPa
# mu_u_het = -0.9730;   mu_upper_het = mu_expression(mu_u_het)   # ~25 GPa
# mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u_het, mu_l_het, degree=5)
# mu_str_het = f"_mul{round(mu_expression(mu_l_het))}u{round(mu_expression(mu_u_het))}"
# print("Heterogeneous structure (K_2LAYER):")
# print(f"  mu_upper = {mu_upper_het:.1f}, mu_lower = {mu_lower_het:.1f}")

# %%
# Define the true model PARAMETERS for INVERSE problem
# use the 3D & 1D velocity models to compute shear modulus, call 'process_velocity_models_hull' inside solvers
nu = 0.25
mtrue_mu_expr_het = None
# contrast_factor = 1.0  # amplification factor
# contrast_factor = 4.0  # amplification factor, too extreme, needs clipping, and not adopted since 03/05/2026
contrast_factor = 2.5  # amplification factor, more reasonable, and adopted since 03/05/2026

# String: _hull indicates process_velocity_models_hull method
if contrast_factor == 1.0:
    mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}_hull"
else:
    # mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}_hull"   # ref global mean
    mu_str_het = f"_DeShon3D_ref1D_{round(contrast_factor)}_hull"   # ref 1D value at each depth layer

print( "Heterogeneous structure:")
print( "Converted from 3D & 1D velocity models to shear modulus, mu_str_het = ", mu_str_het)

# %%
# Station targets — locations of surface observations
ntargets = data.shape[0]
targets = np.zeros([ntargets, dim])
targets[:,0] = np.array(data['x'])*1e3   # km to m
targets[:,1] = np.array(data['y'])*1e3
targets[:,2] = np.array(data['z'])*1e3
print(targets.shape)

# %%
# Decide the weights of the horizontal, vertical components
f_h, f_v = 1, 1     # same as in coseismic case
print(f"Data weight horizontal / vertical: {f_h:.2f} / {f_v:.2f}")

# %%
# Define regularization weights
# In a Bayesian inference setting, the ratio rho = sqrt(gamma/delta) plays the role of the
# correlation length in the prior term. For our case, the station separation is around 20 km,
# and the mesh size on the fault is 4-20 km.
rho_s = 2.5e8           # allows variations of slip of the order of ~15 km, close to the maximum resolution
gamma_val_H1 = 3e3
delta_val_L2 = gamma_val_H1 / rho_s

# %%
# Take the inverse for saving the name of the weights
w_h, w_v = int(1/f_h), int(1/f_v)

# %%
# ============================================================
# RUN MODE: set RUN_LCURVE = False for a quick single test run
#           set RUN_LCURVE = True  to run the 3D Het L-curve
# ============================================================
RUN_LCURVE = True    # <--- flip to False for single test run
# RUN_LCURVE = False    # <--- flip to False for single test run

# %%
# Amplitude transform mode:
#   True  → m_phys = amp_max*(tanh(m)+1)/2  (bounded [0, amp_max]; m=0 → amp_max/2)
#   False → m_phys = m                       (unbounded linear; m=0 → 0)
# Outputs from the two modes are kept separate via the filename tag amp_tag.
use_tanh_amp = False
amp_tag = '' if use_tanh_amp else 'u'

import gc

rho_s    = 2.5e8

### gammas for bounded inversion (tanh mode)
# gammas_s = [4e2, 2e2, 1e2]
# gammas_s = [4e2, 8e2, 1e3, 4e3, 8e3]
# gammas_s = [8e2, 1e3, 2e3, 4e3, 8e3]   # HOM-ONLY RUN: large-γ points missing from Hom L-curve
# gammas_s = [3e2]   # HOM-ONLY RUN: large-γ points missing from Hom L-curve
# gammas_s = [3e2, 1e2, 6e1]
# gammas_s = [2e1, 4e1, 6e1, 8e1, 1e2, 1.5e2, 2e2, 3e2, 4e2, 8e2, 1e3]
# gammas_s = [2e1, 4e1, 8e1, 1.5e2, 2e2, 4e2, 8e2, 1e3]

### gammas for unbounded inversion (linear mode)
# gammas_s = [4e2, 8e2, 1e3, 2e3, 4e3, 1e4, 5e4,                # 1st-round (az=45)
#             2e2, 6e2, 3e3, 6e3]                               # extra (az=45)
gammas_s = [4e3, 3e3]   # two-point comparison at az=33.5° bracketing the az=45° L-curve corner

# ============================================================
# SINGLE TEST RUN (RUN_LCURVE = False): 3D μ (Het), large damping
# ============================================================
if not RUN_LCURVE:
    gamma_s_test = 1e3   # smaller than 1e4 (tanh mode: lets amplitude move away from amp_max/2; linear mode: lets it move away from 0)
    delta_s_test = gamma_s_test / rho_s
    # inv_str and mu_str_inv are used as globals inside solveCoseismicInversion_AzSlip
    inv_str    = f"_lockbothaz{azimuth_deg:.0f}{amp_tag}_w{w_h}{w_v}_gs{gamma_s_test:.0e}_ds{delta_s_test:.0e}_TEST"
    mtrue_mu_expr_inv = mtrue_mu_expr_het
    mu_str_inv = mu_str_het
    # Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
    if mtrue_mu_expr_inv is None:  # 3D heterogeneous model via process_velocity_models_hull
        CG_mu_deg = 2
        mu_str_inv = mu_str_inv + f"_CG{CG_mu_deg}"
    else:  # Homogeneous or K_2LAYER
        CG_mu_deg = 1
    print(sep, "SINGLE TEST RUN — 3D μ, Az-constrained scalar amplitude inversion", sep)
    print(f"  azimuth_deg = {azimuth_deg:.1f}°,  amp_max = {amp_max*1e3:.1f} mm/yr,  use_tanh_amp = {use_tanh_amp}")
    print(f"  gamma_s = {gamma_s_test:.1e},  delta_s = {delta_s_test:.1e}  (large damping for fast run)")
    print("Inverse problem identifier:", inv_str)
    print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")

    results_test = solveCoseismicInversion_AzSlip(
        k, targets, m0_amp_expr, mtrue_mu_expr_inv, gamma_s_test, delta_s_test,
        c_strike_fun, c_dip_fun, amp_max,
        CG_mu_deg=CG_mu_deg,
        use_tanh_amp=use_tanh_amp,
        savefiles=True, verbose=True)

    print(f"  data misfit = {float(results_test[-2]):.6e}")
    print(f"  model misfit = {float(results_test[-1]):.6e}")
    del results_test
    gc.collect()
    print("Test run complete. Set RUN_LCURVE = True to run full L-curves.")

# %%
# ============================================================
# L-curve: Het μ model, Az-constrained amplitude
# ============================================================
if RUN_LCURVE:
    # ============================================================
    # L-curve: 3D Het μ, Az-constrained amplitude
    # ============================================================
    print(sep, "3D Het L-curve — Az-constrained scalar amplitude inversion", sep)
    print(f"  azimuth_deg = {azimuth_deg:.1f}°,  amp_max = {amp_max*1e3:.1f} mm/yr,  use_tanh_amp = {use_tanh_amp}")

    dmisfit_3d = []
    mmisfit_3d = []
    outFileName_lc = None   # set inside the loop after mu_str_inv (with _CG2 suffix) is known
    for gamma_s in gammas_s:
        delta_s = gamma_s / rho_s
        inv_str = f"_lockbothaz{azimuth_deg:.0f}{amp_tag}_w{w_h}{w_v}_gs{gamma_s:.0e}_ds{delta_s:.0e}"
        print("Inverse problem identifier:", inv_str)
        print(f"****** Computing solution with gamma_s = {gamma_s:.1e}, "
              f"delta_s = {delta_s:.1e}, and rho_s = {rho_s:.1e} ******")

        mtrue_mu_expr_inv = mtrue_mu_expr_het   # = None → 3D model inside solver
        mu_str_inv = mu_str_het
        # Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
        if mtrue_mu_expr_inv is None:  # 3D heterogeneous model via process_velocity_models_hull
            CG_mu_deg = 2
            mu_str_inv = mu_str_inv + f"_CG{CG_mu_deg}"
        else:  # Homogeneous or K_2LAYER
            CG_mu_deg = 1
        print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")

        # Compute the L-curve filename once (first iteration); same across gammas.
        if outFileName_lc is None:
            outFileName_lc = f"Lcurvelockbothaz{azimuth_deg:.0f}{amp_tag}_rs{rho_s:.0e}_{meshname}{mu_str_inv}.txt"
            print("L-curve file:", outFileName_lc)

        results = solveCoseismicInversion_AzSlip(
            k, targets, m0_amp_expr, mtrue_mu_expr_inv, gamma_s, delta_s,
            c_strike_fun, c_dip_fun, amp_max,
            CG_mu_deg=CG_mu_deg,
            use_tanh_amp=use_tanh_amp,
            savefiles=True, verbose=False)

        misfit_d = float(results[-2])
        misfit_m = float(results[-1])
        dmisfit_3d.append(misfit_d)
        mmisfit_3d.append(misfit_m)
        del results
        gc.collect()

        # Append this gamma's row to the L-curve file immediately (open/write/close
        # per iteration so disk state is always flushed and visible)
        with open(resultpath + outFileName_lc, 'a') as csvoutput:
            csvoutput.write("%.6e %.6e %.1e %.0e\n" % (misfit_d, misfit_m, gamma_s, rho_s))

        try:
            import psutil
            mem_gb = psutil.Process().memory_info().rss / 1024**3
            print(f"  Memory after cleanup: {mem_gb:.2f} GB")
        except ImportError:
            pass

    print("All 3D Het Az Inversion Finished!")
