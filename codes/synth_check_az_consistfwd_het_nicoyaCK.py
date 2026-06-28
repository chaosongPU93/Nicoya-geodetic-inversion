# %% [markdown]
# # Synthetic forward (Az-consistent) — CHECKERBOARD, Het μ
#
# Based on: synth_check_az_consistfwd_nicoyaCK.py (Hom) + synth_stripe_az_consistfwd_het_nicoyaCK.py
#
# PURPOSE: Test the AzSlip-consistent forward scheme to isolate and diagnose
# the light-spot artifacts seen in scalar amplitude recovery plots.
#
# SCHEME: AzSlip-consistent forward
#   True slip: m_phys(x) = amp in locked patches, 0 in creeping (scalar, CG1)
#   Forward:   PDEVarf_AzSlip_Direct  —  slip = m_phys * (c_strike_fun, c_dip_fun)
#              SAME operator as solveCoseismicInversion_AzSlip
#   Inversion: DISABLED (forward only in this script)
#
# CONTRAST WITH ORIGINAL FORWARD:
#   Original: create_fault_local_checkerboard_az → 2-comp vector in GLOBAL frame
#   This:     m_phys * (c_s, c_d) where (c_s, c_d) are LOCAL CG1-averaged coefficients
#   Flat fault → identical; undulations → differ by O(1 - ||c||) ≈ 3-10%
#
# OUTPUTS:
#   caz_diag_<meshname>_az<deg>.txt       per-vertex: xf yf zf c_s c_d c_mag
#   slip_true_cf_<meshname>...txt         per-vertex: xf yf zf m_phys s_strike s_dip slip_mag
#   d_obs_cf_<meshname>...txt             station displacements (consistent forward)
#   d_obs_grid_cf_<meshname>...txt        dense grid displacements
#   u_cf_<meshname>...xdmf               displacement field (ParaView)

# %%
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
from pointwiseStateObs_weights import PointwiseStateObservation as PSBW
from pointwiseStateObs import PointwiseStateObservation as PSB
dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
sep = "\n"+"#"*80+"\n"
print(sep, "START of Az-consistent forward (checkerboard, Het mu)", sep)

# %%
# Define data directory
datadir = "/home/staff/chao/SSEinv/Nicoya/data/"
obs_disp_name = "CKfig6_data_final.csv"
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1,
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car',
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

lon0, lat0 = -84, 7
rot = 45
x_rot, y_rot = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)
x0, y0 = 130e3, 350e3
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3
data['z'] = 0.0
data['vx_Car'], data['vy_Car'] = ut.rot_xy(data['vx_Car'], data['vy_Car'], rot)
print("Number of stations:", len(data))

volc_file = "GVP_Holocene_Volcano_loc.csv"
volc = pd.read_csv(datadir + volc_file, sep=",", skiprows=1,
                   names=['id', 'lat', 'lon', 'elv'])
volc = volc[(volc['lat'] >= 8) & (volc['lat'] <= 12) & (volc['lon'] >= -88) & (volc['lon'] <= -83)]
x_rot, y_rot = ut.LL2ckmd(volc['lon'], volc['lat'], lon0, lat0, rot)
volc['x'], volc['y'] = x_rot - x0, y_rot - y0
volc['z'] = 0.0

# %%
# ============================================================================
# HELPER FUNCTIONS FOR UNEVEN MESH TOP BOUNDARY HANDLING
# ============================================================================
def extract_top_boundary_surface(mesh, boundaries, top_id):
    all_coords = mesh.coordinates()
    top_facets = [facet for facet in dl.facets(mesh) if boundaries[facet] == top_id]
    top_vertex_indices = set()
    for facet in top_facets:
        for vertex in dl.vertices(facet):
            top_vertex_indices.add(vertex.index())
    top_coords = all_coords[list(top_vertex_indices)]
    print(f"Extracted {len(top_coords)} vertices from top boundary")
    print(f"  x range: [{top_coords[:,0].min()/1e3:.1f}, {top_coords[:,0].max()/1e3:.1f}] km")
    print(f"  y range: [{top_coords[:,1].min()/1e3:.1f}, {top_coords[:,1].max()/1e3:.1f}] km")
    print(f"  z range: [{top_coords[:,2].min()/1e3:.1f}, {top_coords[:,2].max()/1e3:.1f}] km")
    return top_coords


def find_valid_surface_depths_raycast(x_query, y_query, mesh, z_start=1000, z_end=-15000,
                                      n_samples=150, verbose=True):
    from scipy.spatial import cKDTree
    tree = dl.BoundingBoxTree()
    tree.build(mesh)
    z_levels = np.linspace(z_start, z_end, n_samples)
    n_pts = len(x_query)
    z_valid = np.zeros(n_pts)
    valid_mask = np.zeros(n_pts, dtype=bool)
    for i in range(n_pts):
        for z in z_levels:
            pt = dl.Point(float(x_query[i]), float(y_query[i]), float(z))
            cell_id = tree.compute_first_entity_collision(pt)
            if cell_id < mesh.num_cells():
                z_valid[i] = z
                valid_mask[i] = True
                break
    n_valid = np.sum(valid_mask)
    n_invalid = n_pts - n_valid
    if verbose:
        print(f"[LEVEL 1 Ray Casting] {n_valid}/{n_pts} valid ({100*n_valid/n_pts:.1f}%)")
        if n_invalid > 0:
            print(f"  {n_invalid} points outside mesh footprint")
    return z_valid, valid_mask


def extrapolate_displacements(targets, d_obs, valid_indices, verbose=True):
    from scipy.interpolate import NearestNDInterpolator
    n_total = targets.shape[0]
    n_valid = len(valid_indices)
    n_invalid = n_total - n_valid
    if verbose and n_invalid > 0:
        print(f"Extrapolating displacements for {n_invalid} points outside mesh...")
    d_full = np.zeros(3 * n_total)
    if n_invalid == 0:
        d_full = d_obs
    else:
        valid_targets = targets[valid_indices, :]
        x_valid = valid_targets[:, 0]; y_valid = valid_targets[:, 1]
        ux_valid = d_obs[0::3]; uy_valid = d_obs[1::3]; uz_valid = d_obs[2::3]
        ext_x = NearestNDInterpolator(list(zip(x_valid, y_valid)), ux_valid)
        ext_y = NearestNDInterpolator(list(zip(x_valid, y_valid)), uy_valid)
        ext_z = NearestNDInterpolator(list(zip(x_valid, y_valid)), uz_valid)
        for i in range(n_total):
            if i in valid_indices:
                j = valid_indices.index(i)
                d_full[3*i] = d_obs[3*j]; d_full[3*i+1] = d_obs[3*j+1]; d_full[3*i+2] = d_obs[3*j+2]
            else:
                x, y = targets[i, 0], targets[i, 1]
                d_full[3*i] = ext_x(x, y); d_full[3*i+1] = ext_y(x, y); d_full[3*i+2] = ext_z(x, y)
    return d_full


# %%
# CREATE DENSE OBSERVATION GRID
print(sep, "Creating dense observation grid", sep)
region = [-87, -84, 8.6, 11.6]
lon_min, lon_max = region[0], region[1]
lat_min, lat_max = region[2], region[3]
grid_spacing_deg = 0.01
lon_grid = np.arange(lon_min, lon_max + grid_spacing_deg, grid_spacing_deg)
lat_grid = np.arange(lat_min, lat_max + grid_spacing_deg, grid_spacing_deg)
LON_GRID, LAT_GRID = np.meshgrid(lon_grid, lat_grid)
lon_2d = LON_GRID.flatten(); lat_2d = LAT_GRID.flatten()
x_rot_2d, y_rot_2d = ut.LL2ckmd(lon_2d, lat_2d, lon0, lat0, rot)
x_2d = (x_rot_2d - x0) / 1e3; y_2d = (y_rot_2d - y0) / 1e3
n_2d = len(x_2d)
lon_dense = lon_2d; lat_dense = lat_2d
x_dense = x_2d; y_dense = y_2d
z_dense = np.zeros(n_2d)
print(f"Dense grid: {len(lon_grid)} x {len(lat_grid)} = {n_2d} points")
dense_data = pd.DataFrame({'lon': lon_dense, 'lat': lat_dense,
                           'x': x_dense, 'y': y_dense, 'z': z_dense})

# %%
resultpath = "/home/staff/chao/SSEinv/Nicoya/syn_slip/"
os.makedirs(resultpath, exist_ok=True)

# %%
# Elastic helper functions
def AEsigma(s, mu, nu):
    A = 1./(2.*mu)*(s - nu/(1+nu*(dim-2))*ufl.tr(s)*ufl.Identity(dim))
    return A

def asym(s):
    if dim == 2:
        as_ = s[1,0] - s[0,1]
    elif dim == 3:
        as_ = ufl.as_vector([s[1,2]-s[2,1], s[2,0]-s[0,2], s[0,1]-s[1,0]])
    return as_

def dir_strike(n):
    z_dir = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    return n_cross_z / ufl.sqrt(ufl.dot(n_cross_z, n_cross_z))

def dir_dip(n):
    return ufl.cross(dir_strike(n), n)

class K_2LAYER(dl.UserExpression):
    def __init__(self, subdomains, k_r, k_l, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains; self.k_r = k_r; self.k_l = k_l
    def eval_cell(self, values, x, cell):
        if self.subdomains[cell.index] == blockright:
            values[0] = self.k_r
        elif self.subdomains[cell.index] == blockleft:
            values[0] = self.k_l
    def value_shape(self):
        return ()

# %%
# Mesh selection
meshname = "nicoyaCKden_une_sm"
# meshname = "nicoyaCKden_une_all"
use_uneven_mesh = "une" in meshname
print(meshname)
print(f"Using uneven mesh: {use_uneven_mesh}")

meshpath = "/home/staff/chao/SSEinv/Nicoya/mesh/"
mesh = dl.Mesh(meshpath + meshname + '.xml')
xmin, xmax = -1000e3, 1000e3
ymin, ymax = -1000e3, 1000e3
zmin, zmax = -400e3, 0.
dim = mesh.topology().dim()
n = dl.FacetNormal(mesh)
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')
top = 1; bottom = 2; west = 3; east = 4; north = 5; south = 6; fault = 7
blockleft = 8; blockright = 9
ds = dl.Measure("ds")(domain=mesh, subdomain_data=boundaries)
dS = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

# %%
# Assign depths to dense grid from mesh top boundary (uneven meshes only)
if use_uneven_mesh:
    print(sep, "Assigning depths from mesh top boundary to observation grid", sep)
    top_coords = extract_top_boundary_surface(mesh, boundaries, top)
    x_query_m = dense_data['x'].values * 1e3
    y_query_m = dense_data['y'].values * 1e3
    z_valid_m, valid_mask = find_valid_surface_depths_raycast(
        x_query_m, y_query_m, mesh, z_start=1000, z_end=-15000, n_samples=150, verbose=True)
    z_valid_km = z_valid_m / 1e3
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        from scipy.interpolate import NearestNDInterpolator
        print(f"\n[LEVEL 1] Extrapolating z-depths for {n_invalid} points outside mesh footprint...")
        x_valid = x_query_m[valid_mask]; y_valid = y_query_m[valid_mask]
        z_from_valid = z_valid_m[valid_mask]
        extrapolator = NearestNDInterpolator(list(zip(x_valid, y_valid)), z_from_valid)
        z_valid_m[~valid_mask] = extrapolator(x_query_m[~valid_mask], y_query_m[~valid_mask])
        z_valid_km = z_valid_m / 1e3
    else:
        print(f"\n[LEVEL 1] All {len(x_query_m)} grid points have mesh coverage.")
    dense_data['z'] = z_valid_km
    print(f"\nFinal grid z range: [{z_valid_km.min():.3f}, {z_valid_km.max():.3f}] km")
else:
    print(sep, "Using z=0 for observation grid (even mesh)", sep)

# %%
def mu_expression(m):
    return 20*(2. + ufl.tanh(m))

# %%
# ============================================================================
# Az COEFFICIENT FUNCTIONS
# ============================================================================
def compute_az_cg1_functions(mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg=45.0):
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
    print(f"  ||c||:    [{mag.min():.4f}, {mag.max():.4f}]  (1.0 = no averaging loss)")
    return c_strike_fun, c_dip_fun


def save_caz_diagnostic(c_strike_fun, c_dip_fun, mesh, boundaries, fault,
                         meshname, azimuth_deg, resultpath):
    """
    Save per-fault-vertex Az coefficient diagnostic file.
    Columns: xf_m  yf_m  zf_m  c_strike  c_dip  c_mag
    c_mag = ||(c_strike, c_dip)|| — how much amplitude is lost at each vertex by CG1 averaging.
    """
    Vm_diag = dl.FunctionSpace(mesh, "CG", 1)
    bc_diag = dl.DirichletBC(Vm_diag, 99.0, boundaries, fault)
    um_diag = dl.Function(Vm_diag); bc_diag.apply(um_diag.vector())
    fault_mask = um_diag.vector().get_local() == 99.0

    xslip = dl.interpolate(dl.Expression("x[0]", degree=1), Vm_diag)
    yslip = dl.interpolate(dl.Expression("x[1]", degree=1), Vm_diag)
    zslip = dl.interpolate(dl.Expression("x[2]", degree=1), Vm_diag)
    xf = xslip.vector().get_local()[fault_mask]
    yf = yslip.vector().get_local()[fault_mask]
    zf = zslip.vector().get_local()[fault_mask]
    cs = c_strike_fun.vector().get_local()[fault_mask]
    cd = c_dip_fun.vector().get_local()[fault_mask]
    cmag = np.sqrt(cs**2 + cd**2)

    outFileName = f"caz_diag_{meshname}_az{azimuth_deg}.txt"
    with open(resultpath + outFileName, 'w') as fout:
        fout.write("# xf_m  yf_m  zf_m  c_strike  c_dip  c_mag\n")
        for i in range(len(xf)):
            fout.write("%.3f %.3f %.3f %.6f %.6f %.6f\n" % (
                xf[i], yf[i], zf[i], cs[i], cd[i], cmag[i]))
    print(f"Saved Az diagnostic: {outFileName}")
    print(f"  ||c|| stats: min={cmag.min():.4f}, mean={cmag.mean():.4f}, max={cmag.max():.4f}")
    print(f"  Amplitude loss: max {(1-cmag.min())*100:.1f}% at worst-undulation vertex")
    return xf, yf, zf, cs, cd, cmag


# %%
# ============================================================================
# PDEVarf_AzSlip_Direct — m IS the physical amplitude (no tanh transform)
# Same slip operator as the inversion, but m = m_phys directly.
# ============================================================================
class PDEVarf_AzSlip_Direct:
    """
    Forward PDE: scalar amplitude m prescribed directly (no tanh wrapping).
    slip = m * (c_strike_fun, c_dip_fun)  at the fault interface.

    Use this for the AzSlip-consistent forward so that the forward and inversion
    operators are identical — eliminates artifacts at fault undulations.
    """
    def __init__(self, mtrue_mu_fun, c_strike_fun, c_dip_fun):
        self.mtrue_mu_fun = mtrue_mu_fun
        self.c_strike_fun = c_strike_fun
        self.c_dip_fun    = c_dip_fun

    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        tau, w, q    = dl.split(p)
        u0 = dl.Constant((0., 0., 0.))
        m_phys = m   # direct physical amplitude — no tanh
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


# %%
# ============================================================================
# HELPER: build scalar CG1 amplitude from a 2-component fault expression
# ============================================================================
def build_scalar_amp_from_vec_expr(mtrue_s_expr, boundaries, fault, mesh):
    """
    Given a 2-component CG1 fault expression (s_strike, s_dip), extract
    the scalar amplitude ||(s_strike, s_dip)|| as a scalar CG1 function.

    Uses vertex_to_dof_map for exact pointwise computation (no L2 projection).
    """
    V_slip_2 = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    bc_tmp = dl.DirichletBC(V_slip_2, mtrue_s_expr, boundaries, fault)
    s_2comp = dl.Function(V_slip_2)
    bc_tmp.apply(s_2comp.vector())

    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)
    amp_fun = dl.Function(Vm_amp)
    s_arr   = s_2comp.vector().get_local()
    v2d_vec = dl.vertex_to_dof_map(V_slip_2)  # length 2*n_verts: [v0c0, v0c1, v1c0, v1c1, ...]
    v2d_sca = dl.vertex_to_dof_map(Vm_amp)    # length n_verts
    a_arr   = amp_fun.vector().get_local()
    n_verts = mesh.num_vertices()
    for vi in range(n_verts):
        dof0 = v2d_vec[2*vi]
        dof1 = v2d_vec[2*vi + 1]
        sx   = s_arr[dof0]
        sy   = s_arr[dof1]
        a_arr[v2d_sca[vi]] = np.sqrt(sx**2 + sy**2)
    amp_fun.vector().set_local(a_arr)
    amp_fun.vector().apply('')

    # Diagnostics
    Vm_diag = dl.FunctionSpace(mesh, "CG", 1)
    bc_diag = dl.DirichletBC(Vm_diag, 99.0, boundaries, fault)
    um_diag = dl.Function(Vm_diag); bc_diag.apply(um_diag.vector())
    fault_mask = um_diag.vector().get_local() == 99.0
    amp_fault = a_arr[fault_mask]
    print(f"  Scalar amplitude from 2-comp expr: [{amp_fault.min():.4f}, {amp_fault.max():.4f}] m")
    print(f"  Non-zero fault vertices: {(amp_fault > 1e-10).sum()} / {len(amp_fault)}")
    return amp_fun


# %%
# ============================================================================
# AZ-CONSISTENT FORWARD: STATION OBSERVATIONS
# ============================================================================
def solveCoseismicForward_AzConsistent(k, targets, mtrue_mu_expr_for,
                                        c_strike_fun, c_dip_fun, amp_fun_true,
                                        pollute=True, pollute_type='uniform',
                                        savefiles=True, verbose=True):
    """
    AzSlip-consistent forward: uses PDEVarf_AzSlip_Direct with scalar amplitude.
    Slip prescription: slip = amp_fun_true * (c_strike_fun, c_dip_fun)
    Same operator as solveCoseismicInversion_AzSlip → no mismatch at undulations.
    """
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)   # scalar parameter space
    Vh = [Vu, Vm_amp, Vu]

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # True parameter: scalar amplitude
    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()
    mtrue.set_local(amp_fun_true.vector().get_local())
    mtrue.apply('')

    if verbose:
        amp_vals = amp_fun_true.vector().get_local()
        print(f"  True amplitude range: [{amp_vals.min():.4f}, {amp_vals.max():.4f}] m")

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    if savefiles:
        print("Saving true shear modulus to .xdmf file")
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
        mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str_for + '.xdmf')
        m_mu_true.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true)

    pde_varf = PDEVarf_AzSlip_Direct(mtrue_mu_fun, c_strike_fun, c_dip_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

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

    idx_d = list(np.nonzero(obs_weights)[0])

    model = hp.Model(pde, hp.BiLaplacianPrior(Vh[hp.PARAMETER], 1., 1e-9, robin_bc=False), misfit)
    u_fwd = pde.generate_state()
    x = [u_fwd, mtrue, None]
    model.solveFwd(u_fwd, x)

    if savefiles:
        u_save = dl.Function(Vh[hp.STATE].sub(1), u_fwd)
        u_save.rename('displacement', 'displacement')
        uid = dl.XDMFFile(resultpath + 'u_cf_' + meshname + slip_str_gt + mu_str_for + '.xdmf')
        uid.write(u_save)

    misfit.B.mult(x[hp.STATE], misfit.Bu)

    if pollute:
        np.random.seed(1)
        if pollute_type == 'uniform':
            noise_h = np.random.normal(0, noise_std_h, targets.shape[0])
            noise_v = np.random.normal(0, noise_std_v, targets.shape[0])
        elif pollute_type == 'datastd':
            noise_h_x = np.random.normal(0, data['vx_std_Car'].to_numpy())
            noise_h_y = np.random.normal(0, data['vy_std_Car'].to_numpy())
            noise_v   = np.random.normal(0, data['vz_std_Car'].to_numpy())
        Bu_arr = misfit.Bu.get_local()
        if pollute_type == 'uniform':
            Bu_arr[9::15]  += noise_h
            Bu_arr[10::15] += noise_h
            Bu_arr[11::15] += noise_v
        elif pollute_type == 'datastd':
            Bu_arr[9::15]  += noise_h_x
            Bu_arr[10::15] += noise_h_y
            Bu_arr[11::15] += noise_v
        misfit.Bu.set_local(Bu_arr); misfit.Bu.apply('')

    d_obs = misfit.Bu[idx_d]

    if savefiles:
        outFileName = 'd_obs_cf_' + meshname + slip_str_gt + mu_str_for + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]))
        csvoutput.close()

    # Fault coordinates (scalar CG1 mask)
    bc_amp = dl.DirichletBC(Vm_amp, 99.0, boundaries, fault)
    um_amp = dl.Function(Vm_amp); bc_amp.apply(um_amp.vector())
    fault_mask_arr = um_amp.vector().get_local() == 99.0

    xslip = dl.interpolate(dl.Expression("x[0]", degree=1), Vm_amp)
    yslip = dl.interpolate(dl.Expression("x[1]", degree=1), Vm_amp)
    zslip = dl.interpolate(dl.Expression("x[2]", degree=1), Vm_amp)
    xf = xslip.vector().get_local()[fault_mask_arr]
    yf = yslip.vector().get_local()[fault_mask_arr]
    zf = zslip.vector().get_local()[fault_mask_arr]

    # True slip at fault: m_phys * (c_s, c_d)
    m_phys_fault = amp_fun_true.vector().get_local()[fault_mask_arr]
    cs_fault = c_strike_fun.vector().get_local()[fault_mask_arr]
    cd_fault = c_dip_fun.vector().get_local()[fault_mask_arr]
    c_mag_fault = np.sqrt(cs_fault**2 + cd_fault**2)
    s_strike_fault = m_phys_fault * cs_fault
    s_dip_fault    = m_phys_fault * cd_fault
    slip_mag_fault  = m_phys_fault * c_mag_fault   # ||slip|| = m_phys * ||c||

    print(f"  True amplitude (locked): [{m_phys_fault[m_phys_fault>0].min():.4f}, {m_phys_fault.max():.4f}] m")
    print(f"  True ||slip|| (= m_phys*||c||): max={slip_mag_fault.max():.4f} m")
    print(f"  Max amplitude loss from ||c||<1: {(m_phys_fault - slip_mag_fault).max()*1e3:.2f} mm/yr")

    if savefiles:
        outFileName = 'slip_true_cf_' + meshname + slip_str_gt + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        csvoutput.write("# xf_m  yf_m  zf_m  m_phys_m  s_strike_m  s_dip_m  slip_mag_m  c_mag\n")
        for i in range(len(xf)):
            csvoutput.write("%.3f %.3f %.3f %.6f %.6f %.6f %.6f %.6f\n" % (
                xf[i], yf[i], zf[i],
                m_phys_fault[i], s_strike_fault[i], s_dip_fault[i],
                slip_mag_fault[i], c_mag_fault[i]))
        csvoutput.close()

    # Seismic moment: mu * ||slip|| * dS = mu * m_phys * c_mag * dS
    c_mag_fun = dl.Function(Vm_amp)
    c_mag_arr = np.sqrt(c_strike_fun.vector().get_local()**2 + c_dip_fun.vector().get_local()**2)
    c_mag_fun.vector().set_local(c_mag_arr); c_mag_fun.vector().apply('')
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    moment_true = dl.assemble(mtrue_mu_fun_expr * GPa2Pa * amp_fun_true * c_mag_fun * dS(fault))
    print(f"True seismic moment (consistent): {moment_true:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment_true)
    print(f"True moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    potency = dl.assemble(amp_fun_true * c_mag_fun * dS(fault))
    print(f"True seismic potency: {potency:.3e} m^3")
    if savefiles:
        with open(resultpath + 'moment_true_cf_' + meshname + slip_str_gt + mu_str_for + '.txt', 'w') as fout:
            fout.write(f"{moment_true:.6e} {M_w3:.4f} {potency:.6e}\n")

    return mtrue_mu, u_fwd, xf, yf, zf, d_obs, m_phys_fault, s_strike_fault, s_dip_fault


# %%
# ============================================================================
# AZ-CONSISTENT FORWARD: DENSE GRID
# ============================================================================
def computeGridDisplacements_AzConsistent(k, targets, mtrue_mu_expr_for,
                                           c_strike_fun, c_dip_fun, amp_fun_true,
                                           pollute=False, pollute_type='uniform',
                                           savefiles=True, verbose=True):
    """
    Dense grid forward using PDEVarf_AzSlip_Direct (AzSlip-consistent).
    Handles points outside mesh via LEVEL 2 nearest-neighbor extrapolation.
    """
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vu, Vm_amp, Vu]

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()
    mtrue.set_local(amp_fun_true.vector().get_local())
    mtrue.apply('')

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    pde_varf = PDEVarf_AzSlip_Direct(mtrue_mu_fun, c_strike_fun, c_dip_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    indicator_vec = dl.interpolate(
        dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()

    # LEVEL 2: identify valid targets inside mesh
    bb_tree = dl.BoundingBoxTree()
    bb_tree.build(mesh)
    valid_indices = []
    for i, pt_coords in enumerate(targets):
        pt = dl.Point(float(pt_coords[0]), float(pt_coords[1]), float(pt_coords[2]))
        cell_id = bb_tree.compute_first_entity_collision(pt)
        if cell_id < mesh.num_cells():
            valid_indices.append(i)

    n_valid = len(valid_indices)
    n_total = targets.shape[0]
    n_invalid = n_total - n_valid
    if verbose:
        print(f"[LEVEL 2] {n_valid}/{n_total} points inside mesh ({100*n_valid/n_total:.1f}%)")
        if n_invalid > 0:
            print(f"  {n_invalid} points will be extrapolated")

    valid_targets = targets[valid_indices, :]
    misfit = PSB(Vh[hp.STATE], valid_targets, indicator_vec=indicator_vec)
    misfit.noise_variance = 1.

    model = hp.Model(pde, hp.BiLaplacianPrior(Vh[hp.PARAMETER], 1., 1e-9, robin_bc=False), misfit)
    u_fwd = pde.generate_state()
    x = [u_fwd, mtrue, None]
    model.solveFwd(u_fwd, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)

    obs_weights = np.zeros(valid_targets.shape[0]*15)
    obs_weights[9::15] = 1; obs_weights[10::15] = 1; obs_weights[11::15] = 1
    idx_d = list(np.nonzero(obs_weights)[0])
    d_obs_valid = misfit.Bu[idx_d]

    d_obs_full = extrapolate_displacements(targets, d_obs_valid, valid_indices, verbose=verbose)

    if savefiles:
        outFileName = 'd_obs_grid_cf_' + meshname + slip_str_gt + mu_str_for + str(grid_spacing_deg) + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_obs_full[3*i], d_obs_full[3*i+1], d_obs_full[3*i+2]))
        csvoutput.close()

    return d_obs_full, valid_indices


# %%
# ============================================================================
# PDEVarf_AzSlip — tanh-parameterized scalar amplitude (used in inversion)
# Identical slip BC to PDEVarf_AzSlip_Direct, but m_phys = amp_max*(tanh(m)+1)/2
# ============================================================================
class PDEVarf_AzSlip:
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


# %%
# ============================================================================
# INVERSION: Az-constrained scalar amplitude
# Reads consistent-forward synthetic data (d_obs_cf_*).
# Output files tagged with _cf_ in inv_str.
# ============================================================================
def solveCoseismicInversion_AzSlip(k, targets, m0_amp_expr, mtrue_mu_expr_inv,
                                   gamma_val_H1, delta_val_L2,
                                   c_strike_fun, c_dip_fun, amp_max,
                                   pollute=True, pollute_type='uniform',
                                   savefiles=True, verbose=True,
                                   true_amp_arr=None, amp_ref=None):
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm_amp = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vu, Vm_amp, Vu]

    if verbose:
        ndofs = [Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()]
        print(sep, "Az scalar amplitude inversion (consistent forward)", sep)
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(*ndofs))
        print(f"Amplitude bound: [0, {amp_max*1e3:.1f} mm/yr]")

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    m0_amp = dl.interpolate(m0_amp_expr, Vh[hp.PARAMETER]).vector()

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    pde_varf = PDEVarf_AzSlip(mtrue_mu_fun, c_strike_fun, c_dip_fun, amp_max)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

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

    # Load consistent-forward synthetic data (global syndata)
    tmp = np.zeros(len(misfit.d))
    tmp[9::15]  = np.array(syndata['ux'])
    tmp[10::15] = np.array(syndata['uy'])
    tmp[11::15] = np.array(syndata['uz'])
    misfit.d.set_local(tmp); misfit.d.apply('')

    idx_d = list(np.nonzero(obs_weights)[0])
    d_obs = misfit.d[idx_d]

    # Fault coordinates
    bc_amp = dl.DirichletBC(Vm_amp, 99.0, boundaries, fault)
    um_amp = dl.Function(Vm_amp); bc_amp.apply(um_amp.vector())
    fault_mask_arr = um_amp.vector().get_local() == 99.0
    xslip = dl.interpolate(dl.Expression("x[0]", degree=1), Vm_amp)
    yslip = dl.interpolate(dl.Expression("x[1]", degree=1), Vm_amp)
    zslip = dl.interpolate(dl.Expression("x[2]", degree=1), Vm_amp)
    xf = xslip.vector().get_local()[fault_mask_arr]
    yf = yslip.vector().get_local()[fault_mask_arr]
    zf = zslip.vector().get_local()[fault_mask_arr]

    reg = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma_val_H1, delta_val_L2, robin_bc=False)
    model = hp.Model(pde, reg, misfit)
    m = m0_amp.copy()

    u = model.generate_vector(hp.STATE)
    p = model.generate_vector(hp.ADJOINT)
    x = [u, m, p]
    mg = model.generate_vector(hp.PARAMETER)
    model.solveFwd(u, x)
    model.solveAdj(p, x)
    model.evalGradientParameter(x, mg)

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

    # Post-processing
    m_fun = hp.vector2Function(m, Vm_amp)
    m_phys_fun = dl.project(amp_max * (ufl.tanh(m_fun) + 1) / 2, Vm_amp)
    s_strike_fun_inv = dl.project(m_phys_fun * c_strike_fun, Vm_amp)
    s_dip_fun_inv    = dl.project(m_phys_fun * c_dip_fun,    Vm_amp)

    if savefiles:
        amp_id = dl.XDMFFile(resultpath + 'slip_amp_' + meshname + slip_str_gt
                             + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        m_phys_fun.rename('slip amplitude', 'slip amplitude'); amp_id.write(m_phys_fun)
        amp_id.close()

    # Seismic moment of inverted solution: mu * m_phys * ||c|| * dS
    c_mag_arr_inv = np.sqrt(c_strike_fun.vector().get_local()**2 + c_dip_fun.vector().get_local()**2)
    c_mag_fun_inv = dl.Function(Vm_amp)
    c_mag_fun_inv.vector().set_local(c_mag_arr_inv); c_mag_fun_inv.vector().apply('')
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    moment_inv = dl.assemble(mtrue_mu_fun_expr * GPa2Pa * m_phys_fun * c_mag_fun_inv * dS(fault))
    print(f"Inverted seismic moment: {moment_inv:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment_inv)
    print(f"Inverted moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    potency_inv = dl.assemble(m_phys_fun * c_mag_fun_inv * dS(fault))
    print(f"Inverted seismic potency: {potency_inv:.3e} m^3")
    if savefiles:
        outFileName = 'moment_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        with open(resultpath + outFileName, 'w') as fout:
            fout.write(f"{moment_inv:.6e} {M_w3:.4f} {potency_inv:.6e}\n")

    # Forward solve with inverted m → predicted data
    x = [u, m, p]
    model.solveFwd(u, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)
    d_cal = misfit.Bu[idx_d]

    if savefiles:
        print("Saving predicted displacement and stress to .xdmf file")
        u_save = dl.Function(Vh[hp.STATE].sub(1), u)
        u_save.rename('displacement', 'displacement')
        uid = dl.XDMFFile(resultpath + 'u_predicted_' + meshname + slip_str_gt
                          + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        uid.write(u_save)
        uid.close()
        sigma_non = dl.Function(Vh[hp.STATE].sub(0), u)
        sigma_save = sigma_non.copy()
        sigma_save.vector()[:] = sigma_non.vector()[:] * GPa2Pa
        sigma_save.rename('stress', 'stress')
        sid = dl.XDMFFile(resultpath + 'stress_predicted_' + meshname + slip_str_gt
                          + mu_str_for + inv_str + mu_str_inv + '.xdmf')
        sid.write(sigma_save)
        sid.close()
        print("Finish saving predicted displacement and stress")

    m_fun2 = dl.Function(Vm_amp, m)
    grad_m   = dl.assemble(ufl.inner(ufl.avg(ufl.nabla_grad(m_fun2)),
                                     ufl.avg(ufl.nabla_grad(m_fun2)))*dS(fault))
    misfitd  = np.linalg.norm((d_cal - d_obs), 2)
    print("Data misfit {0:.6e}; Model misfit {1:.6e}".format(misfitd, grad_m))

    if savefiles:
        outFileName = 'd_cal_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_cal[3*i], d_cal[3*i+1], d_cal[3*i+2]))
        csvoutput.close()

    m_amp_arr  = m_phys_fun.vector().get_local()
    m_amp_fault = m_amp_arr[fault_mask_arr]
    cs_fault = c_strike_fun.vector().get_local()[fault_mask_arr]
    cd_fault = c_dip_fun.vector().get_local()[fault_mask_arr]
    m_sx_fault = m_amp_fault * cs_fault
    m_sy_fault = m_amp_fault * cd_fault

    if savefiles:
        outFileName = 'm_s_fault_' + meshname + slip_str_gt + mu_str_for + inv_str + mu_str_inv + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(m_sx_fault.shape[0]):
            csvoutput.write("%.6f %.6f\n" % (m_sx_fault[i], m_sy_fault[i]))
        csvoutput.close()

    # Slip recovery (only when ground truth is provided — synthetic tests)
    if savefiles and true_amp_arr is not None:
        amp_nom = amp_ref if amp_ref is not None else true_amp_arr.max()
        recovery_global = m_amp_fault / amp_nom
        eps = 0.01 * amp_nom
        valid = true_amp_arr > eps
        recovery_local = np.where(valid, m_amp_fault / np.where(valid, true_amp_arr, 1.0), np.nan)
        print(f"  Recovery (global): [{recovery_global.min():.3f}, {recovery_global.max():.3f}]")
        print(f"  Recovery (local):  [{np.nanmin(recovery_local):.3f}, {np.nanmax(recovery_local):.3f}]")
        outFileName = ('slip_recovery_' + meshname + slip_str_gt
                       + mu_str_for + inv_str + mu_str_inv + '.txt')
        with open(resultpath + outFileName, 'w') as csvoutput:
            csvoutput.write("# xf_m  yf_m  zf_m  true_amp_m  recovered_amp_m  ratio_global  ratio_local\n")
            for i in range(len(xf)):
                csvoutput.write("%.3f %.3f %.3f %.6f %.6f %.6f %.6f\n" % (
                    xf[i], yf[i], zf[i],
                    true_amp_arr[i], m_amp_fault[i],
                    recovery_global[i], recovery_local[i]))
        print("Recovery file saved:", outFileName)

    return xf, yf, zf, m, u, m_phys_fun, d_obs, d_cal, m_amp_fault, misfitd, grad_m


# %%
# ============================================================================
# COMMON PARAMETERS
# ============================================================================
k = 2
f = dl.Constant((0., 0., 0.))
GPa2Pa = 1e9
nu = 0.25
m0_amp_expr = dl.Constant(0.)

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

mu_l_het = 0.9730;  mu_lower_het = mu_expr_fn(mu_l_het)
mu_u_het = -0.9730; mu_upper_het = mu_expr_fn(mu_u_het)
mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u_het, mu_l_het, degree=5)
mu_str_het = f"_mul{round(mu_expr_fn(mu_l_het))}u{round(mu_expr_fn(mu_u_het))}"
print("Heterogeneous structure:")
print(f"  mu_upper = {mu_upper_het:.1f}, mu_lower = {mu_lower_het:.1f}")

# %%
# Az configuration
azimuth_deg       = 33.5   # CW from North; N33.5E
mesh_rotation_deg = rot   # CCW rotation used to build the Nicoya mesh (matches `rot` defined above)

V_norm       = 78.5   # mm/yr, trench-normal (Cocos-Caribbean)
V_para       = 27     # mm/yr, trench-parallel
V_para_const = 11     # mm/yr, correction for N33.5E azimuth
if azimuth_deg == 45.0:
    amp = V_norm
elif azimuth_deg == 26.0:
    amp = round(np.sqrt(V_norm**2 + V_para**2))
elif azimuth_deg == 33.5:
    amp = round(np.sqrt(V_norm**2 + (V_para - V_para_const)**2))
amp     = amp / 1e3   # mm/yr → m/yr
amp_max = amp

print(sep + "Az-constrained slip setup" + sep)
print(f"  Plate convergence azimuth: {azimuth_deg:.1f}° CW from North")
print(f"  Amplitude bound: [0, {amp_max*1e3:.1f} mm/yr]")

print("Computing Az coefficient CG1 functions ...")
c_strike_fun, c_dip_fun = compute_az_cg1_functions(
    mesh, boundaries, fault, azimuth_deg, mesh_rotation_deg)

# %%
# ============================================================================
# SLIP PRESCRIPTION  (same as reference — checker or stripe, Az-directed)
# ============================================================================
slip_pattern = "checker"
# slip_pattern = "stripe"

pattern_option = 1  # 1: along-strike/dip; 2: along N-E
slip_azimuth_deg = azimuth_deg

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
        mesh_rotation_deg=rot, degree=5)
    slip_str_gt = (f"_check_x{x0_p/1e3:g}_y{y0_p/1e3:g}_dx{dx/1e3:g}_dy{dy/1e3:g}"
                   f"_rot{pattern_rot_deg:g}_ms{amp:g}_az{slip_azimuth_deg:g}")

elif slip_pattern == "stripe":
    print("Creating fault-local stripes (Az)...")
    if pattern_option == 1:
        x_len = 80e3; y_len = 300e3; dx = 35e3
        stripe_spacing = x_len + dx; x0_p = 0; y0_p = -45e3; pattern_rot_deg = 0.0
    elif pattern_option == 2:
        x_len = 40e3; y_len = 300e3; dx = 100e3
        stripe_spacing = x_len + dx; x0_p = 0; y0_p = 12.5e3; pattern_rot_deg = 0.0
    mtrue_s_expr_gt = ut.create_fault_local_stripes_az(
        mesh=mesh, boundaries=boundaries, fault_id=fault,
        amp=amp, stripe_width=x_len, stripe_spacing=stripe_spacing, stripe_length=y_len,
        x0=x0_p, y0=y0_p, rotation_deg=pattern_rot_deg,
        azimuth_deg=slip_azimuth_deg, mesh_rotation_deg=rot, degree=5)
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
targets_grid = np.zeros([ntargets_dense, 3])
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
# Forward shear modulus — Het
mtrue_mu_expr_for = mtrue_mu_expr_het
mu_str_for = mu_str_het
print("Forward problem based on: Het mu")

# %%
# Noise configuration
pollute = True
pollute_type = 'uniform'
print(pollute, pollute_type)

if pollute:
    if pollute_type == 'uniform':
        noise_std_h = 0.5 * (data['vx_std_Car'].mean() + data['vy_std_Car'].mean())
        noise_std_v = data['vz_std_Car'].mean()
        print("Average horizontal 1-sigma obs error: %.6f" % noise_std_h)
        print("Average vertical 1-sigma obs error: %.6f" % noise_std_v)
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
# ============================================================================
# SAVE Az COEFFICIENT DIAGNOSTIC
# ============================================================================
print(sep, "Saving Az coefficient diagnostic (c_strike, c_dip, ||c||) at fault vertices", sep)
xf_diag, yf_diag, zf_diag, cs_diag, cd_diag, cmag_diag = save_caz_diagnostic(
    c_strike_fun, c_dip_fun, mesh, boundaries, fault,
    meshname, azimuth_deg, resultpath)

# %%
# ============================================================================
# BUILD SCALAR AMPLITUDE FUNCTION FROM CHECKER EXPRESSION
# ============================================================================
print(sep, "Building scalar amplitude function from 2-component checker expression", sep)
amp_fun_true = build_scalar_amp_from_vec_expr(mtrue_s_expr_gt, boundaries, fault, mesh)

# # %%
# # ============================================================================
# # AZ-CONSISTENT FORWARD: STATION OBSERVATIONS
# # ============================================================================
# print(sep, "AzSlip-consistent forward — stations", sep)
# results_fwd = solveCoseismicForward_AzConsistent(
#     k, targets, mtrue_mu_expr_for,
#     c_strike_fun, c_dip_fun, amp_fun_true,
#     pollute=pollute, pollute_type=pollute_type,
#     savefiles=True, verbose=True)
# mtrue_mu, u_fwd, xf, yf, zf, d_obs, m_phys_fault, s_strike_fault, s_dip_fault = results_fwd
# print("Station forward done!")

# # %%
# # ============================================================================
# # AZ-CONSISTENT FORWARD: DENSE GRID
# # ============================================================================
# print(sep, "AzSlip-consistent forward — dense grid", sep)
# d_obs_grid, valid_grid_indices = computeGridDisplacements_AzConsistent(
#     k, targets_grid, mtrue_mu_expr_for,
#     c_strike_fun, c_dip_fun, amp_fun_true,
#     pollute=False, pollute_type=pollute_type,
#     savefiles=True, verbose=True)
# n_total_grid = targets_grid.shape[0]
# n_valid_grid = len(valid_grid_indices)
# print(f"Dense grid: {n_valid_grid}/{n_total_grid} valid ({100*n_valid_grid/n_total_grid:.1f}%)")
# print("Grid forward done!")

# %%
print(sep, "All forward computations complete.", sep)
print("Files saved to:", resultpath)
print(f"  caz_diag_{meshname}_az{azimuth_deg}.txt")
print(f"  slip_true_cf_{meshname}{slip_str_gt}.txt")
print(f"  d_obs_cf_{meshname}{slip_str_gt}{mu_str_for}.txt")
print(f"  d_obs_grid_cf_{meshname}{slip_str_gt}{mu_str_for}{grid_spacing_deg}.txt")
print(f"  u_cf_{meshname}{slip_str_gt}{mu_str_for}.xdmf")

# %%
# ============================================================================
# LOAD CONSISTENT-FORWARD SYNTHETIC DATA
# ============================================================================
outFileName = 'd_obs_cf_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x','y','z','ux','uy','uz'])
print("Loaded syndata:", outFileName, "—", len(syndata), "stations")

# %%
# Regularization / initial guess
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
# Mode switch: False = single run, True = L-curve
# ============================================================
RUN_LCURVE = True
# RUN_LCURVE = False

# %%
# ============================================================
# SINGLE TEST RUN  (ACTIVE when RUN_LCURVE = False)
# ============================================================
if not RUN_LCURVE:
    gamma_s = 5e1
    delta_s = gamma_s / rho_s
    mu_str_inv = mu_str_het
    inv_str = f"_cf_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    print(sep, "SINGLE TEST RUN — Az scalar amplitude inversion (consistent forward, Het mu)", sep)
    print(f"  gamma_s={gamma_s:.1e}  delta_s={delta_s:.1e}  rho_s={rho_s:.1e}")
    print("  inv_str:", inv_str)

    # Load ground truth amplitude for recovery computation
    slip_cf_data = np.loadtxt(resultpath + 'slip_true_cf_' + meshname + slip_str_gt + '.txt',
                              comments='#')
    true_amp_arr = slip_cf_data[:, 3]   # m_phys column (m/yr)

    results = solveCoseismicInversion_AzSlip(
        k, targets, m0_amp_expr, mtrue_mu_expr_het,
        gamma_s, delta_s, c_strike_fun, c_dip_fun, amp_max,
        pollute=pollute, pollute_type=pollute_type,
        savefiles=True, verbose=True,
        true_amp_arr=true_amp_arr, amp_ref=amp)
    
    del results; gc.collect()

    # Hom inversion (mismatched: forward=het, inversion=hom)
    mu_str_inv = mu_str_hom
    inv_str = f"_cf_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
    print(sep, "SINGLE TEST RUN — hom inversion (mismatched model)", sep)
    results = solveCoseismicInversion_AzSlip(
        k, targets, m0_amp_expr, mtrue_mu_expr_hom,
        gamma_s, delta_s, c_strike_fun, c_dip_fun, amp_max,
        pollute=pollute, pollute_type=pollute_type,
        savefiles=True, verbose=True,
        true_amp_arr=true_amp_arr, amp_ref=amp)
    

# %%
# ============================================================
# L-CURVE  (ACTIVE when RUN_LCURVE = True)
# ============================================================
if RUN_LCURVE:
    # Priority gammas (around the expected L-curve corner ~5e1)
    # gammas_s_priority = [2e1, 3e1, 4e1, 5e1, 6e1, 8e1]
    gammas_s_priority = [4e1, 5e1]
    gammas_s_priority1 = [5e1]
    # Remaining gammas (needed for full L-curve plot)
    # gammas_s_rest     = [1e1, 7e1, 1e2, 3e2, 5e2, 8e2, 1e3]

    outFileName_het = (f"Lcurvesynscaamp_cf_azbd_rs{rho_s:.0e}_{meshname}"
                       f"_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_het}.txt")
    outFileName_hom = (f"Lcurvesynscaamp_cf_azbd_rs{rho_s:.0e}_{meshname}"
                       f"_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_hom}.txt")

    # Load ground truth amplitude once for recovery computation at each gamma
    slip_cf_data = np.loadtxt(resultpath + 'slip_true_cf_' + meshname + slip_str_gt + '.txt',
                              comments='#')
    true_amp_arr = slip_cf_data[:, 3]   # m_phys column (m/yr)

    # Passes: (label, gammas, outFileName, mtrue_mu_expr, mu_str_inv_val)
    # Order: Het priority → Hom priority → Het rest → Hom rest
    _passes = [
        ("Het L-curve (priority)", gammas_s_priority1, outFileName_het, mtrue_mu_expr_het, mu_str_het),
        ("Hom L-curve (priority)", gammas_s_priority, outFileName_hom, mtrue_mu_expr_hom, mu_str_hom),
        # ("Het L-curve (rest)",     gammas_s_rest,     outFileName_het, mtrue_mu_expr_het, mu_str_het),
        # ("Hom L-curve (rest)",     gammas_s_rest,     outFileName_hom, mtrue_mu_expr_hom, mu_str_hom),
    ]

    for _label, _gammas, _outFile, _mtrue_mu_expr, _mu_str_inv in _passes:
        print(sep, f"{_label} — Az scalar amplitude inversion (consistent forward)", sep)
        # csvoutput = open(resultpath + _outFile, 'a')
        for gamma_s in _gammas:
            delta_s = gamma_s / rho_s
            inv_str = f"_cf_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s:.1e}_ds{delta_s:.1e}"
            mu_str_inv = _mu_str_inv
            print(f"****** gamma_s={gamma_s:.1e}, delta_s={delta_s:.1e} ******")
            results = solveCoseismicInversion_AzSlip(
                k, targets, m0_amp_expr, _mtrue_mu_expr,
                gamma_s, delta_s, c_strike_fun, c_dip_fun, amp_max,
                pollute=pollute, pollute_type=pollute_type,
                savefiles=True, verbose=True,
                true_amp_arr=true_amp_arr, amp_ref=amp)
            misfitd = float(results[-2]); grad_m = float(results[-1])
            # csvoutput.write("%.6e %.6e %.1e %.0e\n" % (misfitd, grad_m, gamma_s, rho_s))
            # csvoutput.flush()
            del results; gc.collect()
        # csvoutput.close()
        print(f"{_label} finished!")
