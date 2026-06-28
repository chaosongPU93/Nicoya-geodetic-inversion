# %% [markdown]
# # Synthetic recovery test — Az-constrained SCALAR amplitude inversion (HET forward)
#
# Based on synth_stripeslip_az_inv_hetmu_uneven_nicoyaCK_lock_noi.py (reference).
# Counterpart to synth_stripeslip_az_scalaramp_inv_hetmu_uneven_nicoyaCK_lock_noi2.py
# (which uses HOM forward). This script uses HETEROGENEOUS forward mu (K_2LAYER, 25/55 GPa).
#
# KEY DIFFERENCE — inversion uses solveCoseismicInversion_AzSlip (Option A):
#   - True slip:   Az-directed (FaultLocalStripesAz/CheckerboardAz), same as reference
#   - Forward:     HET mu (K_2LAYER: upper=25 GPa, lower=55 GPa) — skipped, files from _noi.py
#   - Inversion:   SCALAR amplitude a(x) ∈ [0, amp_max] via tanh; direction fixed to Az
#                  Run with both Het mu and Hom mu to quantify mu-model effect
#   - Assessment:  amplitude recovery = recovered_amp / true_amp  at each fault vertex
#
# SlipTransformation / PDEVarf_TanhSlip / solveCoseismicInversion_TanhSlip removed.
# compute_az_cg1_functions / PDEVarf_AzSlip / solveCoseismicInversion_AzSlip added.

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
print(sep, "START of Az scalar-amplitude synth recovery test", sep)

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
# HELPER FUNCTIONS FOR UNEVEN MESH TOP BOUNDARY HANDLING  (identical to reference)
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
# Elastic helper functions  (identical to reference)
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
# Mesh selection — identical to reference
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
# FORWARD FUNCTIONS  (identical to reference — no changes)
# ============================================================================
def solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=True,
                          pollute_type='uniform', savefiles=True, verbose=True):
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    V_slip = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    if mtrue_s_expr is None:
        slip = dl.Function(V_slip)
        slip.vector()[:] = slip_gt[:]
        mtrue_s = slip.vector()
    else:
        bc_fault_slip = dl.DirichletBC(V_slip, mtrue_s_expr, boundaries, fault)
        mtrue_s_function = dl.Function(V_slip)
        bc_fault_slip.apply(mtrue_s_function.vector())
        mtrue_s = mtrue_s_function.vector()

    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()
    tmp = mtrue_s.copy()[:]
    mtrue.set_local(tmp)
    mtrue.apply('')

    if verbose:
        ut.validate_fault_slip_pattern(mtrue_s=mtrue_s, mesh=mesh,
                                       boundaries=boundaries, fault_id=fault)

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    if savefiles:
        print("Saving true shear modulus structure to .xdmf file")
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
        mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str_for + '.xdmf')
        m_mu_true.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true)

    class PDEVarf_Fwd:
        def __init__(self, mtrue_mu_fun):
            self.mtrue_mu_fun = mtrue_mu_fun
        def __call__(self, u, m, p):
            sigma, uu, r = dl.split(u)
            tau, w, q    = dl.split(p)
            u0 = dl.Constant((0., 0., 0.))
            mu = mu_expression(self.mtrue_mu_fun)
            J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
              + ufl.inner(ufl.div(tau), uu)*ufl.dx \
              + ufl.inner(asym(tau), r)*ufl.dx \
              + ufl.inner(ufl.div(sigma), w)*ufl.dx \
              + ufl.inner(asym(sigma), q)*ufl.dx \
              + ufl.inner(f, w)*dl.dx \
              - ufl.inner(u0, tau*n)*ds(bottom) \
              - ufl.inner(dir_strike(n('+'))*ufl.avg(m[0]) + dir_dip(n('+'))*ufl.avg(m[1]),
                          tau('+')*n('+'))*dS(fault)
            return J

    pde_varf = PDEVarf_Fwd(mtrue_mu_fun)
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
    u_mtrue = pde.generate_state()
    x = [u_mtrue, mtrue, None]
    model.solveFwd(u_mtrue, x)

    if savefiles:
        u_save = dl.Function(Vh[hp.STATE].sub(1), u_mtrue)
        u_save.rename('displacement', 'displacement')
        uid = dl.XDMFFile(resultpath + 'u_' + meshname + slip_str_gt + mu_str_for + '.xdmf')
        uid.write(u_save)

    misfit.B.mult(x[hp.STATE], misfit.Bu)

    # Add noise and save synthetic data
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
        outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]))
        csvoutput.close()

    # Fault coordinates
    CG_s = dl.VectorFunctionSpace(mesh, "CG", degree=1)
    bc1 = dl.DirichletBC(CG_s, (10, 10, 10), boundaries, fault)
    um = dl.Function(CG_s); bc1.apply(um.vector())
    xslip = dl.interpolate(dl.Expression(("x[0]","x[0]","x[0]"), degree=5), CG_s)
    yslip = dl.interpolate(dl.Expression(("x[1]","x[1]","x[1]"), degree=5), CG_s)
    zslip = dl.interpolate(dl.Expression(("x[2]","x[2]","x[2]"), degree=5), CG_s)
    xf = xslip.vector()[um.vector() == 10]
    yf = yslip.vector()[um.vector() == 10]
    zf = zslip.vector()[um.vector() == 10]

    # True slip at fault
    bc2 = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2 = dl.Function(Vm); bc2.apply(um2.vector())
    mtrue_s_fault = mtrue_s[um2.vector() == 99]
    mtrue_sx_fault = mtrue_s_fault[0::2]
    mtrue_sy_fault = mtrue_s_fault[1::2]
    print(f"  True strike range: [{mtrue_sx_fault.min():.6f}, {mtrue_sx_fault.max():.6f}]")
    print(f"  True dip   range: [{mtrue_sy_fault.min():.6f}, {mtrue_sy_fault.max():.6f}]")

    if savefiles:
        outFileName = 'mtrue_s_fault_' + meshname + slip_str_gt + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(mtrue_sx_fault.shape[0]):
            csvoutput.write("%.6f %.6f\n" % (mtrue_sx_fault[i], mtrue_sy_fault[i]))
        csvoutput.close()

    # Seismic moment of true model
    mtrue_fun = dl.Function(Vh[hp.PARAMETER], mtrue)
    s_mag_true = ufl.sqrt(ufl.dot(mtrue_fun, mtrue_fun))
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
    moment_true = dl.assemble(m_mu_true * GPa2Pa * s_mag_true * dS(fault))
    print(f"True scalar seismic moment: {moment_true:.3e} N·m")
    M_w1, M_w2, M_w3 = ut.moment2mag(moment_true)
    print(f"True moment magnitude: {M_w1:.2f}; {M_w2:.2f}; {M_w3:.2f}")
    potency_true = dl.assemble(s_mag_true * dS(fault))
    print(f"True seismic potency: {potency_true:.3e} m^3")
    if savefiles:
        outFileName = 'moment_true_' + meshname + slip_str_gt + mu_str_for + '.txt'
        with open(resultpath + outFileName, 'w') as fout:
            fout.write(f"{moment_true:.6e} {M_w3:.4f} {potency_true:.6e}\n")

    return mtrue, mtrue_mu, u_mtrue, xf, yf, zf, d_obs, mtrue_s_fault


def computeGridDisplacements(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=False,
                             pollute_type='uniform', savefiles=True, verbose=True):
    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME_element = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu = dl.FunctionSpace(mesh, ME_element)
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh = [Vu, Vm, Vu]

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    V_slip = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    if mtrue_s_expr is None:
        slip = dl.Function(V_slip); slip.vector()[:] = slip_gt[:]
        mtrue_s = slip.vector()
    else:
        bc_fault_slip = dl.DirichletBC(V_slip, mtrue_s_expr, boundaries, fault)
        mtrue_s_function = dl.Function(V_slip)
        bc_fault_slip.apply(mtrue_s_function.vector())
        mtrue_s = mtrue_s_function.vector()

    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()
    mtrue.set_local(mtrue_s.copy()[:])
    mtrue.apply('')

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)

    class PDEVarf_Fwd2:
        def __init__(self, mtrue_mu_fun):
            self.mtrue_mu_fun = mtrue_mu_fun
        def __call__(self, u, m, p):
            sigma, uu, r = dl.split(u)
            tau, w, q    = dl.split(p)
            u0 = dl.Constant((0., 0., 0.))
            mu = mu_expression(self.mtrue_mu_fun)
            J = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
              + ufl.inner(ufl.div(tau), uu)*ufl.dx \
              + ufl.inner(asym(tau), r)*ufl.dx \
              + ufl.inner(ufl.div(sigma), w)*ufl.dx \
              + ufl.inner(asym(sigma), q)*ufl.dx \
              + ufl.inner(f, w)*dl.dx \
              - ufl.inner(u0, tau*n)*ds(bottom) \
              - ufl.inner(dir_strike(n('+'))*ufl.avg(m[0]) + dir_dip(n('+'))*ufl.avg(m[1]),
                          tau('+')*n('+'))*dS(fault)
            return J

    pde_varf = PDEVarf_Fwd2(mtrue_mu_fun)
    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    type_solver = "mumps"
    pde.solver = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_fwd_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)
    pde.solver_adj_inc = hp.PETScLUSolver(mesh.mpi_comm(), type_solver)

    indicator_vec = dl.interpolate(
        dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()

    # Build bounding box tree for LEVEL 2 check
    bb_tree = dl.BoundingBoxTree()
    bb_tree.build(mesh)

    # Identify which targets are inside the mesh
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
    u_mtrue = pde.generate_state()
    x = [u_mtrue, mtrue, None]
    model.solveFwd(u_mtrue, x)
    misfit.B.mult(x[hp.STATE], misfit.Bu)

    obs_weights = np.zeros(valid_targets.shape[0]*15)
    obs_weights[9::15] = 1; obs_weights[10::15] = 1; obs_weights[11::15] = 1
    idx_d = list(np.nonzero(obs_weights)[0])
    d_obs_valid = misfit.Bu[idx_d]

    # LEVEL 2 extrapolation
    d_obs_full = extrapolate_displacements(targets, d_obs_valid, valid_indices, verbose=verbose)

    if savefiles:
        outFileName = 'd_obs_grid' + meshname + slip_str_gt + mu_str_for + str(grid_spacing_deg) + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                targets[i,0], targets[i,1], targets[i,2],
                d_obs_full[3*i], d_obs_full[3*i+1], d_obs_full[3*i+2]))
        csvoutput.close()

    return d_obs_full, valid_indices


# %%
# ============================================================================
# NEW: Az-constrained scalar amplitude inversion functions
# ============================================================================

def compute_az_cg1_functions(mesh, boundaries, fault_id, azimuth_deg, mesh_rotation_deg=45.0):
    """
    Project per-facet Az coefficients (c_strike, c_dip) to CG1 vertex functions
    by averaging over adjacent fault facets. Identical to real-data inversion version.
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
                                   pollute=True, pollute_type='uniform',
                                   savefiles=True, verbose=True):
    """
    Synth recovery version of Az-constrained scalar amplitude inversion.

    Reads synthetic data from global `syndata` (loaded from d_obs_* file).
    Weights controlled by pollute / pollute_type (same as solveCoseismicInversion_TanhSlip).
    Outputs scalar amplitude + derived (s_strike, s_dip) for comparison with truth.

    Recovery metric saved to: slip_recovery_<meshname><...>.txt
      columns: xf_m  yf_m  zf_m  true_amp_m  recovered_amp_m  recovery_ratio
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

    CG_mu = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
    mtrue_mu_fun = hp.vector2Function(mtrue_mu, CG_mu)
    if savefiles:
        print("Saving true shear modulus to .xdmf file")
        mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
        m_mu_true_save = dl.project(mtrue_mu_fun_expr, CG_mu)
        mu_id = dl.XDMFFile(resultpath + 'mu_true_' + meshname + mu_str_inv + '.xdmf')
        m_mu_true_save.rename('shear modulus', 'shear modulus')
        mu_id.write(m_mu_true_save)

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

    # Weights: same logic as reference solveCoseismicInversion_TanhSlip
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

    # Fault coordinates (scalar CG1, consistent with Vm_amp → N elements, not 3*N)
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
    mtrue_mu_fun_expr = mu_expression(mtrue_mu_fun)
    m_mu_true = dl.project(mtrue_mu_fun_expr, CG_mu)
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

mu_l_het = 0.9730;  mu_lower_het = mu_expr_fn(mu_l_het)
mu_u_het = -0.9730; mu_upper_het = mu_expr_fn(mu_u_het)
mtrue_mu_expr_het = K_2LAYER(subdomains, mu_u_het, mu_l_het, degree=5)
mu_str_het = f"_mul{round(mu_expr_fn(mu_l_het))}u{round(mu_expr_fn(mu_u_het))}"
print("Heterogeneous structure:")
print(f"  mu_upper = {mu_upper_het:.1f}, mu_lower = {mu_lower_het:.1f}")

# %%
# Az configuration — same as real-data inversion
# slip_azimuth_deg  = 45.0   # CW from North; N45E = Cocos-Caribbean trench-normal convergence
# slip_azimuth_deg  = 26.0   # CW from North; N26E = roughly oblique convergence
azimuth_deg       = 33.5   # CW from North; N33.5E, a little oblique convergence
mesh_rotation_deg = 45.0

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
# SLIP PRESCRIPTION  (identical to reference — stripe or checker, Az-directed)
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
# Forward shear modulus (heterogeneous for synthetics, as in reference _noi.py)
mtrue_mu_expr_for = mtrue_mu_expr_het
mu_str_for = mu_str_het
print("Solving forward problem based on:", mu_str_for)

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
# True slip expression for forward solve
mtrue_s_expr_for = mtrue_s_expr_gt

# %%
# Forward solve — SKIPPED: output files already generated by reference script
# (synth_stripeslip_az_inv_hetmu_uneven_nicoyaCK_lock_noi.py with same parameters)
# Uncomment if forward files are missing or parameters have changed.
#
# solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for,
#                       pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
# print("Forward problem done!!!")
#
# d_obs_grid, valid_grid_indices = computeGridDisplacements(
#     k, targets_grid, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for,
#     pollute=False, pollute_type=pollute_type, savefiles=True, verbose=True)
# n_total_grid = targets_grid.shape[0]; n_valid_grid = len(valid_grid_indices)
# print(f"Dense grid: {n_valid_grid}/{n_total_grid} valid ({100*n_valid_grid/n_total_grid:.1f}%)")

# %%
# Load synthetic data for inversion
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x','y','z','ux','uy','uz'])

# %%
# Regularization parameters — same rho as reference
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
# SINGLE TEST RUN — Het μ, large damping, Az scalar amplitude
# ============================================================
if not RUN_LCURVE:
    gamma_s_test = 1e3
    delta_s_test = gamma_s_test / rho_s
    inv_str    = f"_synscaamp_azbd_w{w_h}{w_v}_gs{gamma_s_test:.1e}_ds{delta_s_test:.1e}"
    mu_str_inv = mu_str_het
    print(sep, "SINGLE TEST RUN — Az scalar amplitude inversion", sep)
    print("Inverse problem identifier:", inv_str)
    print(f"gamma_s = {gamma_s_test:.1e}, delta_s = {delta_s_test:.1e}, rho_s = {rho_s:.1e}")
    print("Solving inverse problem based on:", mu_str_inv)

    results_test = solveCoseismicInversion_AzSlip(
        k, targets, m0_amp_expr, mtrue_mu_expr_het,
        gamma_s_test, delta_s_test,
        c_strike_fun, c_dip_fun, amp_max,
        pollute=pollute, pollute_type=pollute_type,
        savefiles=True, verbose=True)

    # Unpack and compute amplitude recovery ratio
    _, xf, yf, zf, _, _, m_phys_fun_t, _, _, d_obs_t, d_cal_t, \
        m_amp_fault_t, m_sx_fault_t, m_sy_fault_t, misfitd_t, grad_m_t = results_test

    # True amplitude at fault vertices (from saved file)
    true_slip_file = resultpath + 'mtrue_s_fault_' + meshname + slip_str_gt + '.txt'
    true_s = np.loadtxt(true_slip_file)  # shape (N, 2): col0=s_strike, col1=s_dip
    true_amp_arr = np.sqrt(true_s[:,0]**2 + true_s[:,1]**2)
    # Mask recovery to vertices with significant true amplitude (stripe patches only)
    eps = 0.01 * amp  # 1% of full amplitude threshold
    valid = true_amp_arr > eps
    recovery = np.where(valid, m_amp_fault_t / np.where(valid, true_amp_arr, 1.0), np.nan)
    print(f"  Amplitude recovery (stripe only): [{np.nanmin(recovery):.3f}, {np.nanmax(recovery):.3f}]  (1.0 = perfect)")

    outFileName = ('slip_recovery_' + meshname + slip_str_gt
                   + mu_str_for + inv_str + mu_str_inv + '.txt')
    csvoutput = open(resultpath + outFileName, 'w+')
    csvoutput.write("# xf_m  yf_m  zf_m  true_amp_m  recovered_amp_m  recovery_ratio\n")
    for i in range(len(xf)):
        csvoutput.write("%.3f %.3f %.3f %.6f %.6f %.6f\n" % (
            xf[i], yf[i], zf[i], true_amp_arr[i], m_amp_fault_t[i], recovery[i]))
    csvoutput.close()
    print("Recovery file saved:", outFileName)
    del results_test; gc.collect()
    print("Single test run complete.")

# %%
# ============================================================
# L-curve: Het μ + Hom μ, Az scalar amplitude
# ============================================================
if RUN_LCURVE:

    # # Priority gammas (likely at the L-curve corner — run first for BOTH cases)
    # gammas_s_priority = [4e2, 5e2, 6e2]   # het has finished, hom just started
    # # Remaining gammas (needed for full L-curve plot — run after BOTH priority cases)
    # gammas_s_rest     = [1e1, 5e1, 1e2, 2e2, 8e2, 1e3, 5e3]

    # Priority gammas (likely at the L-curve corner — run first for BOTH cases)
    gammas_s_priority = [1e2, 2e2]
    # Remaining gammas (needed for full L-curve plot — run after BOTH priority cases)
    gammas_s_rest     = [1.5e2, 2.5e1, 5e1, 1e2, 2e2, 8e2, 1e3]

    outFileName_het = f"Lcurvesynscaamp_azbd_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_het}.txt"
    outFileName_hom = f"Lcurvesynscaamp_azbd_rs{rho_s:.0e}_{meshname}_{slip_pattern}_{pattern_option}_{mu_str_for}_{mu_str_hom}.txt"

    # Passes: (label, gammas, outFileName, mtrue_mu_expr, mu_str_inv_val)
    # Order: Het priority → Hom priority → Het rest → Hom rest
    _passes = [
        ("Het L-curve (priority)", gammas_s_priority, outFileName_het, mtrue_mu_expr_het, mu_str_het),
        ("Hom L-curve (priority)", gammas_s_priority, outFileName_hom, mtrue_mu_expr_hom, mu_str_hom),
        # ("Het L-curve (rest)",     gammas_s_rest,     outFileName_het, mtrue_mu_expr_het, mu_str_het),
        # ("Hom L-curve (rest)",     gammas_s_rest,     outFileName_hom, mtrue_mu_expr_hom, mu_str_hom),
    ]

    for _label, _gammas, _outFile, _mtrue_mu_expr, _mu_str_inv in _passes:
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
