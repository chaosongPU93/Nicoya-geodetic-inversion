"""
sandbox_az_checker_test.py
==========================
Sandbox to validate FaultLocalCheckerboardAz.

Mirrors sandbox_az_forward_test.py but uses the CHECKERBOARD pattern.
Runs OLD (pure dip-slip) vs NEW (uniform azimuth Az) checkerboard.
Analyses:
  1. Slip coefficient statistics (c_strike, c_dip) at fault DOFs
  2. Horizontal angle of reconstructed slip vs. prescribed azimuth
  3. Displacement statistics

Uses the even-top mesh (nicoyaCKden_sm).
Saves all outputs to a sandbox subdirectory separate from existing results.

Run with:
    conda activate fenics
    python sandbox_az_checker_test.py
"""

# ---------------------------------------------------------------------------
# Thread limits (same as driver)
# ---------------------------------------------------------------------------
import sys, os
os.environ['OMP_NUM_THREADS'] = '5'
os.environ['OPENBLAS_NUM_THREADS'] = '5'
os.environ['MKL_NUM_THREADS'] = '5'
os.environ['VECLIB_MAXIMUM_THREADS'] = '5'
os.environ['NUMEXPR_NUM_THREADS'] = '5'

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import dolfin as dl
import ufl
import pandas as pd
import numpy as np
import utils as ut
from mpi4py import MPI
import hippylib as hp
from pointwiseStateObs import PointwiseStateObservation as PSB

dl.parameters["form_compiler"]["quadrature_degree"] = 5
dl.parameters["form_compiler"]["optimize"] = True
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

sep = "\n" + "#"*80 + "\n"

# ---------------------------------------------------------------------------
# Paths — SANDBOX: separate from existing results
# ---------------------------------------------------------------------------
datadir    = "/home/staff/chao/SSEinv/Nicoya/data/"
meshpath   = "/home/staff/chao/SSEinv/Nicoya/mesh/"
resultpath = "/home/staff/chao/SSEinv/Nicoya/syn_slip/sandbox_az/"
os.makedirs(resultpath, exist_ok=True)
print(f"Sandbox output directory: {resultpath}")

# ---------------------------------------------------------------------------
# GNSS station data (observation targets)
# ---------------------------------------------------------------------------
obs_disp_name = "CKfig6_data_final.csv"
data = pd.read_csv(datadir + obs_disp_name, sep=",", skiprows=1,
                   names=['lon', 'lat', 'vx_Car', 'vy_Car', 'vz_Car',
                          'vx_std_Car', 'vy_std_Car', 'vz_std_Car'])

lon0, lat0 = -84, 7
rot = 45                          # CCW degrees — mesh x-axis = N45E
x0, y0 = 130e3, 350e3            # mesh offset (m)
x_rot, y_rot = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)
data['x'] = (x_rot - x0) / 1e3   # km, mesh coords
data['y'] = (y_rot - y0) / 1e3
data['z'] = 0.0                   # even mesh — surface at z=0

print(f"Number of GNSS stations: {len(data)}")

# ---------------------------------------------------------------------------
# Mesh (even-top, no uneven mesh handling needed)
# ---------------------------------------------------------------------------
meshname = "nicoyaCKden_sm"
mesh      = dl.Mesh(meshpath + meshname + '.xml')
dim       = mesh.topology().dim()
n         = dl.FacetNormal(mesh)
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')

top       = 1;  bottom = 2;  west = 3;  east = 4
north     = 5;  south  = 6;  fault = 7
blockleft = 8;  blockright = 9

ds  = dl.Measure("ds")(domain=mesh, subdomain_data=boundaries)
dS  = dl.Measure("dS")(domain=mesh, subdomain_data=boundaries)

print(f"Mesh: {meshname}, {mesh.num_cells()} cells, {mesh.num_vertices()} vertices")

# ---------------------------------------------------------------------------
# FEniCS building blocks (copied verbatim from driver)
# ---------------------------------------------------------------------------
GPa2Pa = 1e9
k      = 2                        # element order
f      = dl.Constant((0., 0., 0.))
nu     = 0.25

def AEsigma(s, mu, nu):
    A = 1./(2.*mu) * (s - nu/(1 + nu*(dim-2)) * ufl.tr(s) * ufl.Identity(dim))
    return A

def asym(s):
    return ufl.as_vector([s[1,2]-s[2,1], s[2,0]-s[0,2], s[0,1]-s[1,0]])

def dir_strike(n):
    z_dir    = dl.Constant((0., 0., 1.))
    n_cross_z = ufl.cross(n, z_dir)
    return n_cross_z / ufl.sqrt(ufl.dot(n_cross_z, n_cross_z))

def dir_dip(n):
    return ufl.cross(dir_strike(n), n)

def mu_expression(m):
    return 20*(2. + ufl.tanh(m))

class K_2LAYER(dl.UserExpression):
    def __init__(self, subdomains, k_r, k_l, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.k_r = k_r;  self.k_l = k_l
    def eval_cell(self, values, x, cell):
        values[0] = self.k_r if self.subdomains[cell.index] == blockright else self.k_l
    def value_shape(self):
        return ()

class PDEVarf:
    def __init__(self, mtrue_mu_fun):
        self.mtrue_mu_fun = mtrue_mu_fun
    def __call__(self, u, m, p):
        sigma, uu, r = dl.split(u)
        m_strike, m_dip = dl.split(m)
        tau, w, q = dl.split(p)
        u0  = dl.Constant((0., 0., 0.))
        mu  = mu_expression(self.mtrue_mu_fun)
        J   = ufl.inner(AEsigma(sigma, mu, nu), tau)*ufl.dx \
            + ufl.inner(ufl.div(tau), uu)*ufl.dx \
            + ufl.inner(asym(tau), r)*ufl.dx \
            + ufl.inner(ufl.div(sigma), w)*ufl.dx \
            + ufl.inner(asym(sigma), q)*ufl.dx \
            + ufl.inner(f, w)*dl.dx \
            - ufl.inner(u0, tau*n)*ds(bottom) \
            - ufl.inner(dir_strike(n('+')) * ufl.avg(m_strike) + dir_dip(n('+')) * ufl.avg(m_dip),
                        tau('+')*n('+'))*dS(fault)
        return J

# ---------------------------------------------------------------------------
# Minimal forward solver (no noise, no dense grid, no inversion)
# ---------------------------------------------------------------------------
def run_forward(slip_expr, mu_expr, targets, tag, savefiles=True):
    """
    Solve the forward problem and return slip and displacement arrays.

    Parameters
    ----------
    slip_expr : dl.UserExpression  (2-component)
    mu_expr   : dl.UserExpression  (scalar)
    targets   : Nx3 ndarray        (m)
    tag       : str                appended to output filenames
    """
    print(sep + f"Forward solve: {tag}" + sep)

    BDM = dl.VectorFunctionSpace(mesh, "BDM", k)
    DGv = dl.VectorFunctionSpace(mesh, "DG", k-1)
    DGr = dl.VectorFunctionSpace(mesh, "DG", k-1)
    ME  = dl.MixedElement([BDM.ufl_element(), DGv.ufl_element(), DGr.ufl_element()])
    Vu  = dl.FunctionSpace(mesh, ME)
    Vm  = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    Vh  = [Vu, Vm, Vu]

    zero_tensor = dl.Expression((("0.","0.","0."),("0.","0.","0."),("0.","0.","0.")), degree=0)
    bc  = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)
    bc0 = dl.DirichletBC(Vh[hp.STATE].sub(0), zero_tensor, boundaries, top)

    # --- Load slip into parameter vector via fault Dirichlet BC ---
    V_slip = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)
    bc_slip = dl.DirichletBC(V_slip, slip_expr, boundaries, fault)
    slip_fun = dl.Function(V_slip)
    bc_slip.apply(slip_fun.vector())
    mtrue_s = slip_fun.vector()

    mtrue = dl.Function(Vh[hp.PARAMETER]).vector()
    tmp   = np.zeros(Vh[hp.PARAMETER].dim())
    tmp[0::2] = mtrue_s.copy()[0::2]
    tmp[1::2] = mtrue_s.copy()[1::2]
    mtrue.set_local(tmp)

    # --- Shear modulus ---
    CG_mu    = dl.FunctionSpace(mesh, "CG", 1)
    mtrue_mu = dl.interpolate(mu_expr, CG_mu).vector()
    mu_fun   = hp.vector2Function(mtrue_mu, CG_mu)

    # --- PDE ---
    pde_varf = PDEVarf(mu_fun)
    pde      = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    pde.solver          = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")
    pde.solver_fwd_inc  = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")
    pde.solver_adj_inc  = hp.PETScLUSolver(mesh.mpi_comm(), "mumps")

    indicator_vec = dl.interpolate(
        dl.Constant((0,0,0, 0,0,0, 0,0,0, 1,1,1, 0,0,0)), Vh[hp.STATE]).vector()
    misfit = PSB(Vh[hp.STATE], targets, indicator_vec=indicator_vec)

    u_state = pde.generate_state()
    x       = [u_state, mtrue, None]
    pde.solveFwd(x[hp.STATE], x)

    misfit.B.mult(x[hp.STATE], misfit.d)
    idx_d = list(np.nonzero(misfit.d)[0])
    d_obs  = misfit.d[idx_d]          # ux,uy,uz interleaved for valid pts

    # --- Extract fault-DOF slip values ---
    bc2  = dl.DirichletBC(Vm, (99, 99), boundaries, fault)
    um2  = dl.Function(Vm)
    bc2.apply(um2.vector())
    fault_mask     = (um2.vector()[:] == 99)
    slip_at_fault  = mtrue[fault_mask]
    s_strike_fault = slip_at_fault[0::2]
    s_dip_fault    = slip_at_fault[1::2]

    # --- Save ---
    if savefiles:
        fname = resultpath + f'slip_fault_{tag}.txt'
        np.savetxt(fname, np.column_stack([s_strike_fault, s_dip_fault]),
                   fmt='%.8f', header='c_strike  c_dip')
        print(f"Saved fault slip to {fname}")

        fname = resultpath + f'd_obs_{tag}.txt'
        n_obs = len(d_obs) // 3
        with open(fname, 'w') as fh:
            for i in range(n_obs):
                fh.write("%.6f %.6f %.6f\n" % (d_obs[3*i], d_obs[3*i+1], d_obs[3*i+2]))
        print(f"Saved {n_obs} displacements to {fname}")

    return s_strike_fault, s_dip_fault, d_obs

# ---------------------------------------------------------------------------
# Observation targets
# ---------------------------------------------------------------------------
ntargets   = data.shape[0]
targets    = np.zeros([ntargets, 3])
targets[:,0] = np.array(data['x']) * 1e3   # km → m
targets[:,1] = np.array(data['y']) * 1e3
targets[:,2] = np.array(data['z']) * 1e3

# ---------------------------------------------------------------------------
# Shear modulus (homogeneous)
# ---------------------------------------------------------------------------
mu_u = -0.9730;  mu_l = 0.9730
mu_expr = K_2LAYER(subdomains, mu_u, mu_l, degree=5)

# ---------------------------------------------------------------------------
# Slip parameters (checkerboard)
# ---------------------------------------------------------------------------
V_norm        = 78.5 / 1e3
amp           = V_norm
dx            = 80e3    # checker patch size along-dip (m)
dy            = 80e3    # checker patch size along-strike (m)
x0_pat        = 0.
y0_pat        = -45e3
rot_deg       = 0.0

slip_azimuth_deg  = 45.0   # N45E — change to test other azimuths
mesh_rot_deg      = 45.0   # Nicoya mesh convention

# OLD: pure dip-slip (averaged normal frame)
slip_expr_old = ut.create_fault_local_checkerboard(
    mesh=mesh, boundaries=boundaries, fault_id=fault,
    amp=amp,
    dx=dx, dy=dy,
    x0=x0_pat, y0=y0_pat, rotation_deg=rot_deg,
    degree=5
)

# NEW: uniform azimuth (per-facet projection)
slip_expr_new = ut.create_fault_local_checkerboard_az(
    mesh=mesh, boundaries=boundaries, fault_id=fault,
    amp=amp,
    dx=dx, dy=dy,
    azimuth_deg=slip_azimuth_deg, mesh_rotation_deg=mesh_rot_deg,
    x0=x0_pat, y0=y0_pat, rotation_deg=rot_deg,
    degree=5
)

# ---------------------------------------------------------------------------
# Forward solves
# ---------------------------------------------------------------------------
tag_old = f"{meshname}_checker_OLD_az{slip_azimuth_deg:g}"
tag_new = f"{meshname}_checker_AZ{slip_azimuth_deg:g}"

s_str_old, s_dip_old, d_old = run_forward(slip_expr_old, mu_expr, targets, tag_old)
s_str_new, s_dip_new, d_new = run_forward(slip_expr_new, mu_expr, targets, tag_new)

# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
print(sep + "ANALYSIS" + sep)

print("\n--- Fault-DOF slip coefficient statistics ---")
print(f"{'':30s}  {'c_strike':>12s}  {'c_dip':>12s}")
for label, cs, cd in [("OLD (pure dip)", s_str_old, s_dip_old),
                       (f"NEW (az={slip_azimuth_deg}°)", s_str_new, s_dip_new)]:
    # Only non-zero DOFs (inside slip patches)
    active = np.abs(cd) > 1e-10
    if active.sum() == 0:
        print(f"  {label:30s}: no active DOFs found!")
        continue
    print(f"\n  {label}")
    print(f"    Active DOFs: {active.sum()} / {len(cd)}")
    print(f"    c_strike  min={cs[active].min():.6f}  max={cs[active].max():.6f}  "
          f"mean={cs[active].mean():.6f}  std={cs[active].std():.6f}")
    print(f"    c_dip     min={cd[active].min():.6f}  max={cd[active].max():.6f}  "
          f"mean={cd[active].mean():.6f}  std={cd[active].std():.6f}")
    slip_mag = np.sqrt(cs[active]**2 + cd[active]**2)
    print(f"    ||slip||  min={slip_mag.min():.6f}  max={slip_mag.max():.6f}  "
          f"mean={slip_mag.mean():.6f}")
    # Checkerboard: expect both +amp and -amp patches — report separately
    pos = (cd > 1e-10);  neg = (cd < -1e-10)
    print(f"    Positive patches: {pos.sum()} DOFs,  Negative patches: {neg.sum()} DOFs")

print("\n--- Direction check for NEW class ---")
print("For each active fault DOF, the physical slip vector should have")
print(f"its horizontal projection at azimuth ≈ {slip_azimuth_deg}° CW from North.")
print("We verify using the averaged fault frame stored in the expression object.\n")

facet_coeffs = slip_expr_new._facet_coeffs   # {facet_idx -> (c_strike, c_dip)}

alpha = np.deg2rad(slip_azimuth_deg)
mrot  = np.deg2rad(mesh_rot_deg)
h_prescribed = np.array([np.sin(alpha + mrot), np.cos(alpha + mrot), 0.0])
print(f"Prescribed horizontal direction h = {h_prescribed}")
# Convert h from mesh coords to geographic (East, North), then compute geographic azimuth.
# Mesh is rotated CCW by mrot from geographic, so the inverse is:
#   East  = h_x * cos(mrot) - h_y * sin(mrot)
#   North = h_x * sin(mrot) + h_y * cos(mrot)
h_east  = h_prescribed[0] * np.cos(mrot) - h_prescribed[1] * np.sin(mrot)
h_north = h_prescribed[0] * np.sin(mrot) + h_prescribed[1] * np.cos(mrot)
az_expected = np.rad2deg(np.arctan2(h_east, h_north))
print(f"Expected azimuth (geographic, from h) = {az_expected:.2f}° CW from North")

avg_strike = slip_expr_new.strike_vector
avg_dip    = slip_expr_new.dip_vector
print(f"\nAveraged fault frame (used for pattern geometry only):")
print(f"  strike_vec = {avg_strike}")
print(f"  dip_vec    = {avg_dip}")

# Reconstruct slip azimuth for active DOFs using avg frame (approximation)
# Use positive-patch DOFs (cd > 0) for clean azimuth reconstruction
pos_new = (s_dip_new > 1e-10)
if pos_new.sum() > 0:
    slip_x = s_str_new[pos_new] * avg_strike[0] + s_dip_new[pos_new] * avg_dip[0]
    slip_y = s_str_new[pos_new] * avg_strike[1] + s_dip_new[pos_new] * avg_dip[1]
    slip_h_mag = np.sqrt(slip_x**2 + slip_y**2)
    valid = slip_h_mag > 1e-10
    az_recon = np.rad2deg(np.arctan2(slip_x[valid], slip_y[valid]))
    print(f"\n  Reconstructed horizontal slip azimuth (positive patches, approx avg frame):")
    print(f"    mean = {az_recon.mean():.2f}°,  std = {az_recon.std():.2f}°,  "
          f"min = {az_recon.min():.2f}°,  max = {az_recon.max():.2f}°")
    print(f"    Expected: {az_expected:.2f}°")
    print(f"    Mean error: {(az_recon - az_expected).mean():.4f}°  "
          f"(non-zero due to avg-frame approximation)")
else:
    print("  No positive-patch DOFs found for NEW class.")

print("\n--- Displacement comparison at GNSS stations ---")
n_sta = len(d_old) // 3
print(f"  Stations: {n_sta}")
for label, d in [("OLD", d_old), ("NEW", d_new)]:
    ux = d[0::3];  uy = d[1::3];  uz = d[2::3]
    mag = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"\n  {label}:")
    print(f"    ux  mean={ux.mean()*1e3:.3f}  std={ux.std()*1e3:.3f} mm/yr")
    print(f"    uy  mean={uy.mean()*1e3:.3f}  std={uy.std()*1e3:.3f} mm/yr")
    print(f"    uz  mean={uz.mean()*1e3:.3f}  std={uz.std()*1e3:.3f} mm/yr")
    print(f"    |d| mean={mag.mean()*1e3:.3f}  max={mag.max()*1e3:.3f} mm/yr")

diff = d_new - d_old
diff_mag = np.sqrt(diff[0::3]**2 + diff[1::3]**2 + diff[2::3]**2)
print(f"\n  |NEW - OLD| displacement difference:")
print(f"    mean = {diff_mag.mean()*1e3:.4f} mm/yr,  max = {diff_mag.max()*1e3:.4f} mm/yr")
print(f"    (should be ~0 for azimuth=45° where Az ≈ pure dip-slip on this fault)")

print(sep + "SANDBOX CHECKER COMPLETE" + sep)
