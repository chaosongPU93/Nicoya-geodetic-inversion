# %% [markdown]
# This is a duplicate of 'synth_stripeslip_inv_hetmu3D_nicoyaCK_lock_noi.py' with limited changes to test
# the goal is test the new mesh 'nicoyaCKden_une_sm' with uneven top bound on left and right sides because the supplied
# plate interface with CK does not reach the surface on the left side. So the left block tops at a median depth of ~7 km,
# whereas the right block tops at the surface.
# To test against the previous mesh 'nicoyaCKden_sm', which has both left and right blocks top at the surface.
# The reason of test is because the mesh 'nicoyaCKden_sm' has a sharp transition to the top, nearly vertical, so we
# see some sharp gradient and artifact in the synthetic ground displacements at the trench.
# Though need not be wrong, we don't know the sharp gradient would affect the inversion results or not.
# L-curve analysis enabled to find optimal regularization parameters. 


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

# define the centroid of relative coordinates, must be consistent with the mesh!
lon0, lat0 = -84, 7     # from Christos's email
# print(lon0, lat0)

# convert to relative locations in meters, and then rotate
rot = 45  # rotation angle in degrees, positive is CCW
x_rot, y_rot  = ut.LL2ckmd(data['lon'], data['lat'], lon0, lat0, rot)

# offset in x and y direction, the same as being done to the mesh in 'Kyriakopoulos2016JGR/convert_exodus_to_msh.ipynb'
x0, y0 = 130e3, 350e3  # offset for x and y coordinates, in m
data['x'], data['y'] = (x_rot - x0) / 1e3, (y_rot - y0) / 1e3   # offset to match the mesh coordinates
data['z'] = 0.0

# print(data[['lon', 'lat', 'x', 'y']].head())
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
# For each 3D point, look up the 1D Vs at the matching depth
# Vp and Vp/Vs are independently inverted; Vs = Vp / (Vp/Vs) is derived
vp1d_at   = vel3d['z'].map(lambda z: vel1d.loc[(vel1d['z'] - z).abs().idxmin(), 'vp'])
vpvs1d_at = vel3d['z'].map(lambda z: vel1d.loc[(vel1d['z'] - z).abs().idxmin(), 'vp_vs_ratio'])
vs1d_at   = vel3d['z'].map(lambda z: vel1d.loc[(vel1d['z'] - z).abs().idxmin(), 'vs'])

# dln(Vs) = dVs / Vs_1D
dln_vp   = (vel3d['vp']           - vp1d_at)   / vp1d_at    # independent
dln_vpvs = (vel3d['vp_vs_ratio']  - vpvs1d_at) / vpvs1d_at  # independent
dln_vs   = (vel3d['vs']           - vs1d_at)   / vs1d_at    # derived: Vs = Vp / (Vp/Vs)
# note: dln_vs ~ dln_vp - dln_vpvs to first order

print(f"dln(Vs) range: {dln_vs.min():.4f} to {dln_vs.max():.4f}")
print(f"  i.e., {dln_vs.min()*100:.1f}% to {dln_vs.max()*100:.1f}%")
print(f"Mean: {dln_vs.mean()*100:.2f}%, Std: {dln_vs.std()*100:.2f}%")
print(f"dln(Vp) range: {dln_vp.min()*100:.1f}% to {dln_vp.max()*100:.1f}%")
print(f"dln(Vp/Vs) range: {dln_vpvs.min()*100:.1f}% to {dln_vpvs.max()*100:.1f}%")

# Breakdown by depth layer
print("\nDepth(km)  Vs_1D   min%     max%     mean%    std%")
for z_m in sorted(vel3d['z'].unique()):
    mask = vel3d['z'] == z_m
    dlnv = dln_vs[mask]
    vs1d_val = vel1d.loc[(vel1d['z'] - z_m).abs().idxmin(), 'vs']
    print(f"{z_m/1e3:8.1f}  {vs1d_val:.3f}  {dlnv.min()*100:7.2f}  "
          f"{dlnv.max()*100:7.2f}  {dlnv.mean()*100:7.2f}  {dlnv.std()*100:7.2f}")
    
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
# ============================================================================
# HELPER FUNCTIONS FOR UNEVEN MESH TOP BOUNDARY HANDLING
# ============================================================================
#
# TWO-LEVEL APPROACH TO HANDLE POINTS OUTSIDE MESH:
#
# LEVEL 1 (Ray Casting - Depth Assignment):
#   - Checks if each (x,y) location has ANY mesh coverage at all
#   - Points outside the mesh's horizontal footprint get z-depths extrapolated
#   - If your grid region matches mesh extent: expect 0% invalid
#
# LEVEL 2 (Displacement Computation):
#   - Handles rare cases where FEniCS can't compute displacements despite valid z
#   - Due to numerical precision or mesh topology edge cases
#   - If grid matches mesh well: expect <<1% or 0% invalid
#
# Both levels use nearest-neighbor extrapolation for missing values to ensure
# continuous displacement fields for visualization/interpolation.
# ============================================================================

def extract_top_boundary_surface(mesh, boundaries, top_id):
    """
    Extract the top boundary surface coordinates from the mesh.

    Args:
        mesh: FEniCS mesh object
        boundaries: MeshFunction marking boundaries
        top_id: boundary ID for the top surface

    Returns:
        top_coords: Nx3 array of (x, y, z) coordinates on the top boundary
    """
    # Get all mesh coordinates
    all_coords = mesh.coordinates()

    # Get facets on the top boundary
    top_facets = []
    for facet in dl.facets(mesh):
        if boundaries[facet] == top_id:
            top_facets.append(facet)

    # Extract unique vertices on top boundary
    top_vertex_indices = set()
    for facet in top_facets:
        for vertex in dl.vertices(facet):
            top_vertex_indices.add(vertex.index())

    # Get coordinates of top boundary vertices
    top_coords = all_coords[list(top_vertex_indices)]

    print(f"Extracted {len(top_coords)} vertices from top boundary")
    print(f"  x range: [{top_coords[:,0].min()/1e3:.1f}, {top_coords[:,0].max()/1e3:.1f}] km")
    print(f"  y range: [{top_coords[:,1].min()/1e3:.1f}, {top_coords[:,1].max()/1e3:.1f}] km")
    print(f"  z range: [{top_coords[:,2].min()/1e3:.1f}, {top_coords[:,2].max()/1e3:.1f}] km")

    return top_coords


def interpolate_top_boundary_depth(x_query, y_query, top_coords, method='nearest'):
    """
    Interpolate z-depth at query (x, y) points from top boundary surface.

    NOTE: This is a simple interpolation and may not guarantee points are inside mesh.
    Use find_valid_surface_depths_raycast() for guaranteed valid points.

    Args:
        x_query: array of x coordinates (in meters) to query
        y_query: array of y coordinates (in meters) to query
        top_coords: Nx3 array of (x, y, z) coordinates on top boundary
        method: 'nearest' or 'linear' interpolation

    Returns:
        z_interp: array of interpolated z-depths at query points
    """
    from scipy.interpolate import griddata, NearestNDInterpolator

    # Extract x, y, z from top boundary coordinates
    x_top = top_coords[:, 0]
    y_top = top_coords[:, 1]
    z_top = top_coords[:, 2]

    if method == 'nearest':
        # Use nearest neighbor interpolation (more robust for irregular meshes)
        interp = NearestNDInterpolator(list(zip(x_top, y_top)), z_top)
        z_interp = interp(x_query, y_query)
    else:
        # Use griddata for linear interpolation
        z_interp = griddata((x_top, y_top), z_top, (x_query, y_query),
                           method=method, fill_value=0.0)

    return z_interp


def find_valid_surface_depths_raycast(x_query, y_query, mesh, z_start=1000, z_end=-15000,
                                      n_samples=100, verbose=True):
    """
    Find valid z-depths at each (x,y) by vertical ray casting to ensure points are inside mesh.

    This is LEVEL 1 checking: Determines if the (x,y) horizontal location has ANY mesh coverage.

    For each (x,y) point, searches downward from z_start to z_end to find the highest
    z-value where the point is inside a mesh cell.

    Points that return np.nan are OUTSIDE THE MESH'S HORIZONTAL FOOTPRINT - meaning the
    (x,y) location itself is beyond the mesh domain, not just at the wrong depth.

    If you've chosen your grid region to match the mesh extent, you should get 0% invalid points.

    Args:
        x_query: array of x coordinates (in meters)
        y_query: array of y coordinates (in meters)
        mesh: FEniCS mesh object
        z_start: starting z for downward search (meters, above surface)
        z_end: ending z for search (meters, deep underground)
        n_samples: number of z-levels to test
        verbose: print progress info

    Returns:
        z_valid: array of valid z-depths (or np.nan if (x,y) is outside mesh footprint)
        valid_mask: boolean array indicating which (x,y) locations have mesh coverage
    """
    # Build bounding box tree for efficient containment testing
    bbtree = dl.BoundingBoxTree()
    bbtree.build(mesh)

    n_points = len(x_query)
    z_valid = np.full(n_points, np.nan)

    # Generate z-levels to test (from top to bottom)
    z_levels = np.linspace(z_start, z_end, n_samples)

    if verbose:
        print(f"Ray casting to find valid depths for {n_points} points...")
        print(f"  Testing {n_samples} z-levels from {z_start/1e3:.1f} to {z_end/1e3:.1f} km")

    # For each query point, find the highest z where it's inside the mesh
    points_found = 0
    for i in range(n_points):
        for z_test in z_levels:
            point = dl.Point(x_query[i], y_query[i], z_test)
            cell_id = bbtree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells():
                z_valid[i] = z_test
                points_found += 1
                break  # Found valid depth, move to next point

        # Progress indicator
        if verbose and (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{n_points} points, {points_found} valid so far")

    valid_mask = ~np.isnan(z_valid)
    n_valid = np.sum(valid_mask)
    n_invalid = n_points - n_valid

    if verbose:
        print(f"Ray casting complete:")
        print(f"  Valid points: {n_valid} ({100*n_valid/n_points:.1f}%)")
        print(f"  Invalid points: {n_invalid} ({100*n_invalid/n_points:.1f}%)")
        if n_valid > 0:
            print(f"  Valid z-range: [{np.nanmin(z_valid)/1e3:.3f}, {np.nanmax(z_valid)/1e3:.3f}] km")

    return z_valid, valid_mask


def extrapolate_displacements(targets, d_obs, valid_indices, verbose=True):
    """
    Extrapolate displacement values for points where FEniCS couldn't compute displacements.

    This is LEVEL 2 handling: These are points where ray casting found a valid z-depth
    (meaning the (x,y) location has mesh coverage), BUT FEniCS still couldn't compute
    the displacement during the forward solve.

    This can happen due to:
    - Numerical precision issues at cell boundaries
    - Mesh topology edge cases
    - Rare discrepancies between BoundingBoxTree and function evaluation

    If your grid matches the mesh well, this should affect <<1% of points or even 0%.

    Args:
        targets: Nx3 array of all target points (x, y, z in meters)
        d_obs: array of displacement values (ux, uy, uz) for valid points only (length = 3*n_valid)
        valid_indices: list of indices where FEniCS successfully computed displacements
        verbose: print progress info

    Returns:
        d_full: array of displacement values for all points (length = 3*n_targets)
                with extrapolated values filled in for invalid points
    """
    from scipy.interpolate import NearestNDInterpolator

    n_total = targets.shape[0]
    n_valid = len(valid_indices)
    n_invalid = n_total - n_valid

    if verbose and n_invalid > 0:
        print(f"Extrapolating displacements for {n_invalid} points outside mesh...")

    # Create full displacement array
    d_full = np.zeros(3 * n_total)

    if n_invalid == 0:
        # All points are valid, just reshape
        d_full = d_obs
    else:
        # Get valid point locations and displacements
        valid_targets = targets[valid_indices, :]
        x_valid = valid_targets[:, 0]
        y_valid = valid_targets[:, 1]

        # Separate displacement components
        ux_valid = d_obs[0::3]
        uy_valid = d_obs[1::3]
        uz_valid = d_obs[2::3]

        # Create extrapolators for each component
        extrapolator_x = NearestNDInterpolator(list(zip(x_valid, y_valid)), ux_valid)
        extrapolator_y = NearestNDInterpolator(list(zip(x_valid, y_valid)), uy_valid)
        extrapolator_z = NearestNDInterpolator(list(zip(x_valid, y_valid)), uz_valid)

        # Extrapolate for all points
        for i in range(n_total):
            if i in valid_indices:
                # Use computed value
                j = valid_indices.index(i)
                d_full[3*i] = d_obs[3*j]
                d_full[3*i+1] = d_obs[3*j+1]
                d_full[3*i+2] = d_obs[3*j+2]
            else:
                # Extrapolate from nearby valid points
                x, y = targets[i, 0], targets[i, 1]
                d_full[3*i] = extrapolator_x(x, y)
                d_full[3*i+1] = extrapolator_y(x, y)
                d_full[3*i+2] = extrapolator_z(x, y)

        if verbose:
            print(f"  Extrapolation complete for {n_invalid} points")

    return d_full


# %%
# CREATE DENSE OBSERVATION GRID IN LAT/LON THEN CONVERT TO MESH COORDINATES
print(sep, "Creating dense observation grid", sep)

# Define regular lat/lon grid covering study area
# Based on your image, approximate coverage around Costa Rica/Nicaragua region
# region=[-87, -84, 8.5, 11.5]    # suitable region for chopping the plate interface grid file
# region=[-86.75, -84.4, 8.75, 11.25]    # suitable region for chopping the plate interface grid file
# region=[-88, -83, 7.5, 12.5]
# region=[-88, -83, 7.4, 12.6]
region=[-87, -84, 8.6, 11.6]
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
x_rot_2d, y_rot_2d = ut.LL2ckmd(lon_2d, lat_2d, lon0, lat0, rot)
x_2d = (x_rot_2d - x0) / 1e3  # convert to km
y_2d = (y_rot_2d - y0) / 1e3

# NOTE: Instead of hardcoding depth_levels = [0], we'll now automatically
# determine the actual top boundary depth at each (x, y) location
print("Note: This code will automatically determine the top boundary depth at each location")
print("      instead of using a hardcoded depth of 0 km")

# We'll assign depths after loading the mesh (see code block after mesh loading)
# For now, create placeholder
n_2d = len(x_2d)

lon_dense = lon_2d
lat_dense = lat_2d
x_dense = x_2d
y_dense = y_2d
z_dense = np.zeros(n_2d)  # Placeholder, will be filled after mesh loading

print(f"Dense grid: {len(lon_grid)} x {len(lat_grid)} = {n_2d} points")

# Create dense grid dataframe (z will be updated later)
dense_data = pd.DataFrame({
    'lon': lon_dense,
    'lat': lat_dense,
    'x': x_dense,
    'y': y_dense,
    'z': z_dense
})

print(f"Dense grid coordinate ranges (before depth assignment):")
print(f"  x: [{x_dense.min():.1f}, {x_dense.max():.1f}] km")
print(f"  y: [{y_dense.min():.1f}, {y_dense.max():.1f}] km")
print(f"  z: will be determined from mesh top boundary for uneven meshes, 0 otherwise")

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
# meshname = "nicoyaCK"   # local interface model from C. Kyriakopoulos_etal2015JGRSE
# meshname = "nicoyaCK2"   # same as above but 5-km mesh size on fault
# meshname = "nicoyaCK3"   # fault zone extended to the whole subduction zone
# meshname = "nicoyaCK4"   # same as CK2, but connecting the trench now

# Meshes with even top boundary at 0 depth
# meshname = "nicoyaCKden_sm"   # based on nicoyaCK3 or 4, but denser mesh size, and smaller fault zone
# meshname = "nicoyaCKden_all"   # based on nicoyaCK3 or 4, but denser mesh size, and all subduction interface

# Mesh with uneven top boundary, left at mean trench depth ~7 km, right at 0 km
meshname = "nicoyaCKden_une_sm"   # uneven top boundary, smaller fault zone, counterpart to 'nicoyaCKden_sm'
# meshname = "nicoyaCKden_une_all"   # uneven top boundary, all subduction interface, counterpart to 'nicoyaCKden_all'

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
# ASSIGN DEPTHS TO OBSERVATION GRID BASED ON MESH TOP BOUNDARY (for uneven meshes)
if use_uneven_mesh:
    print(sep, "Assigning depths from mesh top boundary to observation grid", sep)

    # Extract the top boundary surface from the mesh
    top_coords = extract_top_boundary_surface(mesh, boundaries, top)

    # Convert from km to m for processing
    x_query_m = dense_data['x'].values * 1e3  # km to m
    y_query_m = dense_data['y'].values * 1e3  # km to m

    # ============================================================================
    # LEVEL 1 CHECK: Find valid z-depths for each (x,y) location using ray casting
    # ============================================================================
    # This checks if the (x,y) horizontal location has ANY mesh coverage at all.
    # Points that fail this check are OUTSIDE THE MESH'S HORIZONTAL FOOTPRINT.
    #
    # If you've chosen your grid region to match the mesh, expect 0% invalid here.
    z_valid_m, valid_mask = find_valid_surface_depths_raycast(
        x_query_m, y_query_m, mesh,
        z_start=1000,      # Start above surface
        z_end=-15000,      # Search down to -15 km
        n_samples=150,     # Test 150 z-levels for accuracy
        verbose=True
    )

    # Convert to km
    z_valid_km = z_valid_m / 1e3

    # Handle points outside mesh footprint by extrapolating from nearby valid points
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        from scipy.interpolate import NearestNDInterpolator
        print(f"\n[LEVEL 1] Extrapolating z-depths for {n_invalid} points outside mesh footprint...")
        print(f"  These (x,y) locations have no mesh coverage at any depth.")

        # Get valid points
        x_valid = x_query_m[valid_mask]
        y_valid = y_query_m[valid_mask]
        z_from_valid = z_valid_m[valid_mask]

        # Create interpolator from valid points
        extrapolator = NearestNDInterpolator(list(zip(x_valid, y_valid)), z_from_valid)

        # Extrapolate for invalid points
        x_invalid = x_query_m[~valid_mask]
        y_invalid = y_query_m[~valid_mask]
        z_extrapolated = extrapolator(x_invalid, y_invalid)

        # Fill in the invalid points
        z_valid_m[~valid_mask] = z_extrapolated
        z_valid_km = z_valid_m / 1e3

        print(f"  Extrapolated z-range: [{z_extrapolated.min()/1e3:.3f}, {z_extrapolated.max()/1e3:.3f}] km")
    else:
        print(f"\n[LEVEL 1] All {len(x_query_m)} grid points have mesh coverage - no extrapolation needed!")

    # Update the dataframe
    dense_data['z'] = z_valid_km

    print(f"\nFinal observation grid with valid depths:")
    print(f"  z range: [{z_valid_km.min():.3f}, {z_valid_km.max():.3f}] km")
    print(f"  z mean: {z_valid_km.mean():.3f} km")
    print(f"  z std: {z_valid_km.std():.3f} km")

    # Check if there are different depth values (confirming uneven surface)
    unique_depths = np.unique(np.round(z_valid_km, 2))
    print(f"Number of unique depth levels (rounded to 0.01 km): {len(unique_depths)}")
    if len(unique_depths) > 1:
        print(f"  Depths range from {unique_depths.min():.2f} to {unique_depths.max():.2f} km")
        print(f"  Confirming uneven top boundary across the mesh!")
    else:
        print(f"  All points at the same depth: {unique_depths[0]:.2f} km")
else:
    # For even meshes, z=0 is already set correctly
    print(sep, "Using z=0 for observation grid (even mesh)", sep)

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
# Create a routine that solves the forward problem from a given slip to the ground displacement based on ANY given elastic structure
def solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=True, \
                          pollute_type='uniform', savefiles=True, verbose=True):
    """
    Solve forward problem to compute surface displacements at target points.

    NOTE: Now includes LEVEL 2 extrapolation for points where FEniCS cannot compute
    displacements, ensuring all target points get displacement values (either computed
    or extrapolated from nearby valid points).

    Returns:
        mtrue, mtrue_mu, u_mtrue, xf, yf, zf, d_full, mtrue_s_fault
        where d_full includes both computed and extrapolated displacements for all targets
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
    # CG_mu_deg = 2  # Use higher-order CG element for mu interpolation, 1 was used previously
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    # If mtrue_mu_expr_for is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_for == None:
        # OLD:
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # # NEW, using convex hull to avoid extrapolation artifacts
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # contrast_factor = 4.0
        # min_mu = 5.0  # Minimum physical value (GPa)
        min_mu = 1.0  # Minimum physical value (GPa)

        # mtrue_mu_fun = mu_func  # Contains shear modulus in GPa
        # contrast_factor = 1.0
        # mtrue_mu_fun.vector()[:] *= contrast_factor

        # ### The original way of scaling re global mean
        # # Amplify the contrast wrt a reference value
        # # mu_ref = mu_expression(0)   # which gives 40 GPa
        # mu_ref = np.mean(mu_func.vector()[:])
        # print(f"Reference mu (gloabl mean): {mu_ref:.2f} GPa")
        # mtrue_mu_fun = dl.Function(CG_mu)
        # new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
        # mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)

        # ### new way of amplication, relative to 1D depth-layered values
        # mtrue_mu_fun = ut.scale_shear_modulus_by_1d(
        #     mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=True)

        # New (linear-interp pair):
        vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull_interp(
            vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)
        
        mtrue_mu_fun = ut.scale_shear_modulus_by_1d_interp(
            mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=False)  
        
    else:
        # Assign the values of the vector
        mtrue_mu_expr = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
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
        filename_base = resultpath + 'mu_true_' + meshname + mu_str_for
        save_function_with_dofs(m_mu_true, filename_base, 'shear modulus', cg_degree=CG_mu_deg)
        print( m_mu_true.vector().min(), m_mu_true.vector().max() )

        # Save true Vs and density structures if using 3D & 1D velocity models
        if mtrue_mu_expr_for == None:
            print( "Saving true Vs structure to .xdmf file" )
            # vs_true = dl.project( vs_func, CG_mu )  #didn't break anything, but redundant, already in CG_mu space
            vs_true = vs_func
            filename_base = resultpath + 'vs_true_' + meshname + mu_str_for
            save_function_with_dofs(vs_true, filename_base, 'shear velocity Vs', cg_degree=CG_mu_deg)
            print( vs_true.vector().min(), vs_true.vector().max() )

            print( "Saving true density structure to .xdmf file" )
            # den_true = dl.project( den_func, CG_mu )    #didn't break anything, but redundant, already in CG_mu space
            den_true = den_func
            filename_base = resultpath + 'pho_true_' + meshname + mu_str_for
            save_function_with_dofs(den_true, filename_base, 'density', cg_degree=CG_mu_deg)
            print( den_true.vector().min(), den_true.vector().max() )

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

    # ============================================================================
    # LEVEL 2 CHECK: Handle points where FEniCS couldn't compute displacements
    # ============================================================================
    # Even though ray casting (LEVEL 1) found valid z-depths for all (x,y),
    # FEniCS might still fail to compute displacements for some points due to:
    # - Numerical precision at cell boundaries
    # - Mesh topology edge cases
    # If your grid matches the mesh well, expect <<1% or 0% invalid here.

    n_valid_points = len(d_obs) // 3
    n_total_points = targets.shape[0]

    if n_valid_points < n_total_points:
        if verbose:
            print(f"\n[LEVEL 2] Only {n_valid_points} out of {n_total_points} points computed successfully")
            print(f"          {n_total_points - n_valid_points} points failed during forward solve")
            print(f"          (These had valid depths from ray casting but FEniCS couldn't evaluate them)")

        # Build bounding box tree to identify which specific points failed
        bbtree = dl.BoundingBoxTree()
        bbtree.build(mesh)

        # Find valid target indices
        valid_indices = []
        for i in range(targets.shape[0]):
            point = dl.Point(targets[i,0], targets[i,1], targets[i,2])
            cell_id = bbtree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells():
                valid_indices.append(i)

        if verbose:
            print(f"          Verified: {len(valid_indices)} points have valid displacements")

        # Extrapolate displacements for points that failed
        d_full = extrapolate_displacements(targets, d_obs, valid_indices, verbose=verbose)
    else:
        # All points computed successfully - no extrapolation needed
        if verbose:
            print(f"\n[LEVEL 2] All {n_total_points} points computed successfully - no extrapolation needed!")
        valid_indices = list(range(targets.shape[0]))
        d_full = d_obs

    if savefiles:
        # Save the observed surface displacement for all points (with extrapolation for invalid points)
        outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(
                targets[i,0], targets[i,1], targets[i,2],
                d_full[3*i], d_full[3*i+1], d_full[3*i+2]) )
        csvoutput.close()

        if verbose:
            print(f"Saved {n_total_points} observations (including {n_total_points - n_valid_points} extrapolated) to {outFileName}")

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
    # m_mu_true = dl.project(mtrue_mu_fun, CG_mu) #didn't break anything, but redundant, already in CG_mu space 
    if isinstance(mtrue_mu_fun, dl.Function):
        m_mu_true = mtrue_mu_fun
    else:
        # mtrue_mu_expr_fun is already in CG_mu space, compute mu = 20*(2+tanh(m)) directly
        m_mu_true = dl.Function(CG_mu)
        m_values = mtrue_mu_expr_fun.vector()[:]
        m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
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

    # Return d_full (all points with extrapolation) instead of d_obs (only valid points)
    return mtrue, mtrue_mu_fun, u_mtrue, xf, yf, zf, d_full, mtrue_s_fault

# %%
def computeGridDisplacements(k, targets, mtrue_mu_expr_for, mtrue_s_expr=None, pollute=False, \
                          pollute_type='uniform', savefiles=True, verbose=True):
    """
    Compute surface displacements at given target points using specified slip and shear modulus.

    NOTE: Now includes LEVEL 2 extrapolation for points where FEniCS cannot compute
    displacements, ensuring all target points get displacement values (either computed
    or extrapolated from nearby valid points).

    Args:
        k: BDM element order
        targets: numpy array of observation points [N x 3] (x, y, z in meters)
        mtrue_mu_expr_for: shear modulus expression (None for 3D velocity model)
        mtrue_s_expr: slip expression (None to use slip_gt)
        pollute: whether to add noise
        pollute_type: 'uniform' or 'datastd'
        savefiles: whether to save results
        verbose: print progress info

    Returns:
        d_full: array of displacement values for all points (length = 3*n_targets)
                with extrapolated values filled in for invalid points
        valid_indices: list of indices where FEniCS successfully computed displacements
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
    # CG_mu_deg = 2  # Use higher-order CG element for mu interpolation, 1 was used previously
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    # If mtrue_mu_expr_for is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_for == None:
        # OLD:
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # # NEW, using convex hull to avoid extrapolation artifacts
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # contrast_factor = 4.0
        # min_mu = 5.0  # Minimum physical value (GPa)
        min_mu = 1.0  # Minimum physical value (GPa)

        # mtrue_mu_fun = mu_func  # Contains shear modulus in GPa
        # contrast_factor = 1.0
        # mtrue_mu_fun.vector()[:] *= contrast_factor

        # ### The original way of scaling re global mean
        # # Amplify the contrast wrt a reference value
        # # mu_ref = mu_expression(0)   # which gives 40 GPa
        # mu_ref = np.mean(mu_func.vector()[:])
        # print(f"Reference mu (gloabl mean): {mu_ref:.2f} GPa")
        # mtrue_mu_fun = dl.Function(CG_mu)
        # new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
        # mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)

        # ### new way of amplication, relative to 1D depth-layered values
        # mtrue_mu_fun = ut.scale_shear_modulus_by_1d(
        #     mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=True)

        # New (linear-interp pair):
        vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull_interp(
            vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)
        
        mtrue_mu_fun = ut.scale_shear_modulus_by_1d_interp(
            mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=False)  
        
    else:
        # Assign the values of the vector
        mtrue_mu_expr = dl.interpolate(mtrue_mu_expr_for, CG_mu).vector()
        mtrue_mu_expr_fun = hp.vector2Function(mtrue_mu_expr, CG_mu)
        mtrue_mu_fun = mu_expression(mtrue_mu_expr_fun)
    
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

    # ============================================================================
    # LEVEL 2 CHECK: Handle points where FEniCS couldn't compute displacements
    # ============================================================================
    # Even though ray casting (LEVEL 1) found valid z-depths for all (x,y),
    # FEniCS might still fail to compute displacements for some points due to:
    # - Numerical precision at cell boundaries
    # - Mesh topology edge cases
    # If your grid matches the mesh well, expect <<1% or 0% invalid here.

    n_valid_points = len(d_obs) // 3
    n_total_points = targets.shape[0]

    if n_valid_points < n_total_points:
        if verbose:
            print(f"\n[LEVEL 2] Only {n_valid_points} out of {n_total_points} points computed successfully")
            print(f"          {n_total_points - n_valid_points} points failed during forward solve")
            print(f"          (These had valid depths from ray casting but FEniCS couldn't evaluate them)")

        # Build bounding box tree to identify which specific points failed
        bbtree = dl.BoundingBoxTree()
        bbtree.build(mesh)

        # Find valid target indices
        valid_indices = []
        for i in range(targets.shape[0]):
            point = dl.Point(targets[i,0], targets[i,1], targets[i,2])
            cell_id = bbtree.compute_first_entity_collision(point)
            if cell_id < mesh.num_cells():
                valid_indices.append(i)

        if verbose:
            print(f"          Verified: {len(valid_indices)} points have valid displacements")

        # Extrapolate displacements for points that failed
        d_full = extrapolate_displacements(targets, d_obs, valid_indices, verbose=verbose)
    else:
        # All points computed successfully - no extrapolation needed
        if verbose:
            print(f"\n[LEVEL 2] All {n_total_points} points computed successfully - no extrapolation needed!")
        valid_indices = list(range(targets.shape[0]))
        d_full = d_obs

    if savefiles:
        # Save surface displacements for all points (with extrapolation for invalid points)
        outFileName = 'd_obs_grid' + meshname + slip_str_gt + mu_str_for + str(grid_spacing_deg) + '.txt'
        csvoutput = open(resultpath + outFileName, 'w+')
        for i in range(targets.shape[0]):
            csvoutput.write( "%.6f %.6f %.6f %.6f %.6f %.6f\n" %(
                targets[i,0], targets[i,1], targets[i,2],
                d_full[3*i], d_full[3*i+1], d_full[3*i+2]) )
        csvoutput.close()

        if verbose:
            print(f"Saved {n_total_points} observations (including {n_total_points - n_valid_points} extrapolated) to {outFileName}")

    return d_full, valid_indices

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
    # CG_mu_deg = 2  # Use higher-order CG element for mu interpolation, 1 was used previously
    CG_mu = dl.FunctionSpace(mesh, "CG", CG_mu_deg)
    # If mtrue_mu_expr_inv is None, use the 3D & 1D velocity models to compute shear modulus
    if mtrue_mu_expr_inv == None:
        # OLD:
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # # NEW, using convex hull to avoid extrapolation artifacts
        # vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull(vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)

        # contrast_factor = 4.0
        # min_mu = 5.0  # Minimum physical value (GPa)
        min_mu = 1.0  # Minimum physical value (GPa)

        # mtrue_mu_fun = mu_func  # Contains shear modulus in GPa
        # contrast_factor = 1.0
        # mtrue_mu_fun.vector()[:] *= contrast_factor

        # ### The original way of scaling re global mean
        # # Amplify the contrast wrt a reference value
        # # mu_ref = mu_expression(0)   # which gives 40 GPa
        # mu_ref = np.mean(mu_func.vector()[:])
        # print(f"Reference mu (gloabl mean): {mu_ref:.2f} GPa")
        # mtrue_mu_fun = dl.Function(CG_mu)
        # new_values = mu_ref + contrast_factor * (mu_func.vector()[:] - mu_ref)
        # mtrue_mu_fun.vector()[:] = np.maximum(new_values, min_mu)

        # ### new way of amplication, relative to 1D depth-layered values
        # mtrue_mu_fun = ut.scale_shear_modulus_by_1d(
        #     mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=True)

        # New (linear-interp pair):
        vs_func, den_func, mu_func, _ = ut.process_velocity_models_hull_interp(
            vel3d, vel1d, den1d, mesh, CG_mu_degree=CG_mu_deg, verbose=False)
        
        mtrue_mu_fun = ut.scale_shear_modulus_by_1d_interp(
            mu_func, vel1d, den1d, contrast_factor, min_mu=min_mu, verbose=False)  

    else:
        # Assign the values of the vector
        mtrue_mu_expr = dl.interpolate(mtrue_mu_expr_inv, CG_mu).vector()
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

    ### Define the misfit based on whether the noise was added, and how it was added
    if pollute:

        ##### USE 'pointwiseStateObs_weights'
        weights = dl.Vector(MPI.COMM_WORLD, targets.shape[0]*15)
        obs_weights = np.zeros(targets.shape[0]*15,)

        if pollute_type == "uniform":
            # ##### USE 'pointwiseStateObs', without 'weight' option
            # misfit = PSB( Vh[hp.STATE], targets, indicator_vec=indicator_vec )
            # misfit.noise_variance = noise_std_h*noise_std_h
            # ### Below is just to get the indice of displacement, 'weights' are NOT used by inversion
            # obs_weights = np.zeros(targets.shape[0]*15,)
            # obs_weights[9::15]  = 1  # horizontal east displacement 
            # obs_weights[10::15] = 1  # horizontal north displacement 
            # obs_weights[11::15] = 1  # vectical displacement 

            obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
            obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
            obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements

        elif pollute_type == "datastd":
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
    # m_mu_true = dl.project(mtrue_mu_fun, CG_mu) #didn't break anything, but redundant, already in CG_mu space  
    if isinstance(mtrue_mu_fun, dl.Function):
        m_mu_true = mtrue_mu_fun
    else:
        # mtrue_mu_expr_fun is already in CG_mu space, compute mu = 20*(2+tanh(m)) directly
        m_mu_true = dl.Function(CG_mu)
        m_values = mtrue_mu_expr_fun.vector()[:]
        m_mu_true.vector()[:] = 20.0 * (2.0 + np.tanh(m_values))
    moment = dl.assemble( m_mu_true * GPa2Pa * s_mag * dS(fault) )        
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

    return mtrue_mu_fun, xf, yf, zf, m, u, s_strike_fun, s_dip_fun, d_obs, d_cal, np.concatenate([m_sx_fault, m_sy_fault]), misfitd, grad_m 

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
# slip_pattern = "checker"
slip_pattern = "stripe"

### stripe pattern options
pattern_option = 1  # 1: 2-stripe, shallow-deep; 2: 1-stripe, intermediate
# pattern_option = 2

if slip_pattern == "checker":
    ##### NEW CHECKERBOARD SLIP MODEL #####
    print("Creating fault-local checkerboard with natural depth range...")

    # from utils import FaultLocalCheckerboard

    # Define true model, m_strike = 0, m_dip = checkerboard, alternating between 0 and max
    V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
    amp = V_norm   # interseismic coupling case, max == complete coupling of 1 == the trench-normal long-term loading
    # amp = 1   # coseismic case, max == 1 m
    
    zmin = -70e3
    zmax = 0
    smin = -120e3
    smax = 120e3

    # Checkerboard pattern parameters
    if pattern_option == 1:
        # Option 1, checkerboard along fault strike-dip directions
        dx = 35e3  # grid spacing in x direction, \lamda_x = 2*dx
        dy = 35e3  # spacing in y direction, \lamda_y = 2*dy
        x0 = -20e3  # offset along strike
        y0 = -60e3  # offset along dip
        rot_deg = 45  # 45° counterclockwise 

    elif pattern_option == 2:
        # Option 2, checkerboard along N-E directions
        dx = 30e3  # grid spacing in x direction, \lamda_x = 2*dx
        dy = 30e3  # spacing in y direction, \lamda_y = 2*dy
        x0 = 20e3  # offset along strike
        y0 = -15e3  # offset along dip
        rot_deg = 0

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
        # zmin=zmin, zmax=zmax,
        # smin=smin, smax=smax,
        degree=5
    )

    slip_str_gt = f"_check_x{x0/1e3:g}_y{y0/1e3:g}_dx{dx/1e3:g}_dy{dy/1e3:g}_rot{rot_deg:g}_ms{amp:g}"


elif slip_pattern == "stripe":
    ##### STRIPE SLIP MODEL #####
    print("Creating fault-local stripes with natural depth range...")

    # from utils import FaultLocalStripes

    # Define true model, m_strike = 0, m_dip = stripes along-strike, alternating between 0 and max
    V_norm = 78.5 / 1e3     # the trench-normal long-term loading of 78.5 mm
    amp = V_norm   # interseismic coupling case, max == complete coupling of 1 == the trench-normal long-term loading
    # amp = 1   # coseismic case, max == 1 m
    
    ### Stripe pattern parameters
    # # Stripe pattern in m_dip, along-strike, 3-stripe, shallow-middle-deep; m_strike = 0
    # x_len = 35e3     # width of each rectangle in dip direction
    # y_len = 300e3   # length of each rectangle in strike direction  
    # dx = 25e3  # gap between rectangles
    # stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
    # x0 = 0     # x center of the pattern
    # y0 = -5e3     # y center of the pattern
    # rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)

    if pattern_option == 1:
        # Stripe pattern in m_dip, along-strike, 2-stripe, shallow-deep; m_strike = 0
        x_len = 80e3     # width of each rectangle in dip direction
        y_len = 300e3   # length of each rectangle in strike direction  
        dx = 35e3  # gap between rectangles
        stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
        x0 = 0     # x center of the pattern
        y0 = -45e3     # y center of the pattern
        rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)

    elif pattern_option == 2:
        # Stripe pattern in m_dip, along-strike, 1-stripe, form complementary pattern to the above; m_strike = 0
        x_len = 40e3     # width of each rectangle in dip direction
        y_len = 300e3   # length of each rectangle in strike direction  
        dx = 100e3  # gap between rectangles
        stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
        x0 = 0     # x center of the pattern
        y0 = 12.5e3     # y center of the pattern
        rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)

    #### Below 2 is specific to 'nicoyaCK4' fault model
    # # Stripe pattern in m_dip, along-strike, 2-stripe, shallow-deep; m_strike = 0, *for nicoyaCK3*
    # x_len = 80e3     # width of each rectangle in dip direction
    # y_len = 240e3   # length of each rectangle in strike direction  
    # dx = 40e3  # gap between rectangles
    # stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
    # x0 = 0e3     # x center of the pattern
    # y0 = -40e3     # y center of the pattern
    # rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)
    # zmin = -70e3
    # zmax = 0

    # # Stripe pattern in m_dip, along-strike, 1-stripe, form complementary pattern to the above; m_strike = 0, *for nicoyaCK3*
    # x_len = 40e3     # width of each rectangle in dip direction
    # y_len = 240e3   # length of each rectangle in strike direction  
    # dx = 100e3  # gap between rectangles
    # stripe_spacing = x_len + dx  # center-to-center distance between rectangles in dip direction
    # x0 = 0e3     # x center of the pattern
    # y0 = 12.5e3     # y center of the pattern
    # # y0 = -40e3     # y center of the pattern
    # rot_deg = 0.0  # rotation angle in degrees (counter-clockwise positive)
    # zmin = -70e3
    # zmax = 0
    
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
        # zmin=zmin, zmax=zmax,
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

# Dense grid
ntargets_dense = len(dense_data)
targets_grid = np.zeros([ntargets_dense, 3])
targets_grid[:,0] = np.array(dense_data['x'])*1e3   # km to m
targets_grid[:,1] = np.array(dense_data['y'])*1e3
targets_grid[:,2] = np.array(dense_data['z'])*1e3
print(targets_grid.shape)

# # Save dense grid coordinates with lat/lon for plotting
# coord_file = f"dense_grid_coordinates_{meshname}_{grid_spacing_deg}.txt"
# # if not os.path.exists(resultpath + coord_file):
# coord_output = open(resultpath + coord_file, 'w+')
# # coord_output.write("# lon lat x_km y_km z_km\n")
# for i in range(0, targets_grid.shape[0]):
#     coord_output.write("%.6f %.6f %.6f %.6f %.6f\n" %(
#         dense_data['lon'].iloc[i], dense_data['lat'].iloc[i],
#         dense_data['x'].iloc[i], dense_data['y'].iloc[i], dense_data['z'].iloc[i]
#     ))
# coord_output.close()

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
# Note: CG1 is the default for homogeneous/K_2LAYER, no suffix needed

print( "Homogeneous structure:")
print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f" %(mu_upper, mu_lower) )
# print( "The shear modulus for the upper plate mu = %.1f and lower plate mu = %.1f and volcano mu = %.1f" %(mu_upper, mu_lower, mu_volcano) )

# %%
# # Define the true model PARAMETERS for INVERSE problem
# nu = 0.25
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
# contrast_factor = 4.0  # amplification factor, too extreme, needs clipping, and not adopted since 03/05/2026
contrast_factor = 2.5  # amplification factor, more reasonable, and adopted since 03/05/2026

# mu_str_het = "_DeShon3D"
# mu_str_het = f"_DeShon3D_{round(contrast_factor)}"
# mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}"

# String: _hull indicates process_velocity_models_hull method
# mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}_hull"
mu_str_het = f"_DeShon3D_ref_{round(contrast_factor)}_hull_test1"

print( "Heterogeneous structure:")
print( "Converted from 3D & 1D velocity models to shear modulus, mu_str_het = ", mu_str_het)


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
        # f_h, f_v = 1, 1/2
        f_h, f_v = 1, 1
        # Print the weights of the data
        print( "Data weight horizontal / vertical: %.2f / %.2f" %(f_h, f_v) )
        
        obs_weights = np.zeros(targets.shape[0]*15,)
        obs_weights[9::15]  = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal east displacement 
        obs_weights[10::15] = (1. / noise_std_h**2) * (1. / f_h**2)  # horizontal north displacement 
        obs_weights[11::15] = (1. / noise_std_v**2) * (1. / f_v**2)  # vertical displacements
        
        slip_str_gt = slip_str_gt + "_pou"

    elif pollute_type == 'datastd':
        # Decide the weights of the horizontal, vertical components
        # f_h, f_v = 1, 1/2
        f_h, f_v = 1, 1
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
# mtrue_s_expr_for = None
mtrue_s_expr_for = mtrue_s_expr_gt
if mtrue_s_expr_for==None:
    print("Ground-truth slip is from inversion")
else:
    print("Ground-truth slip is customized")  

# %%
# # Solve the coseismic forward problem to generate the synthetic data within a Homogeneous shear modulus half-space
# mtrue_mu_expr_for = mtrue_mu_expr_hom
# mu_str_for = mu_str_hom

# Solve the coseismic forward problem to generate the synthetic data within a Heterogeneous shear modulus half-space
mtrue_mu_expr_for = mtrue_mu_expr_het
mu_str_for = mu_str_het

# Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
if mtrue_mu_expr_for is None:  # 3D heterogeneous model via process_velocity_models_hull
    CG_mu_deg = 2
    mu_str_for = mu_str_for + f"_CG{CG_mu_deg}"
else:  # Homogeneous or K_2LAYER
    CG_mu_deg = 1

print(f"Solving forward problem based on: {mu_str_for}, CG_mu_deg = {CG_mu_deg}")

# %%
# Solve forward problem to generate synthetics
solveCoseismicForward(k, targets, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
                      pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True)
print("Forward problem done!!!")

# Solve forward problem to generate synthetics at a dense grid
d_obs_grid, valid_grid_indices = computeGridDisplacements(k, targets_grid, mtrue_mu_expr_for, mtrue_s_expr=mtrue_s_expr_for, \
                      pollute=False, pollute_type=pollute_type, savefiles=True, verbose=True)
n_total_grid = targets_grid.shape[0]
n_valid_grid = len(valid_grid_indices)
n_extrapolated = n_total_grid - n_valid_grid
print(f"Dense grid forward problem done!")
print(f"  Total points: {n_total_grid}")
print(f"  Valid (computed): {n_valid_grid} ({100*n_valid_grid/n_total_grid:.1f}%)")
print(f"  Extrapolated: {n_extrapolated} ({100*n_extrapolated/n_total_grid:.1f}%)")

# %% [markdown]
# Use the synthetic displacement for inversion within any $\mu$ structure model

# %%
# Load the synethetic displacement data from the forward problem
outFileName = 'd_obs_' + meshname + slip_str_gt + mu_str_for + '.txt'
syndata = pd.read_csv(resultpath + outFileName, sep=r'\s+', names=['x', 'y', 'z', 'ux', 'uy', 'uz'])

# %%
# SLIP TRANSFORMATION SETUP
print(sep + "SLIP TRANSFORMATION SETUP" + sep)

BOUNDED = True
# BOUNDED = False

BOUND_TYPE = 'both'
# BOUND_TYPE = 'dip'

if BOUNDED:
    # Define slip bounds based on your problem
    V_para = 16/1e3
    if BOUND_TYPE == 'both':
        slip_transformer = SlipTransformation(
            strike_bounds=(-2e-3, 2e-3),    # just a little mm. although the truth is 0
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

if not BOUNDED:
    if meshname == "nicoyaCK3":    # large fault extending to the whole subduction zone
        # gamma_val_H1 = 1e3  
        # delta_val_L2 = 1e-4
        # delta_val_L2 = 1e-5
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 2.5e-5
        # gamma_val_H1 = 2.5e3  
        # delta_val_L2 = 5e-5
        gamma_val_H1 = 5e3    # best so far
        delta_val_L2 = 5e-6
        # gamma_val_H1 = 7.5e3  
        # delta_val_L2 = 7.5e-6

else:
    if meshname == "nicoyaCK4":     # small fault connecting trench
        rho_s = 1e9   # allows variations of slip of the order of ~30 km 

        # rho_s = 1e8   # allows variations of slip of the order of ~3 km, close to the maximum resolution
        # gamma_val_H1 = 5e1  
        gamma_val_H1 = 1e2    # best so far
        # gamma_val_H1 = 4e2
        # gamma_val_H1 = 1e3    
        # gamma_val_H1 = 1e4    
        delta_val_L2 = gamma_val_H1 / rho_s  


# newest preferred values for the dense mesh, as of 12/10/2025
rho_s = 1e9   # allows variations of slip of the order of ~30 km
gamma_val_H1 = 2.5e2  # used as of 12/10/2025, used gamma_val_H1:.1e
delta_val_L2 = gamma_val_H1 / rho_s


# %%
if pollute:
    if pollute_type == 'uniform':
        # Take the inverse for saving the name of the weights
        w_h, w_v = int(1/noise_std_h), int(1/noise_std_v)
    elif pollute_type == 'datastd':
        # Take the inverse for saving the name of the weights
        w_h, w_v = int(1/f_h), int(1/f_v)
else:
    # Take the inverse for saving the name of the weights
    w_h, w_v = int(1/f_h), int(1/f_v)

# file identifier
if BOUNDED:
    inv_str = f"_synlockbd_w{w_h}{w_v}_gs{gamma_val_H1:.1e}_ds{delta_val_L2:.1e}"
else:
    inv_str = f"_synlock_w{w_h}{w_v}_gs{gamma_val_H1:.1e}_ds{delta_val_L2:.1e}"
print("Inverse problem identifier: ", inv_str)

# %%
# Solve slip inverse problem within the heterogeneous shear modulus half-space
mtrue_mu_expr_inv = mtrue_mu_expr_het
mu_str_inv = mu_str_het
# Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
if mtrue_mu_expr_inv is None:  # 3D heterogeneous model via process_velocity_models_hull
    CG_mu_deg = 2
    mu_str_inv = mu_str_inv + f"_CG{CG_mu_deg}"
else:  # Homogeneous or K_2LAYER
    CG_mu_deg = 1
print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")
results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
        slip_transformer,  # KEY: Pass slip transformation
        pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    )
print("Het Inversion finished!!!")

# %%
# ===== AGGRESSIVE MEMORY CLEANUP =====
# Delete the results tuple containing 11 other large objects
del results

# Force Python garbage collector
import gc  # for explicit memory management
gc.collect()

# Optional: Monitor memory
try:
    import psutil
    mem_gb = psutil.Process().memory_info().rss / 1024**3
    print(f"  Memory after cleanup: {mem_gb:.2f} GB")
except ImportError:
    pass
# ======================================

# %%
# Solve slip inverse problem within the homogeneous shear modulus half-space
mtrue_mu_expr_inv = mtrue_mu_expr_hom
mu_str_inv = mu_str_hom
# Set CG degree based on model type: CG2 for 3D model, CG1 (default) otherwise
if mtrue_mu_expr_inv is None:  # 3D heterogeneous model via process_velocity_models_hull
    CG_mu_deg = 2
    mu_str_inv = mu_str_inv + f"_CG{CG_mu_deg}"
else:  # Homogeneous or K_2LAYER
    CG_mu_deg = 1
print(f"Solving inverse problem based on: {mu_str_inv}, CG_mu_deg = {CG_mu_deg}")
results = solveCoseismicInversion_TanhSlip(
        k, targets, m0_s_expr, mtrue_mu_expr_inv, gamma_val_H1, delta_val_L2,
        slip_transformer,  # KEY: Pass slip transformation
        pollute=pollute, pollute_type=pollute_type, savefiles=True, verbose=True
    )
print("Hom Inversion finished!!!")
