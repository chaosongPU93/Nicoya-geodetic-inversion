#!/usr/bin/env python3
"""
Diagnose which observation points are outside the mesh and why
"""
import argparse
import dolfin as dl
import numpy as np
import pandas as pd
import utils as ut

parser = argparse.ArgumentParser(description='Diagnose which observation points are outside the mesh')
parser.add_argument('--mesh', '-m', type=str, default='nicoyaCK_uneven',
                    help='Mesh name (default: nicoyaCK_uneven)')
parser.add_argument('--meshpath', '-p', type=str, default='/home/staff/chao/SSEinv/Nicoya/mesh/',
                    help='Path to mesh directory')
args = parser.parse_args()

meshpath = args.meshpath
meshname = args.mesh

# Load mesh and boundaries
mesh = dl.Mesh(meshpath + meshname + '.xml')
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')
top = 1

print("="*80)
print("DIAGNOSING EXCLUDED OBSERVATION POINTS")
print("="*80)

# Extract top boundary
def extract_top_boundary_surface(mesh, boundaries, top_id):
    all_coords = mesh.coordinates()
    top_facets = []
    for facet in dl.facets(mesh):
        if boundaries[facet] == top_id:
            top_facets.append(facet)

    top_vertex_indices = set()
    for facet in top_facets:
        for vertex in dl.vertices(facet):
            top_vertex_indices.add(vertex.index())

    return all_coords[list(top_vertex_indices)]

top_coords = extract_top_boundary_surface(mesh, boundaries, top)
print(f"Top boundary: {len(top_coords)} vertices")
print(f"  x-range: [{top_coords[:,0].min()/1e3:.1f}, {top_coords[:,0].max()/1e3:.1f}] km")
print(f"  y-range: [{top_coords[:,1].min()/1e3:.1f}, {top_coords[:,1].max()/1e3:.1f}] km")
print(f"  z-range: [{top_coords[:,2].min()/1e3:.3f}, {top_coords[:,2].max()/1e3:.3f}] km")
print()

# Create observation grid (same as in script)
lon0, lat0 = -84, 7
rot = 45  # rotation angle in degrees
x0, y0 = 130e3, 350e3
region = [-88, -83, 7.4, 12.6]
grid_spacing_deg = 0.01

lon_min, lon_max = region[0], region[1]
lat_min, lat_max = region[2], region[3]

lon_grid = np.arange(lon_min, lon_max + grid_spacing_deg, grid_spacing_deg)
lat_grid = np.arange(lat_min, lat_max + grid_spacing_deg, grid_spacing_deg)
LON_GRID, LAT_GRID = np.meshgrid(lon_grid, lat_grid)

lon_2d = LON_GRID.flatten()
lat_2d = LAT_GRID.flatten()

x_rot_2d, y_rot_2d = ut.LL2ckmd(lon_2d, lat_2d, lon0, lat0, rot)
x_2d = (x_rot_2d - x0) / 1e3  # km
y_2d = (y_rot_2d - y0) / 1e3

# Interpolate z from top boundary
from scipy.interpolate import NearestNDInterpolator
x_top = top_coords[:, 0]
y_top = top_coords[:, 1]
z_top = top_coords[:, 2]

interp = NearestNDInterpolator(list(zip(x_top, y_top)), z_top)
x_2d_m = x_2d * 1e3
y_2d_m = y_2d * 1e3
z_2d_m = interp(x_2d_m, y_2d_m)

print(f"Observation grid: {len(x_2d)} points")
print(f"  x-range: [{x_2d.min():.1f}, {x_2d.max():.1f}] km")
print(f"  y-range: [{y_2d.min():.1f}, {y_2d.max():.1f}] km")
print(f"  z-range: [{z_2d_m.min()/1e3:.3f}, {z_2d_m.max()/1e3:.3f}] km")
print()

# Check which points are inside mesh
bbtree = dl.BoundingBoxTree()
bbtree.build(mesh)

inside_mask = np.zeros(len(x_2d), dtype=bool)
for i in range(len(x_2d)):
    point = dl.Point(x_2d_m[i], y_2d_m[i], z_2d_m[i])
    cell_id = bbtree.compute_first_entity_collision(point)
    if cell_id < mesh.num_cells():
        inside_mask[i] = True

n_inside = np.sum(inside_mask)
n_outside = len(x_2d) - n_inside

print(f"Points inside mesh: {n_inside} ({100*n_inside/len(x_2d):.1f}%)")
print(f"Points outside mesh: {n_outside} ({100*n_outside/len(x_2d):.1f}%)")
print()

if n_outside > 0:
    # Analyze the excluded points
    x_outside = x_2d[~inside_mask]
    y_outside = y_2d[~inside_mask]
    z_outside = z_2d_m[~inside_mask] / 1e3

    print("EXCLUDED POINTS ANALYSIS:")
    print(f"  x-range: [{x_outside.min():.1f}, {x_outside.max():.1f}] km")
    print(f"  y-range: [{y_outside.min():.1f}, {y_outside.max():.1f}] km")
    print(f"  z-range: [{z_outside.min():.3f}, {z_outside.max():.3f}] km")
    print()

    # Check if excluded points are on the edges
    x_is_edge = (x_outside < x_2d.min() + 10) | (x_outside > x_2d.max() - 10)
    y_is_edge = (y_outside < y_2d.min() + 10) | (y_outside > y_2d.max() - 10)
    edge_points = np.sum(x_is_edge | y_is_edge)

    print(f"Points near grid edges (within 10 km): {edge_points} ({100*edge_points/n_outside:.1f}% of excluded)")
    print()

    # Check spatial distribution
    print("Spatial distribution of excluded points:")
    # Divide into quadrants
    x_mid = (x_2d.min() + x_2d.max()) / 2
    y_mid = (y_2d.min() + y_2d.max()) / 2

    q1 = np.sum((x_outside < x_mid) & (y_outside < y_mid))
    q2 = np.sum((x_outside >= x_mid) & (y_outside < y_mid))
    q3 = np.sum((x_outside < x_mid) & (y_outside >= y_mid))
    q4 = np.sum((x_outside >= x_mid) & (y_outside >= y_mid))

    print(f"  Q1 (x<{x_mid:.1f}, y<{y_mid:.1f}): {q1}")
    print(f"  Q2 (x>={x_mid:.1f}, y<{y_mid:.1f}): {q2}")
    print(f"  Q3 (x<{x_mid:.1f}, y>={y_mid:.1f}): {q3}")
    print(f"  Q4 (x>={x_mid:.1f}, y>={y_mid:.1f}): {q4}")
    print()

    # Save excluded points for visualization
    outfile = '/home/staff/chao/SSEinv/Nicoya/syn_slip/excluded_points.txt'
    with open(outfile, 'w') as f:
        f.write("# x_km y_km z_km\n")
        for i in range(len(x_outside)):
            f.write(f"{x_outside[i]:.6f} {y_outside[i]:.6f} {z_outside[i]:.6f}\n")
    print(f"Saved excluded points to: {outfile}")

    # Also save included points for comparison
    x_inside = x_2d[inside_mask]
    y_inside = y_2d[inside_mask]
    z_inside = z_2d_m[inside_mask] / 1e3

    outfile = '/home/staff/chao/SSEinv/Nicoya/syn_slip/included_points.txt'
    with open(outfile, 'w') as f:
        f.write("# x_km y_km z_km\n")
        for i in range(len(x_inside)):
            f.write(f"{x_inside[i]:.6f} {y_inside[i]:.6f} {z_inside[i]:.6f}\n")
    print(f"Saved included points to: {outfile}")

print("="*80)
