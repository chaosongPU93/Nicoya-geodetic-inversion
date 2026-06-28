#!/usr/bin/env python3
"""
Check which observation points are inside vs outside the mesh
"""
import argparse
import dolfin as dl
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Check which observation points are inside vs outside the mesh')
parser.add_argument('--mesh', '-m', type=str, default='nicoyaCK_uneven',
                    help='Mesh name (default: nicoyaCK_uneven)')
parser.add_argument('--meshpath', '-p', type=str, default='/home/staff/chao/SSEinv/Nicoya/mesh/',
                    help='Path to mesh directory')
args = parser.parse_args()

meshpath = args.meshpath
meshname = args.mesh

# Load mesh
mesh = dl.Mesh(meshpath + meshname + '.xml')

# Get mesh bounding box
coords = mesh.coordinates()
x_min, x_max = coords[:,0].min(), coords[:,0].max()
y_min, y_max = coords[:,1].min(), coords[:,1].max()
z_min, z_max = coords[:,2].min(), coords[:,2].max()

print("="*80)
print("MESH BOUNDING BOX (in km)")
print("="*80)
print(f"x: [{x_min/1e3:.1f}, {x_max/1e3:.1f}] km")
print(f"y: [{y_min/1e3:.1f}, {y_max/1e3:.1f}] km")
print(f"z: [{z_min/1e3:.1f}, {z_max/1e3:.1f}] km")
print()

# Simulate the observation grid creation (from the script)
import utils as ut

# Reference point and rotation (from lines 64-67 of the script)
lon0, lat0, rot = 275.0, 9.5, 36.0

# Offsets (from lines 75-76)
x0, y0 = 130e3, 350e3

# Grid definition (from lines 179-202)
region = [-88, -83, 7.4, 12.6]
lon_min, lon_max = region[0], region[1]
lat_min, lat_max = region[2], region[3]
grid_spacing_deg = 0.01

lon_grid = np.arange(lon_min, lon_max + grid_spacing_deg, grid_spacing_deg)
lat_grid = np.arange(lat_min, lat_max + grid_spacing_deg, grid_spacing_deg)
LON_GRID, LAT_GRID = np.meshgrid(lon_grid, lat_grid)

lon_2d = LON_GRID.flatten()
lat_2d = LAT_GRID.flatten()

# Convert to mesh coordinates
x_rot_2d, y_rot_2d = ut.LL2ckmd(lon_2d, lat_2d, lon0, lat0, rot)
x_2d = (x_rot_2d - x0) / 1e3  # km
y_2d = (y_rot_2d - y0) / 1e3

x_2d_m = x_2d * 1e3  # convert to meters
y_2d_m = y_2d * 1e3

print("OBSERVATION GRID RANGE (in km)")
print("="*80)
print(f"x: [{x_2d.min():.1f}, {x_2d.max():.1f}] km")
print(f"y: [{y_2d.min():.1f}, {y_2d.max():.1f}] km")
print(f"Total points: {len(x_2d)}")
print()

# Check which points are inside the mesh bounding box
inside_x = (x_2d_m >= x_min) & (x_2d_m <= x_max)
inside_y = (y_2d_m >= y_min) & (y_2d_m <= y_max)
inside_bbox = inside_x & inside_y

print("POINTS INSIDE MESH BOUNDING BOX")
print("="*80)
print(f"Inside x-range: {np.sum(inside_x)} / {len(x_2d)} ({100*np.sum(inside_x)/len(x_2d):.1f}%)")
print(f"Inside y-range: {np.sum(inside_y)} / {len(y_2d)} ({100*np.sum(inside_y)/len(y_2d):.1f}%)")
print(f"Inside both x & y: {np.sum(inside_bbox)} / {len(x_2d)} ({100*np.sum(inside_bbox)/len(x_2d):.1f}%)")
print()

# Check with actual mesh cell containment (more accurate)
print("Checking actual cell containment (this may take a moment)...")
bbtree = dl.BoundingBoxTree()
bbtree.build(mesh)

points_in_mesh = 0
for i in range(len(x_2d_m)):
    point = dl.Point(x_2d_m[i], y_2d_m[i], 0.0)  # use z=0 as placeholder
    cell_id = bbtree.compute_first_entity_collision(point)
    if cell_id < mesh.num_cells():
        points_in_mesh += 1

print(f"Points inside mesh cells: {points_in_mesh} / {len(x_2d)} ({100*points_in_mesh/len(x_2d):.1f}%)")
print(f"Points outside mesh: {len(x_2d) - points_in_mesh}")
print()

expected_displacements = points_in_mesh * 3
print(f"Expected displacement values: {expected_displacements} (points_in_mesh * 3)")
print("="*80)
