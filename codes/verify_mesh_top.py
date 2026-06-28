#!/usr/bin/env python3
"""
Quick script to verify the top boundary statistics of a mesh
"""
import argparse
import dolfin as dl
import numpy as np

parser = argparse.ArgumentParser(description='Verify mesh top boundary statistics')
parser.add_argument('--mesh', '-m', type=str, default='nicoyaCK_uneven',
                    help='Mesh name (default: nicoyaCK_uneven)')
parser.add_argument('--meshpath', '-p', type=str, default='/home/staff/chao/SSEinv/Nicoya/mesh/',
                    help='Path to mesh directory')
args = parser.parse_args()

meshpath = args.meshpath
meshname = args.mesh

# Load mesh
mesh = dl.Mesh(meshpath + meshname + '.xml')
boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')

# Get all mesh coordinates
all_coords = mesh.coordinates()

print("="*80)
print("MESH STATISTICS")
print("="*80)
print(f"Total vertices: {len(all_coords)}")
print(f"Overall z-range: [{all_coords[:,2].min()/1e3:.3f}, {all_coords[:,2].max()/1e3:.3f}] km")
print()

# Check top boundary (boundary ID = 1)
top_id = 1
top_facets = []
for facet in dl.facets(mesh):
    if boundaries[facet] == top_id:
        top_facets.append(facet)

print(f"Top boundary facets: {len(top_facets)}")

# Extract unique vertices on top boundary
top_vertex_indices = set()
for facet in top_facets:
    for vertex in dl.vertices(facet):
        top_vertex_indices.add(vertex.index())

top_coords = all_coords[list(top_vertex_indices)]

print(f"Top boundary vertices: {len(top_coords)}")
print(f"Top boundary z-range: [{top_coords[:,2].min()/1e3:.3f}, {top_coords[:,2].max()/1e3:.3f}] km")
print(f"Top boundary x-range: [{top_coords[:,0].min()/1e3:.3f}, {top_coords[:,0].max()/1e3:.3f}] km")
print(f"Top boundary y-range: [{top_coords[:,1].min()/1e3:.3f}, {top_coords[:,1].max()/1e3:.3f}] km")
print()

# Check z-distribution
z_values = top_coords[:,2] / 1e3
print("Z-distribution analysis:")
print(f"  Mean: {z_values.mean():.3f} km")
print(f"  Std: {z_values.std():.3f} km")
print(f"  Median: {np.median(z_values):.3f} km")
print(f"  25th percentile: {np.percentile(z_values, 25):.3f} km")
print(f"  75th percentile: {np.percentile(z_values, 75):.3f} km")
print()

# Count how many vertices at different depth ranges
at_surface = np.sum(np.abs(z_values) < 0.1)  # within 100m of surface
deep_7km = np.sum(z_values < -6.5)  # deeper than 6.5 km
deep_10km = np.sum(z_values < -9.5)  # deeper than 9.5 km

print("Depth distribution:")
print(f"  At surface (|z| < 0.1 km): {at_surface} vertices ({100*at_surface/len(z_values):.1f}%)")
print(f"  Deeper than -6.5 km: {deep_7km} vertices ({100*deep_7km/len(z_values):.1f}%)")
print(f"  Deeper than -9.5 km: {deep_10km} vertices ({100*deep_10km/len(z_values):.1f}%)")
print()

# Check if there are distinct regions
print("Checking for left/right block distinction...")
x_values = top_coords[:,0] / 1e3

# Split by x=0 (rough approximation)
left_mask = x_values < 0
right_mask = x_values >= 0

if np.sum(left_mask) > 0:
    z_left = z_values[left_mask]
    print(f"Left block (x < 0): {np.sum(left_mask)} vertices")
    print(f"  z-range: [{z_left.min():.3f}, {z_left.max():.3f}] km")
    print(f"  z-mean: {z_left.mean():.3f} km")
    print(f"  z-median: {np.median(z_left):.3f} km")

if np.sum(right_mask) > 0:
    z_right = z_values[right_mask]
    print(f"Right block (x >= 0): {np.sum(right_mask)} vertices")
    print(f"  z-range: [{z_right.min():.3f}, {z_right.max():.3f}] km")
    print(f"  z-mean: {z_right.mean():.3f} km")
    print(f"  z-median: {np.median(z_right):.3f} km")

print("="*80)
