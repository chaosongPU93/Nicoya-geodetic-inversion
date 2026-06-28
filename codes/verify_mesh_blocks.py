#!/usr/bin/env python3
"""
Verify the actual left/right block structure and their top boundary depths
"""
import argparse
import dolfin as dl
import numpy as np

parser = argparse.ArgumentParser(description='Verify mesh block structure and top boundary depths')
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
subdomains = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_physical_region.xml')

# Boundary and subdomain IDs
top = 1
blockleft = 8
blockright = 9

print("="*80)
print("SUBDOMAIN AND TOP BOUNDARY ANALYSIS")
print("="*80)

# Get all mesh coordinates and cells
all_coords = mesh.coordinates()
print(f"Total vertices: {len(all_coords)}")
print(f"Total cells: {mesh.num_cells()}")
print()

# Analyze subdomains
cells_left = 0
cells_right = 0
cells_other = 0

for cell in dl.cells(mesh):
    subdomain_id = subdomains[cell]
    if subdomain_id == blockleft:
        cells_left += 1
    elif subdomain_id == blockright:
        cells_right += 1
    else:
        cells_other += 1

print(f"Subdomain distribution:")
print(f"  Block left (ID={blockleft}): {cells_left} cells")
print(f"  Block right (ID={blockright}): {cells_right} cells")
print(f"  Other: {cells_other} cells")
print()

# Now find which vertices belong to each block's top boundary
# Strategy: For each top boundary vertex, find adjacent cells and check their subdomain

# First, get all top boundary vertices
top_facets = []
for facet in dl.facets(mesh):
    if boundaries[facet] == top:
        top_facets.append(facet)

top_vertex_indices = set()
for facet in top_facets:
    for vertex in dl.vertices(facet):
        top_vertex_indices.add(vertex.index())

top_coords = all_coords[list(top_vertex_indices)]
print(f"Top boundary vertices: {len(top_coords)}")
print()

# Build vertex-to-subdomain mapping by checking adjacent cells
vertex_to_subdomains = {}
for cell in dl.cells(mesh):
    subdomain_id = subdomains[cell]
    for vertex in dl.vertices(cell):
        v_idx = vertex.index()
        if v_idx not in vertex_to_subdomains:
            vertex_to_subdomains[v_idx] = set()
        vertex_to_subdomains[v_idx].add(subdomain_id)

# Classify top boundary vertices by subdomain
top_left_indices = []
top_right_indices = []
top_other_indices = []

for v_idx in top_vertex_indices:
    subdomain_set = vertex_to_subdomains.get(v_idx, set())
    # A vertex might be on the boundary between blocks
    # Priority: if connected to left block, assign to left; else if to right, assign to right
    if blockleft in subdomain_set:
        top_left_indices.append(v_idx)
    elif blockright in subdomain_set:
        top_right_indices.append(v_idx)
    else:
        top_other_indices.append(v_idx)

print("Top boundary vertices by subdomain:")
print(f"  Connected to block left: {len(top_left_indices)}")
print(f"  Connected to block right: {len(top_right_indices)}")
print(f"  Other: {len(top_other_indices)}")
print()

# Analyze z-distribution for each block
if len(top_left_indices) > 0:
    top_left_coords = all_coords[top_left_indices]
    z_left = top_left_coords[:, 2] / 1e3
    x_left = top_left_coords[:, 0] / 1e3
    y_left = top_left_coords[:, 1] / 1e3

    print("BLOCK LEFT - Top boundary:")
    print(f"  x-range: [{x_left.min():.1f}, {x_left.max():.1f}] km")
    print(f"  y-range: [{y_left.min():.1f}, {y_left.max():.1f}] km")
    print(f"  z-range: [{z_left.min():.3f}, {z_left.max():.3f}] km")
    print(f"  z-mean: {z_left.mean():.3f} km")
    print(f"  z-median: {np.median(z_left):.3f} km")
    print(f"  z-std: {z_left.std():.3f} km")

    at_surface = np.sum(np.abs(z_left) < 0.1)
    deep_7km = np.sum(z_left < -6.5)
    print(f"  At surface (|z| < 0.1 km): {at_surface} vertices ({100*at_surface/len(z_left):.1f}%)")
    print(f"  Deeper than -6.5 km: {deep_7km} vertices ({100*deep_7km/len(z_left):.1f}%)")
    print()

if len(top_right_indices) > 0:
    top_right_coords = all_coords[top_right_indices]
    z_right = top_right_coords[:, 2] / 1e3
    x_right = top_right_coords[:, 0] / 1e3
    y_right = top_right_coords[:, 1] / 1e3

    print("BLOCK RIGHT - Top boundary:")
    print(f"  x-range: [{x_right.min():.1f}, {x_right.max():.1f}] km")
    print(f"  y-range: [{y_right.min():.1f}, {y_right.max():.1f}] km")
    print(f"  z-range: [{z_right.min():.3f}, {z_right.max():.3f}] km")
    print(f"  z-mean: {z_right.mean():.3f} km")
    print(f"  z-median: {np.median(z_right):.3f} km")
    print(f"  z-std: {z_right.std():.3f} km")

    at_surface = np.sum(np.abs(z_right) < 0.1)
    deep_7km = np.sum(z_right < -6.5)
    print(f"  At surface (|z| < 0.1 km): {at_surface} vertices ({100*at_surface/len(z_right):.1f}%)")
    print(f"  Deeper than -6.5 km: {deep_7km} vertices ({100*deep_7km/len(z_right):.1f}%)")

print("="*80)
