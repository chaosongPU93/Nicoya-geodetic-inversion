"""
Standalone helper: write fault_basis_<meshname>.txt without running the full inversion.

The fault basis depends only on the mesh (vertex (x,y,z) and per-facet UFL
strike/dip directions), so this script just loads the mesh, calls
ut.compute_fault_basis_per_vertex, and writes the same file the inversion
scripts produce.

Usage:
    python save_fault_basis.py                    # uses default meshname below
    python save_fault_basis.py -m nicoyaCKden_sm  # override meshname
"""

import os
import argparse
import dolfin as dl
import utils as ut


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--mesh', default='nicoyaCKden_une_all',
                        help='Mesh name (without .xml extension)')
    parser.add_argument('-p', '--meshpath', default='/home/staff/chao/SSEinv/Nicoya/mesh/',
                        help='Directory containing the .xml mesh and facet/region files')
    parser.add_argument('-o', '--outpath', default='/home/staff/chao/SSEinv/Nicoya/rst_locking/',
                        help='Output directory for fault_basis_<meshname>.txt')
    parser.add_argument('--fault-tag', type=int, default=7,
                        help='Boundary marker value identifying fault facets (default 7)')
    args = parser.parse_args()

    meshname = args.mesh
    meshpath = args.meshpath
    resultpath = args.outpath
    fault = args.fault_tag

    os.makedirs(resultpath, exist_ok=True)

    # --- Load mesh and facet markers ---
    print(f"Loading mesh: {meshpath}{meshname}.xml")
    mesh = dl.Mesh(meshpath + meshname + '.xml')
    boundaries = dl.MeshFunction("size_t", mesh, meshpath + meshname + '_facet_region.xml')

    # 2-component CG1 VectorFunctionSpace — same layout the inversion scripts use
    # for the slip parameter; required by compute_fault_basis_per_vertex to recover
    # the m_s_fault row order.
    Vm = dl.VectorFunctionSpace(mesh, "CG", degree=1, dim=2)

    # --- Compute per-vertex strike / dip basis ---
    basis_coords, strike_vec, dip_vec = ut.compute_fault_basis_per_vertex(
        mesh, boundaries, Vm, fault, verbose=True)

    # --- Write fault_basis_<meshname>.txt (same format as inversion scripts) ---
    outfile = resultpath + 'fault_basis_' + meshname + '.txt'
    with open(outfile, 'w+') as fout:
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
    print(f"Wrote {basis_coords.shape[0]} fault vertices to: {outfile}")


if __name__ == '__main__':
    main()
