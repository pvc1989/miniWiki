import CGNS.MAP as cgm
import wrapper

import sys
import numpy as np
import argparse


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Shift the points in a given mesh to their given coordinates.')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the mesh to be shifted')
    parser.add_argument('--coords', type=str, help='the CSV file of shifted coords')
    parser.add_argument('--output', type=str, help='the output mesh file')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    output = args.output
    if output is None:
        output = f'shifted_{args.mesh}'

    # load the mesh to be shifted
    mesh_cgns, _, _ = cgm.load(args.mesh)
    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = wrapper.getNodeData(mesh_zone)
    n_node = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    if args.verbose:
        print(f'n_node = {n_node}, n_cell = {n_cell}')
    mesh_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]
    assert n_node == len(mesh_coords_x) == len(mesh_coords_y) == len(mesh_coords_z)

    # load the shifted coords
    new_coords = np.loadtxt(args.coords, delimiter=',')
    assert (n_node, 3) == new_coords.shape

    # update the coords
    mesh_coords_x[:] = new_coords[:, X]
    mesh_coords_y[:] = new_coords[:, Y]
    mesh_coords_z[:] = new_coords[:, Z]

    # write the shifted mesh
    if args.verbose:
        print('write to ', output)
    cgm.save(output, mesh_cgns)
