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
    parser.add_argument('--cad_points', type=str, help='the CSV file of points on CAD surfaces')
    parser.add_argument('--stl_points', type=str, help='the CSV file of points on STL surfaces')
    parser.add_argument('--x_cut', type=float, default=-1e100, help='the position of the cutting plane (x <= x_cut ? stl : cad)')
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
    mesh_points = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_points_x, mesh_points_y, mesh_points_z = mesh_points[X][1], mesh_points[Y][1], mesh_points[Z][1]
    assert n_node == len(mesh_points_x) == len(mesh_points_y) == len(mesh_points_z)

    # load the shifted coords
    cad_points = np.loadtxt(args.cad_points, delimiter=',')
    assert (n_node, 3) == cad_points.shape

    stl_points = np.loadtxt(args.stl_points, delimiter=',')
    assert (n_node, 3) == stl_points.shape

    # update the coords
    for i_node in range(n_node):
        x = mesh_points_x[i_node]
        if x < args.x_cut:
            mesh_points_x[i_node] = stl_points[i_node, X]
            mesh_points_y[i_node] = stl_points[i_node, Y]
            mesh_points_z[i_node] = stl_points[i_node, Z]
        else:
            mesh_points_x[i_node] = cad_points[i_node, X]
            mesh_points_y[i_node] = cad_points[i_node, Y]
            mesh_points_z[i_node] = cad_points[i_node, Z]

    # write the shifted mesh
    if args.verbose:
        print('writing to', output)
    cgm.save(output, mesh_cgns)
