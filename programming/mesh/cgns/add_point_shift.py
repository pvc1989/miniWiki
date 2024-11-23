import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import wrapper

import sys
import numpy as np
import argparse


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Add the shift of each point as a vector field.')
    parser.add_argument('--folder', type=str, help='the working folder containing input and output files')
    parser.add_argument('--input', type=str, help='the CGNS file of the mesh to be shifted')
    parser.add_argument('--cad_points', type=str, help='the NPY file of points on CAD surfaces')
    parser.add_argument('--stl_points', type=str, help='the NPY file of points on STL surfaces')
    parser.add_argument('--x_cut', type=float, default=-1e100, help='the position of the cutting plane (x <= x_cut ? stl : cad)')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    # load the mesh to be shifted
    mesh_cgns, _, _ = cgm.load(f'{args.folder}/{args.input}')
    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = wrapper.getNodeData(mesh_zone)
    n_node = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    print(f'n_node = {n_node}, n_cell = {n_cell}')

    # read the original coords
    _, old_x, old_y, old_z = wrapper.readPoints(mesh_zone, mesh_zone_size)

    # load the shifted coords
    cad_points = np.load(args.cad_points)
    assert (n_node, 3) == cad_points.shape

    stl_points = np.load(args.stl_points)
    assert (n_node, 3) == stl_points.shape

    # get the new coords
    new_x = np.ndarray(old_x.shape)
    new_y = np.ndarray(old_y.shape)
    new_z = np.ndarray(old_z.shape)
    for i_node in range(n_node):
        if args.verbose:
            print(f'shifting node {i_node} / {n_node}')
        x = old_x[i_node]
        if x < args.x_cut:
            new_x[i_node] = stl_points[i_node, X]
            new_y[i_node] = stl_points[i_node, Y]
            new_z[i_node] = stl_points[i_node, Z]
        else:
            new_x[i_node] = cad_points[i_node, X]
            new_y[i_node] = cad_points[i_node, Y]
            new_z[i_node] = cad_points[i_node, Z]

    # write the shifts as a field of point data
    point_data = wrapper.getSolutionByLocation(mesh_zone, 'Vertex', 'PointShift')
    cgl.newDataArray(point_data, 'ShiftX', new_x - old_x)
    cgl.newDataArray(point_data, 'ShiftY', new_y - old_y)
    cgl.newDataArray(point_data, 'ShiftZ', new_z - old_z)

    # write the shifted mesh
    output = f'{args.folder}/shifted.cgns'
    print('writing to', output)
    cgm.save(output, mesh_cgns)

    print('\n[Done]', args)
