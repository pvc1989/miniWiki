import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper

import numpy as np
from scipy.spatial import KDTree
import argparse
import sys


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Increase the order of geometric approximation.')
    parser.add_argument('--input', type=str, help='the input linear mesh file')
    parser.add_argument('--output', type=str, help='the output high-order mesh file')
    parser.add_argument('--order', type=int, default=2, help='order of the output mesh')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    # load the linear mesh
    mesh_cgns, _, _ = cgm.load(args.input)

    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = wrapper.getNodeData(mesh_zone)
    n_node = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    print(f'in linear mesh: n_node = {n_node}, n_cell = {n_cell}')

    mesh_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    section = wrapper.getUniqueChildByType(mesh_zone, 'Elements_t')
    element_type = wrapper.getNodeData(section)
    assert element_type[0] == 7  # QUAD_4
    old_connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementConnectivity'))

    # add center and mid-edge points:
    n_node_new = n_node + n_cell * 5  # TODO(PVC): support higher orders
    new_coords_x = np.ndarray((n_node_new,), dtype=mesh_coords_x.dtype)
    new_coords_x[0 : n_node] = mesh_coords_x
    new_coords_y = np.ndarray((n_node_new,), dtype=mesh_coords_y.dtype)
    new_coords_y[0 : n_node] = mesh_coords_y
    new_coords_z = np.ndarray((n_node_new,), dtype=mesh_coords_z.dtype)
    new_coords_z[0 : n_node] = mesh_coords_z
    i_node_next = n_node
    new_connectivity = np.ndarray((old_connectivity.shape[0] * 9 // 4,),
        dtype=old_connectivity.dtype)
    for i_cell_mesh in range(n_cell):
        old_first = i_cell_mesh * 4
        new_fisrt = i_cell_mesh * 9
        new_connectivity[new_fisrt : new_fisrt + 4] = old_connectivity[old_first : old_first + 4]
        # build the mid-point on the bottom edge
        x = (mesh_coords_x[old_first + 0] + mesh_coords_x[old_first + 1]) / 2
        y = (mesh_coords_y[old_first + 0] + mesh_coords_y[old_first + 1]) / 2
        z = (mesh_coords_z[old_first + 0] + mesh_coords_z[old_first + 1]) / 2
        new_coords_x[i_node_next] = x
        new_coords_y[i_node_next] = y
        new_coords_z[i_node_next] = z
        i_node_next += 1
        new_connectivity[new_fisrt + 4] = i_node_next
        # build the mid-point on the right edge
        x = (mesh_coords_x[old_first + 1] + mesh_coords_x[old_first + 2]) / 2
        y = (mesh_coords_y[old_first + 1] + mesh_coords_y[old_first + 2]) / 2
        z = (mesh_coords_z[old_first + 1] + mesh_coords_z[old_first + 2]) / 2
        new_coords_x[i_node_next] = x
        new_coords_y[i_node_next] = y
        new_coords_z[i_node_next] = z
        i_node_next += 1
        new_connectivity[new_fisrt + 5] = i_node_next
        # build the mid-point on the top edge
        x = (mesh_coords_x[old_first + 2] + mesh_coords_x[old_first + 3]) / 2
        y = (mesh_coords_y[old_first + 2] + mesh_coords_y[old_first + 3]) / 2
        z = (mesh_coords_z[old_first + 2] + mesh_coords_z[old_first + 3]) / 2
        new_coords_x[i_node_next] = x
        new_coords_y[i_node_next] = y
        new_coords_z[i_node_next] = z
        i_node_next += 1
        new_connectivity[new_fisrt + 6] = i_node_next
        # build the mid-point on the left edge
        x = (mesh_coords_x[old_first + 3] + mesh_coords_x[old_first + 0]) / 2
        y = (mesh_coords_y[old_first + 3] + mesh_coords_y[old_first + 0]) / 2
        z = (mesh_coords_z[old_first + 3] + mesh_coords_z[old_first + 0]) / 2
        new_coords_x[i_node_next] = x
        new_coords_y[i_node_next] = y
        new_coords_z[i_node_next] = z
        i_node_next += 1
        new_connectivity[new_fisrt + 7] = i_node_next
        # build the center point
        index = old_connectivity[old_first : old_first + 4] - 1
        x = np.sum(mesh_coords_x[index]) / 4
        y = np.sum(mesh_coords_y[index]) / 4
        z = np.sum(mesh_coords_z[index]) / 4
        new_coords_x[i_node_next] = x
        new_coords_y[i_node_next] = y
        new_coords_z[i_node_next] = z
        i_node_next += 1
        new_connectivity[new_fisrt + 8] = i_node_next
        if args.verbose:
            print(f'QUAD_4[{i_cell_mesh}] converted')
    assert i_node_next == n_node_new
    n_node = n_node_new
    print(f'in quadratic mesh: n_node = {n_node}, n_cell = {n_cell}')

    # update the cells
    cgu.removeChildByName(section, 'ElementConnectivity')
    cgl.newDataArray(section, 'ElementConnectivity', new_connectivity)

    # write the new mesh out
    mesh_zone_size[0][0] = n_node
    element_type[0] = 9  # QUAD_9
    mesh_coords[X][1] = new_coords_x
    mesh_coords[Y][1] = new_coords_y
    mesh_coords[Z][1] = new_coords_z

    output = args.output
    if output is None:
        output = f'order={args.order}_{args.input}'
    print('writing to', output)
    cgm.save(output, mesh_cgns)
