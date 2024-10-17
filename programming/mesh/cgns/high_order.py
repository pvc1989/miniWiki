import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk

import numpy as np
from scipy.spatial import KDTree
import argparse
import wrapper


def getFootOnTriangle(point: np.ndarray, kdtree: KDTree, connectivity: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    distance_to_center, i_cell = kdtree.query(point)
    first = i_cell * 3
    a, b, c = connectivity[first : first + 3] - 1
    point_a = np.array([x[a], y[a], z[a]])
    point_b = np.array([x[b], y[b], z[b]])
    point_c = np.array([x[c], y[c], z[c]])
    normal = np.cross(point_b - point_a, point_c - point_a)
    normal /= np.linalg.norm(normal)  # (A, B, C)
    assert (np.linalg.norm(normal) - 1) < 1e-6, np.linalg.norm(normal)
    offset = -normal.dot(point_a)  # D
    assert abs(normal.dot(point_a) + offset) < 1e-6, normal.dot(point_a) + offset
    assert abs(normal.dot(point_b) + offset) < 1e-6, normal.dot(point_b) + offset
    assert abs(normal.dot(point_c) + offset) < 1e-6, normal.dot(point_c) + offset
    ratio = -normal.dot(point) - offset
    foot = point + ratio * normal
    assert abs(normal.dot(foot) + offset) < 1e-6, normal.dot(foot) + offset
    distance_to_triangle = np.linalg.norm(foot - point)
    assert distance_to_triangle <= distance_to_center, (distance_to_triangle, distance_to_center)
    return foot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 high_order.py')
    parser.add_argument('--cad', type=str, help='the input CAD file (currently in CGNS)')
    parser.add_argument('--mesh', type=str, help='the input linear mesh file')
    parser.add_argument('--output', type=str, help='the output high-order mesh file')
    parser.add_argument('--order', type=int, default=2, help='order of the output mesh')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    X, Y, Z = 0, 1, 2

    # load the CAD model
    cad_cgns, _, _ = cgm.load(args.cad)
    cad_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(cad_cgns, 'CGNSBase_t'), 'Zone_t')
    zone_size = wrapper.getNodeData(cad_zone)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    if args.verbose:
        print(f'in CAD: n_node = {n_node}, n_cell = {n_cell}')

    cad_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(cad_zone, 'GridCoordinates_t'), 'DataArray_t')
    cad_coords_x, cad_coords_y, cad_coords_z = cad_coords[X][1], cad_coords[Y][1], cad_coords[Z][1]

    cad_connectivity = np.ndarray((n_cell * 3,), dtype='int')
    sections = wrapper.getChildrenByType(cad_zone, 'Elements_t')
    centers = np.ndarray((n_cell, 3))
    i_cell_global = 0
    for section in sections:
        if args.verbose and False:
            print(section)
        element_type = wrapper.getNodeData(section)
        assert element_type[0] == 5  # TRI_3
        # build DS for quick nearest neighbor query
        connectivity = wrapper.getNodeData(
            wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        element_range = wrapper.getNodeData(wrapper.getUniqueChildByName(section, 'ElementRange'))
        first = element_range[0] - 1
        last = element_range[1]  # inclusive to exclusive
        if args.verbose:
            print('element_range =', element_range, first, last)
        cad_connectivity[first * 3 : last * 3] = connectivity
        for i_cell_local in range(connectivity.shape[0] // 3):
            first = i_cell_local * 3
            index = connectivity[first : first + 3] - 1
            centers[i_cell_global][X] = np.sum(cad_coords_x[index]) / 3
            centers[i_cell_global][Y] = np.sum(cad_coords_y[index]) / 3
            centers[i_cell_global][Z] = np.sum(cad_coords_z[index]) / 3
            i_cell_global += 1
    assert i_cell_global == n_cell, i_cell_global
    cad_kdtree = KDTree(centers)

    # load the linear mesh
    mesh_cgns, _, _ = cgm.load(args.mesh)

    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    zone_size = wrapper.getNodeData(mesh_zone)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    if args.verbose:
        print(f'in linear mesh: n_node = {n_node}, n_cell = {n_cell}')

    mesh_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    section = wrapper.getUniqueChildByType(mesh_zone, 'Elements_t')
    element_type = wrapper.getNodeData(section)
    assert element_type[0] == 7  # QUAD_4
    connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementConnectivity'))

    # add center and mid-edge points:
    n_node_new = n_node + n_cell * 5  # TODO(PVC): remove duplicates
    new_coords_x = np.ndarray((n_node_new,), dtype=mesh_coords_x.dtype)
    new_coords_x[0 : n_node] = mesh_coords_x
    new_coords_y = np.ndarray((n_node_new,), dtype=mesh_coords_y.dtype)
    new_coords_y[0 : n_node] = mesh_coords_y
    new_coords_z = np.ndarray((n_node_new,), dtype=mesh_coords_z.dtype)
    new_coords_z[0 : n_node] = mesh_coords_z
    i_node_next = n_node
    new_connectivity = np.ndarray((connectivity.shape[0] * 9 // 4,), dtype=connectivity.dtype)
    for i_cell_mesh in range(n_cell):
        old_first = i_cell_mesh * 4
        new_fisrt = i_cell_mesh * 9
        new_connectivity[new_fisrt : new_fisrt + 4] = connectivity[old_first : old_first + 4]
        # build the mid-point on the bottom edge
        x = (mesh_coords_x[old_first + 0] + mesh_coords_x[old_first + 1]) / 2
        y = (mesh_coords_y[old_first + 0] + mesh_coords_y[old_first + 1]) / 2
        z = (mesh_coords_z[old_first + 0] + mesh_coords_z[old_first + 1]) / 2
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 4] = i_node_next
        # build the mid-point on the right edge
        x = (mesh_coords_x[old_first + 1] + mesh_coords_x[old_first + 2]) / 2
        y = (mesh_coords_y[old_first + 1] + mesh_coords_y[old_first + 2]) / 2
        z = (mesh_coords_z[old_first + 1] + mesh_coords_z[old_first + 2]) / 2
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 5] = i_node_next
        # build the mid-point on the top edge
        x = (mesh_coords_x[old_first + 2] + mesh_coords_x[old_first + 3]) / 2
        y = (mesh_coords_y[old_first + 2] + mesh_coords_y[old_first + 3]) / 2
        z = (mesh_coords_z[old_first + 2] + mesh_coords_z[old_first + 3]) / 2
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 6] = i_node_next
        # build the mid-point on the left edge
        x = (mesh_coords_x[old_first + 3] + mesh_coords_x[old_first + 0]) / 2
        y = (mesh_coords_y[old_first + 3] + mesh_coords_y[old_first + 0]) / 2
        z = (mesh_coords_z[old_first + 3] + mesh_coords_z[old_first + 0]) / 2
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 7] = i_node_next
        # build the center point
        index = connectivity[old_first : old_first + 4] - 1
        x = np.sum(mesh_coords_x[index]) / 4
        y = np.sum(mesh_coords_y[index]) / 4
        z = np.sum(mesh_coords_z[index]) / 4
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 8] = i_node_next
        if args.verbose:
            print(f'QUAD_4[{i_cell_mesh}] -> QUAD_9[{i_cell_mesh}]')
    assert i_node_next == n_node_new
    if args.verbose:
        print(f'in quadratic mesh: n_node = {n_node_new}, n_cell = {n_cell}')

    # write the new mesh out
    zone_size[0][0] = n_node_new
    element_type[0] = 9  # QUAD_9
    mesh_coords[X][1] = new_coords_x
    mesh_coords[Y][1] = new_coords_y
    mesh_coords[Z][1] = new_coords_z

    cgu.removeChildByName(section, 'ElementConnectivity')
    cgl.newDataArray(section, 'ElementConnectivity', new_connectivity)

    cgm.save(args.output, mesh_cgns)
