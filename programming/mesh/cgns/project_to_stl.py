import argparse
import sys

import numpy as np
from scipy.spatial import KDTree

import CGNS.MAP as cgm
import pycgns_wrapper
from pycgns_wrapper import X, Y, Z


def getNearestPoint(point_p: np.ndarray, kdtree: KDTree) -> np.ndarray:
    distance_to_center, i_cell = kdtree.query(point_p)
    return kdtree.data[i_cell]


def getFootOnTriangle(point_p: np.ndarray, kdtree: KDTree, connectivity: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    # return point_p
    distance_to_center, i_cell = kdtree.query(point_p)
    first = i_cell * 3
    index = connectivity[first : first + 3] - 1
    a, b, c = index
    point_a = np.array([x[a], y[a], z[a]])
    point_b = np.array([x[b], y[b], z[b]])
    point_c = np.array([x[c], y[c], z[c]])
    normal = np.cross(point_b - point_a, point_c - point_a)
    normal /= np.linalg.norm(normal)  # (A, B, C)
    # assert (np.linalg.norm(normal) - 1) < 1e-6, np.linalg.norm(normal)
    offset = -normal.dot(point_a)  # D
    # assert abs(normal.dot(point_a) + offset) < 1e-6, normal.dot(point_a) + offset
    # assert abs(normal.dot(point_b) + offset) < 1e-6, normal.dot(point_b) + offset
    # assert abs(normal.dot(point_c) + offset) < 1e-6, normal.dot(point_c) + offset
    ratio = -normal.dot(point_p) - offset
    foot = point_p + ratio * normal
    # assert abs(normal.dot(foot) + offset) < 1e-6, normal.dot(foot) + offset
    # distance_to_triangle = np.linalg.norm(foot - point_p)
    # print('p =', point_p)
    # print('a =', point_a, np.linalg.norm(point_p - point_a))
    # print('b =', point_b, np.linalg.norm(point_p - point_b))
    # print('c =', point_c, np.linalg.norm(point_p - point_c))
    # center = (point_a + point_b + point_c) / 3
    # if not (center == kdtree.data[i_cell]).all():
    #     print('i_cell =', i_cell)
    #     print('center =', center)
    #     print('kdtree.data[i] =', kdtree.data[i_cell])
    #     print(np.linalg.norm(center - kdtree.data[i_cell]))
    #     assert False
    # assert distance_to_triangle <= np.linalg.norm(point_p - point_a)
    # assert distance_to_triangle <= np.linalg.norm(point_p - point_b)
    # assert distance_to_triangle <= np.linalg.norm(point_p - point_c)
    # assert distance_to_triangle <= distance_to_center, (distance_to_triangle, distance_to_center)
    return foot


def getKdtreeFromSTL(setKdtreePoints: callable):
    # stl_coords already captured in setKdtreePoints
    stl_connectivity = np.zeros(n_cell * 3, dtype=int) - 1
    assert stl_connectivity.shape == (n_cell * 3,)
    assert (stl_connectivity == -1).all()
    sections = pycgns_wrapper.getChildrenByType(stl_zone, 'Elements_t')
    print('n_section =', len(sections))
    i_cell_global = 0
    for section in sections:
        if args.verbose:
            print(section)
        element_type = pycgns_wrapper.getNodeData(section)
        assert element_type[0] == 5  # TRI_3
        connectivity = pycgns_wrapper.getNodeData(
            pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        element_range = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByName(section, 'ElementRange'))
        first_global = element_range[0] - 1
        last_global = element_range[1]  # inclusive to exclusive
        if args.verbose:
            print('element_range =', element_range, first_global, last_global)
        stl_connectivity[first_global * 3 : last_global * 3] = connectivity[:]
        n_cell_local = last_global - first_global
        assert n_cell_local == connectivity.shape[0] // 3
        for i_cell_local in range(n_cell_local):
            first_local = i_cell_local * 3
            node_index_tuple = connectivity[first_local : first_local + 3] - 1
            # assert (0 <= index).all() and (index < n_node).all()
            cell_index = first_global + i_cell_local
            setKdtreePoints(cell_index, node_index_tuple)
            i_cell_global += 1
    assert i_cell_global == n_cell, i_cell_global
    assert (1 <= stl_connectivity).all() and (stl_connectivity <= n_node).all()

    stl_kdtree = KDTree(kdtree_points)
    return stl_connectivity, stl_kdtree


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[1]}.py',
        description = 'Project points in CGNS file to the STL model represented by another CGNS file.')
    parser.add_argument('--stl', type=str, help='the input STL file (currently in CGNS)')
    parser.add_argument('--mesh', type=str, help='the input mesh file')
    parser.add_argument('--output', type=str, help='the output high-order mesh file')
    parser.add_argument('--target', choices=['corner', 'center', 'foot'], default='corner',
        help='which kind of point to be shifhted to')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    output = args.output
    if output is None:
        i = args.STL.index('max=') + 4
        j = args.STL[i:].index('mm')
        h_max = float(args.STL[i : i + j])
        output = f'{args.mesh[:-5]}_STL={h_max:3.1e}mm_shift={args.target}.cgns'

    # load the STL model
    stl_cgns, _, _ = cgm.load(args.STL)
    stl_zone = pycgns_wrapper.getUniqueChildByType(
        pycgns_wrapper.getUniqueChildByType(stl_cgns, 'CGNSBase_t'), 'Zone_t')
    stl_zone_size = pycgns_wrapper.getNodeData(stl_zone)
    n_node = stl_zone_size[0][0]
    n_cell = stl_zone_size[0][1]
    if args.verbose:
        print(f'in STL: n_node = {n_node}, n_cell = {n_cell}')

    stl_coords = pycgns_wrapper.getChildrenByType(
        pycgns_wrapper.getUniqueChildByType(stl_zone, 'GridCoordinates_t'), 'DataArray_t')
    stl_coords_x, stl_coords_y, stl_coords_z = stl_coords[X][1], stl_coords[Y][1], stl_coords[Z][1]

    if args.target == 'corner':
        kdtree_points = np.zeros((n_cell * 3, 3), dtype=float)  # 3 point for each triangle
        def setKdtreePoints(cell_index, node_index_tuple):
            first = cell_index * 3
            for i in range(3):
                j = first + i
                k = node_index_tuple[i]
                kdtree_points[j] = stl_coords_x[k], stl_coords_y[k], stl_coords_z[k]
    elif args.target == 'center' or args.target == 'foot':
        kdtree_points = np.zeros((n_cell, 3), dtype=float)  # 1 point for each triangle
        def setKdtreePoints(cell_index, node_index_tuple):
            kdtree_points[cell_index] = (
                np.sum(stl_coords_x[node_index_tuple]) / 3,
                np.sum(stl_coords_y[node_index_tuple]) / 3,
                np.sum(stl_coords_z[node_index_tuple]) / 3)
    else:
        assert False

    # TODO(PVC): (args.target == corner) do not need stl_connectivity
    stl_connectivity, stl_kdtree = getKdtreeFromSTL(setKdtreePoints)

    # load the linear mesh
    mesh_cgns, _, _ = cgm.load(args.mesh)

    mesh_zone = pycgns_wrapper.getUniqueChildByType(
        pycgns_wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = pycgns_wrapper.getNodeData(mesh_zone)
    n_node = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    if args.verbose:
        print(f'in linear mesh: n_node = {n_node}, n_cell = {n_cell}')

    mesh_coords = pycgns_wrapper.getChildrenByType(
        pycgns_wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    # update the coords
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    if args.target == 'corner' or args.target == 'center':
        getShiftedPoint = lambda query_point: getNearestPoint(query_point, stl_kdtree)
    elif args.target == 'foot':
        getShiftedPoint = lambda query_point: getFootOnTriangle(query_point, stl_kdtree,
            stl_connectivity, stl_coords_x, stl_coords_y, stl_coords_z)
    else:
        assert False

    # shift nodes to the (exactly or nearly) closest points on the wall
    for i in range(n_node):
        query_point = np.array([mesh_coords_x[i], mesh_coords_y[i], mesh_coords_z[i]])
        # x, y, z = Refine(query_point, getShiftedPoint(query_point))
        x, y, z = getShiftedPoint(query_point)
        mesh_coords_x[i] = x
        mesh_coords_y[i] = y
        mesh_coords_z[i] = z
        print(f'Shift i_node / n_node = {i} / {n_node}')

    if args.verbose:
        print('writing to', output)
    cgm.save(output, mesh_cgns)
