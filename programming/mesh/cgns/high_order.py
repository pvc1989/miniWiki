import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk

import numpy as np
from scipy.spatial import KDTree
import argparse
import wrapper


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

    cad_connectivity = np.zeros(n_cell * 3, dtype=int) - 1
    # assert cad_connectivity.shape == (n_cell * 3,)
    # assert (cad_connectivity == -1).all()
    sections = wrapper.getChildrenByType(cad_zone, 'Elements_t')
    print('n_section =', len(sections))
    centers = np.zeros((n_cell, 3), dtype=float)
    i_cell_global = 0
    for section in sections:
        if args.verbose and False:
            print(section)
        element_type = wrapper.getNodeData(section)
        # assert element_type[0] == 5  # TRI_3
        # build DS for quick nearest neighbor query
        connectivity = wrapper.getNodeData(
            wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        element_range = wrapper.getNodeData(wrapper.getUniqueChildByName(section, 'ElementRange'))
        first_global = element_range[0] - 1
        last_global = element_range[1]  # inclusive to exclusive
        if args.verbose:
            print('element_range =', element_range, first_global, last_global)
        cad_connectivity[first_global * 3 : last_global * 3] = connectivity[:]
        n_cell_local = last_global - first_global
        # assert n_cell_local == connectivity.shape[0] // 3
        for i_cell_local in range(n_cell_local):
            first_local = i_cell_local * 3
            index = connectivity[first_local : first_local + 3] - 1
            # assert (0 <= index).all() and (index < n_node).all()
            # a, b, c = index
            # centers[i_cell_global][X] = (cad_coords_x[a] + cad_coords_x[b] + cad_coords_x[c]) / 3            
            # centers[i_cell_global][Y] = (cad_coords_y[a] + cad_coords_y[b] + cad_coords_y[c]) / 3
            # centers[i_cell_global][Z] = (cad_coords_z[a] + cad_coords_z[b] + cad_coords_z[c]) / 3
            curr_global = first_global + i_cell_local
            centers[curr_global][X] = np.sum(cad_coords_x[index]) / 3
            centers[curr_global][Y] = np.sum(cad_coords_y[index]) / 3
            centers[curr_global][Z] = np.sum(cad_coords_z[index]) / 3
            i_cell_global += 1
    # assert i_cell_global == n_cell, i_cell_global
    # assert (1 <= cad_connectivity).all() and (cad_connectivity <= n_node).all()

    section_names = []
    for section in sections:
        section_names.append(wrapper.getNodeName(section))
    # assert len(section_names) == len(sections)
    for name in section_names:
        cgu.removeChildByName(cad_zone, name)
    # assert len(wrapper.getChildrenByType(cad_zone, 'Elements_t')) == 0
    cgl.newElements(cad_zone, 'SingleSection', etype='TRI_3', erange=np.array([1, n_cell]), econnectivity=cad_connectivity)
    print(len(wrapper.getChildrenByType(cad_zone, 'Elements_t')))
    cgm.save('cad-merged.cgns', cad_cgns)

    cad_kdtree = KDTree(centers)
    for i in range(n_cell):
        pass
        # assert np.linalg.norm(centers[i] - cad_kdtree.data[i]) == 0

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
    # assert element_type[0] == 7  # QUAD_4
    old_connectivity = wrapper.getNodeData(
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
        index = old_connectivity[old_first : old_first + 4] - 1
        x = np.sum(mesh_coords_x[index]) / 4
        y = np.sum(mesh_coords_y[index]) / 4
        z = np.sum(mesh_coords_z[index]) / 4
        foot = getFootOnTriangle(np.array([x, y, z]), cad_kdtree, cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
        new_coords_x[i_node_next] = foot[X]
        new_coords_y[i_node_next] = foot[Y]
        new_coords_z[i_node_next] = foot[Z]
        i_node_next += 1
        new_connectivity[new_fisrt + 8] = i_node_next
        print(f'QUAD_4[{i_cell_mesh}] -> QUAD_9[{i_cell_mesh}]')
    # assert i_node_next == n_node_new
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
