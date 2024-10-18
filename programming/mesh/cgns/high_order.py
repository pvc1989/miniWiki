import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper

import numpy as np
from scipy.spatial import KDTree
import argparse
import vtk


X, Y, Z = 0, 1, 2


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


def getKdtreeFromCAD(setKdtreePoints: callable):
    # cad_coords already captured in setKdtreePoints
    cad_connectivity = np.zeros(n_cell * 3, dtype=int) - 1
    assert cad_connectivity.shape == (n_cell * 3,)
    assert (cad_connectivity == -1).all()
    sections = wrapper.getChildrenByType(cad_zone, 'Elements_t')
    print('n_section =', len(sections))
    i_cell_global = 0
    for section in sections:
        if args.verbose and False:
            print(section)
        element_type = wrapper.getNodeData(section)
        assert element_type[0] == 5  # TRI_3
        connectivity = wrapper.getNodeData(
            wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        element_range = wrapper.getNodeData(wrapper.getUniqueChildByName(section, 'ElementRange'))
        first_global = element_range[0] - 1
        last_global = element_range[1]  # inclusive to exclusive
        if args.verbose:
            print('element_range =', element_range, first_global, last_global)
        cad_connectivity[first_global * 3 : last_global * 3] = connectivity[:]
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
    assert (1 <= cad_connectivity).all() and (cad_connectivity <= n_node).all()

    # section_names = []
    # for section in sections:
    #     section_names.append(wrapper.getNodeName(section))
    # assert len(section_names) == len(sections)
    # for name in section_names:
    #     cgu.removeChildByName(cad_zone, name)
    # assert len(wrapper.getChildrenByType(cad_zone, 'Elements_t')) == 0
    # cgl.newElements(cad_zone, 'SingleSection', etype='TRI_3', erange=np.array([1, n_cell]), econnectivity=cad_connectivity)
    # print(len(wrapper.getChildrenByType(cad_zone, 'Elements_t')))
    # cgm.save('cad-merged.cgns', cad_cgns)
    cad_kdtree = KDTree(kdtree_points)
    return cad_connectivity, cad_kdtree


def getMinimumLengths(kdtree: KDTree) -> np.ndarray:
    coords = cad_kdtree.data
    n_node = coords.shape[0]
    assert coords.shape[1] == 3
    min_lengths = np.ndarray((n_node), float)
    for i_node in range(n_node):
        distance_ij, j_node = kdtree.query(coords[i_node])
        min_lengths[i_node] = distance_ij
    return min_lengths


def getRefinedKdtree(cad_nodes: np.ndarray, cad_connectivity: np.ndarray,
        mesh_kdtree: KDTree, mesh_min_lengths: np.ndarray) -> KDTree:
    # cad_nodes = cad_kdtree.data ⚠️
    assert cad_nodes.shape[1] == 3
    n_cad_cell = cad_connectivity.shape[0] // 3
    refined_points = []
    refined_tuples = []
    n_gap = 4
    area_coords = np.arange(1, n_gap, 1) / n_gap
    for i_cad_cell in range(n_cad_cell):
        first = i_cad_cell * 3
        node_index_tuple = cad_connectivity[first : first + 3] - 1
        corners = cad_nodes[node_index_tuple, :]
        assert corners.shape == (3, 3)
        need_refinement = False
        for corner in corners:
            _, i_mesh_point = mesh_kdtree.query(corner)
            if mesh_min_lengths[i_mesh_point] < 1:  # in inch
                need_refinement = True
                break
        if not need_refinement:
            continue
        print('add refined points in CAD cell', i_cad_cell)
        coord_a, coord_b, coord_c = corners[0, :], corners[1, :], corners[2, :]
        for area_a in area_coords:
            for area_b in area_coords:
                area_c = 1. - area_a - area_b
                if area_c <= 0.01:
                    continue
                new_point = coord_a * area_a + coord_b * area_b + coord_c * area_c
                refined_points.append(new_point)
        # refined_points.append(coord_a)
        # refined_points.append(coord_b)
        # refined_points.append(coord_c)
        # i, j, k = node_index_tuple
        # refined_tuples.append((i, j, k))
    print(len(refined_points), 'points added')
    return KDTree(refined_points), refined_points, refined_tuples


def writePointsToVtu(name, xyz_tuples, ijk_tuples):
    if isinstance(xyz_tuples, list):
        xyz_tuples = np.array(xyz_tuples)
    n_points = len(xyz_tuples)
    # n_cells = len(ijk_tuples)

    grid = vtk.vtkUnstructuredGrid()

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(n_points)
    for i in range(n_points):
        vtk_points.InsertPoint(i, xyz_tuples[i])
        vtk_cell = vtk.vtkVertex()
        vtk_id_list = vtk_cell.GetPointIds()
        vtk_id_list.SetId(0, i)
        grid.InsertNextCell(vtk_cell.GetCellType(), vtk_id_list)

    # for c in range(n_cells):
    #     vtk_cell = vtk.vtkTriangle()
    #     vtk_id_list = vtk_cell.GetPointIds()
    #     i, j, k = ijk_tuples[c]
    #     vtk_id_list.SetId(0, i)
    #     vtk_id_list.SetId(1, j)
    #     vtk_id_list.SetId(2, k)
    #     vtk_cell = vtk.vtkVertex()
    #     vtk_id_list = vtk_cell.GetPointIds()
    #     vtk_id_list.SetId(0, c)
    #     grid.InsertNextCell(vtk_cell.GetCellType(), vtk_id_list)

    grid.SetPoints(vtk_points)
    writer = vtk.vtkXMLDataSetWriter()
    writer.SetInputData(grid)
    output = f'refined_{name}.vtu'
    print('writing to', output)
    writer.SetFileName(output)
    writer.SetDataModeToBinary()
    writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 high_order.py')
    parser.add_argument('--cad', type=str, help='the input CAD file (currently in CGNS)')
    parser.add_argument('--mesh', type=str, help='the input linear mesh file')
    parser.add_argument('--output', type=str, help='the output high-order mesh file')
    parser.add_argument('--order', type=int, default=1, help='order of the output mesh')
    parser.add_argument('--target', choices=['corner', 'center', 'foot'], default='corner',
        help='which kind of point to be shifhted to')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    output = args.output
    if output is None:
        i = args.cad.index('max=') + 4
        j = args.cad[i:].index('mm')
        h_max = float(args.cad[i : i + j])
        output = f'{args.mesh[:-5]}_order={args.order}_cad={h_max:3.1e}mm_shift={args.target}.cgns'

    # load the CAD model
    cad_cgns, _, _ = cgm.load(args.cad)
    cad_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(cad_cgns, 'CGNSBase_t'), 'Zone_t')
    cad_zone_size = wrapper.getNodeData(cad_zone)
    n_node = cad_zone_size[0][0]
    n_cell = cad_zone_size[0][1]
    if args.verbose:
        print(f'in CAD: n_node = {n_node}, n_cell = {n_cell}')

    cad_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(cad_zone, 'GridCoordinates_t'), 'DataArray_t')
    cad_coords_x, cad_coords_y, cad_coords_z = cad_coords[X][1], cad_coords[Y][1], cad_coords[Z][1]

    if args.target == 'corner':
        kdtree_points = np.zeros((n_cell * 3, 3), dtype=float)  # 3 point for each triangle
        def setKdtreePoints(cell_index, node_index_tuple):
            first = cell_index * 3
            for i in range(3):
                j = first + i
                k = node_index_tuple[i]
                kdtree_points[j] = cad_coords_x[k], cad_coords_y[k], cad_coords_z[k]
    elif args.target == 'center' or args.target == 'foot':
        kdtree_points = np.zeros((n_cell, 3), dtype=float)  # 1 point for each triangle
        def setKdtreePoints(cell_index, node_index_tuple):
            kdtree_points[cell_index] = (
                np.sum(cad_coords_x[node_index_tuple]) / 3,
                np.sum(cad_coords_y[node_index_tuple]) / 3,
                np.sum(cad_coords_z[node_index_tuple]) / 3)
    else:
        assert False

    # TODO(PVC): (args.target == corner) do not need cad_connectivity
    cad_connectivity, cad_kdtree = getKdtreeFromCAD(setKdtreePoints)

    # load the linear mesh
    mesh_cgns, _, _ = cgm.load(args.mesh)

    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = wrapper.getNodeData(mesh_zone)
    n_node = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    if args.verbose:
        print(f'in linear mesh: n_node = {n_node}, n_cell = {n_cell}')

    mesh_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    section = wrapper.getUniqueChildByType(mesh_zone, 'Elements_t')
    element_type = wrapper.getNodeData(section)
    assert element_type[0] == 7  # QUAD_4
    old_connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementConnectivity'))

    mesh_coords_x, mesh_coords_y, mesh_coords_z
    mesh_kdtree_points = np.ndarray((len(mesh_coords_x), 3))
    mesh_kdtree_points[:, X] = mesh_coords_x
    mesh_kdtree_points[:, Y] = mesh_coords_y
    mesh_kdtree_points[:, Z] = mesh_coords_z
    mesh_kdtree = KDTree(mesh_kdtree_points)
    mesh_min_lengths = getMinimumLengths(mesh_kdtree)

    cad_nodes = np.ndarray((len(cad_coords_x), 3))
    cad_nodes[:, X] = cad_coords_x
    cad_nodes[:, Y] = cad_coords_y
    cad_nodes[:, Z] = cad_coords_z
    refined_kdtree, refined_points, refined_tuples = getRefinedKdtree(cad_nodes, cad_connectivity, mesh_kdtree, mesh_min_lengths)
    def Refine(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # capture refined_kdtree
        d, i = refined_kdtree.query(p)
        if np.linalg.norm(p - q) > d:
            q = refined_kdtree.data[i]
        return q

    writePointsToVtu(args.cad, refined_points, refined_tuples)

    if args.order == 2:
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
            print(f'Convert QUAD_4[{i_cell_mesh}] -> QUAD_9[{i_cell_mesh}]')
        assert i_node_next == n_node_new
        n_node = n_node_new
        if args.verbose:
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

    # update the coords
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    if args.target == 'corner' or args.target == 'center':
        getShiftedPoint = lambda query_point: getNearestPoint(query_point, cad_kdtree)
    elif args.target == 'foot':
        getShiftedPoint = lambda query_point: getFootOnTriangle(query_point, cad_kdtree,
            cad_connectivity, cad_coords_x, cad_coords_y, cad_coords_z)
    else:
        assert False

    # shift nodes to the (exactly or nearly) closest points on the wall
    for i in range(n_node):
        query_point = np.array([mesh_coords_x[i], mesh_coords_y[i], mesh_coords_z[i]])
        x, y, z = Refine(query_point, getShiftedPoint(query_point))
        mesh_coords_x[i] = x
        mesh_coords_y[i] = y
        mesh_coords_z[i] = z
        print(f'Shift i_node / n_node = {i} / {n_node}')

    if args.verbose:
        print('write to ', output)
    cgm.save(output, mesh_cgns)
