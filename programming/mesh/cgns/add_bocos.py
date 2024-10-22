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

quad4_faces = (
    np.array([0, 3, 2, 1], dtype=int),
    np.array([0, 1, 5, 4], dtype=int),
    np.array([1, 2, 6, 5], dtype=int),
    np.array([2, 3, 7, 6], dtype=int),
    np.array([0, 4, 7, 3], dtype=int),
    np.array([4, 5, 6, 7], dtype=int))


def getSurfaceKdtree(file_name: str) -> KDTree:
    cgns, _, _ = cgm.load(file_name)
    zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(cgns, 'CGNSBase_t'), 'Zone_t')
    zone_size = wrapper.getNodeData(zone)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    # load coordinates
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(zone, 'GridCoordinates_t'),
        'DataArray_t')
    X, Y, Z = 0, 1, 2
    coords_x, coords_y, coords_z = coords[X][1], coords[Y][1], coords[Z][1]
    assert len(coords_x) == len(coords_y) == len(coords_z) == n_node
    # load connectivity
    section = wrapper.getUniqueChildByType(zone, 'Elements_t')
    element_type = wrapper.getNodeData(section)
    assert element_type[0] == 7  # QUAD_4
    connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
    assert len(connectivity) == n_cell * 4
    # build centers of cells
    centers = np.ndarray((n_cell, 3))
    for i_cell in range(n_cell):
        first = i_cell * 4
        node_index_tuple = connectivity[first : first + 4] - 1
        centers[i_cell][X] = np.sum(coords_x[node_index_tuple]) / 4
        centers[i_cell][Y] = np.sum(coords_y[node_index_tuple]) / 4
        centers[i_cell][Z] = np.sum(coords_z[node_index_tuple]) / 4
    return KDTree(centers)


def getConnectivity(face_center: np.ndarray, boco_kdtrees, boco_connectivities) -> list[int]:
    min_distance = 1e100
    for i_boco in range(n_boco):
        kdtree_i = boco_kdtrees[i_boco]
        assert isinstance(kdtree_i, KDTree)
        distance, _ = kdtree_i.query(face_center)
        min_distance = min(distance, min_distance)
        if distance < 1e-10:
            return boco_connectivities[i_boco]
    assert False, min_distance


def addZoneBC(zone_bc, section, bc_name: str, bc_type: str):
    element_range = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementRange'))
    bc_range = np.array([element_range])
    new_boco = cgl.newBoundary(zone_bc, bname=bc_name, brange=bc_range,
        btype=bc_type, pttype='PointRange')
    cgl.newGridLocation(new_boco, 'FaceCenter')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Add surface elements and BCs to a volume mesh.')
    parser.add_argument('--volume', type=str, help='the input volume mesh file')
    parser.add_argument('--output', type=str, help='the output mesh with surface elements and BCs')
    parser.add_argument('--bc_names', type=str, nargs='+', help='list of BC names')
    parser.add_argument('--bc_types', type=str, nargs='+', help='list of BC types')
    parser.add_argument('--bc_grids', type=str, nargs='+', help='list of BC grids')
    parser.add_argument('--index_maps', type=str, nargs='+', help='list of volume-to-surface index map files')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    # load the volume mesh
    volume_cgns, _, _ = cgm.load(args.volume)
    volume_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(volume_cgns, 'CGNSBase_t'), 'Zone_t')
    volume_zone_size = wrapper.getNodeData(volume_zone)
    n_volume_node = volume_zone_size[0][0]
    n_volume_cell = volume_zone_size[0][1]
    print(f'in volume mesh: n_node = {n_volume_node}, n_cell = {n_volume_cell}')

    # load volume mesh coordiantes
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(volume_zone, 'GridCoordinates_t'),
        'DataArray_t')
    X, Y, Z = 0, 1, 2
    coords_x, coords_y, coords_z = coords[X][1], coords[Y][1], coords[Z][1]
    assert len(coords_x) == len(coords_y) == len(coords_z) == n_volume_node

    # load volume mesh connectivity
    volume_section = wrapper.getUniqueChildByType(volume_zone, 'Elements_t')
    element_type = wrapper.getNodeData(volume_section)
    assert element_type[0] == 17  # HEXA_8
    volume_connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(volume_section, 'ElementConnectivity'))
    assert len(volume_connectivity) == n_volume_cell * 8

    # get the index set of boundary points
    n_boco = len(args.bc_names)
    assert n_boco == len(args.index_maps)
    boco_points = set()  # 0-based
    for i_boco in range(n_boco):
        vol_to_surf = np.loadtxt(args.index_maps[i_boco], dtype=int)
        for i_vol in range(len(vol_to_surf)):
            if vol_to_surf[i_vol] >= 0:
                boco_points.add(i_vol)
    print(f'n_point_on_boundary = {len(boco_points)}')

    # get the index set of boundary cells
    boco_values = np.zeros((n_volume_cell,), dtype=int)
    boco_cells = set()
    for i_cell in range(n_volume_cell):
        on_boundary = False
        first_global = i_cell * 8
        for i_node_local in range(8):
            i_node_global = i_node_local + first_global
            if volume_connectivity[i_node_global] - 1 in boco_points:
                boco_values[i_cell] = 1
                boco_cells.add(i_cell)
                on_boundary = True
                break
        if args.verbose:
            print(f'[{i_cell / n_volume_cell:.2f}] Is cell[{i_cell}] on some boundary? {on_boundary}')
    print(f'n_cell_on_boundary = {len(boco_cells)}')

    # bc_in_cell_data = cgl.newFlowSolution(volume_zone, 'FlowSolution', 'CellCenter')
    # cgl.newDataArray(bc_in_cell_data, 'BCType', boco_values)

    # build a kd-tree for each surface mesh
    print('building surface kd-trees')
    boco_kdtrees = []
    for surf_grid in args.bc_grids:
        boco_kdtrees.append(getSurfaceKdtree(surf_grid))

    def all_on_surface(face_nodes_global) -> bool:
        for face_node_global in face_nodes_global:
            if face_node_global not in boco_points:
                return False
        return True

    # build a surface element for each BC cell
    print('building surface elements')
    surf_connectivities = np.ndarray((n_boco,), dtype=list)
    for i_boco in range(n_boco):
        surf_connectivities[i_boco] = list()
    for i_cell in boco_cells:
        offset = i_cell * 8
        for face_nodes_local in quad4_faces:
            face_nodes_global = volume_connectivity[face_nodes_local + offset] - 1
            if not all_on_surface(face_nodes_global):
                continue
            face_center = np.array([
                np.sum(coords_x[face_nodes_global]) / 4,
                np.sum(coords_y[face_nodes_global]) / 4,
                np.sum(coords_z[face_nodes_global]) / 4,
            ])
            try:
                boco_i = getConnectivity(face_center, boco_kdtrees, surf_connectivities)
            except Exception:
                print(i_boco, i_cell, face_nodes_global)
            for i_node_global in face_nodes_global:
                boco_i.append(i_node_global + 1)  # CGNS use 1-based index

    # check consistency on face elements
    for i_boco in range(n_boco):
        assert len(surf_connectivities[i_boco]) == boco_kdtrees[i_boco].n * 4

    # build Elements_t and ZoneBC_t objects
    print('building Elements_t\'s and ZoneBC_t\'s')
    i_cell_next = n_volume_cell + 1  # again, 1-based
    zone_bc = cgl.newZoneBC(volume_zone)
    for i_boco in range(n_boco):
        surf_connectivity = surf_connectivities[i_boco]
        n_face = len(surf_connectivity) // 4
        first = i_cell_next
        last = i_cell_next + n_face - 1  # inclusive
        name = f'{args.bc_names[i_boco]}'
        erange = np.array([first, last], dtype=int)
        section = cgl.newElements(volume_zone, name, 'QUAD_4', erange,
            econnectivity=np.array(surf_connectivity))
        addZoneBC(zone_bc, section, name, args.bc_types[i_boco])
        i_cell_next = last + 1

    # write the BC-added mesh
    output = args.output
    if output is None:
        output = f'BCadded_{args.volume}'
    print('writing to', output)
    cgm.save(output, volume_cgns)
