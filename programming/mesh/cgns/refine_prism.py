import argparse
import sys

import numpy as np
import pycgns_wrapper
import CGNS.MAP as cgm
import CGNS.PAT.cgnskeywords as cgk
import CGNS.PAT.cgnslib as cgl

from scipy.spatial import KDTree, Delaunay


def get_bc_point_set(zone, section_names: list[str]) -> set:
    bc_point_set = set()
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        name = pycgns_wrapper.getNodeName(section)
        if name not in section_names:
            continue
        conn_node = pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity')
        conn_global = pycgns_wrapper.getNodeData(conn_node) - 1
        for i_point in conn_global:
            bc_point_set.add(i_point)
    print(f'{len(bc_point_set)} boundary points found')
    return bc_point_set


def multiple_to_unique(xyz_multiple: list, bc_point_multiple: list, radius=1e-8) -> tuple[np.ndarray, list]:
    points = np.array(xyz_multiple)
    kdtree = KDTree(points)
    i_unique_set = set()
    bc_point_unique = list()
    xyz_unique = list()
    for i_multiple in range(len(points)):
        point_i = points[i_multiple]
        neighbors = kdtree.query_ball_point(point_i, radius)
        i_unique = np.min(neighbors)
        if i_unique not in i_unique_set:
            i_unique_set.add(i_unique)
            assert bc_point_multiple[i_unique] == bc_point_multiple[i_multiple]
            bc_point_unique.append(bc_point_multiple[i_unique])
            xyz_unique.append(point_i)
    print(f'xyz_multiple = {len(xyz_multiple)}')
    print(f'n_unique = {len(xyz_unique)}')
    return np.array(xyz_unique), bc_point_unique


def add_midedge_point(u: int, v: int, conn_local: np.ndarray, xyz: np.ndarray, bc_point_set: set,
        xyz_new: list, bc_point_list_new: list):
    u, v = conn_local[u], conn_local[v]
    bc_point_list_new.append(u in bc_point_set and v in bc_point_set)
    xyz_new.append((xyz[u] + xyz[v]) / 2)


def refine(xyz, bc_point_set: set, section, level: int, levels: np.ndarray, xyz_new: list, bc_point_list_new: list):
    erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
    len_xyz_new = len(xyz_new)
    conn_node = pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity')
    conn_global = pycgns_wrapper.getNodeData(conn_node)
    npe = 6
    n_refined = 0
    offset = erange[0] - 1
    for i_cell in range(erange[0] - 1, erange[1]):
        if levels[i_cell] > level:
            continue
        n_refined += 1
        first = (i_cell - offset) * npe
        conn_local = conn_global[first : first + npe] - 1
        # add nodes on top
        add_midedge_point(0, 1, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
        add_midedge_point(1, 2, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
        add_midedge_point(2, 0, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
        # add nodes on bottom
        add_midedge_point(3, 4, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
        add_midedge_point(4, 5, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
        add_midedge_point(5, 3, conn_local, xyz, bc_point_set, xyz_new, bc_point_list_new)
    print(f'{n_refined} prisms refined by adding {len(xyz_new) - len_xyz_new} points in {pycgns_wrapper.getNodeName(section)}')


def filter_tetra(new_tetra: np.ndarray, bc_point_set: set) -> np.ndarray:
    print(f'before filter_tetra: new_tetra.shape = {new_tetra.shape}')
    new_tetra_list = list()
    for tetra in new_tetra:
        all_bc = True
        for i in tetra:
            if i not in bc_point_set:
                all_bc = False
                break
        if not all_bc:
            new_tetra_list.append(tetra)
    new_tetra = np.array(new_tetra_list)
    print(f'after filter_tetra: new_tetra.shape = {new_tetra.shape}')
    return new_tetra


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Refine prisms in a Prism-Tetra dominated mesh.')
    parser.add_argument('--input', type=str, help='the input CGNS mesh')
    parser.add_argument('--level', type=int, default=10, help='max level of prisms to be refined')
    parser.add_argument('--sections', type=str, nargs='+', help='the given list of sections')
    args = parser.parse_args()

    # read the input CGNS file
    input = args.input
    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(input)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    print(f'in volume mesh: n_node = {n_node}, n_cell = {n_cell}')
    xyz, x, y, z = pycgns_wrapper.readPoints(zone, zone_size)

    levels = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByName(
        pycgns_wrapper.getUniqueChildByType(zone, 'FlowSolution_t'), 'CellLevel'))

    bc_point_set = get_bc_point_set(zone, args.sections)

    # add new points
    xyz_new = list()
    bc_point_list_new = list()
    assert n_node == len(xyz)
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        type_val = pycgns_wrapper.getNodeData(section)[0]
        type_str = cgk.ElementType_l[type_val]
        if 'PENTA' in type_str:
            refine(xyz, bc_point_set, section, args.level, levels, xyz_new, bc_point_list_new)

    xyz_new, bc_point_list_new = multiple_to_unique(xyz_new, bc_point_list_new)
    for i_new in range(len(xyz_new)):
        if bc_point_list_new[i_new]:
            bc_point_set.add(n_node + i_new)

    n_node += len(xyz_new)

    # create a new file and zone
    new_cgns = cgl.newCGNSTree()
    new_base = cgl.newCGNSBase(new_cgns, 'RefinedBase', 3, 3)
    new_zone_size = np.array([[n_node, n_cell, 0]])
    new_zone = cgl.newZone(new_base, 'RefinedZone',
        new_zone_size, ztype='Unstructured')
    pycgns_wrapper.mergePointList([xyz, xyz_new], n_node, new_zone, new_zone_size)
    new_xyz, _, _, _ = pycgns_wrapper.readPoints(new_zone, new_zone_size)

    # get new connectivity by Delaunay
    new_tetra = Delaunay(new_xyz).simplices
    new_tetra = filter_tetra(new_tetra, bc_point_set)

    n_cell = len(new_tetra)
    new_zone_size[0][1] = n_cell

    # add the new conn to new_zone
    new_erange = np.array([1, n_cell])
    new_connectivity_list = new_tetra.flatten()
    cgl.newElements(new_zone, 'Elements', erange=new_erange, etype='TETRA_4',
        econnectivity=new_connectivity_list + 1)

    print(new_zone_size)
    print(f'after refining, n_node = {n_node}, n_cell = {n_cell}')

    dot_pos = input.rfind('.')
    output = f'{input[:dot_pos]}_refined.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, new_cgns)
    print(args)
