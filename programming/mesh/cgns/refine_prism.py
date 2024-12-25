import argparse
import sys

import numpy as np
import pycgns_wrapper
import CGNS.MAP as cgm
import CGNS.PAT.cgnskeywords as cgk
import CGNS.PAT.cgnslib as cgl

from scipy.spatial import KDTree, Delaunay


def unique(points: list, radius=1e-8):
    points = np.array(points)
    i_min_set = set()
    kdtree = KDTree(points)
    for i in range(len(points)):
        neighbors = kdtree.query_ball_point(points[i], radius)
        i_min_set.add(np.min(neighbors))
    return points[list(i_min_set)]


def refine(xyz, section, level: int, levels: np.ndarray):
    erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
    xyz_new = list()
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
        A, B, C = 0, 1, 2
        xyz_new.append((xyz[conn_local[A]] + xyz[conn_local[B]]) / 2)
        xyz_new.append((xyz[conn_local[B]] + xyz[conn_local[C]]) / 2)
        xyz_new.append((xyz[conn_local[C]] + xyz[conn_local[A]]) / 2)
        # add nodes on bottom
        A, B, C = 3, 4, 5
        xyz_new.append((xyz[conn_local[A]] + xyz[conn_local[B]]) / 2)
        xyz_new.append((xyz[conn_local[B]] + xyz[conn_local[C]]) / 2)
        xyz_new.append((xyz[conn_local[C]] + xyz[conn_local[A]]) / 2)
    xyz_new = unique(xyz_new)
    print(f'{n_refined} prisms refined by adding {len(xyz_new)} points in {pycgns_wrapper.getNodeName(section)}')
    return xyz_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Refine prisms in a Prism-Tetra dominated mesh.')
    parser.add_argument('--input', type=str, help='the input CGNS mesh')
    parser.add_argument('--level', type=int, default=10, help='max level of prisms to be refined')
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

    # add new points
    xyz_list = [xyz]
    assert n_node == len(xyz)
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        type_val = pycgns_wrapper.getNodeData(section)[0]
        type_str = cgk.ElementType_l[type_val]
        if 'PENTA' in type_str:
            xyz_new = refine(xyz, section, args.level, levels)
            xyz_list.append(xyz_new)
            n_node += len(xyz_new)
            erange_old = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByName(section, 'ElementRange'))

    # create a new file and zone
    new_cgns = cgl.newCGNSTree()
    new_base = cgl.newCGNSBase(new_cgns, 'RefinedBase', 3, 3)
    new_zone_size = np.array([[n_node, n_cell, 0]])
    new_zone = cgl.newZone(new_base, 'RefinedZone',
        new_zone_size, ztype='Unstructured')
    pycgns_wrapper.mergePointList(xyz_list, n_node, new_zone, new_zone_size)
    new_xyz, _, _, _ = pycgns_wrapper.readPoints(new_zone, new_zone_size)

    # get new connectivity by Delaunay
    new_tetra = Delaunay(new_xyz).simplices
    print(new_tetra.shape)
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
