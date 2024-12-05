import argparse
import os
import sys

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import pycgns_wrapper
from pycgns_wrapper import X, Y, Z


type_to_local_coords = dict()
type_to_local_coords['QUAD_4'] = np.array([
    [-1, -1], [+1, -1], [+1, +1], [-1, +1]
], dtype=float)
type_to_local_coords['QUAD_9'] = np.array([
    [-1, -1], [+1, -1], [+1, +1], [-1, +1],
    [0, -1], [+1, 0], [0, +1], [-1, 0], [0, 0]
], dtype=float)
type_to_local_coords['HEXA_8'] = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]
], dtype=float)
type_to_local_coords['HEXA_27'] = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    # mid-edge points:
    [0, -1, -1], [+1, 0, -1], [0, +1, -1], [-1, 0, -1],
    [-1, -1, 0], [+1, -1, 0], [+1, +1, 0], [-1, +1, 0],
    [0, -1, +1], [+1, 0, +1], [0, +1, +1], [-1, 0, +1],
    # mid-face points:
    [0, 0, -1],
    [0, -1, 0], [+1, 0, 0], [0, +1, 0], [-1, 0, 0],
    [0, 0, +1],
    # mid-cell point:
    [0, 0, 0]
], dtype=float)


type_to_shape_functions = dict()
type_to_shape_functions['QUAD_4'] = lambda local_fixed, local_query: \
    (1 + local_query[X] * local_fixed[:, X]) * \
    (1 + local_query[Y] * local_fixed[:, Y]) / 4.0
type_to_shape_functions['HEXA_8'] = lambda local_fixed, local_query: \
    (1 + local_query[X] * local_fixed[:, X]) * \
    (1 + local_query[Y] * local_fixed[:, Y]) * \
    (1 + local_query[Z] * local_fixed[:, Z]) / 8.0


def get_type_info(type_val: int) -> tuple[str, int]:
    return cgk.ElementType_l[type_val], cgk.ElementTypeNPE_l[type_val]


def get_new_type_info(old_type_val: int, order: int) -> tuple[str, int]:
    # TODO(PVC): support higher orders
    old_type_str, old_type_npe = get_type_info(old_type_val)
    new_type_str = str()
    if old_type_str == 'QUAD_4':
        if order == 2:
            new_type_str = 'QUAD_9'
        else:
            assert False, (old_type_str, order)
    elif old_type_str == 'HEXA_8':
        if order == 2:
            new_type_str = 'HEXA_27'
        else:
            assert False, (old_type_str, order)
    else:
        assert False, (old_type_str, order)
    new_type_npe = int(new_type_str[new_type_str.find('_') + 1:])
    return new_type_str, new_type_npe


def add_points(section, xyz_old: np.ndarray, n_node_old: int, args) -> np.ndarray:
    element_type = pycgns_wrapper.getNodeData(section)

    # get old and new type info
    old_type_val = element_type[0]
    old_type_str, old_type_npe = get_type_info(old_type_val)
    new_type_str, new_type_npe = get_new_type_info(old_type_val, args.order)
    print(f'converting ({old_type_str}, {old_type_npe}) -> ({new_type_str}, {new_type_npe})')
    old_local_coords = type_to_local_coords[old_type_str]
    assert old_type_npe == len(old_local_coords)
    new_local_coords = type_to_local_coords[new_type_str]
    assert new_type_npe == len(new_local_coords)
    old_shape_function = type_to_shape_functions[old_type_str]

    old_connectivity = pycgns_wrapper.getNodeData(
        pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))

    # add new points and update connectivity:
    n_cell = len(old_connectivity) // old_type_npe
    n_node_add = n_cell * (new_type_npe - old_type_npe)
    n_node_new = n_node_old + n_node_add
    xyz_new = np.ndarray((n_node_add, 3))
    i_node_next = n_node_old
    assert old_connectivity.shape == (n_cell * old_type_npe,)
    new_connectivity = np.resize(old_connectivity, n_cell * new_type_npe)
    for i_cell in range(n_cell):
        old_first = i_cell * old_type_npe
        new_fisrt = i_cell * new_type_npe
        # copy original connectivity
        old_index = old_connectivity[old_first : old_first + old_type_npe]
        new_connectivity[new_fisrt : new_fisrt + old_type_npe] = old_index
        for i_local in range(old_type_npe, new_type_npe):
            shape = old_shape_function(old_local_coords, new_local_coords[i_local])
            xyz_new[i_node_next - n_node_old] = np.dot(shape, xyz_old[old_index - 1])
            i_node_next += 1
            new_connectivity[new_fisrt + i_local] = i_node_next
        if args.verbose:
            print(f'cell[{i_cell} / {n_cell}] converted')
    assert i_node_next == n_node_new
    print(f'after converting: n_node = {n_node_new}, n_cell = {n_cell}')

    # update the connectivity of cells
    new_type_val = cgk.ElementType_l.index(new_type_str)
    assert get_type_info(new_type_val)[0] == new_type_str
    element_type[0] = new_type_val
    cgu.removeChildByName(section, 'ElementConnectivity')
    cgl.newDataArray(section, 'ElementConnectivity', new_connectivity)

    return xyz_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Increase the order of elements without merging new points.')
    parser.add_argument('--input', type=str, help='the input linear mesh file')
    parser.add_argument('--order', type=int, default=2, help='order of the output mesh')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    # load the linear mesh
    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.input)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    print(f'before converting: n_node = {n_node}, n_cell = {n_cell}')

    xyz_old, _, _, _ = pycgns_wrapper.readPoints(zone, zone_size)

    xyz_list = [xyz_old]
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        xyz_new = add_points(section, xyz_old, n_node, args)
        xyz_list.append(xyz_new)
        n_node += len(xyz_new)
    pycgns_wrapper.mergePointList(xyz_list, n_node, zone, zone_size)

    output_folder = f'{pycgns_wrapper.folder(args.input)}/order={args.order}'
    os.makedirs(output_folder, exist_ok=True)
    output = f'{output_folder}/to_be_merged.cgns'
    print('writing to', output)
    cgm.save(output, cgns)
