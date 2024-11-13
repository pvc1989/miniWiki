import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper

import numpy as np
import argparse
import sys
import os


X, Y, Z = 0, 1, 2


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Increase the order of elements without merging new points.')
    parser.add_argument('--folder', type=str, help='the working directory')
    parser.add_argument('--input', type=str, help='the input linear mesh file')
    parser.add_argument('--order', type=int, default=2, help='order of the output mesh')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    # load the linear mesh
    mesh_cgns, _, _ = cgm.load(f'{args.folder}/{args.input}')

    mesh_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(mesh_cgns, 'CGNSBase_t'), 'Zone_t')
    mesh_zone_size = wrapper.getNodeData(mesh_zone)
    n_node_old = mesh_zone_size[0][0]
    n_cell = mesh_zone_size[0][1]
    print(f'before converting: n_node = {n_node_old}, n_cell = {n_cell}')

    mesh_coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(mesh_zone, 'GridCoordinates_t'), 'DataArray_t')
    mesh_coords_x, mesh_coords_y, mesh_coords_z = mesh_coords[X][1], mesh_coords[Y][1], mesh_coords[Z][1]

    section = wrapper.getUniqueChildByType(mesh_zone, 'Elements_t')
    element_type = wrapper.getNodeData(section)

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

    old_connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(section, 'ElementConnectivity'))

    # add new points and update connectivity:
    n_node_new = n_node_old + n_cell * (new_type_npe - old_type_npe)
    new_coords_x = np.resize(mesh_coords_x, n_node_new)
    new_coords_y = np.resize(mesh_coords_y, n_node_new)
    new_coords_z = np.resize(mesh_coords_z, n_node_new)
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
            new_coords_x[i_node_next] = np.dot(mesh_coords_x[old_index - 1], shape)
            new_coords_y[i_node_next] = np.dot(mesh_coords_y[old_index - 1], shape)
            new_coords_z[i_node_next] = np.dot(mesh_coords_z[old_index - 1], shape)
            i_node_next += 1
            new_connectivity[new_fisrt + i_local] = i_node_next
        if args.verbose:
            print(f'cell[{i_cell} / {n_cell}] converted')
    assert i_node_next == n_node_new
    print(f'after converting: n_node = {n_node_new}, n_cell = {n_cell}')

    # update the cells
    cgu.removeChildByName(section, 'ElementConnectivity')
    cgl.newDataArray(section, 'ElementConnectivity', new_connectivity)

    # write the new mesh out
    mesh_zone_size[0][0] = n_node_new
    new_type_val = cgk.ElementType_l.index(new_type_str)
    assert get_type_info(new_type_val)[0] == new_type_str
    element_type[0] = new_type_val
    mesh_coords[X][1] = new_coords_x
    mesh_coords[Y][1] = new_coords_y
    mesh_coords[Z][1] = new_coords_z

    output_folder = f'{args.folder}/order={args.order}'
    os.makedirs(output_folder, exist_ok=True)
    output = f'{output_folder}/to_be_merged.cgns'
    print('writing to', output)
    cgm.save(output, mesh_cgns)
