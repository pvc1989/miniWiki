import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk

import os
import sys
import numpy as np
import argparse
import wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Filter a CGNS file by a given field value.')
    parser.add_argument('--folder', type=str, default='.', help='the working folder containing input and output files')
    parser.add_argument('--input', type=str, help='the CGNS file to be filtered')
    parser.add_argument('--field', type=str, default='BCType', help='the field for filtering')
    parser.add_argument('--value', type=int, default=3, help='the value to be keeped')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # CGNSTree_t level
    old_tree, links, paths = cgm.load(f'{args.folder}/{args.input}')
    if args.verbose:
        print()
        print(old_tree)
    new_tree = cgl.newCGNSTree()

    # CGNSBase_t level
    old_base = wrapper.getUniqueChildByType(old_tree, 'CGNSBase_t')
    cell_dim, phys_dim = wrapper.getDimensions(old_base)
    print()
    print('cell_dim =', cell_dim)
    print('phys_dim =', phys_dim)
    new_base = cgl.newCGNSBase(new_tree,
        wrapper.getNodeName(old_base), cell_dim, phys_dim)

    # Zone_t level
    old_zone = wrapper.getUniqueChildByType(old_base, 'Zone_t')
    zone_name = wrapper.getNodeName(old_zone)
    zone_size = wrapper.getNodeData(old_zone)
    print('zone_size =', zone_size)
    assert zone_size.shape == (1, 3)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(old_zone, 'GridCoordinates_t'),
        'DataArray_t')
    X, Y, Z = 0, 1, 2
    coords_x, coords_y, coords_z = coords[X], coords[Y], coords[Z]
    assert 'CoordinateX' == wrapper.getNodeName(coords_x)
    assert 'CoordinateY' == wrapper.getNodeName(coords_y)
    assert 'CoordinateZ' == wrapper.getNodeName(coords_z)
    values_x, values_y, values_z = coords_x[1], coords_y[1], coords_z[1]
    assert (n_node,) == values_x.shape == values_y.shape == values_z.shape

    # Elements_t level
    section = wrapper.getUniqueChildByType(old_zone, 'Elements_t')
    element_type = wrapper.getUniqueChildByType(section, 'ElementType_t')
    assert element_type is None
    # TODO(gaomin): add ElementType_t(QUAD_4)
    connectivity = wrapper.getUniqueChildByName(section, 'ElementConnectivity')
    assert wrapper.getNodeLabel(connectivity) == 'DataArray_t'
    if args.verbose:
        print()
        print(connectivity)
    old_connectivity_list = wrapper.getNodeData(connectivity)
    n_node_per_cell = 4  # FLEXI only supports `QUAD_4`
    assert old_connectivity_list.shape == (n_node_per_cell * n_cell,)

    # FlowSolution_t level
    solution = wrapper.getUniqueChildByType(old_zone, 'FlowSolution_t')
    if args.verbose:
        print()
        print(solution)
    assert wrapper.getUniqueChildByType(solution, 'GridLocation_t') is None
    # TODO(gaomin): add GridLocation_t(Vertex)
    values = wrapper.getNodeData(wrapper.getUniqueChildByName(solution, args.field))
    assert values.shape == (n_node,)

    # filter cells by types
    filtered_cells = list()
    filtered_nodes = list()
    for i_cell in range(n_cell):
        first = n_node_per_cell * i_cell
        value_i = values[old_connectivity_list[first] - 1]  # connectivity given in 1-based
        for curr in range(first + 1, first + n_node_per_cell):
            # all nodes share the same value if 
            assert value_i == values[old_connectivity_list[curr] - 1]
        if value_i == args.value:
            filtered_cells.append(i_cell)
            for curr in range(first, first + n_node_per_cell):
                filtered_nodes.append(old_connectivity_list[curr] - 1)
                # TODO(pvc): support non-duplicated nodes
    print(f'{len(filtered_cells)} of {n_cell} cells filtered out')
    print(f'{len(filtered_nodes)} of {n_node} nodes filtered out')
    old_to_new = np.ndarray((n_node,), dtype=old_connectivity_list.dtype)
    for i_new in range(len(filtered_nodes)):
        i_old = filtered_nodes[i_new]
        old_to_new[i_old] = i_new

    # create new Zone_t
    n_node = len(filtered_nodes)
    n_cell = len(filtered_cells)
    n_rind = 0
    new_zone = cgl.newZone(new_base, wrapper.getNodeName(old_zone),
        zsize=np.array([[n_node, n_cell, n_rind]]), ztype='Unstructured')
    assert zone_size.shape == wrapper.getNodeData(new_zone).shape

    # create new GridCoordinate_
    new_coords = cgl.newGridCoordinates(new_zone, 'GridCoordinates')
    cgl.newDataArray(new_coords, 'CoordinateX', values_x[filtered_nodes])
    cgl.newDataArray(new_coords, 'CoordinateY', values_y[filtered_nodes])
    cgl.newDataArray(new_coords, 'CoordinateZ', values_z[filtered_nodes])

    # create new Elements_t
    new_connectivity_list = np.ndarray((n_node_per_cell * n_cell,), dtype=old_connectivity_list.dtype)
    for i_cell in range(n_cell):
        new_first = n_node_per_cell * i_cell
        old_first = n_node_per_cell * filtered_cells[i_cell]
        for i in range(n_node_per_cell):
            i_node_old = old_connectivity_list[old_first + i]
            i_node_new = old_to_new[i_node_old - 1] + 1
            # both i_node's are 1-based
            new_connectivity_list[new_first + i] = i_node_new
    assert 1 == np.min(new_connectivity_list) <= np.max(new_connectivity_list) == n_node
    erange_old = wrapper.getNodeData(wrapper.getUniqueChildByName(section, 'ElementRange'))
    erange_new = np.ndarray(erange_old.shape, erange_old.dtype)
    erange_new[0] = 1
    erange_new[1] = n_cell
    cgl.newElements(new_zone, 'Elements', erange=erange_new, etype='QUAD_4',
        econnectivity=new_connectivity_list)

    if args.verbose:
        print()
        print(new_tree)

    # write the new CGNSTree_t
    output_folder = f'{args.folder}/{args.field}={args.value}'
    os.makedirs(output_folder, exist_ok=True)
    output = f'{output_folder}/filtered.cgns'
    print('writing to', output)
    cgm.save(output, new_tree)
