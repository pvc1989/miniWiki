import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk

import numpy as np
import argparse
import wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 get_surface.py',
        description = 'Get elements and the nodes they used on surfaces specified by a given `FLEXI::BCType`.')
    parser.add_argument('--input', type=str, help='the CGNS file to be processed')
    parser.add_argument('--output', type=str, default='surface.cgns',
        help='the CGNS file containing elements and nodes on the specified surfaces')
    parser.add_argument('--bctype', type=int, default=3,
        help='the BCType of the specified surfaces')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # CGNSTree_t level
    old_tree, links, paths = cgm.load(args.input)
    new_tree = cgl.newCGNSTree()

    # CGNSBase_t level
    old_base = wrapper.getUniqueChildByType(old_tree, 'CGNSBase_t')
    cell_dim, phys_dim = wrapper.getDimensions(old_base)
    print('cell_dim =', cell_dim)
    print('phys_dim =', phys_dim)
    new_base = cgl.newCGNSBase(new_tree,
        wrapper.getNodeName(old_base), cell_dim, phys_dim)

    # Zone_t level
    old_zone = wrapper.getUniqueChildByType(old_base, 'Zone_t')
    zone_name = wrapper.getNodeName(old_zone)
    zone_size = old_zone[1]
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
    sections = wrapper.getChildrenByType(old_zone, 'Elements_t')
    print('n_section =', len(sections))
    for section in sections:
        element_type = wrapper.getUniqueChildByType(section, 'ElementType_t')
        assert element_type is None
        # TODO(gaomin): add ElementType_t(QUAD_4)
        connectivity = wrapper.getUniqueChildByName(section, 'ElementConnectivity')
        assert wrapper.getNodeLabel(connectivity) == 'DataArray_t'
        print(connectivity)
        connectivity_list = wrapper.getNodeData(connectivity)
        n_node_per_cell = 4  # FLEXI only supports `QUAD_4`
        assert connectivity_list.shape == (n_node_per_cell * n_cell,)

    # FlowSolution_t level
    solution = wrapper.getUniqueChildByType(old_zone, 'FlowSolution_t')
    assert wrapper.getUniqueChildByType(solution, 'GridLocation_t') is None
    # TODO(gaomin): add GridLocation_t(Vertex)
    type_values = wrapper.getNodeData(wrapper.getUniqueChildByName(solution, 'BCType'))
    assert type_values.shape == (n_node,)

    # filter cells by types
    filtered_cells = list()
    filtered_nodes = list()
    for i_cell in range(n_cell):
        first = n_node_per_cell * i_cell
        type_i = type_values[connectivity_list[first] - 1]  # connectivity given in 1-based
        for curr in range(first + 1, first + n_node_per_cell):
            assert type_i == type_values[connectivity_list[curr] - 1]
        if type_i == args.bctype:
            filtered_cells.append(i_cell)
            for curr in range(first, first + n_node_per_cell):
                filtered_nodes.append(connectivity_list[curr] - 1)
                # TODO(pvc): support non-duplicated nodes
    print(f'{len(filtered_cells)} of {n_cell} cells filtered out')
    print(f'{len(filtered_nodes)} of {n_node} nodes filtered out')
