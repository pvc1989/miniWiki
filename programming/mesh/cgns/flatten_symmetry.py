import argparse
import sys

import numpy as np

import CGNS.MAP as cgm
import pycgns_wrapper


def select_component(x, y, z, args):
    if args.component == 'X':
        return x
    elif args.component == 'Y':
        return y
    elif args.component == 'Z':
        return z
    else:
        assert False


def select_points(zone, args) -> list:
    i_node_set = set()
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        name = pycgns_wrapper.getNodeName(section)
        if name not in args.sections:
            continue
        connectivity = pycgns_wrapper.getNodeData(
            pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        for i_node in connectivity:
            i_node_set.add(i_node - 1)
    return list(i_node_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Flatten the symmetry plane by setting coordinates in a given list of sections to 0.')
    parser.add_argument('--folder', type=str, default='.', help='the working folder containing input and output files')
    parser.add_argument('--input', type=str, help='the CGNS file to be filtered')
    parser.add_argument('--sections', type=str, nargs='+', default=['Symmetry'],
        help='the given list of sections to be fixed')
    parser.add_argument('--component', choices=['X', 'Y', 'Z'], default='Y',
        help='the coordinate component to be fixed')
    args = parser.parse_args()
    print(args)

    input = f'{args.folder}/{args.input}'
    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(input)
    _, coord_x, coord_y, coord_z = pycgns_wrapper.readPoints(zone, zone_size)

    coord = select_component(coord_x, coord_y, coord_z, args)
    i_node_list = select_points(zone, args)

    print(f'On selected sections, n_node = {len(i_node_list)}')

    print(f'before flattening: bound = [{np.min(coord[i_node_list])}, {np.max(coord[i_node_list])}]')

    coord[i_node_list] = 0.0

    print(f'after flattening: bound = [{np.min(coord[i_node_list])}, {np.max(coord[i_node_list])}]')

    output = f'{input[:-5]}_flattened.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
