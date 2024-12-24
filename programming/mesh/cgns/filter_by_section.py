import argparse
import os
import sys

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnsutils as cgu
import pycgns_wrapper
from pycgns_wrapper import X, Y, Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Filter a CGNS file by a given list of sections.')
    parser.add_argument('--input', type=str, help='the CGNS file to be filtered')
    parser.add_argument('--output', type=str, help='the output folder')
    parser.add_argument('--sections', type=str, nargs='+', help='the given list of sections')
    args = parser.parse_args()
    print(args)

    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.input)
    xyz_old, _, _, _ = pycgns_wrapper.readPoints(zone, zone_size)
    print(f'before filtering: n_node = {zone_size[0][0]}, n_cell = {zone_size[0][1]}')

    # remove sections not in args.sections:
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    to_be_removed = []
    n_cell_new = 0
    for section in sections:
        name = pycgns_wrapper.getNodeName(section)
        if name not in args.sections:
            to_be_removed.append(name)
            continue
        # update erange
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByName(section, 'ElementRange'))
        # print(f'old erange of Elements_t({name}) = {erange}')
        first = n_cell_new + 1
        last = erange[1] - erange[0] + first
        erange[0] = first
        erange[1] = last
        n_cell_new = last
        print(f'new erange of Elements_t({name}) = {erange}')
    for name in to_be_removed:
        print(f'remove Elements_t({name})')
        cgu.removeChildByName(zone, name)

    # filter points and update connectivities
    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    old_to_new = -np.ones(len(xyz_old) + 1, dtype=int)
    new_to_old = []  # 0-based
    for section in sections:
        name = pycgns_wrapper.getNodeName(section)
        connectivity = pycgns_wrapper.getNodeData(
            pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        for i in range(len(connectivity)):
            i_old = connectivity[i]
            if old_to_new[i_old] == -1:
                new_to_old.append(i_old - 1)
                old_to_new[i_old] = len(new_to_old)
            connectivity[i] = old_to_new[i_old]
    xyz_new = xyz_old[new_to_old]
    n_node_new = len(xyz_new)
    assert n_node_new == len(new_to_old)
    print(f'after filtering: n_node = {n_node_new}, n_cell = {n_cell_new}')
    zone_size[0][0] = n_node_new
    zone_size[0][1] = n_cell_new
    pycgns_wrapper.setDimensions(pycgns_wrapper.getUniqueChildByType(cgns, 'CGNSBase_t'), 2, 3)

    # update DataArray_t's in the unique GridCoordinates_t
    coords = pycgns_wrapper.getUniqueChildByName(zone, 'GridCoordinates')
    pycgns_wrapper.getUniqueChildByName(coords, 'CoordinateX')[1] = xyz_new[:, X]
    pycgns_wrapper.getUniqueChildByName(coords, 'CoordinateY')[1] = xyz_new[:, Y]
    pycgns_wrapper.getUniqueChildByName(coords, 'CoordinateZ')[1] = xyz_new[:, Z]

    # remove other children of the unique Zont_t:
    to_be_removed = []
    for child in zone[2]:
        node_type = pycgns_wrapper.getNodeLabel(child)
        node_name = pycgns_wrapper.getNodeName(child)
        if node_type not in ('ZoneType_t', 'GridCoordinates_t', 'Elements_t'):
            to_be_removed.append((node_type, node_name))
    for node_type, node_name in to_be_removed:
        print(f'remove Zone_t/{node_type}({node_name})')
        cgu.removeChildByName(zone, node_name)

    input_folder = pycgns_wrapper.folder(args.input)
    output_folder = f'{input_folder}/{args.output}'
    os.makedirs(output_folder, exist_ok=True)

    output = f'{output_folder}/unique_points.npy'
    print(f'writing to {output} ...')
    np.save(output, xyz_new)

    output = f'{output_folder}/merged.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
