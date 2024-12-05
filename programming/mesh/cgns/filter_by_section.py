import argparse
import os
import sys

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
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
    n_node = len(xyz_old)
    print(f'before filtering: n_node = {n_node}')

    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    i_node_set = set()
    for section in sections:
        name = pycgns_wrapper.getNodeName(section)
        if name not in args.sections:
            continue
        connectivity = pycgns_wrapper.getNodeData(
            pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        for i_node in connectivity:
            i_node_set.add(i_node - 1)

    xyz_new = xyz_old[list(i_node_set)]
    n_node = len(xyz_new)
    print(f'after filtering: n_node = {n_node}')

    input_folder = pycgns_wrapper.folder(args.input)
    output_folder = f'{input_folder}/{args.output}'
    os.makedirs(output_folder, exist_ok=True)

    output = f'{output_folder}/unique_points.npy'
    print(f'writing to {output} ...')
    np.save(output, xyz_new)

    # TODO(PVC): writing the connectivities as well as the coordinates
    cgns = cgl.newCGNSTree(3.4)
    base = cgl.newCGNSBase(cgns, 'FilteredBase', 3, 3)
    zone = cgl.newZone(base, 'FilteredZone',
        zsize=np.array([[n_node, 0, 0]]), ztype='Unstructured')
    new_coords = cgl.newGridCoordinates(zone, 'GridCoordinates')
    cgl.newDataArray(new_coords, 'CoordinateX', xyz_new[:, X])
    cgl.newDataArray(new_coords, 'CoordinateY', xyz_new[:, Y])
    cgl.newDataArray(new_coords, 'CoordinateZ', xyz_new[:, Z])
    output = f'{output_folder}/merged.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
