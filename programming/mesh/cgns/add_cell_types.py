import argparse
import sys

import numpy as np
import pycgns_wrapper
import CGNS.MAP as cgm
import CGNS.PAT.cgnskeywords as cgk
import CGNS.PAT.cgnslib as cgl


def is_3d(type_str: str):
    return ('TETRA' in type_str) or ('PYRA' in type_str) or ('PENTA' in type_str) or ('HEXA' in type_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Add the type of each cell as a field of cell data.')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    args = parser.parse_args()

    # read the cgns file
    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.mesh)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    print(f'in volume mesh: n_node = {n_node}, n_cell = {n_cell}')

    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    offset = 1e100
    for section in sections:
        type_val = pycgns_wrapper.getNodeData(section)[0]
        type_str = cgk.ElementType_l[type_val]
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
        if is_3d(type_str):
            offset = min(offset, erange[0] - 1)
        print(type_val, type_str, erange, offset)

    print(f'offset = {offset}')

    type_vals = -np.ones((n_cell,), dtype=np.int32)
    for section in sections:
        type_val = pycgns_wrapper.getNodeData(section)[0]
        type_str = cgk.ElementType_l[type_val]
        if not is_3d(type_str):
            continue
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
        for i_cell in range(erange[0] - 1, erange[1]):
            type_vals[i_cell - offset] = type_val

    cell_data = pycgns_wrapper.getSolutionByLocation(zone, 'CellCenter', 'CellData')
    cgl.newDataArray(cell_data, 'CellType', type_vals)

    # write to the original file
    output = args.mesh[:-5] + '_with_types.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
