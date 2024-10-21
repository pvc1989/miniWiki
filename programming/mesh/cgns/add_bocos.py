import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper

import numpy as np
from scipy.spatial import KDTree
import argparse
import sys


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Add surface elements and BCs to a volume mesh.')
    parser.add_argument('--volume', type=str, help='the input volume mesh file')
    parser.add_argument('--output', type=str, help='the output mesh with surface elements and BCs')
    parser.add_argument('--bctypes', type=str, nargs='+', help='list of BC types')
    parser.add_argument('--index_maps', type=str, nargs='+', help='list of volume-to-surface index map files')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    print(args)

    # load the volume mesh
    volume_cgns, _, _ = cgm.load(args.volume)
    volume_zone = wrapper.getUniqueChildByType(
        wrapper.getUniqueChildByType(volume_cgns, 'CGNSBase_t'), 'Zone_t')
    volume_zone_size = wrapper.getNodeData(volume_zone)
    n_volume_node = volume_zone_size[0][0]
    n_volume_cell = volume_zone_size[0][1]
    print(f'in volume mesh: n_node = {n_volume_node}, n_cell = {n_volume_cell}')

    # load volume mesh connectivity
    volume_section = wrapper.getUniqueChildByType(volume_zone, 'Elements_t')
    element_type = wrapper.getNodeData(volume_section)
    assert element_type[0] == 17  # HEXA_8
    volume_connectivity = wrapper.getNodeData(
        wrapper.getUniqueChildByName(volume_section, 'ElementConnectivity'))
    assert len(volume_connectivity) == n_volume_cell * 8

    # get the index set of boundary points
    n_boco = len(args.bctypes)
    assert n_boco == len(args.index_maps)
    boco_points = set()
    for i_boco in range(n_boco):
        vol_to_surf = np.loadtxt(args.index_maps[i_boco], dtype=int)
        for i_vol in range(len(vol_to_surf)):
            if vol_to_surf[i_vol] >= 0:
                boco_points.add(i_vol)
    print(f'n_point_on_boundary = {len(boco_points)}')

    # get the index set of boundary cells
    boco_values = np.zeros((n_volume_cell,), dtype=int)
    boco_cells = set()
    for i_cell in range(n_volume_cell):
        on_boundary = False
        first_global = i_cell * 8
        for i_node_local in range(8):
            i_node_global = i_node_local + first_global
            if volume_connectivity[i_node_global] - 1 in boco_points:
                boco_values[i_cell] = 1
                boco_cells.add(i_cell)
                on_boundary = True
                break
        if args.verbose:
            print(f'[{i_cell / n_volume_cell:.2f}] Is cell[{i_cell}] on some boundary? {on_boundary}')
    print(f'n_cell_on_boundary = {len(boco_cells)}')

    bc_in_cell_data = cgl.newFlowSolution(volume_zone, 'FlowSolution', 'CellCenter')
    cgl.newDataArray(bc_in_cell_data, 'BCType', boco_values)

    output = args.output
    if output is None:
        output = f'BCadded_{args.volume}'
    print('write to ', output)
    cgm.save(output, volume_cgns)
