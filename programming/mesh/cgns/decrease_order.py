import argparse
import sys

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnskeywords as cgk
import CGNS.PAT.cgnsutils as cgu
import pycgns_wrapper


A, B, C, D = 0, 1, 2, 3
AB, AC, AD, BC, BD, CD = 4, 5, 6, 7, 8, 9
ABC, ABD, ACD, BCD = 10, 11, 12, 13
ABCD = 14

A2, B2, C2 = 0, 1, 2
AB2, AC2, BC2 = 3, 4, 5
ABC2 = 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Shift mid-edge points to make low-order tetrahedra.')
    parser.add_argument('--input', type=str, help='the CGNS file of the quadratic tetrahedral mesh')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.input)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    # print(zone)
    print(f'before splitting, n_node = {n_node}, n_cell = {n_cell}')
    xyz, x, y, z = pycgns_wrapper.readPoints(zone, zone_size)

    sections = pycgns_wrapper.getChildrenByType(zone, 'Elements_t')
    n_face = 0  # faces are also cells, but not counted in zone_size[0][1]
    n_cell = 0
    i_cell_next = 1
    for section in sections:
        i_type = pycgns_wrapper.getNodeData(section)[0]
        conn_node = pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity')
        conn_data = pycgns_wrapper.getNodeData(conn_node)
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
        erange_old = tuple(erange)
        if cgk.ElementType_l[i_type] in ('TETRA_10',):
            first = 0
            last = 10
            while last <= len(conn_data):
                i_global = np.array(conn_data[first : last]) - 1
                if i_cell_next % 1000 == 0:
                    print(i_cell_next)
                point_a = xyz[i_global[A]]
                point_b = xyz[i_global[B]]
                point_c = xyz[i_global[C]]
                point_d = xyz[i_global[D]]
                xyz[i_global[4]] = (point_a + point_b) / 2
                xyz[i_global[5]] = (point_b + point_c) / 2
                xyz[i_global[6]] = (point_a + point_c) / 2
                xyz[i_global[7]] = (point_a + point_d) / 2
                xyz[i_global[8]] = (point_b + point_d) / 2
                xyz[i_global[9]] = (point_c + point_d) / 2
                first += 10
                last += 10
                i_cell_next += 1
        elif cgk.ElementType_l[i_type] in ('TRI_6',):
            # xyz_new, conn_new = tri_to_quad(linear, n_node, xyz, conn_old, i_type)
            # xyz_list.append(xyz_new)
            # n_node += len(xyz_new)
            # assert (1 <= conn_new).all() and (conn_new <= n_node).all(), (section, n_node, np.min(conn_new), np.max(conn_new))
            # n_cell_in_curr_section = len(conn_new) // 4
            # n_cell += n_cell_in_curr_section
            # n_face += n_cell_in_curr_section
            # # update the data in the Elements_t node
            # pycgns_wrapper.getNodeData(section)[0] = cgk.ElementType_l.index('QUAD_4')
            # conn_node[1] = conn_new
            pass
        else:
            assert False, (i_type, cgk.ElementType_l[i_type])
        # erange[0] = i_cell_next
        # i_cell_next += n_cell_in_curr_section
        # erange[1] = i_cell_next - 1
        # erange_new = tuple(erange)
        # erange_old_to_new[erange_old] = erange_new

    # update the arrays of coordinates in the file
    x[:] = xyz[:, 0]
    y[:] = xyz[:, 1]
    z[:] = xyz[:, 2]

    output = f'{pycgns_wrapper.folder(args.input)}/linear.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
    print(args)
