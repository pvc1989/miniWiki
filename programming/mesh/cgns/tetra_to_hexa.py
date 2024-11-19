import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial import KDTree
import sys
import argparse
from timeit import default_timer as timer
from concurrent.futures import ProcessPoolExecutor, wait

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper


X, Y, Z = 0, 1, 2

A, B, C, D = 0, 1, 2, 3
AB, AC, AD, BC, BD, CD = 4, 5, 6, 7, 8, 9
ABC, ABD, ACD, BCD = 10, 11, 12, 13
ABCD = 14


def get_area_and_normal(arrow_ab: np.ndarray, arrow_ac: np.ndarray) -> tuple[float, np.ndarray]:
    normal = np.cross(arrow_ab, arrow_ac)
    norm = np.linalg.norm(normal)
    return norm / 2, normal / norm


def get_foot_on_face(point_q: np.ndarray, point_a: np.ndarray, normal: np.ndarray) -> np.ndarray:
    arrow_aq = point_q - point_a
    arrow_hq = np.dot(arrow_aq, normal) * normal
    arrow_ah = arrow_aq - arrow_hq
    return point_a + arrow_ah


def get_foot_on_edge(point_q: np.ndarray, point_a: np.ndarray, arrow_ab: np.ndarray) -> np.ndarray:
    arrow_aq = point_q - point_a
    ratio_ah = np.dot(arrow_ab, arrow_aq) / np.dot(arrow_ab, arrow_ab)
    arrow_ah = ratio_ah * arrow_ab
    return point_a + arrow_ah

# add_hexa(i_tetra, xyz_tetra[index_tetra], xyz_hexa, conn_hexa)
def add_hexa(i_tetra: int, n_point_old: int, xyz_tetra_local: np.ndarray,
        xyz_hexa: np.ndarray, conn_hexa: np.ndarray):
    point_a = xyz_tetra_local[A]
    point_b = xyz_tetra_local[B]
    point_c = xyz_tetra_local[C]
    point_d = xyz_tetra_local[D]
    arrow_ab = point_b - point_a
    arrow_ac = point_c - point_a
    arrow_ad = point_d - point_a
    arrow_bc = point_c - point_b
    arrow_bd = point_d - point_b
    arrow_cd = point_d - point_c
    v_abcd = np.abs(np.dot(arrow_ab, np.cross(arrow_ac, arrow_ad))) / 6.0
    s_abc, n_abc = get_area_and_normal(arrow_ab, arrow_ac)
    s_abd, n_abd = get_area_and_normal(arrow_ab, arrow_ad)
    s_acd, n_acd = get_area_and_normal(arrow_ac, arrow_ad)
    s_bcd, n_bcd = get_area_and_normal(arrow_bc, arrow_bd)
    s_abcd = s_abc + s_abd + s_acd + s_bcd
    radius = 3 * v_abcd / (s_abcd)
    center = (s_abc * point_d + s_abd * point_c + s_acd * point_b + s_bcd * point_a) / s_abcd
    # add new points
    xyz_offset = i_tetra * 15
    # copy the corners
    xyz_hexa[xyz_offset + A] = point_a
    xyz_hexa[xyz_offset + B] = point_b
    xyz_hexa[xyz_offset + C] = point_c
    xyz_hexa[xyz_offset + D] = point_d
    # add the foot on edges
    xyz_hexa[xyz_offset + AB] = get_foot_on_edge(center, point_a, arrow_ab)
    xyz_hexa[xyz_offset + AC] = get_foot_on_edge(center, point_a, arrow_ac)
    xyz_hexa[xyz_offset + AD] = get_foot_on_edge(center, point_a, arrow_ad)
    xyz_hexa[xyz_offset + BC] = get_foot_on_edge(center, point_b, arrow_bc)
    xyz_hexa[xyz_offset + BD] = get_foot_on_edge(center, point_b, arrow_bd)
    xyz_hexa[xyz_offset + CD] = get_foot_on_edge(center, point_c, arrow_cd)
    # add the foot on faces
    xyz_hexa[xyz_offset + ABC] = get_foot_on_face(center, point_a, n_abc)
    xyz_hexa[xyz_offset + ABD] = get_foot_on_face(center, point_a, n_abd)
    xyz_hexa[xyz_offset + ACD] = get_foot_on_face(center, point_a, n_acd)
    xyz_hexa[xyz_offset + BCD] = get_foot_on_face(center, point_b, n_bcd)
    # add the center
    xyz_hexa[xyz_offset + ABCD] = center
    # add new connectivity
    xyz_offset += n_point_old
    conn_offset = i_tetra * 4 * 8
    conn_hexa[conn_offset : conn_offset + 8] = np.array([
        AC, A, AB, ABC, ACD, AD, ABD, ABCD ]) + xyz_offset
    conn_offset += 8
    conn_hexa[conn_offset : conn_offset + 8] = np.array([
        AB, B, BC, ABC, ABD, BD, BCD, ABCD ]) + xyz_offset
    conn_offset += 8
    conn_hexa[conn_offset : conn_offset + 8] = np.array([
        BC, C, AC, ABC, BCD, CD, ACD, ABCD ]) + xyz_offset
    conn_offset += 8
    conn_hexa[conn_offset : conn_offset + 8] = np.array([
        CD, D, AD, ACD, BCD, BD, ABD, ABCD ]) + xyz_offset


def tetra_to_hexa(n_point_old: int, xyz_tetra: np.ndarray, conn_tetra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_tetra = len(conn_tetra) // 4
    # each tetra is converted to 4 hexa
    n_hexa = n_tetra * 4
    conn_hexa = -np.ones((n_hexa * 8,), dtype=int)
    # each tetra has 4(corner) + 6(edge) + 4(face) + 1(center) points
    xyz_hexa = np.ndarray((n_tetra * 15, 3))
    for i_tetra in range(n_tetra):
        first_tetra = i_tetra * 4
        index_tetra = conn_tetra[first_tetra : first_tetra + 4] - 1
        add_hexa(i_tetra, n_point_old, xyz_tetra[index_tetra], xyz_hexa, conn_hexa)
    assert (0 <= conn_hexa).all(), conn_hexa[-64:]
    return xyz_hexa, conn_hexa + 1


def merge_xyz_list(xyz_list, n_node) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_new = np.ndarray((n_node,))
    y_new = np.ndarray((n_node,))
    z_new = np.ndarray((n_node,))
    first = 0
    for xyz in xyz_list:
        last = first + len(xyz)
        x_new[first : last] = np.array(xyz[:, X])
        y_new[first : last] = np.array(xyz[:, Y])
        z_new[first : last] = np.array(xyz[:, Z])
        first = last
    assert first == n_node
    return x_new, y_new, z_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Split each tetrahedron to four hexahedra.')
    parser.add_argument('--input', type=str, help='the mesh of tetrahedra')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    cgns, zone, zone_size = wrapper.getUniqueZone(args.input)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    # print(zone)
    print(f'before splitting, n_node = {n_node}, n_cell = {n_cell}')
    xyz, x, y, z = wrapper.readPoints(zone, zone_size)

    sections = wrapper.getChildrenByType(zone, 'Elements_t')
    n_cell = 0
    i_cell_next = 1
    xyz_list = [xyz]
    for section in sections:
        i_type = wrapper.getNodeData(section)[0]
        conn_node = wrapper.getUniqueChildByName(section, 'ElementConnectivity')
        conn_old = wrapper.getNodeData(conn_node)
        erange = wrapper.getNodeData(wrapper.getUniqueChildByType(section, 'IndexRange_t'))
        if cgk.ElementType_l[i_type] == 'TETRA_4':
            xyz_new, conn_new = tetra_to_hexa(n_node, xyz, conn_old)
            xyz_list.append(xyz_new)
            n_node += len(xyz_new)
            assert (1 <= conn_new).all() and (conn_new <= n_node).all(), (section, n_node, np.min(conn_new), np.max(conn_new))
            n_cell_in_curr_section = len(conn_new) // 8
            n_cell += n_cell_in_curr_section
            # update the data in the Elements_t node
            # print(section)
            wrapper.getNodeData(section)[0] = cgk.ElementType_l.index('HEXA_8')
            conn_node[1] = conn_new
            # print(section)
        elif cgk.ElementType_l[i_type] == 'TRI_3':
            n_cell_in_curr_section = erange[1] - erange[0] + 1
        else:
            assert False, (i_type, cgk.ElementType_l[i_type])
        print(wrapper.getNodeName(section), erange)
        erange[0] = i_cell_next
        i_cell_next += n_cell_in_curr_section
        erange[1] = i_cell_next - 1
        print(wrapper.getNodeName(section), erange)

    x_new, y_new, z_new = merge_xyz_list(xyz_list, n_node)
    cgu.removeChildByName(zone, 'GridCoordinates')
    new_coords = cgl.newGridCoordinates(zone, 'GridCoordinates')
    cgl.newDataArray(new_coords, 'CoordinateX', x_new)
    cgl.newDataArray(new_coords, 'CoordinateY', y_new)
    cgl.newDataArray(new_coords, 'CoordinateZ', z_new)
    zone_size[0][0] = n_node
    zone_size[0][1] = n_cell
    # print(zone)
    print(f'after splitting, n_node = {n_node}, n_cell = {n_cell}')

    cgm.save(f'{args.input[:-5]}_splitted.cgns', cgns)
    print(args)
