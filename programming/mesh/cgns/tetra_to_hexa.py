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


def get_hexa_points_by_inscribed(point_a: np.ndarray, point_b: np.ndarray,
        point_c: np.ndarray, point_d: np.ndarray, xyz_hexa_with_offset: np.ndarray):
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
    # copy the corners
    xyz_hexa_with_offset[A] = point_a
    xyz_hexa_with_offset[B] = point_b
    xyz_hexa_with_offset[C] = point_c
    xyz_hexa_with_offset[D] = point_d
    # add the foot on edges
    xyz_hexa_with_offset[AB] = get_foot_on_edge(center, point_a, arrow_ab)
    xyz_hexa_with_offset[AC] = get_foot_on_edge(center, point_a, arrow_ac)
    xyz_hexa_with_offset[AD] = get_foot_on_edge(center, point_a, arrow_ad)
    xyz_hexa_with_offset[BC] = get_foot_on_edge(center, point_b, arrow_bc)
    xyz_hexa_with_offset[BD] = get_foot_on_edge(center, point_b, arrow_bd)
    xyz_hexa_with_offset[CD] = get_foot_on_edge(center, point_c, arrow_cd)
    # add the foot on faces
    xyz_hexa_with_offset[ABC] = get_foot_on_face(center, point_a, n_abc)
    xyz_hexa_with_offset[ABD] = get_foot_on_face(center, point_a, n_abd)
    xyz_hexa_with_offset[ACD] = get_foot_on_face(center, point_a, n_acd)
    xyz_hexa_with_offset[BCD] = get_foot_on_face(center, point_b, n_bcd)
    # add the center
    xyz_hexa_with_offset[ABCD] = center


def get_hexa_points_by_centroid(point_a: np.ndarray, point_b: np.ndarray,
        point_c: np.ndarray, point_d: np.ndarray, xyz_hexa_with_offset: np.ndarray):
    # copy the corners
    xyz_hexa_with_offset[A] = point_a
    xyz_hexa_with_offset[B] = point_b
    xyz_hexa_with_offset[C] = point_c
    xyz_hexa_with_offset[D] = point_d
    # add the centroids of edges
    xyz_hexa_with_offset[AB] = (point_a + point_b) / 2
    xyz_hexa_with_offset[AC] = (point_a + point_c) / 2
    xyz_hexa_with_offset[AD] = (point_a + point_d) / 2
    xyz_hexa_with_offset[BC] = (point_b + point_c) / 2
    xyz_hexa_with_offset[BD] = (point_b + point_d) / 2
    xyz_hexa_with_offset[CD] = (point_c + point_d) / 2
    # add the centroids of faces
    xyz_hexa_with_offset[ABC] = (point_a + point_b + point_c) / 3
    xyz_hexa_with_offset[ABD] = (point_a + point_b + point_d) / 3
    xyz_hexa_with_offset[ACD] = (point_a + point_c + point_d) / 3
    xyz_hexa_with_offset[BCD] = (point_b + point_c + point_d) / 3
    # add the centroid of cell
    xyz_hexa_with_offset[ABCD] = (point_d + point_c + point_b + point_a) / 4


def tri_6_shapes(lambda_a: float, lambda_b: float) -> np.ndarray:
    shapes = np.zeros(6)
    lambda_c = 1 - lambda_a - lambda_b
    twice_lambda_a = 2 * lambda_a
    twice_lambda_b = 2 * lambda_b
    twice_lambda_c = 2 * lambda_c
    shapes[A2] = lambda_a * (twice_lambda_a - 1)
    shapes[B2] = lambda_b * (twice_lambda_b - 1)
    shapes[C2] = lambda_c * (twice_lambda_c - 1)
    shapes[AB2] = twice_lambda_a * twice_lambda_b
    shapes[AC2] = twice_lambda_a * twice_lambda_c
    shapes[BC2] = twice_lambda_b * twice_lambda_c
    return shapes


def tetra10_to_midface_points(point_a, point_b, point_c, point_d,
        point_ab, point_ac, point_ad, point_bc, point_bd, point_cd):
    shapes = tri_6_shapes(1.0/3, 1.0/3)
    point_abc = shapes.dot([point_a, point_b, point_c, point_ab, point_ac, point_bc])
    point_abd = shapes.dot([point_a, point_b, point_d, point_ab, point_ad, point_bd])
    point_acd = shapes.dot([point_a, point_c, point_d, point_ac, point_ad, point_cd])
    point_bcd = shapes.dot([point_b, point_c, point_d, point_bc, point_bd, point_cd])
    return point_abc, point_abd, point_acd, point_bcd

def get_hexa_points_from_general_tetra(xyz_tetra_local: np.ndarray, tetra_npe: int,
    xyz_hexa_with_offset: np.ndarray):
    point_a = xyz_tetra_local[A]
    point_b = xyz_tetra_local[B]
    point_c = xyz_tetra_local[C]
    point_d = xyz_tetra_local[D]
    if tetra_npe == 10:
        point_ab = xyz_tetra_local[4]
        point_bc = xyz_tetra_local[5]
        point_ac = xyz_tetra_local[6]
        point_ad = xyz_tetra_local[7]
        point_bd = xyz_tetra_local[8]
        point_cd = xyz_tetra_local[9]
        point_abc, point_abd, point_acd, point_bcd = tetra10_to_midface_points(
            point_a, point_b, point_c, point_d,
            point_ab, point_ac, point_ad, point_bc, point_bd, point_cd)
    # copy the corners
    xyz_hexa_with_offset[A] = point_a
    xyz_hexa_with_offset[B] = point_b
    xyz_hexa_with_offset[C] = point_c
    xyz_hexa_with_offset[D] = point_d
    # add the centroids of edges
    xyz_hexa_with_offset[AB] = point_ab
    xyz_hexa_with_offset[AC] = point_ac
    xyz_hexa_with_offset[AD] = point_ad
    xyz_hexa_with_offset[BC] = point_bc
    xyz_hexa_with_offset[BD] = point_bd
    xyz_hexa_with_offset[CD] = point_cd
    # add the centroids of faces
    xyz_hexa_with_offset[ABC] = point_abc
    xyz_hexa_with_offset[ABD] = point_abd
    xyz_hexa_with_offset[ACD] = point_acd
    xyz_hexa_with_offset[BCD] = point_bcd
    # add the centroid of cell
    xyz_hexa_with_offset[ABCD] = (point_abc + point_abd + point_acd + point_bcd) / 4


def add_hexa(i_tetra: int, tetra_npe: int, n_point_old: int, xyz_tetra_local: np.ndarray,
        xyz_hexa: np.ndarray, conn_hexa: np.ndarray):
    xyz_offset = i_tetra * 15
    # add new points
    if tetra_npe == 4:
        point_a = xyz_tetra_local[A]
        point_b = xyz_tetra_local[B]
        point_c = xyz_tetra_local[C]
        point_d = xyz_tetra_local[D]
        get_hexa_points_by_centroid(point_a, point_b, point_c, point_d,
            xyz_hexa[xyz_offset : xyz_offset + 15])
    elif tetra_npe == 10:
        get_hexa_points_from_general_tetra(xyz_tetra_local, tetra_npe,
            xyz_hexa[xyz_offset : xyz_offset + 15])
    else:
        assert False
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


def tetra_to_hexa(n_point_old: int, xyz_tetra: np.ndarray, conn_tetra: np.ndarray, i_type: int) -> tuple[np.ndarray, np.ndarray]:
    tetra_npe = cgk.ElementTypeNPE_l[i_type]
    n_tetra = len(conn_tetra) // tetra_npe
    # each tetra is converted to 4 hexa
    n_hexa = n_tetra * 4
    conn_hexa = -np.ones((n_hexa * 8,), dtype=int)
    # each tetra has 4(corner) + 6(edge) + 4(face) + 1(center) points
    xyz_hexa = np.ndarray((n_tetra * 15, 3))
    for i_tetra in range(n_tetra):
        first_tetra = i_tetra * tetra_npe
        index_tetra = conn_tetra[first_tetra : first_tetra + tetra_npe] - 1
        add_hexa(i_tetra, tetra_npe, n_point_old, xyz_tetra[index_tetra],
            xyz_hexa, conn_hexa)
    assert (0 <= conn_hexa).all(), conn_hexa[-64:]
    return xyz_hexa, conn_hexa + 1


def get_quad_points_from_general_tri(xyz_tri_local: np.ndarray, tri_npe: int,
        xyz_hexa_with_offset: np.ndarray):
    point_a = xyz_tri_local[A2]
    point_b = xyz_tri_local[B2]
    point_c = xyz_tri_local[C2]
    if tri_npe == 3:
        point_ab = (point_a + point_b) / 2
        point_ac = (point_a + point_c) / 2
        point_bc = (point_b + point_c) / 2
        point_abc = (point_a + point_b + point_c) / 3
    elif tri_npe == 6:
        # In CGNS, (A, B, C, AB, BC, CA) = (0, 1, 2, 3, 4, 5)
        point_ab = xyz_tri_local[3]
        point_bc = xyz_tri_local[4]
        point_ac = xyz_tri_local[5]
        point_abc = tri_6_shapes(1.0/3, 1.0/3).dot(xyz_tri_local)
    else:
        assert False
    # copy the corners
    xyz_hexa_with_offset[A2] = point_a
    xyz_hexa_with_offset[B2] = point_b
    xyz_hexa_with_offset[C2] = point_c
    # add the centroids on edges
    xyz_hexa_with_offset[AB2] = point_ab
    xyz_hexa_with_offset[AC2] = point_ac
    xyz_hexa_with_offset[BC2] = point_bc
    # add the centroid on face
    xyz_hexa_with_offset[ABC2] = point_abc


def add_quad(i_tri: int, tri_npe: int, n_point_old: int, xyz_tri_local: np.ndarray,
        xyz_quad: np.ndarray, conn_quad: np.ndarray):
    # add new points
    xyz_offset = i_tri * 7
    get_quad_points_from_general_tri(xyz_tri_local, tri_npe,
        xyz_quad[xyz_offset : xyz_offset + 7])
    # add new connectivity
    xyz_offset += n_point_old
    conn_offset = i_tri * 3 * 4
    conn_quad[conn_offset : conn_offset + 4] = np.array([
        AC2, A2, AB2, ABC2 ]) + xyz_offset
    conn_offset += 4
    conn_quad[conn_offset : conn_offset + 4] = np.array([
        AB2, B2, BC2, ABC2 ]) + xyz_offset
    conn_offset += 4
    conn_quad[conn_offset : conn_offset + 4] = np.array([
        BC2, C2, AC2, ABC2 ]) + xyz_offset


def tri_to_quad(n_point_old: int, xyz_tri: np.ndarray, conn_tri: np.ndarray, i_type: int) -> tuple[np.ndarray, np.ndarray]:
    tri_npe = cgk.ElementTypeNPE_l[i_type]
    n_tri = len(conn_tri) // tri_npe
    # each tri is converted to 3 quad
    n_quad = n_tri * 3
    conn_quad = -np.ones((n_quad * 4,), dtype=int)
    # each tri has 3(corner) + 3(edge) + 1(face) points
    xyz_quad = np.ndarray((n_tri * 7, 3))
    for i_tri in range(n_tri):
        first_tri = i_tri * tri_npe
        index_tri = conn_tri[first_tri : first_tri + tri_npe] - 1
        add_quad(i_tri, tri_npe, n_point_old, xyz_tri[index_tri],
            xyz_quad, conn_quad)
    assert (0 <= conn_quad).all(), conn_quad[-12:]
    return xyz_quad, conn_quad + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Split each tetrahedron to four hexahedra.')
    parser.add_argument('--input', type=str, help='the CGNS file to be splitted')
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
    xyz_list = [xyz]
    erange_old_to_new = dict()
    for section in sections:
        i_type = pycgns_wrapper.getNodeData(section)[0]
        conn_node = pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity')
        conn_old = pycgns_wrapper.getNodeData(conn_node)
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(section, 'IndexRange_t'))
        erange_old = tuple(erange)
        if cgk.ElementType_l[i_type] in ('TETRA_4', 'TETRA_10'):
            xyz_new, conn_new = tetra_to_hexa(n_node, xyz, conn_old, i_type)
            xyz_list.append(xyz_new)
            n_node += len(xyz_new)
            assert (1 <= conn_new).all() and (conn_new <= n_node).all(), (section, n_node, np.min(conn_new), np.max(conn_new))
            n_cell_in_curr_section = len(conn_new) // 8
            n_cell += n_cell_in_curr_section
            # update the data in the Elements_t node
            pycgns_wrapper.getNodeData(section)[0] = cgk.ElementType_l.index('HEXA_8')
            conn_node[1] = conn_new
        elif cgk.ElementType_l[i_type] in ('TRI_3', 'TRI_6'):
            xyz_new, conn_new = tri_to_quad(n_node, xyz, conn_old, i_type)
            xyz_list.append(xyz_new)
            n_node += len(xyz_new)
            assert (1 <= conn_new).all() and (conn_new <= n_node).all(), (section, n_node, np.min(conn_new), np.max(conn_new))
            n_cell_in_curr_section = len(conn_new) // 4
            n_cell += n_cell_in_curr_section
            n_face += n_cell_in_curr_section
            # update the data in the Elements_t node
            pycgns_wrapper.getNodeData(section)[0] = cgk.ElementType_l.index('QUAD_4')
            conn_node[1] = conn_new
        else:
            assert False, (i_type, cgk.ElementType_l[i_type])
        erange[0] = i_cell_next
        i_cell_next += n_cell_in_curr_section
        erange[1] = i_cell_next - 1
        erange_new = tuple(erange)
        erange_old_to_new[erange_old] = erange_new
        if args.verbose:
            print('Elements_t', pycgns_wrapper.getNodeName(section), erange_old, erange_new)

    # update element ranges in ZoneBC_t
    zone_bc = pycgns_wrapper.getUniqueChildByType(zone, 'ZoneBC_t')
    bocos = pycgns_wrapper.getChildrenByType(zone_bc, 'BC_t')
    to_be_removed = []
    for boco in bocos:
        boco_name = pycgns_wrapper.getNodeName(boco)
        if 'V_' in boco_name:
            # Gmsh creates a 'V_'-prefixed BC_t for each physical volume, which is not used in our workflow.
            to_be_removed.append(boco_name)
        erange = pycgns_wrapper.getNodeData(pycgns_wrapper.getUniqueChildByType(boco, 'IndexRange_t'))
        erange_old = tuple(erange[0])
        erange_new = erange_old_to_new[erange_old]
        erange[0] = np.array(erange_new)
        if args.verbose:
            print('BC_t', pycgns_wrapper.getNodeName(boco), erange_old, erange_new)

    print(to_be_removed)
    print(zone_bc)
    for boco in to_be_removed:
        cgu.removeChildByName(zone_bc, boco)
    print(zone_bc)

    pycgns_wrapper.mergePointList(xyz_list, n_node, zone, zone_size)
    zone_size[0][1] = n_cell - n_face
    # print(zone)
    print(f'after splitting, n_node = {n_node}, n_cell = {n_cell}')

    output = f'{pycgns_wrapper.folder(args.input)}/splitted.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
    print(args)
