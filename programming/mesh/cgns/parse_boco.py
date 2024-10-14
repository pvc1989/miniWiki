import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import numpy as np
import sys
import argparse


def getChildrenByType(node, type):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if cgu.checkNodeType(child, type):
            children.append(child)
    return children


def getUniqueChildByType(node, type):
    children = getChildrenByType(node, type)
    if 1 == len(children):
        return children[0]
    elif 0 == len(children):
        return None
    else:
        assert False


def getNodeName(node) -> str:
    return node[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 parse_boco.py',
        description = 'Parse the BCs in https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/Committee_Grids/C-HLCRM_Str1to1_GridPro/FullGap/CGNS/')
    parser.add_argument('--input', type=str, help='the CGNS file for parsing')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    tree, links, paths = cgm.load(args.input)
    base = getUniqueChildByType(tree, 'CGNSBase_t')

    n_boco = 0
    wall = []
    symmetry = []
    farfield = []

    zones = getChildrenByType(base, 'Zone_t')
    for zone in zones:
        zone_name = getNodeName(zone)
        zone_bc = getUniqueChildByType(zone, 'ZoneBC_t')
        if zone_bc is None:
            continue
        if args.verbose:
            print(zone_name)
        bocos = zone_bc[2]
        for boco in bocos:
            n_boco += 1
            boco_name = boco[0]
            boco_type = boco[1]
            if args.verbose:
                print('  ', boco_name, boco_type)
            if boco_type[2] == b'W':
                wall.append(boco_name)
            elif boco_type[2] == b'S':
                symmetry.append(boco_name)
            elif boco_type[2] == b'F':
                farfield.append(boco_name)
            else:
                assert False
    assert n_boco == len(wall) + len(symmetry) + len(farfield)
    print('n_boco =', n_boco)
    print('wall:\n  ', wall)
    print('symmetry:\n  ', symmetry)
    print('farfield:\n  ', farfield)
