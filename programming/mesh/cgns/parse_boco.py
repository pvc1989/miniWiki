import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import numpy as np
import sys
import argparse
import wrapper


def arr2str(arr) -> str:
    """Convert a string defined as an `np.ndarray` to a standard Python `str`.

    Some CGNS files, e.g. https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/Committee_Grids/C-HLCRM_Str1to1_GridPro/FullGap/CGNS/, use an `np.ndarray` of `bytes`s rather than a `BCType_t` object to store the type of a BC.
    """
    assert isinstance(arr, np.ndarray)
    chars = []
    for row in arr:
        assert isinstance(row, bytes)
        assert 1 == len(row)
        chars.append(row.decode())
    return "".join(chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 parse_boco.py',
        description = 'Parse the BCs in https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/Committee_Grids/C-HLCRM_Str1to1_GridPro/FullGap/CGNS/')
    parser.add_argument('--input', type=str, help='the CGNS file to be parsed')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    tree, links, paths = cgm.load(args.input)
    base = wrapper.getUniqueChildByType(tree, 'CGNSBase_t')

    n_boco = 0
    boco_type_to_names = dict()

    zones = wrapper.getChildrenByType(base, 'Zone_t')
    for zone in zones:
        zone_name = wrapper.getNodeName(zone)
        zone_bc = wrapper.getUniqueChildByType(zone, 'ZoneBC_t')
        if zone_bc is None:
            continue
        if args.verbose:
            print(f'"{zone_name}"')
        bocos = wrapper.getChildrenByType(zone_bc, 'BC_t')
        for boco in bocos:
            n_boco += 1
            boco_name = wrapper.getNodeName(boco)
            boco_type_arr = wrapper.getNodeData(boco)
            if args.verbose:
                print(f'  "{boco_name}" "{boco_type_arr}"')
            boco_type = arr2str(boco_type_arr)
            if boco_type not in boco_type_to_names:
                boco_type_to_names[boco_type] = list()
            assert isinstance(boco_type_to_names[boco_type], list)
            boco_type_to_names[boco_type].append(boco_name)
    n_boco_check = 0
    for boco_type, boco_names in boco_type_to_names.items():
        print(f'"{boco_type}"')
        for boco_name in boco_names:
            n_boco_check += 1
            print(f'  "{boco_name}"')
    assert n_boco == n_boco_check
    print('n_boco =', n_boco)
