import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl

import sys
import argparse
import wrapper

import numpy as np
from scipy.spatial import KDTree


def getUniqueZone(filename):
    tree, _, _ = cgm.load(filename)
    base = wrapper.getUniqueChildByType(tree, 'CGNSBase_t')
    zone = wrapper.getUniqueChildByType(base, 'Zone_t')
    zone_size = wrapper.getNodeData(zone)
    assert zone_size.shape == (1, 3)
    return tree, zone, zone_size


def readPoints(zone, zone_size):
    n_node = zone_size[0][0]
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(zone, 'GridCoordinates_t'),
        'DataArray_t')
    X, Y, Z = 0, 1, 2
    coords_x, coords_y, coords_z = coords[X][1], coords[Y][1], coords[Z][1]
    assert (n_node,) == coords_x.shape == coords_y.shape == coords_z.shape
    point_dict = dict()
    point_arr = np.ndarray((n_node, 3))
    for i in range(n_node):
        point_dict[(coords_x[i], coords_y[i], coords_z[i])] = i
        point_arr[i] = coords_x[i], coords_y[i], coords_z[i]
    return point_dict, point_arr, coords_x, coords_y, coords_z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Check the consistency between volume and surface grids.')
    parser.add_argument('--volume', type=str, help='the CGNS file of volume')
    parser.add_argument('--surface', type=str, help='the CGNS file of surface')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    volume_cgns, volume_zone, volume_zone_size = getUniqueZone(args.volume)
    volume_dict, volume_arr, _, _, _ = readPoints(volume_zone, volume_zone_size)
    volume_tree = KDTree(volume_arr)
    surface_cgns, surface_zone, surface_zone_size = getUniqueZone(args.surface)
    surface_dict, _, _, _, _ = readPoints(surface_zone, surface_zone_size)

    i = 0
    for p_surface in surface_dict:
        if args.verbose:
            print('surface: ', i)
        i += 1
        if p_surface not in volume_dict:
            d_ij, j = volume_tree.query(p_surface)
            print(i, d_ij)
            if d_ij > 1e-10:
                print('Failed!')
                exit(-1)
    print('Passed!')
