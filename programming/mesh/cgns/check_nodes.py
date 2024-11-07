import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl

import sys
import argparse
import wrapper

import numpy as np
from scipy.spatial import KDTree


def buildPointDict(point_x: np.ndarray, point_y: np.ndarray, point_z: np.ndarray) -> dict:
    n_point = len(point_x)
    point_dict = dict()
    for i in range(n_point):
        point_dict[(point_x[i], point_y[i], point_z[i])] = i
    return point_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Check whether all surface nodes are also volume nodes.')
    parser.add_argument('--volume', type=str, help='the CGNS file of volume')
    parser.add_argument('--surface', type=str, help='the CGNS file of surface')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    volume_cgns, volume_zone, volume_zone_size = wrapper.getUniqueZone(args.volume)
    volume_arr, volume_x, volume_y, volume_z = wrapper.readPoints(volume_zone, volume_zone_size)
    volume_dict = buildPointDict(volume_x, volume_y, volume_z)
    volume_tree = KDTree(volume_arr)
    surface_cgns, surface_zone, surface_zone_size = wrapper.getUniqueZone(args.surface)
    _, surface_x, surface_y, surface_z = wrapper.readPoints(surface_zone, surface_zone_size)
    surface_dict = buildPointDict(surface_x, surface_y, surface_z)

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
