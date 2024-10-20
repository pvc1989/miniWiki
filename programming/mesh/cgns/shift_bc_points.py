import CGNS.MAP as cgm
import wrapper
from check_nodes import getUniqueZone, readPoints

import sys
import numpy as np
from scipy.spatial import KDTree
import argparse


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Shift the points in a given volume mesh by surface meshes.')
    parser.add_argument('--volume', type=str, help='the CGNS file of the volume mesh')
    parser.add_argument('--surface_old', type=str, help='the CGNS file of the surface mesh extracted from the volume mesh')
    parser.add_argument('--surface_new', type=str, help='the CGNS file of the shifted surface mesh, which the topology unchanged')
    parser.add_argument('--output', type=str, help='the CGNS file of the shifted volume mesh')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # load the volume mesh to be shifted
    vol_cgns, vol_zone, vol_zone_size = getUniqueZone(args.volume)
    _, vol_xyz, vol_x, vol_y, vol_z = readPoints(vol_zone, vol_zone_size)
    vol_point_size = vol_zone_size[0][0]

    # load the unshifted surface mesh
    _, surface_old_zone, surface_old_zone_size = getUniqueZone(args.surface_old)
    _, surface_old_point_arr, _, _, _ = readPoints(surface_old_zone, surface_old_zone_size)
    surface_old_kdtree = KDTree(surface_old_point_arr)

    # load the shifted surface mesh
    _, surface_new_zone, surface_new_zone_size = getUniqueZone(args.surface_new)
    _, _, surface_new_x, surface_new_y, surface_new_z = readPoints(surface_new_zone, surface_new_zone_size)

    # shift volume points on surface
    for i_volume in range(vol_point_size):
        progress = f'[{i_volume} / {vol_point_size}]'
        d_ij, j_surface = surface_old_kdtree.query(vol_xyz[i_volume])
        if d_ij < 1e-11:
            print(progress, 'Shifting point', i_volume, d_ij)
            vol_x[i_volume] = surface_new_x[j_surface]
            vol_y[i_volume] = surface_new_y[j_surface]
            vol_z[i_volume] = surface_new_z[j_surface]
        else:
            print(progress, 'Skipping point', i_volume, d_ij)

    # write the shifted mesh
    output = args.output
    if output is None:
        output = f'shifted_{args.volume}'
    if args.verbose:
        print('write to ', output)
    cgm.save(output, vol_cgns)
