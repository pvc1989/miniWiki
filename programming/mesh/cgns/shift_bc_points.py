import CGNS.MAP as cgm
import wrapper

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
    parser.add_argument('--surface', type=str, help='the CGNS file of the shifted surface mesh')
    parser.add_argument('--index_map', type=str, help='the NPY file containing the volume-to-surface node index map')
    parser.add_argument('--output', type=str, help='the CGNS file of the shifted volume mesh')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # load the volume mesh to be shifted
    volume_cgns, volume_zone, volume_zone_size = wrapper.getUniqueZone(args.volume)
    volume_xyz, volume_x, volume_y, volume_z = wrapper.readPoints(volume_zone, volume_zone_size)
    volume_point_size = volume_zone_size[0][0]

    # load the shifted surface mesh
    _, surface_zone, surface_zone_size = wrapper.getUniqueZone(args.surface)
    surface_xyz, surface_x, surface_y, surface_z = wrapper.readPoints(surface_zone, surface_zone_size)

    # load the volume-to-surface node index map
    volume_to_surface = np.load(args.index_map)

    # shift volume points on surface
    for i_volume in range(volume_point_size):
        j_surface = volume_to_surface[i_volume]
        if j_surface < 0:
            continue
        d_ij = np.linalg.norm(volume_xyz[i_volume] - surface_xyz[j_surface])
        print(f'[{i_volume / volume_point_size:.2f}] Shifting point[{i_volume}] by {d_ij:.2e}')
        volume_x[i_volume] = surface_x[j_surface]
        volume_y[i_volume] = surface_y[j_surface]
        volume_z[i_volume] = surface_z[j_surface]

    # write the shifted mesh
    output = args.output
    if output is None:
        output = f'shifted_{args.volume}'
    if args.verbose:
        print('writing to', output)
    cgm.save(output, volume_cgns)
