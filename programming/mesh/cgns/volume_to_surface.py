import argparse
import sys

import numpy as np
from scipy.spatial import KDTree

import pycgns_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Build a volume_node_index-to-surface_node_index map.')
    parser.add_argument('--volume', type=str, help='the CGNS file of the volume mesh')
    parser.add_argument('--surface', type=str, help='the CGNS file of the surface mesh')
    parser.add_argument('--distance', type=float, default=1e-10,
        help='if the distance of two points is less than this value, then treated them as a single one')
    parser.add_argument('--output', type=str, help='the NPY file containing the map')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # load the volume mesh
    volume_cgns, volume_zone, volume_zone_size = pycgns_wrapper.getUniqueZone(args.volume)
    volume_xyz, _, _, _ = pycgns_wrapper.readPoints(volume_zone, volume_zone_size)
    volume_point_size = volume_zone_size[0][0]
    volume_to_surface = np.zeros((volume_point_size,), dtype=int) - 1

    # load the surface mesh
    _, surface_zone, surface_zone_size = pycgns_wrapper.getUniqueZone(args.surface)
    surface_point_arr, _, _, _ = pycgns_wrapper.readPoints(surface_zone, surface_zone_size)
    surface_kdtree = KDTree(surface_point_arr)

    # look for the nearest point on the surface
    for i_volume in range(volume_point_size):
        progress = f'[{i_volume} / {volume_point_size}]'
        d_ij, j_surface = surface_kdtree.query(volume_xyz[i_volume])
        if d_ij < args.distance:
            print(progress, f'volume[{i_volume}] to surface[{j_surface}] = {d_ij}')
            volume_to_surface[i_volume] = j_surface
        else:
            print(progress, f'volume[{i_volume}] is not on the given surface')

    # write the mesh
    output = args.output
    if output is None:
        output = f'({args.volume[:-5]})-to-({args.surface[:-5]}).npy'
    if args.verbose:
        print('writing to', output)
    np.save(output, volume_to_surface)
