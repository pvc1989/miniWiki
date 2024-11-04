import numpy as np
import argparse
import sys
from scipy.spatial import KDTree

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import wrapper


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description = 'Get the distance between each point and its k-th nearest neighbor.')
    parser.add_argument('--input', type=str, help='the CGNS file containing the coordinates of each point')
    parser.add_argument('--k_neighbor', type=int, default=2, help='number of neighbors in the support')
    parser.add_argument('--output', type=str, help='the output mesh file')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    cgns, zone, zone_size = wrapper.getUniqueZone(args.input)
    points, _, _, _ = wrapper.readPoints(zone, zone_size)

    assert points.shape[1] == 3
    kdtree = KDTree(points)
    n_point = len(points)
    radius = np.ndarray((n_point,))
    k_neighbor = args.k_neighbor
    for i_point in range(n_point):
        point_i = points[i_point]
        distances, neighbors = kdtree.query(point_i, k_neighbor)
        if args.verbose:
            print(i_point, neighbors[-1], distances[-1])
        radius[i_point] = distances[-1]
    print('r_min =', np.min(radius))
    print('r_max =', np.max(radius))

    # write the radius as a field of point data
    wrapper.removeSolutionsByLocation(zone, 'Vertex')
    point_data = cgl.newFlowSolution(zone, 'FlowSolutionHelper', 'Vertex')
    cgl.newDataArray(point_data, 'SupportRadius', radius)

    # write the modified CGNSTree_t
    output = args.output
    if output is None:
        output = f'R(k={args.k_neighbor})_{args.input}'
    if args.verbose:
        print()
        print('writing to', output)
    cgm.save(output, cgns)
