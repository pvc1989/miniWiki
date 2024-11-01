import numpy as np
import argparse
import sys
from scipy.spatial import KDTree


X, Y, Z = 0, 1, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description = 'Get the distance between each point and its k-th nearest neighbor.')
    parser.add_argument('--input', type=str, help='the NPY file containing the coordinates of each point')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    points = np.load(args.input)
    assert points.shape[1] == 3
    kdtree = KDTree(points)
    n_point = len(points)
    radius = np.ndarray((n_point,))
    for i_point in range(n_point):
        point_i = points[i_point]
        distances, neighbors = kdtree.query(point_i, k=2)
        print(i_point, neighbors[1], distances[1])
        radius[i_point] = distances[1]
    print('r_min =', np.min(radius))
    print('r_max =', np.max(radius))
