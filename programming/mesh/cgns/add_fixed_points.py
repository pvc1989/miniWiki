import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial import KDTree
import sys
import argparse
from timeit import default_timer as timer
from concurrent.futures import ProcessPoolExecutor, wait

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import wrapper


X, Y, Z = 0, 1, 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Append fixed points after shifted points.')
    parser.add_argument('--before', type=str, help='the NPY file of boundary points before shifting')
    parser.add_argument('--after', type=str, help='the NPY file of boundary points after shifting')
    parser.add_argument('--fixed', type=str, nargs='+', help='the NPY file(s) of boundary points that are fixed')
    parser.add_argument('--output', type=str, help='the folder (without \'/\') for saving the output')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    points_before_shifting = np.load(args.before)
    kdtree_before_shifting = KDTree(points_before_shifting)

    list_of_points = []
    list_of_points.append(points_before_shifting)
    list_of_ranges = []
    list_of_ranges.append((0, len(points_before_shifting)))

    for fixed in args.fixed:
        rows = []
        points = np.load(fixed)
        n_point = len(points)
        for i_point in range(n_point):
            point_i = points[i_point]
            d_ij, j_point = kdtree_before_shifting.query(point_i)
            if d_ij < 1e-6:
                if args.verbose:
                    print(f'd({j_point} in {args.before},\n  {i_point} in {fixed}) = {d_ij:.2e}')
            else:
                rows.append(i_point)
        list_of_points.append(points[rows])
        first = list_of_ranges[-1][1]
        last = first + len(rows)
        list_of_ranges.append((first, last))
    print('list_of_ranges :', list_of_ranges)

    all_points_before_shifting = np.ndarray((list_of_ranges[-1][1], 3))
    for i in range(len(list_of_ranges)):
        first, last = list_of_ranges[i]
        all_points_before_shifting[first : last] = list_of_points[i]

    all_points_after_shifting = np.ndarray((list_of_ranges[-1][1], 3))
    list_of_points[0] = np.load(args.after)
    for i in range(len(list_of_ranges)):
        first, last = list_of_ranges[i]
        all_points_after_shifting[first : last] = list_of_points[i]

    output = f'{args.output}/old_points.npy'
    print(f'writing into "{output}" ...')
    np.save(output, all_points_before_shifting)
  
    output = f'{args.output}/new_points.npy'
    print(f'writing into "{output}" ...')
    np.save(output, all_points_after_shifting)

    print(args)
