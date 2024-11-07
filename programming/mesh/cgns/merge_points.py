import CGNS.MAP as cgm
import wrapper
import numpy as np
import argparse
import sys
from scipy.spatial import KDTree


X, Y, Z = 0, 1, 2


def multiple_to_unique(multiple_points: np.ndarray, radius: float, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    assert multiple_points.shape[1] == 3
    n_multiple = len(multiple_points)

    multiple_kdtree = KDTree(multiple_points)
    i_multiple_to_i_minimum = np.zeros((n_multiple,), dtype='int') - 1
    i_multiple_to_i_unique = np.zeros((n_multiple,), dtype='int') - 1
    i_minimum_to_i_unique = dict()
    i_unique = 0

    for i_multiple in range(n_multiple):
        point_i = multiple_points[i_multiple]
        neighbors = multiple_kdtree.query_ball_point(point_i, radius)
        i_minimum = np.min(neighbors)
        assert 0 <= i_minimum < n_multiple
        if verbose:
            print(f'{i_multiple} -> {i_minimum}, {neighbors}')
        i_multiple_to_i_minimum[i_multiple] = i_minimum
        if i_minimum not in i_minimum_to_i_unique:
            i_minimum_to_i_unique[i_minimum] = i_unique
            i_unique += 1
        i_multiple_to_i_unique[i_multiple] = i_minimum_to_i_unique[i_minimum]
    n_unique = len(i_minimum_to_i_unique)
    assert i_unique == n_unique
    assert (0 <= i_multiple_to_i_unique).all() and (i_multiple_to_i_unique < n_unique).all()

    unique_points = np.ndarray((n_unique, 3), dtype=multiple_points.dtype)
    for i_minimum, i_unique in i_minimum_to_i_unique.items():
        unique_points[i_unique] = multiple_points[i_minimum]

    for i_multiple in range(n_multiple):
        i_unique = i_multiple_to_i_unique[i_multiple]
        assert np.linalg.norm(multiple_points[i_multiple] - unique_points[i_unique]) < radius

    np.save('i_multiple_to_i_unique.npy', i_multiple_to_i_unique)
    np.save('unique_points.npy', unique_points)
    print('in multiple_to_unique:', n_unique, n_multiple)
    return unique_points, i_multiple_to_i_unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Merge geometrically duplicated points in a single unstructured `Zone_t`.')
    parser.add_argument('--input', type=str, help='the CGNS file to be merged')
    parser.add_argument('--output', type=str, help='the merged CGNS file')
    parser.add_argument('--radius', type=float, default=1e-8, help='radius of the ball within which two points are treated as duplicated')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # get the unique Zone_t
    cgns, zone, zone_size = wrapper.getUniqueZone(args.input)
    multiple_points, _, _, _ = wrapper.readPoints(zone, zone_size)
    assert zone_size[0][0] == len(multiple_points)
    unique_points, i_multiple_to_i_unique = multiple_to_unique(multiple_points, args.radius, args.verbose)
    n_unique = len(unique_points)

    # update zone_size and GridCoordinates_t
    zone_size[0][0] = n_unique
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(zone, 'GridCoordinates_t'), 'DataArray_t')
    if args.verbose:
        print('\ncoords before merging:\n', coords)
    coords[X][1] = np.array(unique_points[:, X])
    coords[Y][1] = np.array(unique_points[:, Y])
    coords[Z][1] = np.array(unique_points[:, Z])
    if args.verbose:
        print('\ncoords after merging:\n', coords)

    # update Elements_t's
    sections = wrapper.getChildrenByType(zone, 'Elements_t')
    for section in sections:
        connectivity = wrapper.getNodeData(
            wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
        n = len(connectivity)
        for i in range(n):
            connectivity[i] = i_multiple_to_i_unique[connectivity[i] - 1] + 1
        assert (1 <= connectivity).all() and (connectivity <= n_unique).all()

    # write the new mesh out
    output = args.output
    if output is None:
        output = f'merged_{args.input}'
    print('writing to', output)
    cgm.save(output, cgns)
