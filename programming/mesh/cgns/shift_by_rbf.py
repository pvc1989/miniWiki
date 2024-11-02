import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial import KDTree
import sys
import argparse
from timeit import default_timer as timer

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import wrapper
from check_nodes import getUniqueZone, readPoints


X, Y, Z = 0, 1, 2


def dimensionaless_rbf(xi: float) -> float:
    if xi < 1e-12:
        return 1
    elif xi > 1.0:
        return 0
    else:
        return 1 - 30 * (xi**2) - 10 * (xi**3) + 45 * (xi**4) - 6 * (xi**5) - 60 * (xi**3) * np.log(xi)


def dimensional_rbf(vector: np.ndarray, radius: float):
    return dimensionaless_rbf(np.linalg.norm(vector) / radius)


def get_neighbors(kdtree: KDTree, point_i: np.ndarray, k_neighbor: int, radius: float) -> tuple[list[int], float]:
    if k_neighbor > 0:
        assert radius == 0.0, radius
        d_neighbors, j_neighbors = kdtree.query(point_i, k_neighbor)
        radius = d_neighbors[-1]
    elif radius > 0:
        assert k_neighbor == 0, k_neighbor
        j_neighbors = kdtree.query_ball_point(point_i, radius)
    return j_neighbors, radius


def build_dok_matrix(kdtree: KDTree, k_neighbor: int, radius: float, verbose: bool) -> sparse.dok_matrix:
    points = kdtree.data
    n_point = len(points)
    assert points.shape[1] == 3
    dok_matrix = sparse.dok_matrix((n_point + 4, n_point + 4))
    for i_point in range(n_point):
        point_i = points[i_point]
        dok_matrix[i_point, n_point] = dok_matrix[n_point, i_point] = 1
        dok_matrix[i_point, n_point + 1] = dok_matrix[n_point + 1, i_point] = point_i[X]
        dok_matrix[i_point, n_point + 2] = dok_matrix[n_point + 2, i_point] = point_i[Y]
        dok_matrix[i_point, n_point + 3] = dok_matrix[n_point + 3, i_point] = point_i[Z]
        j_neighbors, radius_out = get_neighbors(kdtree, point_i, k_neighbor, radius)
        for j_point in j_neighbors:
            point_j = kdtree.data[j_point]
            distance_ij = np.linalg.norm(point_j - point_i)
            assert distance_ij <= radius_out + 1e-10, (distance_ij, radius_out)
            if verbose and n_point <= 100:
                print(f'i = {i_point:2d}, j = {j_point:2d}, d_ij = {distance_ij:.2e}')
            dok_matrix[i_point, j_point] = dimensional_rbf(distance_ij, radius_out)
        if verbose:
            print(f'point[{i_point}] has {len(j_neighbors)} neighbors')
    if verbose and n_point <= 100:
        print('points =\n', points)
        print('matrix =\n', dok_matrix.todense())
    return dok_matrix


def test_on_random_points(n_point: int, radius: float, verbose: bool):
    np.random.seed(123456789)
    points = np.random.rand(n_point, 3)
    if verbose:
        print('building the KDTree ...')
    kdtree = KDTree(points)
    if verbose:
        print('building the dok_matrix ...')
    dok_matrix = build_dok_matrix(kdtree, 0, radius, verbose)
    sparsity = dok_matrix.nnz / (n_point * n_point)
    print(f'nnz / (n * n) = {dok_matrix.nnz} / {n_point * n_point} = {sparsity:.2e}')
    csc_matrix = dok_matrix.tocsc()
    csc_file_name = f'csc_matrix_{n_point}.npz'
    sparse.save_npz(csc_file_name, csc_matrix)
    csc_matrix = sparse.load_npz(csc_file_name)
    u_bc = np.random.rand(n_point + 4)
    u_bc[-4:] = 0
    u_coeffs, exit_code = sparse.linalg.lgmres(csc_matrix, u_bc, maxiter=20000, rtol=1e-7)
    print('exit_code =', exit_code)
    print(np.linalg.norm(csc_matrix.dot(u_coeffs) - u_bc))
    print(np.allclose(csc_matrix.dot(u_coeffs), u_bc))
    den_matrix = dok_matrix.todense()
    u_coeffs = np.linalg.solve(den_matrix, u_bc)
    print(np.linalg.norm(den_matrix.dot(u_coeffs) - u_bc))
    print(np.allclose(den_matrix.dot(u_coeffs), u_bc))


def build_rbf_matrix(old_points: np.ndarray, k_neighbor: int, radius: float, verbose: bool) -> str:
    assert old_points.shape[1] == 3
    print('shape =', old_points.shape)
    n_point = len(old_points)
    if verbose:
        print('building the KDTree ...')
        print(old_points)
    kdtree = KDTree(old_points)
    if verbose:
        print('building the dok_matrix ...')
    dok_matrix = build_dok_matrix(kdtree, k_neighbor, radius, verbose)
    sparsity = dok_matrix.nnz / (n_point * n_point)
    print(f'nnz / (n * n) = {dok_matrix.nnz} / {n_point * n_point} = {sparsity:.2e}')
    csc_matrix = dok_matrix.tocsc()
    if k_neighbor > 0:
        assert radius == 0.0
        csc_file_name = f'csc_matrix_n={n_point}_k={k_neighbor}.npz'
    elif radius > 0:
        assert k_neighbor == 0
        csc_file_name = f'csc_matrix_n={n_point}_r={radius:.2e}.npz'
    sparse.save_npz(csc_file_name, csc_matrix)
    return csc_file_name


def solve_rbf_system(rbf_matrix_npz: str, rhs_columns: np.ndarray, verbose: bool) -> str:
    rbf_matrix = sparse.load_npz(rbf_matrix_npz)
    n_point = len(rhs_columns)
    n_column = rhs_columns.shape[1]
    sol_columns = np.ndarray((n_point + 4, n_column))
    for i_column in range(n_column):
        column_i = np.zeros((n_point + 4,))
        column_i[0 : n_point] = rhs_columns[:, i_column]
        if verbose:
            print(f'\nsolving column[{i_column}] ...')
        start = timer()
        sol_columns[:, i_column], exit_code = sparse.linalg.lgmres(
            rbf_matrix, column_i, maxiter=20000, atol=0.0, rtol=1e-7)
        print(f'column[{i_column}] solved with exit_code = {exit_code}')
        end = timer()
        print(f'costs {end - start:.2e} seconds')
    sol_file_name = f'solved_{rbf_matrix_npz}'
    np.save(sol_file_name, sol_columns)
    return sol_file_name


def shift_interior_points(input: str, rbf_solutions_npy: str, bc_points: np.ndarray,
        k_neighbor: int, radius: float, max_radius: float, verbose: bool):
    if verbose:
        print('loading the CGNS tree ...')
    cgns, zone, zone_size = getUniqueZone(input)
    _, _, coords_x, coords_y, coords_z = readPoints(zone, zone_size)
    n_point = zone_size[0][0]
    assert n_point == len(coords_x) == len(coords_y) == len(coords_z)

    if verbose:
        print('loading the RBF solution ...')
    rbf_solutions = np.load(rbf_solutions_npy)
    u_rbf = rbf_solutions[:, X]
    v_rbf = rbf_solutions[:, Y]
    w_rbf = rbf_solutions[:, Z]

    if verbose:
        print('building the KDTree ...')
    kdtree = KDTree(bc_points)

    for i_point in range(n_point):
        x, y, z = coords_x[i_point], coords_y[i_point], coords_z[i_point]
        if (x * x + y * y + z * z > max_radius * max_radius):
            if verbose:
                print(f'{i_point} skipped')
            continue
        point_i = np.array([x, y, z])
        j_neighbors, radius_out = get_neighbors(kdtree, point_i, k_neighbor, radius)
        u = u_rbf[-4] + np.dot(u_rbf[-3:], point_i)
        v = v_rbf[-4] + np.dot(v_rbf[-3:], point_i)
        w = w_rbf[-4] + np.dot(w_rbf[-3:], point_i)
        for j_point in j_neighbors:
            point_j = kdtree.data[j_point]
            distance_ij = np.linalg.norm(point_j - point_i)
            assert distance_ij <= radius_out + 1e-10, (distance_ij, radius_out)
            phi_ij = dimensional_rbf(distance_ij, radius_out)
            u += u_rbf[j_point] * phi_ij
            v += v_rbf[j_point] * phi_ij
            w += w_rbf[j_point] * phi_ij
        coords_x[i_point] += u
        coords_y[i_point] += v
        coords_z[i_point] += w
        if verbose:
            print(f'{i_point} shifted')
    # write back the coords
    cgm.save(f'shifted_{input}', cgns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0][:-3]}.py',
        description='Shift interior points by RBF interpolation of boundary points.')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the mesh to be shifted')
    parser.add_argument('--old_points', type=str, help='the NPY file of boundary points before shifting')
    parser.add_argument('--new_points', type=str, help='the NPY file of boundary points after shifting')
    parser.add_argument('--k_neighbor', type=int, default=0, help='number of neighbors in the support of the RBF basis')
    parser.add_argument('--radius', type=float, default=0.0, help='the radius of the RBF basis')
    parser.add_argument('--output', type=str, help='the output mesh file')
    parser.add_argument('--verbose', default=True, action='store_true')
    args = parser.parse_args()

    old_points = np.load(args.old_points)
    new_points = np.load(args.new_points)
    assert old_points.shape == new_points.shape

    rbf_matrix_npz = build_rbf_matrix(old_points, args.k_neighbor, args.radius, args.verbose)
    rbf_solutions_npy = solve_rbf_system(rbf_matrix_npz, new_points - old_points, args.verbose)

    shift_interior_points(args.mesh, rbf_solutions_npy, old_points, args.k_neighbor, args.radius, 5.0e3, args.verbose)

    # n_point = int(sys.argv[1])
    # test_on_random_points(n_point, radius=0.1, verbose=True)
