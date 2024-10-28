import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial import KDTree
import sys


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


def build_dok_matrix(points: np.ndarray, radius: float, verbose: bool) -> sparse.dok_matrix:
    assert points.shape[1] == 3
    n_point = len(points)
    kdtree = KDTree(points)
    dok_matrix = sparse.dok_matrix((n_point + 4, n_point + 4))
    for i_point in range(n_point):
        point_i = points[i_point]
        dok_matrix[i_point, n_point] = dok_matrix[n_point, i_point] = 1
        dok_matrix[i_point, n_point + 1] = dok_matrix[n_point + 1, i_point] = point_i[X]
        dok_matrix[i_point, n_point + 2] = dok_matrix[n_point + 2, i_point] = point_i[Y]
        dok_matrix[i_point, n_point + 3] = dok_matrix[n_point + 3, i_point] = point_i[Z]
        j_neighbors = kdtree.query_ball_point(point_i, radius)
        for j_point in j_neighbors:
            point_j = kdtree.data[j_point]
            distance_ij = np.linalg.norm(point_j - point_i)
            assert distance_ij <= radius
            if verbose and n_point <= 100:
                print(f'i = {i_point:2d}, j = {j_point:2d}, d_ij = {distance_ij:.2e}')
            dok_matrix[i_point, j_point] = distance_ij
    if verbose and n_point <= 100:
        print('points =\n', points)
        print('matrix =\n', dok_matrix.todense())
    return dok_matrix


if __name__ == '__main__':
    n_point = int(sys.argv[1])
    np.random.seed(123456789)
    points = np.random.rand(n_point, 3)
    dok_matrix = build_dok_matrix(points, radius=0.1, verbose=True)
    print(f'nnz / (n * n) = {dok_matrix.nnz} / {n_point * n_point}')
    csc_matrix = dok_matrix.tocsc()
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
