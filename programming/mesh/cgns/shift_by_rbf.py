import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.spatial import KDTree
import os
import sys
import shutil
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, wait

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import pycgns_wrapper
from pycgns_wrapper import X, Y, Z

import parallel

def dimensionaless_rbf_0(xi: float) -> float:
    return (1 - xi)**8 * (32 * (xi**3) + 25 * (xi**2) + 8 * xi + 1)


def dimensionaless_rbf_1(xi: float) -> float:
    return (1 - xi)**2


def dimensionaless_rbf_2(xi: float) -> float:
    return (1 - xi)**4 * (4 * xi + 1)


def dimensionaless_rbf_3(xi: float) -> float:
    return (1 - xi)**6 * (35 / 3 * (xi**2) + 6 * xi + 1)


def dimensionaless_rbf_4(xi: float) -> float:
    return  (1 - xi)**8 * (32 * (xi**3) + 25 * (xi**2) + 8 * xi + 1)


def dimensionaless_rbf_5(xi: float) -> float:
    return (1.0 - xi)**5


def dimensionaless_rbf_6(xi: float) -> float:
    if xi < 1e-12:
        return 1
    return 1 + 80 / 3 * (xi**2) - 40 * (xi**3) + 15 * (xi**4) - 8 / 3 * (xi**5) + 20 * (xi**2) * np.log(xi)


def dimensionaless_rbf_7(xi: float) -> float:
    if xi < 1e-12:
        return 1
    return 1 - 30 * (xi**2) - 10 * (xi**3) + 45 * (xi**4) - 6 * (xi**5) - 60 * (xi**3) * np.log(xi)


def dimensionaless_rbf_8(xi: float) -> float:
    if xi < 1e-12:
        return 1
    return 1 - 20 * (xi**2) + 80 * (xi**3) - 45 * (xi**4) - 16 * (xi**5) + 60 * (xi**4) * np.log(xi)


def dimensionaless_rbf(xi: float) -> float:
    return dimensionaless_rbf_2(xi)


def dimensional_rbf(vector: np.ndarray, radius: float):
    xi = np.linalg.norm(vector) / radius
    if xi >= 1:
        return 0
    return dimensionaless_rbf(xi)


def get_neighbors(kdtree: KDTree, point_i: np.ndarray, k_neighbor: int, radius: float) -> tuple[list[int], float]:
    if k_neighbor > 0:
        assert radius == 0.0, radius
        d_neighbors, j_neighbors = kdtree.query(point_i, k_neighbor)
        radius = d_neighbors[-1]
    elif radius > 0:
        assert k_neighbor == 0, k_neighbor
        j_neighbors = kdtree.query_ball_point(point_i, radius)
    return j_neighbors, radius


def build_dok_matrix(kdtree: KDTree, args: str) -> sparse.dok_matrix:
    points = kdtree.data
    n_point = len(points)
    assert points.shape[1] == 3
    dok_matrix = sparse.dok_matrix((n_point + 4, n_point + 4))
    k_neighbor = args.k_neighbor
    radius = args.radius
    verbose = args.verbose
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
            print(f'point[{i_point} / {n_point}] has {len(j_neighbors)} neighbors')
    if verbose and n_point <= 100:
        print('points =\n', points)
        print('matrix =\n', dok_matrix.todense())
    return dok_matrix


def build_rbf_matrix(rbf_folder: str, old_points: np.ndarray):
    csc_file_name = f'{rbf_folder}/rbf_matrix.npz'
    if os.path.exists(csc_file_name):
        print(f'\n{csc_file_name} already exists')
        return

    start = time.time()
    assert old_points.shape[1] == 3
    n_point = len(old_points)
    print('building the KDTree ...')
    kdtree = KDTree(old_points)
    print('building the dok_matrix ...')
    dok_matrix = build_dok_matrix(kdtree, args)
    sparsity = dok_matrix.nnz / (n_point * n_point)
    print(f'nnz / (n * n) = {dok_matrix.nnz} / {n_point * n_point} = {sparsity:.2e}')
    csc_matrix = dok_matrix.tocsc()
    sparse.save_npz(csc_file_name, csc_matrix)
    end = time.time()
    print(f'building RBF matrix costs {end - start} seconds')


def solve_rbf_system(rbf_folder: str, rhs_columns: np.ndarray):
    sol_file_name = f'{rbf_folder}/rbf_solutions.npy'
    if os.path.exists(sol_file_name):
        print(f'\n{sol_file_name} already exists')
        return

    rbf_matrix = sparse.load_npz(f'{rbf_folder}/rbf_matrix.npz')
    n_point = len(rhs_columns)
    n_column = rhs_columns.shape[1]
    sol_columns = np.ndarray((n_point + 4, n_column))
    for i_column in range(n_column):
        column_i = np.zeros((n_point + 4,))
        column_i[0 : n_point] = rhs_columns[:, i_column]
        print(f'\nsolving column[{i_column}] ...')
        start = time.time()
        sol_columns[:, i_column], exit_code = sparse.linalg.lgmres(
            rbf_matrix, column_i, maxiter=20000, atol=0.0, rtol=1e-7)
        print(f'column[{i_column}] solved with exit_code = {exit_code}')
        end = time.time()
        print(f'costs {end - start:.2e} seconds')
    np.save(sol_file_name, sol_columns)


def shift_points_by_futures_process(i_task: int, n_task: int, temp_folder: str, args: str):
    global kdtree
    global u_rbf, v_rbf, w_rbf
    global global_x, global_y, global_z

    log = open(f'{temp_folder}/log_{i_task}.txt', 'w')

    start = time.time()

    n_global = len(global_x)
    first, last = parallel.get_range(i_task, n_task, n_global)
    n_local = last - first
    log.write(f'range[{i_task}] = [{first}, {last})\n')

    k_neighbor = args.k_neighbor
    radius = args.radius
    shift = np.zeros((n_local, 3), global_x.dtype)
    for i_local in range(n_local):
        i_global = i_local + first
        x, y, z = global_x[i_global], global_y[i_global], global_z[i_global]
        point_i = np.array([x, y, z])
        j_neighbors, radius_out = get_neighbors(kdtree, point_i, k_neighbor, radius)
        u = u_rbf[-4] + np.dot(u_rbf[-3:], point_i)
        v = v_rbf[-4] + np.dot(v_rbf[-3:], point_i)
        w = w_rbf[-4] + np.dot(w_rbf[-3:], point_i)
        for j_global in j_neighbors:
            point_j = kdtree.data[j_global]
            distance_ij = np.linalg.norm(point_j - point_i)
            assert distance_ij <= radius_out + 1e-10, (distance_ij, radius_out)
            phi_ij = dimensional_rbf(distance_ij, radius_out)
            u += u_rbf[j_global] * phi_ij
            v += v_rbf[j_global] * phi_ij
            w += w_rbf[j_global] * phi_ij
        shift[i_local, :] = u, v, w
        log.write(f'{i_global} shifted by ({u:.2e}, {v:.2e}, {w:.2e})\n')
    end = time.time()
    log.write(f'wall-clock cost: {(end - start):.2f}\n')
    log.close()
    np.save(f'{temp_folder}/shift_{i_task}.npy', shift)
    print(f'task {i_task} costs {(end - start):.2f} seconds')
    return first, last


def shift_interior_points(rbf_folder: str, bc_points: np.ndarray, args: str):
    print('\nloading the volume mesh ...')
    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.mesh)
    global global_x, global_y, global_z
    _, global_x, global_y, global_z = pycgns_wrapper.readPoints(zone, zone_size)
    n_point = zone_size[0][0]
    n_cell = zone_size[0][1]
    assert n_point == len(global_x) == len(global_y) == len(global_z)

    output = f'{rbf_folder}/shifted_{n_cell}cells_{n_point}points.cgns'
    if os.path.exists(output):
        print(f'\n{output} already exists')
        return

    print('loading the RBF solution ...')
    rbf_solutions = np.load(f'{rbf_folder}/rbf_solutions.npy')
    global u_rbf, v_rbf, w_rbf
    u_rbf = np.array(rbf_solutions[:, X])
    v_rbf = np.array(rbf_solutions[:, Y])
    w_rbf = np.array(rbf_solutions[:, Z])

    print('building the KDTree ...')
    global kdtree
    kdtree = KDTree(bc_points)

    print('shifting interior points by futures ...')
    temp_folder = f'{rbf_folder}/temp{np.random.randint(0, 2**32)}'
    os.makedirs(temp_folder, exist_ok=True)
    executor = ProcessPoolExecutor(args.n_worker)
    n_task = args.n_task
    tasks = []
    for i_task in range(n_task):
        tasks.append(executor.submit(
            shift_points_by_futures_process, i_task, n_task, temp_folder, args))
    done, not_done = wait(tasks)
    assert len(not_done) == 0, not_done
    # write back the coords
    global_shift_x = np.ndarray((n_point,), global_x.dtype)
    global_shift_y = np.ndarray((n_point,), global_x.dtype)
    global_shift_z = np.ndarray((n_point,), global_x.dtype)
    for i_task in range(n_task):
        first, last = tasks[i_task].result()
        print(i_task, first, last)
        local_shift = np.load(f'{temp_folder}/shift_{i_task}.npy')
        global_shift_x[first : last] = local_shift[:, X]
        global_shift_y[first : last] = local_shift[:, Y]
        global_shift_z[first : last] = local_shift[:, Z]
    shutil.rmtree(temp_folder)
    global_x += global_shift_x
    global_y += global_shift_y
    global_z += global_shift_z

    # write the shifts as a vector field of point data
    point_data = pycgns_wrapper.getSolutionByLocation(zone, 'Vertex', 'FlowSolutionHelper')
    cgl.newDataArray(point_data, 'ShiftX', global_shift_x)
    cgl.newDataArray(point_data, 'ShiftY', global_shift_y)
    cgl.newDataArray(point_data, 'ShiftZ', global_shift_z)

    print(f'writing to {output} ... ')
    cgm.save(output, cgns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Shift interior points by RBF interpolation of boundary points.')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the mesh to be shifted')
    parser.add_argument('--old_points', type=str, help='the NPY file of boundary points before shifting')
    parser.add_argument('--new_points', type=str, help='the NPY file of boundary points after shifting')
    parser.add_argument('--folder', type=str, help='the folder containing new_points and output')
    parser.add_argument('--n_worker',  type=int, help='number of workers for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_worker')
    parser.add_argument('--k_neighbor', type=int, default=0, help='number of neighbors in the support of the RBF basis')
    parser.add_argument('--radius', type=float, default=0.0, help='the radius of the RBF basis')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    x = np.linspace(0, 1, 101)
    y = np.ndarray(x.shape)
    rbfs = (dimensionaless_rbf_0, dimensionaless_rbf_1, dimensionaless_rbf_2, dimensionaless_rbf_3, dimensionaless_rbf_4, dimensionaless_rbf_5, dimensionaless_rbf_6, dimensionaless_rbf_7, dimensionaless_rbf_8)
    for i_rbf in range(len(rbfs)):
        rbf = rbfs[i_rbf]
        for i in range(len(x)):
            y[i] = rbf(x[i])
        plt.plot(x, y, label=f'RBF_{i_rbf}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('RBF.svg')

    old_points = np.load(args.old_points)
    new_points = np.load(f'{args.folder}/{args.new_points}')
    assert old_points.shape == new_points.shape

    if args.k_neighbor > 0:
        assert args.radius == 0.0
        rbf_folder = f'{args.folder}/k={args.k_neighbor}'
    elif args.radius > 0:
        assert args.k_neighbor == 0
        rbf_folder = f'{args.folder}/r={args.radius:.2e}'

    os.makedirs(rbf_folder, exist_ok=True)

    build_rbf_matrix(rbf_folder, old_points)
    solve_rbf_system(rbf_folder, new_points - old_points)
    shift_interior_points(rbf_folder, old_points, args)
