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
    sol_file_name = f'solved_{rbf_matrix_npz[:-4]}.npy'
    np.save(sol_file_name, sol_columns)
    return sol_file_name


def shift_points_by_futures_process(i_task: int, n_task: int, k_neighbor: int, radius: float, max_radius: float):
    global kdtree
    global u_rbf, v_rbf, w_rbf
    global global_x, global_y, global_z

    log = open(f'log_{i_task}.txt', 'w')

    start = timer()

    n_global = len(global_x)
    n_local = (n_global + n_task - 1) // n_task
    first = n_local * i_task
    last = n_local * (i_task + 1)
    last = min(last, n_global)
    if i_task + 1 == n_task:
        n_local = last - first
        assert last == n_global
    log.write(f'range[{i_task}] = [{first}, {last})\n')
    log.flush()

    shift = np.zeros((n_local, 3), global_x.dtype)
    for i_local in range(n_local):
        i_global = i_local + first
        x, y, z = global_x[i_global], global_y[i_global], global_z[i_global]
        if (x * x + y * y + z * z > max_radius * max_radius):
            log.write(f'{i_global} skipped\n')
            log.flush()
            continue
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
        log.flush()
    end = timer()
    log.write(f'wall-clock cost: {(end - start):.2f}\n')
    log.close()
    np.save(f'shift_{i_task}.npy', shift)
    print(f'task {i_task} costs {(end - start):.2f} seconds')
    return first, last


def shift_interior_points(input: str, rbf_solutions_npy: str, bc_points: np.ndarray,
        k_neighbor: int, radius: float, max_radius: float, verbose: bool):
    if verbose:
        print('loading the CGNS tree ...')
    cgns, zone, zone_size = wrapper.getUniqueZone(input)
    global global_x, global_y, global_z
    _, global_x, global_y, global_z = wrapper.readPoints(zone, zone_size)
    print(np.linalg.norm(global_x), np.linalg.norm(global_y), np.linalg.norm(global_z))
    n_point = zone_size[0][0]
    assert n_point == len(global_x) == len(global_y) == len(global_z)

    if verbose:
        print('loading the RBF solution ...')
    rbf_solutions = np.load(rbf_solutions_npy)
    global u_rbf, v_rbf, w_rbf
    u_rbf = np.array(rbf_solutions[:, X])
    v_rbf = np.array(rbf_solutions[:, Y])
    w_rbf = np.array(rbf_solutions[:, Z])

    if verbose:
        print('building the KDTree ...')
    global kdtree
    kdtree = KDTree(bc_points)

    executor = ProcessPoolExecutor(max_workers=16)
    n_task = 1024
    tasks = []
    for i_task in range(n_task):
        tasks.append(executor.submit(
            shift_points_by_futures_process, i_task, n_task, k_neighbor, radius, max_radius))
    done, not_done = wait(tasks)
    print('done:')
    for x in done:
        print(x)
    print('not_done:')
    for x in not_done:
        print(x)
    # write back the coords
    global_shift_x = np.ndarray((n_point,), global_x.dtype)
    global_shift_y = np.ndarray((n_point,), global_x.dtype)
    global_shift_z = np.ndarray((n_point,), global_x.dtype)
    for i_task in range(n_task):
        first, last = tasks[i_task].result()
        print(i_task, first, last)
        local_shift = np.load(f'shift_{i_task}.npy')
        global_shift_x[first : last] = local_shift[:, X]
        global_shift_y[first : last] = local_shift[:, Y]
        global_shift_z[first : last] = local_shift[:, Z]
    global_x += global_shift_x
    global_y += global_shift_y
    global_z += global_shift_z
    print(np.linalg.norm(global_x), np.linalg.norm(global_y), np.linalg.norm(global_z))
    if k_neighbor > 0:
        assert radius == 0.0
        suffix = f'k={k_neighbor}'
    elif radius > 0:
        assert k_neighbor == 0
        suffix = f'r={radius:.2e}'
    output = f'shifted_{input[:-5]}_{suffix}.cgns'
    if verbose:
        print(f'writing to {output} ... ')

    # write the shifts as a vector field of point data
    point_data = wrapper.getChildrenByType(zone, 'FlowSolution_t')
    for data in point_data:
        cgu.removeChildByName(zone, wrapper.getNodeName(data))
    point_data = cgl.newFlowSolution(zone, 'FlowSolutionHelper', 'Vertex')
    cgl.newDataArray(point_data, 'ShiftX', global_shift_x)
    cgl.newDataArray(point_data, 'ShiftY', global_shift_y)
    cgl.newDataArray(point_data, 'ShiftZ', global_shift_z)
    cgm.save(output, cgns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Shift interior points by RBF interpolation of boundary points.')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the mesh to be shifted')
    parser.add_argument('--old_points', type=str, help='the NPY file of boundary points before shifting')
    parser.add_argument('--new_points', type=str, help='the NPY file of boundary points after shifting')
    parser.add_argument('--k_neighbor', type=int, default=0, help='number of neighbors in the support of the RBF basis')
    parser.add_argument('--radius', type=float, default=0.0, help='the radius of the RBF basis')
    parser.add_argument('--output', type=str, help='the output mesh file')
    parser.add_argument('--verbose', default=True, action='store_true')
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
    new_points = np.load(args.new_points)
    assert old_points.shape == new_points.shape

    rbf_matrix_npz = build_rbf_matrix(old_points, args.k_neighbor, args.radius, args.verbose)
    rbf_solutions_npy = solve_rbf_system(rbf_matrix_npz, new_points - old_points, args.verbose)
    shift_interior_points(args.mesh, rbf_solutions_npy, old_points, args.k_neighbor, args.radius, 5.0e3, args.verbose)

    # n_point = int(sys.argv[1])
    # test_on_random_points(n_point, radius=0.1, verbose=True)
