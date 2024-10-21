from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Vertex
from check_nodes import getUniqueZone, readPoints

import numpy as np
import argparse
import sys
from mpi4py import MPI


def multiple_to_unique(multiple_points: np.ndarray) -> np.ndarray:
    old_coords = multiple_points

    print(old_coords[0, :])
    print(old_coords[-1, :])

    n_old = len(old_coords)

    coord_to_i_new = dict()

    i_new = 0
    for xyz in old_coords:
        x, y, z = xyz
        coord = (x, y, z)
        if coord not in coord_to_i_new:
            coord_to_i_new[coord] = i_new
            i_new += 1
    assert i_new == len(coord_to_i_new)
    n_new = i_new

    i_old_to_i_new = np.ndarray((n_old, 1), dtype='int')
    for i_old in range(n_old):
        xyz = old_coords[i_old]
        coord = (xyz[0], xyz[1], xyz[2])
        i_old_to_i_new[i_old] = coord_to_i_new[coord]
    np.savetxt('i_old_to_i_new.csv', i_old_to_i_new, fmt='%d', delimiter=',')

    new_coords = np.ndarray((n_new, 3), old_coords.dtype)
    for coord, i_new in coord_to_i_new.items():
        new_coords[i_new] = coord

    unique_points = new_coords
    np.savetxt('unique_points.csv', unique_points, delimiter=',')
    print(n_new, n_old)
    return unique_points


def unique_to_multiple(unique_points: np.ndarray) -> np.ndarray:
    new_coords = unique_points
    i_old_to_i_new = np.loadtxt('i_old_to_i_new.csv', dtype=int, delimiter=',')
    n_old = len(i_old_to_i_new)

    # new_coords = np.loadtxt('shifted_unique_coords.csv', delimiter=',')
    old_coords = np.ndarray((n_old, 3), dtype=new_coords.dtype)

    for i_old in range(n_old):
        i_new = i_old_to_i_new[i_old]
        old_coords[i_old] = new_coords[i_new]

    multiple_points = old_coords
    np.savetxt('multiple_points.csv', multiple_points, delimiter=',')


def project_points(cad_file: str, inch_csv: str, comm_rank: int, comm_size: int) -> np.ndarray:
    log = open(f'log_{comm_rank}.txt', 'w')

    log.write(f'reading CAD file on rank {comm_rank}\n')
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(cad_file)
    if status == 1:
        step_reader.TransferRoots()
        cad_model = step_reader.OneShape()
    else:
        raise Exception("Error reading STEP file.")

    log.write(f'reading CSV file on rank {comm_rank}\n')
    old_coords = np.loadtxt(inch_csv, delimiter=',')
    assert old_coords.shape[1] == 3
    old_coords *= 25.4  # old_coords in inch, convert it to mm

    n_node_global = len(old_coords)
    n_node_local = n_node_global // comm_size
    first = n_node_local * comm_rank
    last = n_node_local * (comm_rank + 1)
    last = min(last, n_node_global)

    new_coords = np.ndarray((n_node_local, 3), old_coords.dtype)

    log.write(f'range[{comm_rank}] = [{first}, {last})\n')
    for i in range(first, last):
        x, y, z = old_coords[i]
        point_vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        dist_shape_shape = BRepExtrema_DistShapeShape(TopoDS_Shape(point_vertex), cad_model)
        dist_shape_shape.Perform()
        point_on_shape = dist_shape_shape.PointOnShape2(1)
        new_coords[i - first] = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
        log.write(f'i = {i}, d = {dist_shape_shape.Value():3.1e}\n')

    new_coords /= 25.4  # now, new_coords in inch

    np.savetxt(f'new_coords_{comm_rank}.csv', new_coords, delimiter=',')
    log.close()
    return new_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Project points in CGNS file to the surface of a CAD model.')
    parser.add_argument('--cad', type=str, help='the STEP file of the CAD model')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the points to be projected')
    parser.add_argument('--output', type=str, help='the CSV file of the projected points')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from process {rank} of {size}!")

    if rank == 0:
        cgns, zone, zone_size = getUniqueZone(args.mesh)
        _, xyz, x, y, z = readPoints(zone, zone_size)
        unique_xyz = multiple_to_unique(xyz)
    comm.Barrier()

    shifted_unique_xyz = project_points(args.cad, 'unique_points.csv', rank, size)
