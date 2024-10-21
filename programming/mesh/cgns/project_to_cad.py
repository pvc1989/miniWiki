from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Vertex

import numpy as np
import argparse
import sys
from mpi4py import MPI


def project_points(cad_file: str, inch_csv: np.ndarray, comm_rank: int, comm_size: int):
    log = open(f'./occt/log_{comm_rank}.txt', 'w')

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

    new_coords = np.ndarray((n_node_local), old_coords.dtype)

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

    np.savetxt(f'projected_coords_{comm_rank}.csv', new_coords, delimiter=',')
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Project points in CGNS file to the surface of a CAD model.')
    parser.add_argument('--cad', type=str, help='the STEP file of the CAD model')
    parser.add_argument('--points', type=str, help='the CSV file of the points to be projected')
    parser.add_argument('--output', type=str, help='the CSV file of the projected points')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from process {rank} of {size}!")

    project_points(args.cad, args.points, rank, size)
