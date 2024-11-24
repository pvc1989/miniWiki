from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepTools import BRepTools_ReShape
from OCC.Core.gp import gp_Pnt
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Vertex

import wrapper
import parallel

from timeit import default_timer as timer
import numpy as np
import argparse
import sys
import os
from mpi4py import MPI
import threading
from concurrent.futures import ProcessPoolExecutor, wait


def get_one_shape_from_cad(cad_file: str):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(cad_file)
    if status != 1:
        raise Exception("Error reading STEP file.")
    step_reader.TransferRoots()
    one_shape = step_reader.OneShape()
    return one_shape


def get_all_faces_from_cad(cad_file):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(cad_file)
    if status != 1:
        raise Exception("Error reading STEP file.")
    step_reader.TransferRoots()
    for i_shape in range(step_reader.NbShapes()):
        shape = step_reader.Shape(1 + i_shape)
        faces = extract_faces(shape)
    return faces


def get_bounding_box(shape: TopoDS_Shape):
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    return box.Get()  # xmin, ymin, zmin, xmax, ymax, zmax


def extract_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        face = explorer.Current()
        faces.append(face)
        explorer.Next()
    return faces


def remove_one_face(shape: TopoDS_Shape, face: TopoDS_Face) -> TopoDS_Shape:
    # builder = BRep_Builder()
    # new_shape = shape.EmptyCopy()
    # builder.MakeShape(new_shape)
    # list_of_faces = TopTools_ListOfShape()
    # list_of_faces.Append(face)
    # print(shape, new_shape, face, list_of_faces)
    # builder.Remove(new_shape, list_of_faces)
    reshape = BRepTools_ReShape()
    reshape.Remove(face)
    new_shape = reshape.Apply(shape)
    return new_shape


def project_by_one_shape(vertex: TopoDS_Vertex, shape: TopoDS_Shape) -> tuple[gp_Pnt|float]:
    dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape)
    dist_shape_shape.Perform()
    point_on_shape = dist_shape_shape.PointOnShape2(1)
    distance = dist_shape_shape.Value()
    return point_on_shape, distance


def project_by_faces(vertex: TopoDS_Vertex, faces: list[TopoDS_Face]) -> tuple[gp_Pnt|float]:
    min_distance = 1e100
    min_distance_point = gp_Pnt()
    for face in faces:
        point_on_shape, distance = project_by_one_shape(vertex, face)
        if distance < min_distance:
            min_distance = distance
            min_distance_point = point_on_shape
    return min_distance_point, min_distance


def project_points_by_mpi_process(args,
        comm_rank: int, comm_size: int) -> np.ndarray:
    log = open(f'{args.folder}/log_{comm_rank}.txt', 'w')

    log.write(f'reading CAD file on rank {comm_rank}\n')
    one_shape = get_one_shape_from_cad(args.cad)
    log.write(f'reading NPY file on rank {comm_rank}\n')
    old_points = np.load(f'{args.folder}/unique_points.npy')
    assert old_points.shape[1] == 3
    old_points *= args.length_unit_ratio  # old_points in inch, convert it to mm

    start = timer()

    n_global = len(old_points)
    first, last = parallel.get_range(comm_rank, comm_size, n_global)
    n_local = last - first

    new_points = np.ndarray((n_local, 3), old_points.dtype)

    log.write(f'range[{comm_rank}] = [{first}, {last})\n')
    for i_global in range(first, last):
        x, y, z = old_points[i_global]
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        point_on_shape, distance = project_by_one_shape(vertex, one_shape)
        # point_on_shape, distance = project_by_faces(vertex, faces)
        i_local = i_global - first
        new_points[i_local] = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
        log.write(f'[{(i_local + 1) / n_local:.4f}] i = {i_global}, d = {distance:3.1e}\n')
        log.flush()

    new_points /= args.length_unit_ratio  # now, new_points in inch

    np.save(f'{args.folder}/new_points_{comm_rank}.npy', new_points)
    end = timer()
    log.write(f'wall-clock cost = {end - start}')
    log.close()

    return new_points


def project_points_by_futures_process(args, i_task: int, n_task: int):
    global one_shape
    global old_points

    log = open(f'{args.folder}/log_{i_task}.txt', 'w')
    assert old_points.shape[1] == 3

    start = timer()

    n_global = len(old_points)
    first, last = parallel.get_range(i_task, n_task, n_global)
    n_local = last - first

    new_points = np.ndarray((n_local, 3), old_points.dtype)

    log.write(f'range[{i_task}] = [{first}, {last})\n')
    print(i_task, old_points.shape, np.linalg.norm(old_points))
    log.write(f'old_points.norm = {np.linalg.norm(old_points)} mm\n')
    log.flush()

    for i_global in range(first, last):
        x, y, z = old_points[i_global]
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        point_on_shape, distance = project_by_one_shape(vertex, one_shape)
        i_local = i_global - first
        new_points[i_local] = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
        log.write(f'[{(i_local + 1) / n_local:.4f}] i = {i_global}, d = {distance:3.1e}\n')
        log.flush()

    new_points /= args.length_unit_ratio  # now, new_points in inch

    np.save(f'{args.folder}/new_points_{i_task}.npy', new_points)
    end = timer()
    log.write(f'task {i_task} on process {os.getpid()} thread {threading.current_thread().ident}\n')
    log.write(f'wall-clock cost = {end - start}')
    log.close()

    # tasks are partitioned to be independent, so no data racing
    # old_points[first : last, :] = new_points

    return new_points


def merge_new_points(folder: str, n_task: int, n_point: int):
    first = 0
    new_points_merged = np.ndarray((n_point, 3))
    for i_task in range(n_task):
        part = np.load(f'{folder}/new_points_{i_task}.npy')
        last = first + len(part)
        new_points_merged[first : last, :] = part
        print(f'merge rank = {i_task}: [{first}, {last})')
        os.remove(f'{folder}/new_points_{i_task}.npy')
        # os.remove(f'{folder}/log_{i_task}.txt')
        first = last
    assert first == n_point, (first, n_point)
    np.save(f'{folder}/new_points_merged.npy', new_points_merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Project points in CGNS file to the surface of a CAD model.')
    parser.add_argument('--cad', type=str, help='the STEP file of the CAD model')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the points to be projected')
    parser.add_argument('--folder', type=str, default='.', help='the working folder to save log and output files')
    parser.add_argument('--output', type=str, help='the NPY file of the projected points')
    parser.add_argument('--parallel',  type=str, choices=['mpi', 'futures'], help='parallel mechanism')
    parser.add_argument('--n_worker',  type=int, help='number of workers for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_worker')
    parser.add_argument('--length_unit_ratio',  type=float, default=25.4, help='CGNS\'s length unit / CAD\' length unit')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.folder != '.':
        os.makedirs(args.folder, exist_ok=True)

    using_mpi = (args.parallel == 'mpi')

    if using_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if args.verbose and rank == 0:
            print(args)
        n_task = size
    else:
        n_task = args.n_task

    if not using_mpi or rank == 0:
        cgns, zone, zone_size = wrapper.getUniqueZone(args.mesh)
        print(zone_size)
        xyz, x, y, z = wrapper.readPoints(zone, zone_size)
        assert zone_size[0][0] == len(xyz) == len(x) == len(y) == len(z)
        xyz *= args.length_unit_ratio  # old_points in inch, convert it to mm
        print(f'xyz.norm = {np.linalg.norm(xyz)} mm')

    if using_mpi:
        comm.Barrier()
        # TODO(PVC): read CAD and NPY by rank 0 and share them with other ranks
        project_points_by_mpi_process(args, rank, size)
        comm.Barrier()
    else:
        global one_shape
        global old_points
        old_points = xyz
        one_shape = get_one_shape_from_cad(args.cad)
        executor = ProcessPoolExecutor(args.n_worker)
        start = timer()
        fs = []  # list of futures
        for i_task in range(n_task):
            fs.append(executor.submit(
                project_points_by_futures_process, args, i_task, n_task))
        done, not_done = wait(fs)
        print('done:')
        for x in done:
            print(x)
        print('not_done:')
        for x in not_done:
            print(x)
        end = timer()
        print(f'wall-clock cost: {(end - start):.2f}')
        executor.shutdown()

    if not using_mpi or rank == 0:
        merge_new_points(args.folder, n_task, len(xyz))
