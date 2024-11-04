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
from merge_points import multiple_to_unique

from timeit import default_timer as timer
import numpy as np
import argparse
import sys
import os
from mpi4py import MPI
import threading
from concurrent.futures import ProcessPoolExecutor, wait


def unique_to_multiple(unique_points: np.ndarray) -> np.ndarray:
    n_unique = len(unique_points)

    # load the multiple_point_index-to-unique_point_index map
    i_multiple_to_i_unique = np.load('i_multiple_to_i_unique.npy')
    n_multiple = len(i_multiple_to_i_unique)
    print('in unique_to_multiple:', n_unique, n_multiple)
    assert (0 <= i_multiple_to_i_unique).all() and (i_multiple_to_i_unique < n_unique).all()

    # build-save-return the multiple points
    multiple_points = np.ndarray((n_multiple, 3), dtype=unique_points.dtype)
    for i_multiple in range(n_multiple):
        i_unique = i_multiple_to_i_unique[i_multiple]
        multiple_points[i_multiple] = unique_points[i_unique]
    np.save('multiple_points.npy', multiple_points)
    return multiple_points


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


def project_points_by_mpi_process(cad_file: str, inch_npy: str,
        comm_rank: int, comm_size: int) -> np.ndarray:
    log = open(f'log_{comm_rank}.txt', 'w')

    log.write(f'reading CAD file on rank {comm_rank}\n')
    one_shape = get_one_shape_from_cad(cad_file)
    log.write(f'reading NPY file on rank {comm_rank}\n')
    old_coords = np.load(inch_npy)
    assert old_coords.shape[1] == 3
    old_coords *= 25.4  # old_coords in inch, convert it to mm

    start = timer()

    n_node_global = len(old_coords)
    n_node_local = (n_node_global + comm_size - 1) // comm_size
    first = n_node_local * comm_rank
    last = n_node_local * (comm_rank + 1)
    last = min(last, n_node_global)
    if comm_rank + 1 == comm_size:
        n_node_local = last - first
        assert last == n_node_global

    new_coords = np.ndarray((n_node_local, 3), old_coords.dtype)

    log.write(f'range[{comm_rank}] = [{first}, {last})\n')
    for i_global in range(first, last):
        x, y, z = old_coords[i_global]
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        point_on_shape, distance = project_by_one_shape(vertex, one_shape)
        # point_on_shape, distance = project_by_faces(vertex, faces)
        i_local = i_global - first
        new_coords[i_local] = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
        log.write(f'[{(i_local + 1) / n_node_local:.4f}] i = {i_global}, d = {distance:3.1e}\n')
        log.flush()

    new_coords /= 25.4  # now, new_coords in inch

    np.save(f'new_coords_{comm_rank}.npy', new_coords)
    end = timer()
    log.write(f'wall-clock cost = {end - start}')
    log.close()

    return new_coords


def project_points_by_futures_process(i_task: int, n_task: int):
    global one_shape
    global old_coords

    log = open(f'log_{i_task}.txt', 'w')
    assert old_coords.shape[1] == 3

    start = timer()

    n_node_global = len(old_coords)
    n_node_local = (n_node_global + n_task - 1) // n_task
    first = n_node_local * i_task
    last = n_node_local * (i_task + 1)
    last = min(last, n_node_global)
    if i_task + 1 == n_task:
        n_node_local = last - first
        assert last == n_node_global

    new_coords = np.ndarray((n_node_local, 3), old_coords.dtype)

    log.write(f'range[{i_task}] = [{first}, {last})\n')
    print(i_task, old_coords.shape, np.linalg.norm(old_coords))
    log.write(f'old_coords.norm = {np.linalg.norm(old_coords)} mm\n')
    log.flush()

    for i_global in range(first, last):
        x, y, z = old_coords[i_global]
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Vertex()
        point_on_shape, distance = project_by_one_shape(vertex, one_shape)
        i_local = i_global - first
        new_coords[i_local] = point_on_shape.X(), point_on_shape.Y(), point_on_shape.Z()
        log.write(f'[{(i_local + 1) / n_node_local:.4f}] i = {i_global}, d = {distance:3.1e}\n')
        log.flush()

    new_coords /= 25.4  # now, new_coords in inch

    np.save(f'new_coords_{i_task}.npy', new_coords)
    end = timer()
    log.write(f'task {i_task} on process {os.getpid()} thread {threading.current_thread().ident}\n')
    log.write(f'wall-clock cost = {end - start}')
    log.close()

    # tasks are partitioned to be independent, so no data racing
    # old_coords[first : last, :] = new_coords

    return new_coords


def merge_new_coords(comm_size: int, n_unique_point: int) -> np.ndarray:
    first = 0
    merged_new_coords = np.ndarray((n_unique_point, 3))
    for comm_rank in range(comm_size):
        part = np.load(f'new_coords_{comm_rank}.npy')
        last = first + len(part)
        merged_new_coords[first : last, :] = part
        print(f'merge rank = {comm_rank}: [{first}, {last})')
        first = last
    assert first == n_unique_point, (first, n_unique_point)
    np.save('new_coords_unique.npy', merged_new_coords)
    return merged_new_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Project points in CGNS file to the surface of a CAD model.')
    parser.add_argument('--cad', type=str, help='the STEP file of the CAD model')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the points to be projected')
    parser.add_argument('--output', type=str, help='the NPY file of the projected points')
    parser.add_argument('--parallel',  type=str, choices=['mpi', 'futures'], help='parallel mechanism')
    parser.add_argument('--n_worker',  type=int, help='number of workers for running futures')
    parser.add_argument('--n_task',  type=int, help='number of tasks for running futures, usually >> n_worker')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

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
        unique_points, i_multiple_to_i_unique = multiple_to_unique(
            multiple_points=xyz, radius=1e-8, verbose=args.verbose)
        unique_points *= 25.4  # old_coords in inch, convert it to mm
        print(f'unique_points.norm = {np.linalg.norm(unique_points)} mm')

    if using_mpi:
        comm.Barrier()
        # TODO(PVC): read CAD and NPY by rank 0 and share them with other ranks
        project_points_by_mpi_process(args.cad, 'unique_points.npy', rank, size)
        comm.Barrier()
    else:
        global one_shape
        global old_coords
        old_coords = unique_points
        one_shape = get_one_shape_from_cad(args.cad)
        executor = ProcessPoolExecutor(args.n_worker)
        start = timer()
        fs = []  # list of futures
        for i_task in range(n_task):
            fs.append(executor.submit(
                project_points_by_futures_process, i_task, n_task))
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
        merged_new_coords = merge_new_coords(n_task, len(unique_points))
        new_coords_multiple = unique_to_multiple(merged_new_coords)
        np.save('new_coords_multiple.npy', new_coords_multiple)
