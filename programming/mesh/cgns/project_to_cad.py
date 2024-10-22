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

from check_nodes import getUniqueZone, readPoints

from timeit import default_timer as timer
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

    i_old_to_i_new = np.ndarray((n_old,), dtype='int')
    for i_old in range(n_old):
        xyz = old_coords[i_old]
        coord = (xyz[0], xyz[1], xyz[2])
        i_old_to_i_new[i_old] = coord_to_i_new[coord]
    assert (0 <= i_old_to_i_new).all() and (i_old_to_i_new < n_new).all()
    np.savetxt('i_old_to_i_new.csv', i_old_to_i_new, fmt='%d', delimiter=',')

    new_coords = np.ndarray((n_new, 3), old_coords.dtype)
    for coord, i_new in coord_to_i_new.items():
        new_coords[i_new] = coord

    unique_points = new_coords
    np.savetxt('unique_points.csv', unique_points, delimiter=',')
    print('in multiple_to_unique:', n_new, n_old)
    return unique_points


def unique_to_multiple(unique_points: np.ndarray) -> np.ndarray:
    new_coords = unique_points
    n_new = len(new_coords)
    i_old_to_i_new = np.loadtxt('i_old_to_i_new.csv', dtype=int, delimiter=',')
    n_old = len(i_old_to_i_new)
    print('in unique_to_multiple:', n_new, n_old)
    assert (0 <= i_old_to_i_new).all() and (i_old_to_i_new < n_new).all()

    # new_coords = np.loadtxt('shifted_unique_coords.csv', delimiter=',')
    old_coords = np.ndarray((n_old, 3), dtype=new_coords.dtype)

    for i_old in range(n_old):
        i_new = i_old_to_i_new[i_old]
        old_coords[i_old] = new_coords[i_new]

    multiple_points = old_coords
    np.savetxt('multiple_points.csv', multiple_points, delimiter=',')
    return multiple_points


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


def project_points(cad_file: str, inch_csv: str, comm_rank: int, comm_size: int) -> np.ndarray:
    log = open(f'log_{comm_rank}.txt', 'w')

    log.write(f'reading CAD file on rank {comm_rank}\n')
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(cad_file)
    if status != 1:
        raise Exception("Error reading STEP file.")
    step_reader.TransferRoots()
    # one_shape = step_reader.OneShape()
    one_shape = TopoDS_Shape()
    for i_shape in range(step_reader.NbShapes()):
        shape = step_reader.Shape(1 + i_shape)
        log.write(f'Shape[{i_shape}] = {shape}\n')
        faces = extract_faces(shape)
        log.write(f'n_face = {len(faces)}\n')
        if len(faces) > 0:
            one_shape = remove_one_face(shape, faces[212])  # ignore the symmetry plane
            log.write(f'n_face in new_shape = {len(extract_faces(one_shape))}\n')
    log.write(f'reading CSV file on rank {comm_rank}\n')
    old_coords = np.loadtxt(inch_csv, delimiter=',')
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
        log.write(f'[{(i_local + 1) / n_node_local:.2f}] i = {i_global}, d = {distance:3.1e}\n')
        log.flush()

    new_coords /= 25.4  # now, new_coords in inch

    np.savetxt(f'new_coords_{comm_rank}.csv', new_coords, delimiter=',')
    end = timer()
    log.write(f'wall-clock cost = {end - start}')
    log.close()

    return new_coords


def merge_new_coords(comm_size: int, n_unique_point: int) -> np.ndarray:
    first = 0
    merged_new_coords = np.ndarray((n_unique_point, 3))
    for comm_rank in range(comm_size):
        part = np.loadtxt(f'new_coords_{comm_rank}.csv', delimiter=',')
        last = first + len(part)
        merged_new_coords[first : last, :] = part
        print(f'merge rank = {comm_rank}: [{first}, {last})')
        first = last
    assert first == n_unique_point, (first, n_unique_point)
    np.savetxt('new_coords_unique.csv', merged_new_coords, delimiter=',')
    return merged_new_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}.py',
        description = 'Project points in CGNS file to the surface of a CAD model.')
    parser.add_argument('--cad', type=str, help='the STEP file of the CAD model')
    parser.add_argument('--mesh', type=str, help='the CGNS file of the points to be projected')
    parser.add_argument('--output', type=str, help='the CSV file of the projected points')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.verbose and rank == 0:
        print(args)

    if rank == 0:
        cgns, zone, zone_size = getUniqueZone(args.mesh)
        print(zone_size)
        _, xyz, x, y, z = readPoints(zone, zone_size)
        assert zone_size[0][0] == len(xyz) == len(x) == len(y) == len(z)
        unique_points = multiple_to_unique(xyz)
    comm.Barrier()

    project_points(args.cad, 'unique_points.csv', rank, size)
    comm.Barrier()

    if rank == 0:
        merged_new_coords = merge_new_coords(size, len(unique_points))
        new_coords_multiple = unique_to_multiple(merged_new_coords)
        np.savetxt('new_coords_multiple.csv', new_coords_multiple, delimiter=',')
