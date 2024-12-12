import numpy as np
import argparse
import sys

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import pycgns_wrapper


hexa_edges_local = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [0, 4], [1, 5], [2, 6], [3, 7],
    [4, 5], [5, 6], [6, 7], [7, 4],
])


def get_edges(connectivity: np.ndarray, edges_local: np.ndarray) -> np.ndarray:
    edges_global = np.ndarray(edges_local.shape, dtype=edges_local.dtype)
    for i in range(len(edges_local)):
        u, v = edges_local[i]
        edges_global[i] = connectivity[u], connectivity[v]
    return edges_global


def get_min_length(points: np.ndarray, edges: np.ndarray) -> float:
    l_min = 1e100
    for i, j in edges:
        l = np.linalg.norm(points[i] - points[j])
        l_min = np.minimum(l_min, l)
    return l_min


def get_aspect_ratio(points: np.ndarray, edges: np.ndarray) -> float:
    l_min, l_max = 1e100, 0.0
    for i, j in edges:
        l = np.linalg.norm(points[i] - points[j])
        l_min = np.minimum(l_min, l)
        l_max = np.maximum(l_max, l)
    return l_max / l_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Get the aspect ratio for each cell and write them as a field on cells.')
    parser.add_argument('--mesh', type=str, help='the CGNS file to be modified')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    cgns, zone, zone_size = pycgns_wrapper.getUniqueZone(args.mesh)
    points, _, _, _ = pycgns_wrapper.readPoints(zone, zone_size)

    # load connectivity
    # TODO(PVC): support multi sections
    n_cell = zone_size[0][1]
    section = pycgns_wrapper.getUniqueChildByType(zone, 'Elements_t')
    element_type = pycgns_wrapper.getNodeData(section)
    assert element_type[0] == 17  # HEXA_8
    connectivity = pycgns_wrapper.getNodeData(
        pycgns_wrapper.getUniqueChildByName(section, 'ElementConnectivity'))
    assert len(connectivity) == n_cell * 8

    # get the aspect ratio of each cell
    aspect_ratios = np.ndarray((n_cell,))
    min_lengths = np.ndarray((n_cell,))
    for i_cell in range(n_cell):
        if args.verbose and i_cell % 10000 == 0:
            print(i_cell, '/', n_cell)
        first = i_cell * 8
        last = first + 8
        min_lengths[i_cell] = get_min_length(points,
            get_edges(connectivity[first : last] - 1, hexa_edges_local))
        aspect_ratios[i_cell] = get_aspect_ratio(points,
            get_edges(connectivity[first : last] - 1, hexa_edges_local))

    # write the aspect ratios as a field of cell data
    point_data = cgl.newFlowSolution(zone, 'FlowSolutionCellHelper', 'CellCenter')
    cgl.newDataArray(point_data, 'AspectRatio', aspect_ratios)
    cgl.newDataArray(point_data, 'MinLength', min_lengths)
    cgm.save(args.mesh, cgns)
