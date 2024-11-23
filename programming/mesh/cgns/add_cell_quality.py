import argparse
import sys
import time

import numpy as np
import gmsh
import wrapper
import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl


def measure(mesh_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmsh.merge(mesh_file)
    types, cells, nodes = gmsh.model.mesh.getElements(dim=3)
    all_hexa = cells[0]
    start = time.time()
    minDetJac = gmsh.model.mesh.getElementQualities(all_hexa, 'minDetJac')
    end = time.time()
    print(f'minDetJac costs: {end - start:.2f}s')
    start = time.time()
    maxDetJac = gmsh.model.mesh.getElementQualities(all_hexa, 'maxDetJac')
    end = time.time()
    print(f'maxDetJac costs: {end - start:.2f}s')
    start = time.time()
    volume = gmsh.model.mesh.getElementQualities(all_hexa, 'volume')
    end = time.time()
    print(f'volume costs: {end - start:.2f}s')
    start = time.time()
    return minDetJac, maxDetJac, volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Measure the quality of each cell.')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    args = parser.parse_args()
    gmsh.initialize()
    min_det_jac, max_det_jac, volume = measure(args.mesh)
    gmsh.finalize()

    # read the cgns file
    cgns, zone, zone_size = wrapper.getUniqueZone(args.mesh)
    cell_data = wrapper.getSolutionByLocation(zone,
        'CellCenter', 'CellQualities')
    cgl.newDataArray(cell_data, 'MinDetJac', min_det_jac)
    cgl.newDataArray(cell_data, 'MaxDetJac', max_det_jac)
    cgl.newDataArray(cell_data, 'Volume', volume)
    # write to the original file
    output = args.mesh[:-5] + '_with_quality.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
