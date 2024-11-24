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
    det_jac_min = gmsh.model.mesh.getElementQualities(all_hexa, 'minDetJac')
    end = time.time()
    print(f'det_jac_min costs: {end - start:.2f}s')
    start = time.time()
    det_jac_max = gmsh.model.mesh.getElementQualities(all_hexa, 'maxDetJac')
    det_jac_ratio = det_jac_min / det_jac_max
    end = time.time()
    print(f'det_jac_ratio costs: {end - start:.2f}s')
    start = time.time()
    volume = gmsh.model.mesh.getElementQualities(all_hexa, 'volume')
    end = time.time()
    print(f'volume costs: {end - start:.2f}s')
    start = time.time()
    return det_jac_min, det_jac_ratio, volume


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Measure the quality of each cell.')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    args = parser.parse_args()
    gmsh.initialize()
    det_jac_min, det_jac_ratio, volume = measure(args.mesh)
    gmsh.finalize()

    print(f'{np.sum(det_jac_min <= 0)} cells have min(det(J)) < 0')
    print(f'{np.sum(det_jac_ratio <= 0.01)} cells have min(det(J)) < max(det(J)) * 0.01')
    print(f'{np.sum(volume <= 0)} cells have negative volume')

    # read the cgns file
    cgns, zone, zone_size = wrapper.getUniqueZone(args.mesh)
    cell_data = wrapper.getSolutionByLocation(zone,
        'CellCenter', 'CellQualities')
    cgl.newDataArray(cell_data, 'DetJacMin', det_jac_min)
    cgl.newDataArray(cell_data, 'DetJacRatio', det_jac_ratio)
    cgl.newDataArray(cell_data, 'Volume', volume)
    # write to the original file
    output = args.mesh[:-5] + '_with_quality.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
