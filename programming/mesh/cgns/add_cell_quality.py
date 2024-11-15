import argparse
import sys

import numpy as np
import gmsh
import wrapper
import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl


def measure(mesh_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmsh.merge(mesh_file)
    types, cells, nodes = gmsh.model.mesh.getElements(dim=3)
    all_hexa = cells[0]
    minDetJac = gmsh.model.mesh.getElementQualities(all_hexa, 'minDetJac')
    maxDetJac = gmsh.model.mesh.getElementQualities(all_hexa, 'maxDetJac')
    return minDetJac, maxDetJac, maxDetJac / minDetJac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Measure the quality of each cell.')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    args = parser.parse_args()
    gmsh.initialize()
    min_det_jac, max_det_jac, ratio_det_jac = measure(args.mesh)
    gmsh.finalize()

    # read the cgns file
    cgns, zone, zone_size = wrapper.getUniqueZone(args.mesh)
    # add the cell qualities as fields of cell data
    cell_data = cgl.newFlowSolution(zone, 'CellQualities', 'CellCenter')
    cgl.newDataArray(cell_data, 'MinDetJac', min_det_jac)
    cgl.newDataArray(cell_data, 'MaxDetJac', max_det_jac)
    cgl.newDataArray(cell_data, 'RatioDetJac', ratio_det_jac)
    # write to the original file
    output = args.mesh[:-5] + '_with_quality.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
