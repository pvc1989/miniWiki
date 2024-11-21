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
    volume = gmsh.model.mesh.getElementQualities(all_hexa, 'volume')
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
    # get the existing FlowSolution_t<CellCenter> or create a new one if not found
    cell_data = None
    solutions = wrapper.getChildrenByType(zone, 'FlowSolution_t')
    for solution in solutions:
        location = wrapper.getUniqueChildByType(solution, 'GridLocation_t')
        location_str = wrapper.arr2str(wrapper.getNodeData(location))
        if location_str == 'CellCenter':
            cell_data = solution
    # add the cell qualities as fields of cell data
    if cell_data is None:
        cell_data = cgl.newFlowSolution(zone, 'CellQualities', 'CellCenter')
    cgl.newDataArray(cell_data, 'MinDetJac', min_det_jac)
    cgl.newDataArray(cell_data, 'MaxDetJac', max_det_jac)
    cgl.newDataArray(cell_data, 'Volume', volume)
    # write to the original file
    output = args.mesh[:-5] + '_with_quality.cgns'
    print(f'writing to {output} ...')
    cgm.save(output, cgns)
