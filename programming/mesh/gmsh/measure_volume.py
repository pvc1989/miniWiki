import argparse
import sys

import gmsh
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Measure the volume of each cell.')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    args = parser.parse_args()
    print(args)

    gmsh.initialize()

    gmsh.merge(args.mesh)

    types, cells, nodes = gmsh.model.mesh.getElements(dim=3)
    all_3d_cells = cells[0]
    volumes = gmsh.model.mesh.getElementQualities(all_3d_cells, 'volume')

    print(f'max(volumes) = {np.max(volumes)}')
    print(f'min(volumes) = {np.min(volumes)}')
    print(f'sum(volumes) = {np.sum(volumes)}')

    gmsh.finalize()
