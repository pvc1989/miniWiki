import argparse

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import pycgns_wrapper
from pycgns_wrapper import X, Y, Z


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 merge_zones.py',
        description = 'Merge multiple structured `Zone_t`s into a single unstructured `Zone_t`.')
    parser.add_argument('--input', type=str, help='the CGNS file to be merged')
    parser.add_argument('--output', type=str, default='new.cgns',
        help='the merged CGNS file')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    # CGNSTree_t level
    old_tree, links, paths = cgm.load(args.input)
    new_tree = cgl.newCGNSTree()

    # CGNSBase_t level
    old_base = pycgns_wrapper.getUniqueChildByType(old_tree, 'CGNSBase_t')
    cell_dim, phys_dim = pycgns_wrapper.getDimensions(old_base)
    assert cell_dim == phys_dim == 3
    new_base = cgl.newCGNSBase(new_tree,
        pycgns_wrapper.getNodeName(old_base), cell_dim, phys_dim)

    # Zone_t level
    old_zones = pycgns_wrapper.getChildrenByType(old_base, 'Zone_t')
    xyz_to_count = dict()
    I, J, K = X, Y, Z
    n_node = 0
    for old_zone in old_zones:
        zone_name = pycgns_wrapper.getNodeName(old_zone)
        zone_size = old_zone[1]
        assert zone_size.shape == (3, 3)
        n_node_i, n_cell_i, _ = zone_size[I]
        n_node_j, n_cell_j, _ = zone_size[J]
        n_node_k, n_cell_k, _ = zone_size[K]
        coords = pycgns_wrapper.getChildrenByType(
            pycgns_wrapper.getUniqueChildByType(old_zone, 'GridCoordinates_t'),
            'DataArray_t')
        coords_x, coords_y, coords_z = coords[X], coords[Y], coords[Z]
        assert 'CoordinateX' == pycgns_wrapper.getNodeName(coords_x)
        assert 'CoordinateY' == pycgns_wrapper.getNodeName(coords_y)
        assert 'CoordinateZ' == pycgns_wrapper.getNodeName(coords_z)
        values_x, values_y, values_z = coords_x[1], coords_y[1], coords_z[1]
        assert (n_node_i, n_node_j, n_node_k) == \
            values_x.shape == values_y.shape == values_z.shape
        for i in range(n_node_i):
            for j in range(n_node_j):
                for k in range(n_node_k):
                    xyz = (values_x[i][j][k], values_y[i][j][k], values_z[i][j][k])
                    if xyz not in xyz_to_count:
                        xyz_to_count[xyz] = 0
                    xyz_to_count[xyz] += 1
        n_node += n_node_i * n_node_j * n_node_k
    print('before merging nodes: n_node =', n_node)
    print('after merging nodes: n_node =', len(xyz_to_count))
    n_node = len(xyz_to_count)
    values_x = np.ndarray(n_node)
    values_y = np.ndarray(n_node)
    values_z = np.ndarray(n_node)
    i_node = 0
    for xyz in xyz_to_count.keys():
        values_x[i_node] = xyz[X]
        values_y[i_node] = xyz[Y]
        values_z[i_node] = xyz[Z]
        i_node += 1
    n_cell = 0
    n_rind = 0
    new_zone = cgl.newZone(new_base, 'Zone 1', zsize=np.array([[n_node], [n_cell], [n_rind]]), ztype='Unstructured')
    new_coords = cgl.newGridCoordinates(new_zone, 'GridCoordinates')
    new_coords_x = cgl.newDataArray(new_coords, 'CoordinateX', values_x)
    new_coords_y = cgl.newDataArray(new_coords, 'CoordinateY', values_y)
    new_coords_z = cgl.newDataArray(new_coords, 'CoordinateZ', values_z)
    cgm.save(args.output, new_tree)
