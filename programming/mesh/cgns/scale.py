import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl

import argparse
import wrapper


# mesh uses inches, while CAD uses mm
inch_to_mm = 25.4
mm_to_inch = 1 / inch_to_mm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 scale.py',
        description = 'Scale the given mesh by a multiplying factor.')
    parser.add_argument('--input', type=str, help='the CGNS file to be processed')
    parser.add_argument('--output', type=str, default=None,
        help='the CGNS file containing the scaled mesh')
    parser.add_argument('--factor', type=float, default=mm_to_inch,
        help='the scaling factor')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)

    # CGNSTree_t level
    tree, links, paths = cgm.load(args.input)

    # CGNSBase_t level
    base = wrapper.getUniqueChildByType(tree, 'CGNSBase_t')
    cell_dim, phys_dim = wrapper.getDimensions(base)

    # Zone_t level
    zone = wrapper.getUniqueChildByType(base, 'Zone_t')
    zone_name = wrapper.getNodeName(zone)
    zone_size = wrapper.getNodeData(zone)
    print('zone_size =', zone_size)
    assert zone_size.shape == (1, 3)
    n_node = zone_size[0][0]
    n_cell = zone_size[0][1]
    coords = wrapper.getChildrenByType(
        wrapper.getUniqueChildByType(zone, 'GridCoordinates_t'),
        'DataArray_t')
    X, Y, Z = 0, 1, 2
    coords_x, coords_y, coords_z = coords[X], coords[Y], coords[Z]
    values_x, values_y, values_z = coords_x[1], coords_y[1], coords_z[1]
    assert (n_node,) == values_x.shape == values_y.shape == values_z.shape

    values_x *= args.factor
    values_y *= args.factor
    values_z *= args.factor

    # write the new CGNSTree_t
    output = args.output
    if output is None:
        output = 'scaled_' + args.input
    print('writing to', output)
    cgm.save(output, tree)
