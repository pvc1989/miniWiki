import sys

import numpy as np

import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu

X, Y, Z = 0, 1, 2


def getChildrenByType(node, cgns_type):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if cgu.checkNodeType(child, cgns_type):
            children.append(child)
    return children


def getChildrenByName(node, name):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if getNodeName(child) == name:
            children.append(child)
    return children


def getUniqueChildByType(node, cgns_type):
    children = getChildrenByType(node, cgns_type)
    if 1 == len(children):
        return children[0]
    elif 0 == len(children):
        return None
    else:
        assert False


def getUniqueChildByName(node, name):
    children = getChildrenByName(node, name)
    if 1 == len(children):
        return children[0]
    elif 0 == len(children):
        return None
    else:
        assert False


def getNodeName(node) -> str:
    return node[0]


def getNodeData(node):
    return node[1]


def getNodeLabel(node) -> str:
    return node[-1]


def arr2str(arr) -> str:
    """Convert a string defined as an `np.ndarray` to a standard Python `str`.

    Some CGNS files, e.g. https://hiliftpw-ftp.larc.nasa.gov/HiLiftPW3/HL-CRM_Grids/Committee_Grids/C-HLCRM_Str1to1_GridPro/FullGap/CGNS/, use an `np.ndarray` of `bytes`s rather than a `BCType_t` object to store the type of a BC.
    """
    assert isinstance(arr, np.ndarray)
    chars = []
    for row in arr:
        assert isinstance(row, bytes)
        assert 1 == len(row)
        chars.append(row.decode())
    return "".join(chars)


def getDimensions(base) -> tuple[int, int]:
    assert 'CGNSBase_t' == getNodeLabel(base)
    return base[1][0], base[1][1]


def printInfo(file_name: str):
    cgns, _, _ = cgm.load(file_name)
    print(cgns)


def getUniqueZone(filename):
    tree, _, _ = cgm.load(filename)
    base = getUniqueChildByType(tree, 'CGNSBase_t')
    zone = getUniqueChildByType(base, 'Zone_t')
    zone_size = getNodeData(zone)
    assert zone_size.shape == (1, 3)
    return tree, zone, zone_size


def readPoints(zone, zone_size):
    n_node = zone_size[0][0]
    coords = getChildrenByType(
        getUniqueChildByType(zone, 'GridCoordinates_t'), 'DataArray_t')
    from pycgns_wrapper import X, Y, Z
    coords_x, coords_y, coords_z = coords[X][1], coords[Y][1], coords[Z][1]
    assert (n_node,) == coords_x.shape == coords_y.shape == coords_z.shape
    point_arr = np.ndarray((n_node, 3))
    for i in range(n_node):
        point_arr[i] = coords_x[i], coords_y[i], coords_z[i]
    return point_arr, coords_x, coords_y, coords_z


def removeSolutionsByLocation(zone, location):
    solutions = getChildrenByType(zone, 'FlowSolution_t')
    for solution in solutions:
        location_of_this_solution = getUniqueChildByType(solution, 'GridLocation_t')
        if location_of_this_solution is None:
            location_of_this_solution = 'Vertex'
        if location_of_this_solution == location:
            print(f'removing FlowSolution_t {solution} ...')
            cgu.removeChildByName(zone, getNodeName(solution))


def getSolutionByLocation(zone, location_given, solution_name):
    """Get an existing FlowSolution_t with a given GridLocation_t or create a new one if not found.
    """
    the_solution = None
    solutions = getChildrenByType(zone, 'FlowSolution_t')
    for solution in solutions:
        location = getUniqueChildByType(solution, 'GridLocation_t')
        if location is None:
            # the `cgnslib.c` implementation of `cg_sol_write` does not create a new `GridLocation_t` node in this case
            location_found = 'Vertex'
        else:
            location_found = arr2str(getNodeData(location))
        if location_found == location_given:
            the_solution = solution
            break
    # add the cell qualities as fields of cell data
    if the_solution is None:
        the_solution = cgl.newFlowSolution(zone, solution_name, location_given)
    return the_solution


def mergePointList(xyz_list: list[np.ndarray], n_node: int, zone, zone_size):
    x_new = np.ndarray((n_node,))
    y_new = np.ndarray((n_node,))
    z_new = np.ndarray((n_node,))
    first = 0
    for xyz in xyz_list:
        last = first + len(xyz)
        x_new[first : last] = np.array(xyz[:, X])
        y_new[first : last] = np.array(xyz[:, Y])
        z_new[first : last] = np.array(xyz[:, Z])
        first = last
    assert first == n_node
    cgu.removeChildByName(zone, 'GridCoordinates')
    new_coords = cgl.newGridCoordinates(zone, 'GridCoordinates')
    cgl.newDataArray(new_coords, 'CoordinateX', x_new)
    cgl.newDataArray(new_coords, 'CoordinateY', y_new)
    cgl.newDataArray(new_coords, 'CoordinateZ', z_new)
    zone_size[0][0] = n_node


if __name__ == '__main__':
    printInfo(sys.argv[1])
