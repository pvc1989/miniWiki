import sys
import numpy as np
from check_nodes import getUniqueZone, readPoints

if __name__ == "__main__":
    tree, zone, zone_size = getUniqueZone(sys.argv[1])
    point_dict, point_arr, coords_x, coords_y, coords_z = readPoints(zone, zone_size)
    np.save(f'{sys.argv[1]}.npy', point_arr)
