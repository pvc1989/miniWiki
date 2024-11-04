import sys
import numpy as np
import wrapper

if __name__ == "__main__":
    tree, zone, zone_size = wrapper.getUniqueZone(sys.argv[1])
    point_arr, _, _, _ = wrapper.readPoints(zone, zone_size)
    np.save(f'{sys.argv[1]}.npy', point_arr)
