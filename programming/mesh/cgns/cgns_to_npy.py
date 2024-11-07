import sys
import numpy as np
import wrapper
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description='Read points from a CGNS file and write them into an NPY file.')
    parser.add_argument('--input', type=str, help='the CGNS file containing the points')
    args = parser.parse_args()
    print(args)

    tree, zone, zone_size = wrapper.getUniqueZone(args.input)
    point_arr, _, _, _ = wrapper.readPoints(zone, zone_size)
    output = f'{args.input[:-5]}.npy'
    print(f'writing to "{output}" ...')
    np.save(output, point_arr)
