import argparse
import sys

import gmsh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Convert a mesh (usually in other format) to the MSH format.')
    parser.add_argument('--folder', type=str, help='the working folder')
    parser.add_argument('--mesh', type=str, help='the input mesh')
    parser.add_argument('--binary', default=False, action='store_true', help='output in binary format')
    args = parser.parse_args()
    print(args)

    gmsh.initialize()

    if args.binary:
        gmsh.option.set_number('Mesh.Binary', 1)

    input = f'{args.folder}/{args.mesh}'
    gmsh.merge(input)

    dot_pos = input.rfind('.')
    output = f'{input[:dot_pos]}.msh'

    print(output)
    gmsh.write(output)

    gmsh.finalize()
