import gmsh
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 mesh2d.py',
        description = 'Mesh the surfaces of a CAD model.')
    parser.add_argument('--input', type=str, help='the STEP file to be meshed')
    parser.add_argument('--output', type=str, default='surface_stl',
        help='name of the CGNS file containing the surface mesh')
    parser.add_argument('--h_max', type=float, default=200,
        help='maximum edge length')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        print(args)
    
    gmsh.initialize(sys.argv)
    gmsh.model.add(args.output)
    v = gmsh.model.occ.import_shapes(args.input)
    gmsh.model.occ.synchronize()

    cad_points = gmsh.model.getEntities(0)
    print('n_cad_points =', len(cad_points))
    gmsh.model.mesh.set_size(cad_points, args.h_max)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write(f'{args.output}.cgns')

    if args.verbose:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
            v[0][0], v[0][1])
        print('bounding box =')
        print('  ', xmin, xmax)
        print('  ', ymin, ymax)
        print('  ', zmin, zmax)

    gmsh.finalize()
