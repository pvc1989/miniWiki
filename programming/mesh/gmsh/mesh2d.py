import gmsh
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'python3 mesh2d.py',
        description = 'Mesh the surfaces of a CAD model.')
    parser.add_argument('--input', type=str, help='the STEP file to be meshed')
    parser.add_argument('--output', type=str, default='surface_stl',
        help='name (prefix) of the CGNS file containing the surface mesh')
    parser.add_argument('--h_max', type=float, default=200,
        help='maximum edge length')
    parser.add_argument('--refine', type=int, default=12,
        help='number of elements per 2 * pi rad')
    parser.add_argument('--skip', type=tuple, default=(213,),
        help='tags of faces to be skipped')
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
    gmsh.option.set_number('Mesh.MeshSizeFromCurvature', args.refine)
    gmsh.model.occ.synchronize()

    faces = gmsh.model.getEntities(2)
    tags = []
    tags_to_be_skipped = args.skip
    for dim, tag in faces:
        if tag in tags_to_be_skipped:
            continue
        else:
            tags.append(tag)
    if args.verbose:
        print(faces, 'n_face =', len(faces))
        print(tags, 'n_tag =', len(tags))
    gmsh.model.addPhysicalGroup(2, tags, name='SolidWall')
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)

    output = f'{args.output}_hmax={args.h_max:3.1e}mm_refine={args.refine}.cgns'
    print('writing to', output)
    gmsh.write(output)

    if args.verbose:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
            v[0][0], v[0][1])
        print('bounding box =')
        print('  ', xmin, xmax)
        print('  ', ymin, ymax)
        print('  ', zmin, zmax)

    gmsh.finalize()
