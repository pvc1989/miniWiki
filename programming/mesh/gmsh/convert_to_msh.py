import argparse
import sys

import gmsh


def print_physical_groups() -> dict:
    print()
    entity_to_group = dict()
    groups = gmsh.model.get_physical_groups()
    for group in groups:
        dim, tag = group
        name = gmsh.model.get_physical_name(dim, tag)
        print(f'{dim}d Physical Group {group} "{name}"')
        tags = gmsh.model.get_entities_for_physical_group(dim, tag)
        for tag in tags:
            entity = (dim, int(tag))
            entity_to_group[entity] = group
            name = gmsh.model.get_entity_name(dim, tag)
            print(f'  {dim}d Elementary Entity {entity} "{name}"')
    return entity_to_group


def print_elementary_entities(entity_to_group: dict):
    print()
    entities = gmsh.model.get_entities()
    for entity in entities:
        dim, tag = entity
        name = gmsh.model.get_entity_name(dim, tag)
        # group = gmsh.model.get_physical_groups_for_entity(dim, tag)
        print(f'{dim}d Elementary Entity {entity} "{name}"')
        if entity in entity_to_group:
            group = entity_to_group[entity]
            dim, tag = group
            name = gmsh.model.get_physical_name(dim, tag)
            print(f'  {dim}d Physical Group {group} "{name}" found')
        else:
            tag = gmsh.model.add_physical_group(dim, [tag], tag, str(entity))
            name = gmsh.model.get_physical_name(dim, tag)
            group = (dim, tag)
            print(f'  {dim}d Physical Group {group} "{name}" created')


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

    entity_to_group = print_physical_groups()
    print_elementary_entities(entity_to_group)
    entity_to_group = print_physical_groups()

    dot_pos = input.rfind('.')
    output = f'{input[:dot_pos]}.msh'

    print(output)
    gmsh.write(output)

    gmsh.finalize()
