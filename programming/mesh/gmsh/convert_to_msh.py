import argparse
import sys

import gmsh


def just_print_entities(prefix=''):
    entities = gmsh.model.get_entities()
    for entity in entities:
        dim, tag = entity
        name = gmsh.model.get_entity_name(dim, tag)
        print(f'{prefix} {entity} "{name}"')


def just_print_groups(prefix=''):
    groups = gmsh.model.get_physical_groups()
    for group in groups:
        dim, tag = group
        name = gmsh.model.get_physical_name(dim, tag)
        print(f'{prefix} {group} "{name}"')


def map_entity_to_group_name() -> dict:
    prefix = '[map_entity_to_group_name]'
    entity_to_group_name = dict()
    groups = gmsh.model.get_physical_groups()
    for group in groups:
        dim, tag = group
        name = gmsh.model.get_physical_name(dim, tag)
        print(f'{prefix} Group {group} "{name}"')
        tags = gmsh.model.get_entities_for_physical_group(dim, tag)
        for tag in tags:
            entity = (dim, int(tag))
            entity_to_group_name[entity] = name
            name = gmsh.model.get_entity_name(dim, tag)
            print(f'{prefix} -> Entity {entity} "{name}"')
    print(f'\n{prefix} entity_to_group_name = {entity_to_group_name}')
    return entity_to_group_name


def update_entities() -> dict:
    """Re-label entities from 1, 2d entities first, then 3d entities.
    """
    prefix = '[update_entities]'
    print(f'\nat the beginning of {prefix}')
    just_print_entities(prefix)
    new_entity_to_old_entity = dict()
    new_tag = 1
    for d in (2, 3):
        old_entities = gmsh.model.get_entities(d)
        for old_entity in old_entities:
            dim, old_tag = old_entity
            assert dim == d
            name = gmsh.model.get_entity_name(dim, old_tag)
            if old_tag != new_tag:
                gmsh.model.set_tag(dim, old_tag, new_tag)
                gmsh.model.set_entity_name(dim, new_tag, name)
            new_entity_to_old_entity[(dim, new_tag)] = old_entity
            new_tag += 1
    print(f'\n{prefix} new_entity_to_old_entity = {new_entity_to_old_entity}')
    print(f'\nat the end of {prefix}')
    just_print_entities(prefix)
    return new_entity_to_old_entity


def update_groups(old_entity_to_group_name: dict, new_entity_to_old_entity: dict):
    prefix = '[update_groups]'
    print(f'\n{prefix} old_entity_to_group_name = {old_entity_to_group_name}')
    print(f'\n{prefix} new_entity_to_old_entity = {new_entity_to_old_entity}')
    print(f'\nat the beginning of {prefix}')
    just_print_groups(prefix)
    gmsh.model.remove_physical_groups()
    names = old_entity_to_group_name.values()
    for name in names:
        gmsh.model.remove_physical_name(name)
    print(f'\nafter removing all groups')
    just_print_groups(prefix)
    print(f'\nadding new groups')
    for new_entity, old_entity in new_entity_to_old_entity.items():
        dim, new_tag = new_entity
        if old_entity in old_entity_to_group_name:
            name = old_entity_to_group_name[old_entity]
        else:
            name = gmsh.model.get_entity_name(dim, new_tag)
        # 'NacelleFin' conflicts with 'Nacelle' in FLEXI
        if name == 'NacelleFin':
            name = 'Fin'
        print(f'{prefix} add_physical_group({dim}, {[new_tag]}, {new_tag}, {name})')
        group_tag = gmsh.model.add_physical_group(dim, [new_tag], new_tag, name)
        assert new_tag == group_tag, (new_tag, group_tag)
        group_name = gmsh.model.get_physical_name(dim, group_tag)
        assert name == group_name, (old_entity, new_entity, (dim, new_tag), name, group_name)
    print(f'\nat the end of {prefix}')
    just_print_groups(prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = f'python3 {sys.argv[0]}',
        description = 'Convert a mesh (usually in other format) to the MSH format.')
    parser.add_argument('--mesh', type=str, help='the mesh to be converted')
    parser.add_argument('--format', choices=['msh1', 'msh2', 'msh22', 'msh3', 'msh4', 'msh40', 'msh41', 'msh'], default='msh2',
        help='the output format')
    parser.add_argument('--binary', default=False, action='store_true',
        help='output in binary format')
    args = parser.parse_args()
    print(args)

    gmsh.initialize()

    if args.binary:
        gmsh.option.set_number('Mesh.Binary', 1)

    input = args.mesh
    gmsh.merge(input)

    # In Gmsh, index start at 1 with empty name to account for unclassified elements.
    # See `GModel::readCGNS` in `src/geo/GModelIO_CGNS.cpp` for details.
    print()
    just_print_entities('[old entities]')
    print()
    just_print_groups('[old groups]')

    old_entity_to_group_name = map_entity_to_group_name()
    new_entity_to_old_entity = update_entities()
    update_groups(old_entity_to_group_name, new_entity_to_old_entity)

    print()
    just_print_entities('[new entities]')
    print()
    just_print_groups('[new groups]')

    # gmsh.option.set_number('Mesh.SaveAll', 1)
    dot_pos = input.rfind('.')
    output = f'{input[:dot_pos]}.{args.format}'
    print(f'\nwriting to {output} ...')
    gmsh.write(output)

    gmsh.finalize()
