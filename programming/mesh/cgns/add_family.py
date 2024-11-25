import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import numpy as np
import sys
import pycgns_wrapper


def copy(old_file):
    tree, links, paths = cgm.load(old_file)
    new_file = 'family_copied.cgns'
    cgm.save(new_file, tree)
    return new_file


def read(file):
    tree, links, paths = cgm.load(file)
    families = cgu.getAllFamilies(tree)
    print('Families:', families)
    for family in families:
        bc_list = cgu.getBCFromFamily(tree, family)
        print(family, bc_list)


def add(old_file, families):
    tree, links, paths = cgm.load(old_file)
    base = pycgns_wrapper.getUniqueChildByType(tree, 'CGNSBase_t')
    for family in families:
        cgl.newFamily(base, family)
    for family in families:
        bc_list = cgu.getBCFromFamily(tree, family)
        print(family, bc_list)
    new_file = 'family_added.cgns'
    cgm.save(new_file, tree)
    return new_file


def rename(old_file):
    tree, links, paths = cgm.load(old_file)
    base = pycgns_wrapper.getUniqueChildByType(tree, 'CGNSBase_t')
    families = pycgns_wrapper.getChildrenByType(base, 'Family_t')
    for family in families:
        family_name = pycgns_wrapper.getNodeName(family)
        print(f'In (Family_t) {family_name}:')
        path = cgu.getPathFromRoot(tree, family)
        children = cgu.getChildrenByPath(tree, path)
        for child in children:
            if not cgu.checkNodeType(child, 'FamilyName_t'):
                continue
            old_name = pycgns_wrapper.getNodeName(child)
            print(f'  Rename (FamilyName_t) {old_name} to FamilyParent')
            cgu.setChildName(family, old_name, 'FamilyParent')
    new_file = 'family_renamed.cgns'
    cgm.save(new_file, tree)


def getFamilyToRoot(families) -> dict:
    family_to_root = dict()
    for family in families:
        children = pycgns_wrapper.getChildrenByType(family, 'FamilyName_t')
        if 1 == len(children):
            child = children[0]
            family_to_root[pycgns_wrapper.getNodeName(family)] = cgu.getValueAsString(child)
        elif 0 == len(children):
            name = pycgns_wrapper.getNodeName(family)
            family_to_root[name] = name
        else:
            assert False
    print(family_to_root)
    return family_to_root


def compress(old_file):
    tree, links, paths = cgm.load(old_file)
    base = pycgns_wrapper.getUniqueChildByType(tree, 'CGNSBase_t')
    families = pycgns_wrapper.getChildrenByType(base, 'Family_t')
    family_to_root = getFamilyToRoot(families)
    for family_name, root_name in family_to_root.items():
        if family_name == root_name:
            print('Keep (Family_t)', family_name)
        else:
            print('Remove (Family_t)', family_name)
            cgu.removeChildByName(base, family_name)
    zones = pycgns_wrapper.getChildrenByType(base, 'Zone_t')
    for zone in zones:
        bc = pycgns_wrapper.getUniqueChildByType(zone, 'ZoneBC_t')
        bc_list = pycgns_wrapper.getChildrenByType(bc, 'BC_t')
        for bc in bc_list:
            bc_name = pycgns_wrapper.getNodeName(bc)
            print(f'In (BC_t) {bc_name}:')
            family = pycgns_wrapper.getUniqueChildByType(bc, 'FamilyName_t')
            family_name = cgu.getValueAsString(family)
            root_name = family_to_root[family_name]
            print(f'  Change (FamilyName_t) {family_name} to {root_name}')
            cgu.setValue(family, cgu.setStringAsArray(root_name))
    new_file = 'family_compressed.cgns'
    cgm.save(new_file, tree)
    return new_file


if __name__ == '__main__':
    old = sys.argv[1]
    copied = copy(old)
    read(copied)
    added = add(copied, ['Fluid', 'Wall', 'Left', 'Right'])
    renamed = rename(added)
    compressed = compress(added)
