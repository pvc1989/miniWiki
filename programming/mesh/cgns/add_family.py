import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import numpy as np
import sys


def copy(old_file):
    tree, links, paths = cgm.load(old_file)
    new_file = 'family_copied.cgns'
    cgm.save(new_file, tree)
    return new_file


def getNodesByType(tree, type):
    paths = cgu.getPathsByTypeSet(tree, [type])
    nodes = []
    for path in paths:
        nodes.append(cgu.getNodeByPath(tree, path))
    return nodes


def read(file):
    tree, links, paths = cgm.load(file)
    for section in getNodesByType(tree, 'Elements_t'):
        print(section)
    families = cgu.getAllFamilies(tree)
    print('Families:', families)
    for family in families:
        bc_list = cgu.getBCFromFamily(tree, family)
        print(family, bc_list)


def add(old_file, families):
    tree, links, paths = cgm.load(old_file)
    paths = cgu.getPathsByTypeSet(tree, ['CGNSBase_t'])
    assert 1 == len(paths)
    base = cgu.getNodeByPath(tree, paths[0])
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
    paths = cgu.getPathsByTypeSet(tree, ['CGNSBase_t'])
    assert 1 == len(paths)
    base = cgu.getNodeByPath(tree, paths[0])
    families = getNodesByType(tree, 'Family_t')
    for family in families:
        family_name = family[0]
        print(f'In (Family_t) {family_name}:')
        path = cgu.getPathFromRoot(tree, family)
        children = cgu.getChildrenByPath(tree, path)
        for child in children:
            if not cgu.checkNodeType(child, 'FamilyName_t'):
                continue
            old_name = child[0]
            print(f'  Rename (FamilyName_t) {old_name} to FamilyParent')
            cgu.setChildName(family, old_name, 'FamilyParent')
    new_file = 'family_renamed.cgns'
    cgm.save(new_file, tree)


if __name__ == '__main__':
    old = sys.argv[1]
    copied = copy(old)
    read(copied)
    added = add(copied, ['Fluid', 'Wall', 'Left', 'Right'])
    renamed = rename(added)
