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

def read(file):
    tree, links, paths = cgm.load(file)
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


if __name__ == '__main__':
    old = sys.argv[1]
    new = copy(old)
    read(new)
    add(new, ['Fluid', 'Wall', 'Left', 'Right'])
    # read(new)
