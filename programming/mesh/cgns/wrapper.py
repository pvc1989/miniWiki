import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk
import sys


def getChildrenByType(node, cgns_type):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if cgu.checkNodeType(child, cgns_type):
            children.append(child)
    return children


def getChildrenByName(node, name):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if getNodeName(child) == name:
            children.append(child)
    return children


def getUniqueChildByType(node, cgns_type):
    children = getChildrenByType(node, cgns_type)
    if 1 == len(children):
        return children[0]
    elif 0 == len(children):
        return None
    else:
        assert False


def getUniqueChildByName(node, name):
    children = getChildrenByName(node, name)
    if 1 == len(children):
        return children[0]
    elif 0 == len(children):
        return None
    else:
        assert False


def getNodeName(node) -> str:
    return node[0]


def getNodeData(node):
    return node[1]


def getNodeLabel(node) -> str:
    return node[-1]


def getDimensions(base) -> tuple[int, int]:
    assert 'CGNSBase_t' == getNodeLabel(base)
    return base[1][0], base[1][1]


def printInfo(file_name: str):
    cgns, _, _ = cgm.load(file_name)
    print(cgns)


if __name__ == '__main__':
    printInfo(sys.argv[1])
