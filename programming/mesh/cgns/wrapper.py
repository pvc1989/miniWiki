import CGNS.MAP as cgm
import CGNS.PAT.cgnslib as cgl
import CGNS.PAT.cgnsutils as cgu
import CGNS.PAT.cgnskeywords as cgk


def getChildrenByType(node, type):
    children = []
    all_children = cgu.getNextChildSortByType(node)
    for child in all_children:
        if cgu.checkNodeType(child, type):
            children.append(child)
    return children


def getUniqueChildByType(node, type):
    children = getChildrenByType(node, type)
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


if __name__ == '__main__':
    pass
