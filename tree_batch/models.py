"""Tree data structures."""


def cumsum(seq):
    r, s = [], 0
    for e in seq:
        l = len(e)
        r.append(l + s)
        s += l
    return r


class Forest:
    def __init__(self, trees):
        # The trees to be ordered if text order is desired to be kept.
        self.trees = trees
        self.max_level = max([t.max_level for t in trees])
        cumsums = cumsum([t.nodes for t in trees])
        for ix_tree in range(len(trees)):
            for node in trees[ix_tree].nodes:
                offset = cumsums[ix_tree - 1] if ix_tree > 0 else 0
                node.forest_ix = node.ix + offset
                node.forest_text_ix = node.text_ix + offset
        self.levels = {}
        for l in range(self.max_level + 1):
            self.levels[l] = []
            for tree in [t for t in trees if l in t.levels.keys()]:
                self.levels[l] += tree.levels[l]


class Node:
    def __init__(self, tag, pos, token, ix, parent_ix, relationship, text_ix,
                 level, is_leaf):
        self.tag = tag
        self.pos = pos
        self.token = token
        self.ix = ix
        self.parent_ix = parent_ix
        self.relationship = relationship
        self.text_ix = text_ix
        self.level = level
        self.is_leaf = is_leaf
        self.has_token = token is not None
        # These ones to be set in case of forestry
        self.forest_ix = None
        self.forest_text_ix = None

    def __repr__(self):
        return '\n'.join(['%s: %s' % (key, value)
                          for key, value
                          in self.__dict__.items()])


class Tree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.size = len(nodes)
        self.max_level = max([n.level for n in nodes])
        self.levels = dict(zip(
            range(self.max_level+1),
            [[n for n in nodes if n.level == l]
             for l in range(self.max_level+1)]))
