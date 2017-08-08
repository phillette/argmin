"""Parsing various data structures to native tree classes."""
from tree_batch import models


class Queue:
    def __init__(self):
        self.data = []

    def empty(self):
        return len(self.data) == 0

    def push(self, token, level):
        self.data.append((token, level))

    def pop(self):
        token, level = self.data[0]
        del self.data[0]
        return token, level


def doc_to_tree(doc):
    nodes = []
    q = Queue()
    head = next(t for t in doc if t.head == t)
    q.push(head, 0)
    while not q.empty():
        token, level = q.pop()
        node = token_to_node(token, level)
        nodes.append(node)
        for child in token.children:
            q.push(child, level + 1)
    return models.Tree(nodes)


def token_to_node(token, level):
    return models.Node(
        tag=token.tag_,
        pos=token.pos_,
        token=token.text,
        ix=token.i,
        parent_ix=token.head.i if token.head.i != token.i else -1,
        relationship=token.dep_,
        text_ix=token.i,
        level=level,
        is_leaf=len(list(token.children)) == 0)
