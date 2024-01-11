"""
For tree logic code.
"""

from collections import defaultdict
from typing import Callable


class Node(defaultdict):
    """
    The base class Node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_tree(node_class: Callable) -> Callable:
    """
    Get a tree from a node class.
    This allows one to do tree["path"]["to"]["node"]
    and it will return a new node if it doesn't exist
    or the current node if it does.
    """
    assert isinstance(node_class(), Node)

    def tree():
        return node_class(tree)

    return tree()


def find_node(tree, path):
    if len(path) == 0:
        return tree
    else:
        return find_node(tree[path[0]], path[1:])


def set_node_value(tree, path, value):
    if len(path) == 0:
        tree.data = value
    else:
        set_node_value(tree[path[0]], path[1:], value)


def walk(path, tree):
    """Walk the entire tree and return the values
    Args:
        tree: the root of the tree to start search
    """
    yield path, tree
    for k, v in tree.items():
        yield from walk(path + "/" + k, v)
