"""Path class
"""


from typing import Tuple

UNICODE = str


class Path:
    """Path class

    Args:
        entries: component parts of the path
    """

    __slots__ = ["entries"]

    def __init__(self, entries: Tuple = tuple()):
        self.entries = entries

    def append(self, other: str) -> "Path":
        """Method that appends a new component and returns new Path

        Args:
            other: _description_
        """
        new_path = self.entries
        for element in other.split("/"):
            if len(element) == 0:
                new_path = tuple()
            else:
                new_path = new_path + (element,)
        return Path(new_path)

    def lower(self):
        """Convert path object to serializable format"""
        return UNICODE("/" + "/".join(self.entries))

    def __hash__(self):
        return hash(self.entries)

    def __eq__(self, other):
        return self.entries == other.entries
