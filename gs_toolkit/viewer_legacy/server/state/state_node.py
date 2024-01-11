from gs_toolkit.viewer_legacy.server.state.node import Node


class StateNode(Node):
    """Node that holds a hierarchy of state nodes"""

    __slots__ = ["data"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = None
        self.data = None
