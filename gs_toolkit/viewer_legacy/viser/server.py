""" Core Viser Server """


from __future__ import annotations

from typing import Callable, Type

import viser.infra
from typing_extensions import override

from .message_api import MessageApi
from .messages import GuiUpdateMessage, GSTKMessage


class ViserServer(MessageApi):
    """Core visualization server. Communicates asynchronously with client applications
    via websocket connections.

    By default, all messages (eg `server.add_frame()`) are broadcasted to all connected
    clients.

    To send messages to an individual client, we can grab a client ID -> handle mapping
    via `server.get_clients()`, and then call `client.add_frame()` on the handle.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
    ):
        super().__init__()

        self._ws_server = viser.infra.Server(
            host, port, http_server_root=None, verbose=False
        )
        self._ws_server.register_handler(GuiUpdateMessage, self._handle_gui_updates)
        self._ws_server.start()

    @override
    def _queue(self, message: GSTKMessage) -> None:
        """Implements message enqueue required by MessageApi.

        Pushes a message onto a broadcast queue."""
        self._ws_server.broadcast(message)

    def register_handler(
        self, message_type: Type[GSTKMessage], handler: Callable[[GSTKMessage], None]
    ) -> None:
        """Register a handler for incoming messages.

        Args:
            handler: A function that takes a message, and does something
        """
        self._ws_server.register_handler(
            message_type, lambda client_id, msg: handler(msg)
        )
