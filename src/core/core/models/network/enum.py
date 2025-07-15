from enum import Enum


class ConnectionStatus(str, Enum):
    """
    Enumeration of connection status states.

    These states represent the lifecycle of a network connection.

    Attributes:
        CONNECTING (str): The connection is currently being established.
        CONNECTED (str): The connection is active and ready for use.
        DISCONNECTED (str): The connection has been lost or closed.
        RECONNECTING (str): The system is attempting to re-establish a lost connection.
        ERROR (str): The connection is in an error state.
    """

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
