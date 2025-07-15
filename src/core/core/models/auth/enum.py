from enum import Enum


class Role(str, Enum):
    """
    Enum for user roles.
    """

    ADMIN = "admin"
    USER = "user"


class Permission(str, Enum):
    """
    Enum for basic permissions.
    """

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
