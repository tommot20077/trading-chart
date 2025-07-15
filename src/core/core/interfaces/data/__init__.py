# ABOUTME: Data interfaces package exports
# ABOUTME: Exports abstract classes for data conversion and data provision

from .converter import AbstractDataConverter
from .provider import AbstractDataProvider

__all__ = [
    "AbstractDataConverter",
    "AbstractDataProvider",
]
