# ABOUTME: Middleware interfaces package for the Core layer
# ABOUTME: Exports abstract interfaces for middleware and pipeline management

from .middleware import AbstractMiddleware
from .pipeline import AbstractMiddlewarePipeline

__all__ = ["AbstractMiddleware", "AbstractMiddlewarePipeline"]
