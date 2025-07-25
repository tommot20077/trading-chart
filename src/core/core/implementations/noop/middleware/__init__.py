# ABOUTME: NoOp middleware implementations package initialization
# ABOUTME: Provides no-operation middleware classes for testing and minimal scenarios

from .pipeline import NoOpMiddlewarePipeline

__all__ = [
    "NoOpMiddlewarePipeline",
]
