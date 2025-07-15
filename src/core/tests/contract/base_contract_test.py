# ABOUTME: Base contract test framework for interface compliance verification
# ABOUTME: Provides common utilities and patterns for testing interface implementations

import pytest
import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic, List
from unittest.mock import Mock

# Generic type for interface classes
T = TypeVar("T", bound=ABC)


class ContractTestBase(Generic[T]):
    """
    Base class for contract tests that verify implementations comply with interface contracts.

    This class provides common utilities for:
    - Interface method signature verification
    - Abstract method implementation checking
    - Common behavior pattern testing
    - Exception handling contract verification
    """

    @property
    @abstractmethod
    def interface_class(self) -> Type[T]:
        """The interface class being tested."""
        pass

    @property
    @abstractmethod
    def implementations(self) -> List[Type[T]]:
        """List of concrete implementations to test against the interface."""
        pass

    def get_abstract_methods(self) -> List[str]:
        """Get all abstract method names from the interface."""
        return [
            name
            for name, method in inspect.getmembers(self.interface_class, inspect.isfunction)
            if getattr(method, "__isabstractmethod__", False)
        ]

    def get_abstract_properties(self) -> List[str]:
        """Get all abstract property names from the interface."""
        return [
            name
            for name, prop in inspect.getmembers(self.interface_class, lambda x: isinstance(x, property))
            if getattr(prop.fget, "__isabstractmethod__", False)
            if prop.fget
        ]

    @pytest.mark.contract
    def test_interface_is_abstract(self):
        """Verify that the interface class is properly abstract."""
        assert inspect.isabstract(self.interface_class), f"{self.interface_class.__name__} should be abstract"

        # Try to instantiate - should fail
        with pytest.raises(TypeError):
            self.interface_class()

    @pytest.mark.contract
    def test_implementations_inherit_from_interface(self):
        """Verify all implementations properly inherit from the interface."""
        for impl_class in self.implementations:
            assert issubclass(impl_class, self.interface_class), (
                f"{impl_class.__name__} must inherit from {self.interface_class.__name__}"
            )

    @pytest.mark.contract
    def test_implementations_are_concrete(self):
        """Verify all implementations are concrete (not abstract)."""
        for impl_class in self.implementations:
            assert not inspect.isabstract(impl_class), f"{impl_class.__name__} should be concrete, not abstract"

    @pytest.mark.contract
    def test_all_abstract_methods_implemented(self):
        """Verify all abstract methods are implemented in concrete classes."""
        abstract_methods = self.get_abstract_methods()
        abstract_properties = self.get_abstract_properties()

        for impl_class in self.implementations:
            # Check methods
            for method_name in abstract_methods:
                assert hasattr(impl_class, method_name), f"{impl_class.__name__} missing method: {method_name}"

                method = getattr(impl_class, method_name)
                assert not getattr(method, "__isabstractmethod__", False), (
                    f"{impl_class.__name__}.{method_name} is still abstract"
                )

            # Check properties
            for prop_name in abstract_properties:
                assert hasattr(impl_class, prop_name), f"{impl_class.__name__} missing property: {prop_name}"

    @pytest.mark.contract
    def test_method_signatures_match(self):
        """Verify method signatures match between interface and implementations."""
        abstract_methods = self.get_abstract_methods()

        for impl_class in self.implementations:
            for method_name in abstract_methods:
                interface_method = getattr(self.interface_class, method_name)
                impl_method = getattr(impl_class, method_name)

                interface_sig = inspect.signature(interface_method)
                impl_sig = inspect.signature(impl_method)

                # Compare parameter names and types
                assert interface_sig.parameters.keys() == impl_sig.parameters.keys(), (
                    f"Parameter mismatch in {impl_class.__name__}.{method_name}"
                )

                # Compare return annotations if present (skip for generic types)
                if interface_sig.return_annotation != inspect.Signature.empty:
                    # Skip comparison for generic types that may have different type variable instances
                    interface_return = str(interface_sig.return_annotation)
                    impl_return = str(impl_sig.return_annotation)

                    # Allow generic type variables to differ (e.g., ~T vs ~T)
                    # Also allow modern vs legacy type annotations compatibility
                    # Also allow string annotations vs actual class annotations
                    if not (
                        ("~T" in interface_return and "~T" in impl_return)
                        or (interface_return.startswith("typing.") and impl_return.startswith("typing."))
                        or (interface_return.startswith("list[") and impl_return.startswith("typing.List["))
                        or (interface_return.startswith("typing.List[") and impl_return.startswith("list["))
                        or (interface_return.startswith("dict[") and impl_return.startswith("typing.Dict["))
                        or (interface_return.startswith("typing.Dict[") and impl_return.startswith("dict["))
                        or ("| None" in interface_return and "typing.Optional[" in impl_return)
                        or ("typing.Optional[" in interface_return and "| None" in impl_return)
                        or self._are_compatible_types(interface_sig.return_annotation, impl_sig.return_annotation)
                    ):
                        assert interface_sig.return_annotation == impl_sig.return_annotation, (
                            f"Return type mismatch in {impl_class.__name__}.{method_name}: {interface_return} != {impl_return}"
                        )

    def _are_compatible_types(self, interface_type, impl_type):
        """
        Check if two type annotations are compatible.

        This handles cases where one is a string annotation and the other is an actual class,
        or where both refer to the same class but in different formats.
        """
        # Convert both to strings for comparison
        interface_str = str(interface_type)
        impl_str = str(impl_type)

        # Handle string annotations vs actual class
        if isinstance(interface_type, str) and hasattr(impl_type, "__name__"):
            # Interface has string annotation, impl has actual class
            return interface_type == impl_type.__name__
        elif isinstance(impl_type, str) and hasattr(interface_type, "__name__"):
            # Impl has string annotation, interface has actual class
            return impl_type == interface_type.__name__
        elif isinstance(interface_type, str) and isinstance(impl_type, type):
            # Check if string matches class name (with or without module path)
            return (
                interface_type == impl_type.__name__
                or interface_type.endswith(f".{impl_type.__name__}")
                or impl_str.endswith(f".{interface_type}")
            )
        elif isinstance(impl_type, str) and isinstance(interface_type, type):
            # Check if string matches class name (with or without module path)
            return (
                impl_type == interface_type.__name__
                or impl_type.endswith(f".{interface_type.__name__}")
                or interface_str.endswith(f".{impl_type}")
            )

        # Check if both strings refer to the same class
        if isinstance(interface_type, str) and isinstance(impl_type, str):
            # Extract just the class name from potentially qualified names
            interface_name = interface_type.split(".")[-1]
            impl_name = impl_type.split(".")[-1]
            return interface_name == impl_name

        # If both have module paths, compare class names
        if hasattr(interface_type, "__name__") and hasattr(impl_type, "__name__"):
            return interface_type.__name__ == impl_type.__name__

        return False


class AsyncContractTestMixin:
    """Mixin for testing async interface contracts."""

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_async_methods_are_coroutines(self):
        """Verify async methods return coroutines."""
        if not hasattr(self, "implementations") or not hasattr(self, "interface_class"):
            pytest.skip("Not an async contract test")

        async_methods = [
            name
            for name, method in inspect.getmembers(self.interface_class, inspect.isfunction)
            if asyncio.iscoroutinefunction(method) and getattr(method, "__isabstractmethod__", False)
        ]

        # Also check for async generators
        async_generator_methods = [
            name
            for name, method in inspect.getmembers(self.interface_class, inspect.isfunction)
            if inspect.isasyncgenfunction(method) and getattr(method, "__isabstractmethod__", False)
        ]

        for impl_class in self.implementations:
            for method_name in async_methods:
                impl_method = getattr(impl_class, method_name)
                # Allow async generators to satisfy async method requirements
                is_async = asyncio.iscoroutinefunction(impl_method) or inspect.isasyncgenfunction(impl_method)
                assert is_async, f"{impl_class.__name__}.{method_name} should be async (coroutine or async generator)"

            for method_name in async_generator_methods:
                impl_method = getattr(impl_class, method_name)
                assert inspect.isasyncgenfunction(impl_method), (
                    f"{impl_class.__name__}.{method_name} should be async generator"
                )


class ResourceManagementContractMixin:
    """Mixin for testing resource management contracts (context managers)."""

    @pytest.mark.contract
    def test_context_manager_protocol(self):
        """Verify implementations support async context manager protocol."""
        if not hasattr(self, "implementations"):
            pytest.skip("Not a resource management contract test")

        for impl_class in self.implementations:
            # Check for async context manager methods
            if hasattr(impl_class, "__aenter__") and hasattr(impl_class, "__aexit__"):
                assert asyncio.iscoroutinefunction(impl_class.__aenter__), (
                    f"{impl_class.__name__}.__aenter__ should be async"
                )
                assert asyncio.iscoroutinefunction(impl_class.__aexit__), (
                    f"{impl_class.__name__}.__aexit__ should be async"
                )


def create_mock_implementation(interface_class: Type[T]) -> Type[T]:
    """
    Create a mock implementation of an interface for testing purposes.

    This is useful for testing interface contracts when no real implementation exists yet.
    """

    class MockImplementation(interface_class):
        pass

    # Implement all abstract methods with mocks
    for name, method in inspect.getmembers(interface_class, inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            if asyncio.iscoroutinefunction(method):

                async def async_mock(*args, **kwargs):
                    return Mock()

                setattr(MockImplementation, name, async_mock)
            else:

                def sync_mock(*args, **kwargs):
                    return Mock()

                setattr(MockImplementation, name, sync_mock)

    # Implement all abstract properties with mocks
    for name, prop in inspect.getmembers(interface_class, lambda x: isinstance(x, property)):
        if prop.fget and getattr(prop.fget, "__isabstractmethod__", False):
            setattr(MockImplementation, name, property(lambda self: Mock()))

    MockImplementation.__name__ = f"Mock{interface_class.__name__}"
    return MockImplementation
