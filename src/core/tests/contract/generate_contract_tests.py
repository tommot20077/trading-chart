# ABOUTME: Automated contract test generator for interface classes
# ABOUTME: Scans interface directory and generates contract test templates

import os
import inspect
import importlib
from pathlib import Path
from typing import List, Type
from abc import ABC


def find_interface_classes() -> List[tuple[str, Type[ABC]]]:
    """Find all abstract interface classes in the core.interfaces module."""
    interfaces = []
    # Use relative path from current working directory
    interfaces_path = Path("core/interfaces")

    for root, dirs, files in os.walk(interfaces_path):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Convert file path to module path
                rel_path = Path(root).relative_to(Path("."))
                module_parts = list(rel_path.parts) + [file[:-3]]  # Remove .py
                module_name = ".".join(module_parts)

                try:
                    module = importlib.import_module(module_name)

                    # Find abstract classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            inspect.isabstract(obj)
                            and issubclass(obj, ABC)
                            and obj != ABC
                            and name.startswith("Abstract")
                        ):
                            interfaces.append((module_name, obj))

                except ImportError as e:
                    print(f"Could not import {module_name}: {e}")

    return interfaces


def generate_contract_test_template(interface_class: Type[ABC], module_name: str) -> str:
    """Generate a contract test template for an interface class."""
    class_name = interface_class.__name__
    test_class_name = f"Test{class_name.replace('Abstract', '')}Contract"

    # Determine the category (auth, storage, event, etc.)
    category = module_name.split(".")[-2] if len(module_name.split(".")) > 2 else "misc"

    # Check if interface has async methods
    has_async = any(
        inspect.iscoroutinefunction(method)
        for name, method in inspect.getmembers(interface_class, inspect.isfunction)
        if getattr(method, "__isabstractmethod__", False)
    )

    # Check if interface has context manager methods
    has_context_manager = (
        hasattr(interface_class, "__aenter__")
        or hasattr(interface_class, "__aexit__")
        or any("close" in name.lower() for name, _ in inspect.getmembers(interface_class, inspect.isfunction))
    )

    # Generate mixins
    mixins = ["ContractTestBase[{}]".format(class_name)]
    if has_async:
        mixins.append("AsyncContractTestMixin")
    if has_context_manager:
        mixins.append("ResourceManagementContractMixin")

    # Get abstract methods for specific tests
    abstract_methods = [
        name
        for name, method in inspect.getmembers(interface_class, inspect.isfunction)
        if getattr(method, "__isabstractmethod__", False)
    ]

    template = f'''# ABOUTME: Contract tests for {class_name} interface
# ABOUTME: Verifies all {class_name.replace("Abstract", "").lower()} implementations comply with the interface contract

import pytest
from typing import Type, List

from {module_name} import {class_name}
# TODO: Import actual implementations when they exist
# from core.implementations.memory.{category}.{class_name.replace("Abstract", "").lower()} import InMemory{class_name.replace("Abstract", "")}
from ..base_contract_test import {", ".join(mixins)}


class {test_class_name}({", ".join(mixins)}):
    """Contract tests for {class_name} interface."""
    
    @property
    def interface_class(self) -> Type[{class_name}]:
        return {class_name}
    
    @property
    def implementations(self) -> List[Type[{class_name}]]:
        return [
            # TODO: Add actual implementations here
            # InMemory{class_name.replace("Abstract", "")},
        ]
'''

    # Add specific contract tests for key methods
    for method_name in abstract_methods[:3]:  # Limit to first 3 methods to avoid too long files
        method = getattr(interface_class, method_name)
        is_async = inspect.iscoroutinefunction(method)

        template += f'''
        @pytest.mark.contract
        {"async " if is_async else ""}def test_{method_name}_contract(self):
            """Test {method_name} method contract behavior."""
            # TODO: Implement specific contract tests for {method_name}
            {"await " if is_async else ""}self._test_method_exists_and_callable("{method_name}")
        
        def _test_method_exists_and_callable(self, method_name: str):
            """Helper to test method exists and is callable."""
            for impl_class in self.implementations:
                if hasattr(impl_class, '__name__'):  # Skip if no implementations yet
                    # TODO: Test actual implementation when available
                    pass
    '''

    return template


def main():
    """Generate contract tests for all interfaces."""
    interfaces = find_interface_classes()

    print(f"Found {len(interfaces)} interface classes:")
    for module_name, interface_class in interfaces:
        print(f"  - {interface_class.__name__} from {module_name}")

    # Create contract test directory structure
    contract_tests_dir = Path("src/core/tests/contract")

    for module_name, interface_class in interfaces:
        # Determine category and create directory
        parts = module_name.split(".")
        if len(parts) >= 4:  # core.interfaces.category.module
            category = parts[-2]
        else:
            category = "misc"

        category_dir = contract_tests_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py if it doesn't exist
        init_file = category_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Contract tests for {} interfaces\n".format(category))

        # Generate test file
        test_filename = f"test_{interface_class.__name__.replace('Abstract', '').lower()}_contract.py"
        test_file = category_dir / test_filename

        if not test_file.exists():  # Don't overwrite existing tests
            template = generate_contract_test_template(interface_class, module_name)
            test_file.write_text(template)
            print(f"Generated: {test_file}")
        else:
            print(f"Skipped (exists): {test_file}")


if __name__ == "__main__":
    main()
