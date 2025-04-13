#!/usr/bin/env python3

import os
import importlib
import pkgutil
import transformermpc

def check_module_exists(module_name):
    """Check if a module exists without actually importing it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ModuleNotFoundError, AttributeError):
        return False

def main():
    """Verify the transformermpc package structure."""
    print(f"Transformermpc version: {transformermpc.__version__}")
    print("\nPackage structure:")

    # Get the package path
    pkg_path = os.path.dirname(transformermpc.__file__)
    print(f"Package path: {pkg_path}")
    
    # List all subpackages and modules
    print("\nSubpackages and modules:")
    for _, name, is_pkg in pkgutil.iter_modules([pkg_path]):
        if is_pkg:
            print(f"  - Subpackage: {name}")
            # Check subpackage modules
            subpkg_path = os.path.join(pkg_path, name)
            for _, submodule, _ in pkgutil.iter_modules([subpkg_path]):
                print(f"      - Module: {name}.{submodule}")
        else:
            print(f"  - Module: {name}")
    
    # Check expected module paths
    expected_modules = [
        "transformermpc.data.dataset",
        "transformermpc.data.qp_generator",
        "transformermpc.models.constraint_predictor",
        "transformermpc.models.warm_start_predictor",
        "transformermpc.utils.metrics",
        "transformermpc.utils.osqp_wrapper",
        "transformermpc.utils.visualization",
        "transformermpc.training.trainer",
        "transformermpc.demo.demo"
    ]
    
    print("\nChecking key modules:")
    for module in expected_modules:
        exists = check_module_exists(module)
        print(f"  - {module}: {'✓' if exists else '✗'}")
    
    print("\nPackage structure verification complete!")

if __name__ == "__main__":
    main() 