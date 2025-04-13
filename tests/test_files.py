#!/usr/bin/env python3

import os
import importlib.util
import sys

def print_file_exists(module_path):
    """Check if a Python module file exists and print the result."""
    exists = os.path.isfile(module_path)
    print(f"{module_path}: {'✓ Exists' if exists else '✗ Missing'}")
    return exists

def main():
    """
    Check if key files in the transformermpc package exist
    without importing them.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    transformermpc_dir = os.path.join(package_dir, "transformermpc")
    
    print(f"\nChecking key files in {transformermpc_dir}...\n")
    
    # Check top-level files
    files_to_check = [
        os.path.join(transformermpc_dir, "__init__.py"),
    ]
    
    # Check data module files
    data_dir = os.path.join(transformermpc_dir, "data")
    for file in ["__init__.py", "dataset.py", "qp_generator.py"]:
        files_to_check.append(os.path.join(data_dir, file))
    
    # Check models module files
    models_dir = os.path.join(transformermpc_dir, "models")
    for file in ["__init__.py", "constraint_predictor.py", "warm_start_predictor.py"]:
        files_to_check.append(os.path.join(models_dir, file))
    
    # Check utils module files
    utils_dir = os.path.join(transformermpc_dir, "utils")
    for file in ["__init__.py", "metrics.py", "osqp_wrapper.py", "visualization.py"]:
        files_to_check.append(os.path.join(utils_dir, file))
    
    # Check training module files
    training_dir = os.path.join(transformermpc_dir, "training")
    for file in ["__init__.py", "trainer.py"]:
        files_to_check.append(os.path.join(training_dir, file))
    
    # Check demo module files
    demo_dir = os.path.join(transformermpc_dir, "demo")
    for file in ["__init__.py", "demo.py"]:
        files_to_check.append(os.path.join(demo_dir, file))
    
    # Check each file
    all_exist = True
    for file_path in files_to_check:
        exists = print_file_exists(file_path)
        all_exist = all_exist and exists
    
    # Print summary
    print("\nSummary:")
    if all_exist:
        print("✓ All key files found in the package!")
    else:
        print("✗ Some files are missing from the package.")
        
    return 0 if all_exist else 1

if __name__ == "__main__":
    sys.exit(main()) 