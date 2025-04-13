#!/usr/bin/env python3
"""
Verify the reorganized TransformerMPC package structure.

This script checks that the package structure follows standard Python
package conventions and that all key modules are present.
"""

import os
import sys

def check_dir_exists(path, name):
    """Check if a directory exists and print the result."""
    exists = os.path.isdir(path)
    print(f"{name} directory: {'✓' if exists else '✗'}")
    return exists

def check_file_exists(path, name):
    """Check if a file exists and print the result."""
    exists = os.path.isfile(path)
    print(f"{name} file: {'✓' if exists else '✗'}")
    return exists

def main():
    """Verify the package structure."""
    # Get project root directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print("\n" + "="*60)
    print(" TransformerMPC Package Structure Verification")
    print("="*60)
    
    # Check top-level directories
    print("\nChecking top-level directories:")
    dirs_ok = True
    for dirname in ["transformermpc", "tests", "scripts", "examples"]:
        path = os.path.join(project_root, dirname)
        dirs_ok = check_dir_exists(path, dirname) and dirs_ok
    
    # Check top-level files
    print("\nChecking top-level files:")
    files_ok = True
    for filename in ["setup.py", "pyproject.toml", "requirements.txt", "MANIFEST.in", "LICENSE", "README.md"]:
        path = os.path.join(project_root, filename)
        files_ok = check_file_exists(path, filename) and files_ok
    
    # Check package structure
    print("\nChecking package structure:")
    pkg_dir = os.path.join(project_root, "transformermpc")
    pkg_ok = True
    
    # Check subdirectories
    for dirname in ["data", "models", "utils", "training", "demo"]:
        path = os.path.join(pkg_dir, dirname)
        pkg_ok = check_dir_exists(path, f"transformermpc/{dirname}") and pkg_ok
    
    # Check key files
    pkg_ok = check_file_exists(os.path.join(pkg_dir, "__init__.py"), "transformermpc/__init__.py") and pkg_ok
    
    # Check for key implementation files
    key_files = [
        ("transformermpc/models/constraint_predictor.py", "Constraint Predictor"),
        ("transformermpc/models/warm_start_predictor.py", "Warm Start Predictor"),
        ("transformermpc/data/dataset.py", "Dataset"),
        ("transformermpc/data/qp_generator.py", "QP Generator"),
        ("transformermpc/utils/metrics.py", "Metrics"),
        ("transformermpc/utils/osqp_wrapper.py", "OSQP Wrapper"),
        ("transformermpc/demo/demo.py", "Demo")
    ]
    
    print("\nChecking key implementation files:")
    for filepath, desc in key_files:
        path = os.path.join(project_root, filepath)
        pkg_ok = check_file_exists(path, desc) and pkg_ok
    
    # Print summary
    print("\n" + "="*60)
    if dirs_ok and files_ok and pkg_ok:
        print("✓ Package structure verification PASSED!")
        print("  The TransformerMPC package has been properly reorganized.")
    else:
        print("✗ Package structure verification FAILED!")
        print("  Some expected files or directories are missing.")
    print("="*60 + "\n")
    
    return 0 if dirs_ok and files_ok and pkg_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 