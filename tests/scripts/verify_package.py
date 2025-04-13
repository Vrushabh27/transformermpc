#!/usr/bin/env python3

import os

def list_py_files(start_dir):
    print(f"Scanning directory: {start_dir}")
    
    for root, dirs, files in os.walk(start_dir):
        py_files = [f for f in files if f.endswith('.py')]
        if py_files:
            rel_path = os.path.relpath(root, start_dir)
            if rel_path == '.':
                rel_path = ''
            print(f"\nIn {rel_path or 'root'}:")
            for py_file in sorted(py_files):
                print(f"  - {py_file}")

if __name__ == "__main__":
    list_py_files("transformermpc") 