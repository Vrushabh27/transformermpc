import sys
import os
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    # Import the demo module
    from transformermpc.transformermpc.demo.demo import main as demo_main
    
    # Run the demo
    demo_main()

if __name__ == "__main__":
    main() 