#!/usr/bin/env python3

"""
Test script to verify the proper operation of the new vanilla transformer models.
"""

import torch
import numpy as np

# Fix the import paths
from transformermpc.models.constraint_predictor import ConstraintPredictor
from transformermpc.models.warm_start_predictor import WarmStartPredictor

def test_constraint_predictor():
    """Test the ConstraintPredictor model."""
    print("Testing ConstraintPredictor...")
    
    # Create model instance
    model = ConstraintPredictor(
        input_dim=20,
        hidden_dim=64,
        num_constraints=10,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Test forward pass with random data
    batch_size = 4
    x = torch.randn(batch_size, 20)
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == torch.Size([batch_size, 10]), f"Expected shape {torch.Size([batch_size, 10])}, got {output.shape}"
    
    # Test predict method
    predictions = model.predict(x)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0, :5]}")
    
    print("ConstraintPredictor test passed!")

def test_warm_start_predictor():
    """Test the WarmStartPredictor model."""
    print("Testing WarmStartPredictor...")
    
    # Create model instance
    model = WarmStartPredictor(
        input_dim=20,
        hidden_dim=64,
        output_dim=8,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Test forward pass with random data
    batch_size = 4
    x = torch.randn(batch_size, 20)
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == torch.Size([batch_size, 8]), f"Expected shape {torch.Size([batch_size, 8])}, got {output.shape}"
    
    # Test predict method
    predictions = model.predict(x)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0, :5]}")
    
    print("WarmStartPredictor test passed!")

def main():
    """Run the tests."""
    print("Starting tests for the new transformer models...")
    
    test_constraint_predictor()
    print("\n")
    test_warm_start_predictor()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 