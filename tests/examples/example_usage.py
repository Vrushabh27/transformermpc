"""
Example usage of TransformerMPC library.

This script demonstrates how to use the TransformerMPC package to accelerate
quadratic programming (QP) solvers for Model Predictive Control (MPC).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import TransformerMPC modules
from transformermpc.data.qp_generator import QPGenerator
from transformermpc.data.dataset import QPDataset
from transformermpc.models.constraint_predictor import ConstraintPredictor
from transformermpc.models.warm_start_predictor import WarmStartPredictor
from transformermpc.trainer.model_trainer import ModelTrainer
from transformermpc.utils.osqp_wrapper import OSQPSolver
from transformermpc.utils.metrics import compute_solve_time_metrics
from transformermpc.utils.visualization import plot_solve_time_comparison

def main():
    """Example of training and using TransformerMPC."""
    print("TransformerMPC Example Usage")
    
    # 1. Generate QP problems
    print("\n1. Generating QP problems...")
    state_dim = 4
    input_dim = 2
    horizon = 10
    num_samples = 500  # Small number for quick example
    
    qp_generator = QPGenerator(
        state_dim=state_dim,
        input_dim=input_dim,
        horizon=horizon,
        seed=42
    )
    
    qp_problems = qp_generator.generate_batch(num_samples)
    print(f"Generated {len(qp_problems)} QP problems")
    
    # 2. Create dataset
    print("\n2. Creating dataset...")
    dataset = QPDataset(
        qp_problems=qp_problems,
        precompute_solutions=True,
        feature_normalization=True
    )
    
    train_dataset, val_dataset = dataset.split(test_size=0.2)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 3. Create models
    print("\n3. Creating models...")
    # Get dimensions from a sample
    sample = train_dataset[0]
    feature_dim = sample['features'].shape[0]
    num_constraints = sample['active_constraints'].shape[0]
    solution_dim = sample['solution'].shape[0]
    
    # Create constraint predictor
    cp_model = ConstraintPredictor(
        input_dim=feature_dim,
        hidden_dim=128,
        num_constraints=num_constraints
    )
    
    # Create warm start predictor
    ws_model = WarmStartPredictor(
        input_dim=feature_dim,
        hidden_dim=256,
        output_dim=solution_dim
    )
    
    # 4. Train models
    print("\n4. Training models...")
    # Train constraint predictor
    cp_trainer = ModelTrainer(
        model=cp_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        target_key='active_constraints',
        output_dir=Path('example_output/models')
    )
    
    cp_metrics = cp_trainer.train(
        num_epochs=10,  # Small number for quick example
        batch_size=32,
        learning_rate=1e-3
    )
    
    # Train warm start predictor
    ws_trainer = ModelTrainer(
        model=ws_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        target_key='solution',
        output_dir=Path('example_output/models')
    )
    
    ws_metrics = ws_trainer.train(
        num_epochs=10,  # Small number for quick example
        batch_size=32,
        learning_rate=1e-3
    )
    
    # 5. Evaluate performance
    print("\n5. Evaluating performance...")
    solver = OSQPSolver()
    
    # List to store timing results
    baseline_times = []
    transformer_times = []
    
    # Sample a few problems for demonstration
    test_indices = np.random.choice(len(val_dataset), size=10, replace=False)
    
    for idx in test_indices:
        # Get problem and features
        sample = val_dataset[idx]
        problem = val_dataset.get_problem(idx)
        features = sample['features']
        
        # Predict active constraints and warm start
        with torch.no_grad():
            pred_active = cp_model(features.unsqueeze(0)).squeeze(0) > 0.5
            pred_solution = ws_model(features.unsqueeze(0)).squeeze(0)
        
        # Baseline solve time (standard OSQP)
        _, baseline_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b
        )
        
        # TransformerMPC solve time
        _, transformer_time, _ = solver.solve_pipeline(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=pred_active.numpy(),
            warm_start=pred_solution.numpy(),
            fallback_on_violation=True
        )
        
        baseline_times.append(baseline_time)
        transformer_times.append(transformer_time)
    
    # Calculate metrics
    baseline_times = np.array(baseline_times)
    transformer_times = np.array(transformer_times)
    
    metrics = compute_solve_time_metrics(baseline_times, transformer_times)
    
    print("\nPerformance Summary:")
    print(f"Mean baseline time: {metrics['mean_baseline_time']:.6f}s")
    print(f"Mean transformer time: {metrics['mean_transformer_time']:.6f}s")
    print(f"Mean speedup: {metrics['mean_speedup']:.2f}x")
    print(f"Median speedup: {metrics['median_speedup']:.2f}x")
    
    # Optional: Plot results
    output_dir = Path('example_output/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_solve_time_comparison(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        save_path=output_dir / 'solve_time_comparison.png'
    )
    
    print(f"\nResults saved to {output_dir}")
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 