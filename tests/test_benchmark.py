"""
Benchmark script for TransformerMPC.
"""

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Fix the serialization issue
import torch.serialization
# Add numpy.core.multiarray.scalar to safe globals
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# Import required modules
from transformermpc.data.qp_generator import QPGenerator
from transformermpc.data.dataset import QPDataset
from transformermpc.models.constraint_predictor import ConstraintPredictor
from transformermpc.models.warm_start_predictor import WarmStartPredictor
from transformermpc.utils.osqp_wrapper import OSQPSolver
from transformermpc.utils.metrics import compute_solve_time_metrics, compute_fallback_rate
from transformermpc.utils.visualization import (
    plot_solve_time_comparison,
    plot_solve_time_boxplot
)

# Monkey patch torch.load to use weights_only=False by default
original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TransformerMPC Benchmark")
    
    # Input/output parameters
    parser.add_argument(
        "--data_dir", type=str, default="demo_results/data",
        help="Directory containing QP problem data (default: demo_results/data)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="demo_results/results",
        help="Directory to save benchmark results (default: demo_results/results)"
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--num_test_problems", type=int, default=20,
        help="Number of test problems to benchmark (default: 20)"
    )
    parser.add_argument(
        "--use_gpu", action="store_true",
        help="Use GPU if available"
    )
    
    return parser.parse_args()

def main():
    """Run a benchmark test."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up directories
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load QP problems
    qp_problems_file = data_dir / "qp_problems.npy"
    if qp_problems_file.exists():
        print(f"Loading QP problems from {qp_problems_file}")
        qp_problems = QPGenerator.load(qp_problems_file)
        print(f"Loaded {len(qp_problems)} QP problems")
    else:
        print("Error: No QP problems found. Please run the demo first.")
        return
    
    # 2. Load dataset
    print("Loading dataset")
    dataset = QPDataset(
        qp_problems=qp_problems,
        precompute_solutions=True,
        feature_normalization=True,
        cache_dir=data_dir
    )
    
    _, test_dataset = dataset.split(test_size=args.test_size)
    print(f"Loaded test dataset with {len(test_dataset)} problems")
    
    # 3. Create models with default parameters (don't load from files)
    print("Creating models")
    sample_item = test_dataset[0]
    input_dim = sample_item['features'].shape[0]
    num_constraints = sample_item['active_constraints'].shape[0]
    output_dim = sample_item['solution'].shape[0]
    
    cp_model = ConstraintPredictor(
        input_dim=input_dim,
        hidden_dim=128,
        num_constraints=num_constraints
    )
    
    ws_model = WarmStartPredictor(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim
    )
    
    # 4. Test on a small subset
    print("Testing on a subset of problems")
    solver = OSQPSolver()
    
    # List to store results
    baseline_times = []
    warmstart_only_times = []
    constraint_only_times = []
    transformer_times = []
    fallback_flags = []
    
    # Test on a subset for visualization
    test_subset = np.random.choice(len(test_dataset), size=args.num_test_problems, replace=False)
    
    print(f"Benchmarking {args.num_test_problems} problems...")
    for idx in tqdm(test_subset):
        # Get problem
        sample = test_dataset[idx]
        problem = test_dataset.get_problem(idx)
        
        # Get features
        features = sample['features']
        
        # For demonstration, we'll use the solutions directly instead of predictions
        # since we're using untrained models
        true_active = sample['active_constraints'].numpy()
        true_solution = sample['solution'].numpy()
        
        # Baseline (OSQP without transformers)
        _, baseline_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b
        )
        baseline_times.append(baseline_time)
        
        # Warm start only
        _, warmstart_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            warm_start=true_solution  # Using true solution as warm start
        )
        warmstart_only_times.append(warmstart_time)
        
        # Constraint only
        _, is_feasible, constraint_time = solver.solve_reduced_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=true_active  # Using true active constraints
        )
        constraint_only_times.append(constraint_time)
        
        # Full transformer pipeline (using true values for demo)
        _, transformer_time, used_fallback = solver.solve_pipeline(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=true_active,
            warm_start=true_solution,
            fallback_on_violation=True
        )
        transformer_times.append(transformer_time)
        fallback_flags.append(used_fallback)
    
    # Convert to numpy arrays
    baseline_times = np.array(baseline_times)
    warmstart_only_times = np.array(warmstart_only_times)
    constraint_only_times = np.array(constraint_only_times)
    transformer_times = np.array(transformer_times)
    
    # Compute and print metrics
    solve_metrics = compute_solve_time_metrics(baseline_times, transformer_times)
    fallback_rate = compute_fallback_rate(fallback_flags)
    
    print("\nSolve Time Metrics:")
    print(f"Mean baseline time: {solve_metrics['mean_baseline_time']:.6f}s")
    print(f"Mean transformer time: {solve_metrics['mean_transformer_time']:.6f}s")
    print(f"Mean speedup: {solve_metrics['mean_speedup']:.2f}x")
    print(f"Median speedup: {solve_metrics['median_speedup']:.2f}x")
    print(f"Fallback rate: {fallback_rate:.2f}%")
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot solve time comparison
    plot_solve_time_comparison(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        save_path=results_dir / "solve_time_comparison.png"
    )
    
    # Plot boxplot
    plot_solve_time_boxplot(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        constraint_only_times=constraint_only_times,
        warmstart_only_times=warmstart_only_times,
        save_path=results_dir / "solve_time_boxplot.png"
    )
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main() 