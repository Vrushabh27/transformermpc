#!/usr/bin/env python3
"""
TransformerMPC Boxplot Demo

This script demonstrates the performance comparison of different QP solving strategies
using randomly generated quadratic programming (QP) problems.

It focuses on showing the boxplot comparison without actual training of transformer models.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import TransformerMPC modules
from transformermpc.data.qp_generator import QPGenerator
from transformermpc.utils.osqp_wrapper import OSQPSolver
from transformermpc.utils.metrics import compute_solve_time_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TransformerMPC Boxplot Demo")
    
    # Data generation parameters
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of QP problems to generate (default: 30)")
    parser.add_argument("--state_dim", type=int, default=4,
                        help="State dimension for MPC problems (default: 4)")
    parser.add_argument("--input_dim", type=int, default=2,
                        help="Input dimension for MPC problems (default: 2)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Time horizon for MPC problems (default: 5)")
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="Directory to save results (default: demo_results)")
    
    return parser.parse_args()

def main():
    """Run the boxplot demo workflow."""
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("TransformerMPC Boxplot Demo".center(60))
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up directory for results
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate QP problems
    print("\nStep 1: Generating QP problems")
    print("-" * 60)
    
    print(f"Generating {args.num_samples} QP problems")
    generator = QPGenerator(
        state_dim=args.state_dim,
        input_dim=args.input_dim,
        horizon=args.horizon,
        num_samples=args.num_samples
    )
    qp_problems = generator.generate()
    print(f"Generated {len(qp_problems)} QP problems")
    
    # Step 2: Performance testing with different strategies
    print("\nStep 2: Performance testing")
    print("-" * 60)
    
    solver = OSQPSolver()
    
    # Lists to store results for different strategies
    baseline_times = []              # Standard OSQP
    reduced_constraint_times = []    # 50% random constraints removed
    warm_start_random_times = []     # Warm start with random initial point
    warm_start_perturbed_times = []  # Warm start with slightly perturbed solution
    combined_strategy_times = []     # Both constraints reduced and warm start
    
    print(f"Testing {len(qp_problems)} problems...")
    for i, problem in enumerate(tqdm(qp_problems)):
        # 1. Baseline (Standard OSQP)
        _, baseline_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b
        )
        baseline_times.append(baseline_time)
        
        # Get the solution for perturbed warm start
        solution = solver.solve(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b
        )
        
        # 2. Reduced constraints (randomly remove 50% of constraints)
        num_constraints = problem.A.shape[0]
        mask = np.random.choice([True, False], size=num_constraints, p=[0.5, 0.5])
        
        A_reduced = problem.A[mask]
        b_reduced = problem.b[mask]
        
        _, reduced_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=A_reduced,
            b=b_reduced
        )
        reduced_constraint_times.append(reduced_time)
        
        # 3. Warm start with random initial point
        warm_start_random = np.random.randn(problem.Q.shape[0])
        _, warm_start_random_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            warm_start=warm_start_random
        )
        warm_start_random_times.append(warm_start_random_time)
        
        # 4. Warm start with slightly perturbed solution (simulate good prediction)
        perturbation = np.random.randn(solution.shape[0]) * 0.1  # Small perturbation
        warm_start_perturbed = solution + perturbation
        
        _, warm_start_perturbed_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            warm_start=warm_start_perturbed
        )
        warm_start_perturbed_times.append(warm_start_perturbed_time)
        
        # 5. Combined strategy (reduced constraints + warm start)
        _, combined_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=A_reduced,
            b=b_reduced,
            warm_start=warm_start_perturbed
        )
        combined_strategy_times.append(combined_time)
    
    # Convert to numpy arrays
    baseline_times = np.array(baseline_times)
    reduced_constraint_times = np.array(reduced_constraint_times)
    warm_start_random_times = np.array(warm_start_random_times)
    warm_start_perturbed_times = np.array(warm_start_perturbed_times)
    combined_strategy_times = np.array(combined_strategy_times)
    
    # Compute and print metrics
    print("\nPerformance Results:")
    print("-" * 60)
    print(f"Mean baseline time: {np.mean(baseline_times):.6f}s")
    print(f"Mean reduced constraints time: {np.mean(reduced_constraint_times):.6f}s")
    print(f"Mean warm start (random) time: {np.mean(warm_start_random_times):.6f}s")
    print(f"Mean warm start (perturbed) time: {np.mean(warm_start_perturbed_times):.6f}s")
    print(f"Mean combined strategy time: {np.mean(combined_strategy_times):.6f}s")
    
    # Calculate speedups
    print("\nSpeedup Factors:")
    print(f"Reduced constraints: {np.mean(baseline_times) / np.mean(reduced_constraint_times):.2f}x")
    print(f"Warm start (random): {np.mean(baseline_times) / np.mean(warm_start_random_times):.2f}x")
    print(f"Warm start (perturbed): {np.mean(baseline_times) / np.mean(warm_start_perturbed_times):.2f}x")
    print(f"Combined strategy: {np.mean(baseline_times) / np.mean(combined_strategy_times):.2f}x")
    
    # Step 3: Generate visualizations
    print("\nStep 3: Generating visualizations")
    print("-" * 60)
    
    # Box plot of solve times
    print("Generating solve time boxplot...")
    plt.figure(figsize=(12, 8))
    box_data = [
        baseline_times, 
        reduced_constraint_times, 
        warm_start_random_times, 
        warm_start_perturbed_times, 
        combined_strategy_times
    ]
    
    box_labels = [
        'Baseline', 
        'Reduced\nConstraints', 
        'Warm Start\n(Random)', 
        'Warm Start\n(Perturbed)', 
        'Combined\nStrategy'
    ]
    
    box_plot = plt.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        showmeans=True
    )
    
    # Add colors to boxes
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightcyan']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Solve Time (s)')
    plt.title('QP Solve Time Comparison')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add mean value annotations
    means = [np.mean(data) for data in box_data]
    for i, mean in enumerate(means):
        plt.text(i+1, mean, f'{mean:.6f}s', 
                 horizontalalignment='center', 
                 verticalalignment='bottom',
                 fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "solve_time_boxplot.png", dpi=300)
    
    # Create speedup bar chart
    print("Generating speedup bar chart...")
    plt.figure(figsize=(10, 6))
    
    speedups = [
        np.mean(baseline_times) / np.mean(reduced_constraint_times),
        np.mean(baseline_times) / np.mean(warm_start_random_times),
        np.mean(baseline_times) / np.mean(warm_start_perturbed_times),
        np.mean(baseline_times) / np.mean(combined_strategy_times)
    ]
    
    speedup_labels = [
        'Reduced\nConstraints', 
        'Warm Start\n(Random)', 
        'Warm Start\n(Perturbed)', 
        'Combined\nStrategy'
    ]
    
    bars = plt.bar(speedup_labels, speedups, color=['lightgreen', 'lightpink', 'lightyellow', 'lightcyan'])
    
    plt.ylabel('Speedup Factor (×)')
    plt.title('Speedup Relative to Baseline')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value annotations
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{speedup:.2f}×',
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "speedup_barchart.png", dpi=300)
    
    # Violin plot of solve times
    print("Generating violin plot...")
    plt.figure(figsize=(12, 8))
    
    violin_plot = plt.violinplot(
        box_data,
        showmeans=True,
        showextrema=True
    )
    
    plt.xticks(range(1, 6), box_labels)
    plt.ylabel('Solve Time (s)')
    plt.title('QP Solve Time Distribution')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "solve_time_violinplot.png", dpi=300)
    
    print(f"\nResults and visualizations saved to {output_dir}/results")
    print("\nDemo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 