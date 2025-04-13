"""
Demo script for TransformerMPC.

This script demonstrates the complete workflow of:
1. Generating QP problems
2. Training the constraint predictor and warm start predictor models
3. Testing the models and comparing performance against baseline
4. Visualizing the results
"""

import os
import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from ..data.qp_generator import QPGenerator
from ..data.dataset import QPDataset
from ..models.constraint_predictor import ConstraintPredictor
from ..models.warm_start_predictor import WarmStartPredictor
from ..training.trainer import ModelTrainer
from ..utils.osqp_wrapper import OSQPSolver
from ..utils.metrics import compute_solve_time_metrics, compute_fallback_rate
from ..utils.visualization import (
    plot_solve_time_comparison,
    plot_solve_time_boxplot,
    plot_active_constraints_histogram,
    plot_fallback_statistics
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TransformerMPC Demo")
    
    parser.add_argument(
        "--num_samples", type=int, default=20000,
        help="Number of QP problems to generate (default: 20000)"
    )
    parser.add_argument(
        "--state_dim", type=int, default=4,
        help="State dimension for MPC problems (default: 4)"
    )
    parser.add_argument(
        "--input_dim", type=int, default=2,
        help="Input dimension for MPC problems (default: 2)"
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Time horizon for MPC problems (default: 10)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2000, 
        help="Number of training epochs (default: 2000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="transformermpc_results",
        help="Directory to save results (default: transformermpc_results)"
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and use pretrained models if available"
    )
    parser.add_argument(
        "--cpu", action="store_true", 
        help="Force using CPU even if GPU is available"
    )
    
    return parser.parse_args()


def main():
    """Run the complete demo workflow."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up directories
    data_dir = output_dir / "data"
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Generate QP problems or load from cache
    qp_problems_file = data_dir / "qp_problems.npy"
    
    if qp_problems_file.exists():
        print(f"Loading QP problems from {qp_problems_file}")
        qp_problems = QPGenerator.load(qp_problems_file)
        print(f"Loaded {len(qp_problems)} QP problems")
    else:
        print(f"Generating {args.num_samples} QP problems")
        generator = QPGenerator(
            state_dim=args.state_dim,
            input_dim=args.input_dim,
            horizon=args.horizon,
            num_samples=args.num_samples
        )
        qp_problems = generator.generate()
        
        # Save problems for future use
        generator.save(qp_problems, qp_problems_file)
        print(f"Saved QP problems to {qp_problems_file}")
    
    # 2. Create dataset and split into train/test
    print("Creating dataset")
    dataset = QPDataset(
        qp_problems=qp_problems,
        precompute_solutions=True,
        feature_normalization=True,
        cache_dir=data_dir
    )
    
    train_dataset, test_dataset = dataset.split(test_size=args.test_size)
    print(f"Created datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # 3. Train or load constraint predictor
    cp_model_file = models_dir / "constraint_predictor.pt"
    
    if args.skip_training and cp_model_file.exists():
        print(f"Loading constraint predictor from {cp_model_file}")
        cp_model = ConstraintPredictor.load(cp_model_file)
    else:
        print("Training constraint predictor")
        # Get input dimension from the dataset
        sample_item = train_dataset[0]
        input_dim = sample_item['features'].shape[0]
        num_constraints = sample_item['active_constraints'].shape[0]
        
        cp_model = ConstraintPredictor(
            input_dim=input_dim,
            hidden_dim=128,
            num_constraints=num_constraints
        )
        
        cp_trainer = ModelTrainer(
            model=cp_model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=args.batch_size,
            learning_rate=1e-4,
            num_epochs=args.num_epochs,
            checkpoint_dir=models_dir,
            device=device
        )
        
        cp_history = cp_trainer.train(log_dir=logs_dir / "constraint_predictor")
        
        # Save model
        cp_model.save(cp_model_file)
        print(f"Saved constraint predictor to {cp_model_file}")
    
    # 4. Train or load warm start predictor
    ws_model_file = models_dir / "warm_start_predictor.pt"
    
    if args.skip_training and ws_model_file.exists():
        print(f"Loading warm start predictor from {ws_model_file}")
        ws_model = WarmStartPredictor.load(ws_model_file)
    else:
        print("Training warm start predictor")
        # Get input dimension from the dataset
        sample_item = train_dataset[0]
        input_dim = sample_item['features'].shape[0]
        output_dim = sample_item['solution'].shape[0]
        
        ws_model = WarmStartPredictor(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=output_dim
        )
        
        ws_trainer = ModelTrainer(
            model=ws_model,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            batch_size=args.batch_size,
            learning_rate=1e-4,
            num_epochs=args.num_epochs,
            checkpoint_dir=models_dir,
            device=device
        )
        
        ws_history = ws_trainer.train(log_dir=logs_dir / "warm_start_predictor")
        
        # Save model
        ws_model.save(ws_model_file)
        print(f"Saved warm start predictor to {ws_model_file}")
    
    # 5. Benchmark against baseline
    print("Benchmarking against baseline")
    solver = OSQPSolver()
    
    # List to store results
    baseline_times = []
    warmstart_only_times = []
    constraint_only_times = []
    transformer_times = []
    fallback_flags = []
    
    # Test on a subset (100 problems) for visualization
    test_subset = np.random.choice(len(test_dataset), size=100, replace=False)
    
    print("Testing on 100 problems from the test set")
    for idx in tqdm(test_subset):
        # Get problem
        sample = test_dataset[idx]
        problem = test_dataset.get_problem(idx)
        
        # Get features
        features = sample['features']
        
        # Get ground truth
        true_active = sample['active_constraints'].numpy()
        true_solution = sample['solution'].numpy()
        
        # Predict active constraints
        pred_active = cp_model.predict(features)[0]
        
        # Predict warm start
        pred_solution = ws_model.predict(features)[0]
        
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
            warm_start=pred_solution
        )
        warmstart_only_times.append(warmstart_time)
        
        # Constraint only
        _, is_feasible, constraint_time = solver.solve_reduced_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=pred_active
        )
        constraint_only_times.append(constraint_time)
        
        # Full transformer pipeline
        _, transformer_time, used_fallback = solver.solve_pipeline(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=pred_active,
            warm_start=pred_solution,
            fallback_on_violation=True
        )
        transformer_times.append(transformer_time)
        fallback_flags.append(used_fallback)
    
    # Convert to numpy arrays
    baseline_times = np.array(baseline_times)
    warmstart_only_times = np.array(warmstart_only_times)
    constraint_only_times = np.array(constraint_only_times)
    transformer_times = np.array(transformer_times)
    
    # 6. Compute and print metrics
    solve_metrics = compute_solve_time_metrics(baseline_times, transformer_times)
    fallback_rate = compute_fallback_rate(fallback_flags)
    
    print("\nSolve Time Metrics:")
    print(f"Mean baseline time: {solve_metrics['mean_baseline_time']:.6f}s")
    print(f"Mean transformer time: {solve_metrics['mean_transformer_time']:.6f}s")
    print(f"Mean speedup: {solve_metrics['mean_speedup']:.2f}x")
    print(f"Median speedup: {solve_metrics['median_speedup']:.2f}x")
    print(f"Fallback rate: {fallback_rate:.2f}%")
    
    # 7. Plot results
    print("\nPlotting results")
    
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
    
    # Plot fallback statistics
    fallback_rates = {
        "Transformer-MPC": fallback_rate
    }
    plot_fallback_statistics(
        fallback_rates=fallback_rates,
        save_path=results_dir / "fallback_rates.png"
    )
    
    print(f"\nResults saved to {results_dir}")
    
    # Save main result plot for README
    plot_solve_time_comparison(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        save_path=output_dir.parent / "benchmarking_results.png"
    )
    print(f"Main benchmarking result saved to {output_dir.parent / 'benchmarking_results.png'}")


if __name__ == "__main__":
    main()
