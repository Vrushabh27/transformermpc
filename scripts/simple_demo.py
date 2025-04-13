#!/usr/bin/env python3
"""
TransformerMPC Simple Demo

This script provides a complete end-to-end demonstration of the TransformerMPC package:
1. Generates a set of quadratic programming (QP) problems
2. Creates training and test datasets
3. Trains both transformer models (constraint predictor and warm start predictor)
4. Tests the models on the test set
5. Plots performance comparisons including box plots

The script is designed to run quickly with a small number of samples and epochs,
but can be modified for more comprehensive training.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from tqdm import tqdm

# Fix potential serialization issues with numpy scalars
import torch.serialization
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])

# Patch torch.load to handle weights_only parameter
original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)
torch.load = patched_torch_load

# Import TransformerMPC modules
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TransformerMPC Simple Demo")
    
    # Data generation parameters
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of QP problems to generate (default: 100)")
    parser.add_argument("--state_dim", type=int, default=4,
                        help="State dimension for MPC problems (default: 4)")
    parser.add_argument("--input_dim", type=int, default=2,
                        help="Input dimension for MPC problems (default: 2)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Time horizon for MPC problems (default: 5)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2)")
    
    # Other parameters
    parser.add_argument("--output_dir", type=str, default="demo_results",
                        help="Directory to save results (default: demo_results)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--test_problems", type=int, default=10,
                        help="Number of test problems for evaluation (default: 10)")
    
    return parser.parse_args()

def train_model(model, train_data, val_data, num_epochs, batch_size, lr=1e-3):
    """Simple training loop for the models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            
            if isinstance(model, ConstraintPredictor):
                targets = batch['active_constraints'].to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            else:
                targets = batch['solution'].to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = torch.nn.functional.mse_loss(outputs, targets)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                
                if isinstance(model, ConstraintPredictor):
                    targets = batch['active_constraints'].to(device)
                    outputs = model(features)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                else:
                    targets = batch['solution'].to(device)
                    outputs = model(features)
                    loss = torch.nn.functional.mse_loss(outputs, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_epoch_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses
    }

def main():
    """Run the demo workflow."""
    # Parse command line arguments
    args = parse_args()
    
    print("=" * 60)
    print("TransformerMPC Simple Demo".center(60))
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up directories
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    
    for directory in [models_dir, results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Step 2: Create dataset and split into train/test
    print("\nStep 2: Creating datasets")
    print("-" * 60)
    
    dataset = QPDataset(
        qp_problems=qp_problems,
        precompute_solutions=True,
        feature_normalization=True
    )
    
    train_dataset, test_dataset = dataset.split(test_size=args.test_size)
    print(f"Created datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Step 3: Train constraint predictor
    print("\nStep 3: Training constraint predictor")
    print("-" * 60)
    
    # Get input dimension from the dataset
    sample_item = train_dataset[0]
    input_dim = sample_item['features'].shape[0]
    num_constraints = sample_item['active_constraints'].shape[0]
    
    cp_model = ConstraintPredictor(
        input_dim=input_dim,
        hidden_dim=64,  # Smaller model for faster training
        num_constraints=num_constraints,
        num_layers=2    # Fewer layers for faster training
    )
    
    # Train constraint predictor with simplified training loop
    cp_history = train_model(
        model=cp_model,
        train_data=train_dataset,
        val_data=test_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    cp_model_file = models_dir / "constraint_predictor.pt"
    torch.save(cp_model.state_dict(), cp_model_file)
    print(f"Saved constraint predictor to {cp_model_file}")
    
    # Step 4: Train warm start predictor
    print("\nStep 4: Training warm start predictor")
    print("-" * 60)
    
    # Get input dimension from the dataset
    output_dim = sample_item['solution'].shape[0]
    
    ws_model = WarmStartPredictor(
        input_dim=input_dim,
        hidden_dim=128,  # Smaller model for faster training
        output_dim=output_dim,
        num_layers=2     # Fewer layers for faster training
    )
    
    # Train warm start predictor with simplified training loop
    ws_history = train_model(
        model=ws_model,
        train_data=train_dataset,
        val_data=test_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    ws_model_file = models_dir / "warm_start_predictor.pt"
    torch.save(ws_model.state_dict(), ws_model_file)
    print(f"Saved warm start predictor to {ws_model_file}")
    
    # Step 5: Test on a subset of problems
    print("\nStep 5: Performance testing")
    print("-" * 60)
    
    solver = OSQPSolver()
    
    # Lists to store results
    baseline_times = []
    transformer_times = []
    constraint_only_times = []
    warmstart_only_times = []
    fallback_flags = []
    
    # Test on a small subset for demonstration
    num_test_problems = min(args.test_problems, len(test_dataset))
    test_subset = np.random.choice(len(test_dataset), size=num_test_problems, replace=False)
    
    print(f"Testing on {num_test_problems} problems...")
    for idx in tqdm(test_subset):
        # Get problem
        sample = test_dataset[idx]
        problem = test_dataset.get_problem(idx)
        
        # Get features
        features = sample['features']
        
        # Predict active constraints and warm start
        with torch.no_grad():
            cp_output = cp_model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
            pred_active = (torch.sigmoid(cp_output) > 0.5).float().squeeze(0).numpy()
            
            ws_output = ws_model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
            pred_solution = ws_output.squeeze(0).numpy()
        
        # 1. Baseline (OSQP without transformers)
        _, baseline_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b
        )
        baseline_times.append(baseline_time)
        
        # 2. Constraint-only
        _, constraint_time, _ = solver.solve_pipeline(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            active_constraints=pred_active,
            warm_start=None,
            fallback_on_violation=True
        )
        constraint_only_times.append(constraint_time)
        
        # 3. Warm-start-only
        _, warmstart_time = solver.solve_with_time(
            Q=problem.Q,
            c=problem.c,
            A=problem.A,
            b=problem.b,
            warm_start=pred_solution
        )
        warmstart_only_times.append(warmstart_time)
        
        # 4. Full transformer pipeline
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
    transformer_times = np.array(transformer_times)
    constraint_only_times = np.array(constraint_only_times)
    warmstart_only_times = np.array(warmstart_only_times)
    
    # Compute and print metrics
    solve_metrics = compute_solve_time_metrics(baseline_times, transformer_times)
    fallback_rate = compute_fallback_rate(fallback_flags)
    
    print("\nPerformance Results:")
    print("-" * 60)
    print(f"Mean baseline time: {solve_metrics['mean_baseline_time']:.6f}s")
    print(f"Mean transformer time: {solve_metrics['mean_transformer_time']:.6f}s")
    print(f"Mean constraint-only time: {np.mean(constraint_only_times):.6f}s")
    print(f"Mean warm-start-only time: {np.mean(warmstart_only_times):.6f}s")
    print(f"Mean speedup: {solve_metrics['mean_speedup']:.2f}x")
    print(f"Median speedup: {solve_metrics['median_speedup']:.2f}x")
    print(f"Fallback rate: {fallback_rate:.2f}%")
    
    # Step 6: Generate visualizations
    print("\nStep 6: Generating visualizations")
    print("-" * 60)
    
    # Plot solve time comparison
    print("Generating solve time comparison plot...")
    plot_solve_time_comparison(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        save_path=results_dir / "solve_time_comparison.png"
    )
    
    # Plot solve time boxplot
    print("Generating solve time boxplot...")
    plot_solve_time_boxplot(
        baseline_times=baseline_times,
        transformer_times=transformer_times,
        constraint_only_times=constraint_only_times,
        warmstart_only_times=warmstart_only_times,
        save_path=results_dir / "solve_time_boxplot.png"
    )
    
    # Plot performance violin plot
    print("Generating performance violin plot...")
    plt.figure(figsize=(10, 6))
    plt.violinplot(
        [baseline_times, constraint_only_times, warmstart_only_times, transformer_times],
        showmeans=True
    )
    plt.xticks([1, 2, 3, 4], ['Baseline', 'Constraint-only', 'WarmStart-only', 'Full Pipeline'])
    plt.ylabel('Solve Time (s)')
    plt.title('QP Solve Time Comparison')
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "performance_violin.png", dpi=300)
    
    # Plot training history
    print("Generating training history plots...")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cp_history['train_loss'], label='Train Loss')
    plt.plot(cp_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Constraint Predictor Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(ws_history['train_loss'], label='Train Loss')
    plt.plot(ws_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Warm Start Predictor Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_history.png", dpi=300)
    
    # Create a summary plot with box plots
    print("Generating summary box plot...")
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    plt.boxplot(
        [baseline_times, transformer_times, constraint_only_times, warmstart_only_times],
        labels=['Baseline', 'Full Pipeline', 'Constraint-only', 'Warm Start-only'],
        showmeans=True
    )
    
    plt.ylabel('Solve Time (s)')
    plt.title('QP Solve Time Comparison (Box Plot)')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add mean value annotations
    means = [
        np.mean(baseline_times),
        np.mean(transformer_times),
        np.mean(constraint_only_times),
        np.mean(warmstart_only_times)
    ]
    
    for i, mean in enumerate(means):
        plt.text(i+1, mean, f'{mean:.6f}s', 
                 horizontalalignment='center', 
                 verticalalignment='bottom',
                 fontweight='bold')
    
    plt.savefig(results_dir / "summary_boxplot.png", dpi=300)
    
    print(f"\nResults and visualizations saved to {output_dir}")
    print("\nDemo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 