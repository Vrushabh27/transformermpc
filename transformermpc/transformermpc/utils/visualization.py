"""
Visualization module for TransformerMPC.

This module provides functions for visualizing training progress,
evaluation metrics, and benchmarking results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import os


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """
    Plot training history metrics.
    
    Parameters:
    -----------
    history : Dict[str, List[float]]
        Dictionary containing training metrics
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(nrows=len(history), figsize=(10, 3*len(history)))
    
    # Handle case with only one metric
    if len(history) == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(history.items()):
        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_constraint_prediction_metrics(precision: List[float], 
                                      recall: List[float],
                                      f1: List[float],
                                      accuracy: List[float],
                                      epochs: List[int],
                                      save_path: Optional[str] = None,
                                      show: bool = True) -> None:
    """
    Plot constraint prediction model metrics.
    
    Parameters:
    -----------
    precision, recall, f1, accuracy : List[float]
        Lists of metric values
    epochs : List[int]
        List of epoch numbers
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, precision, 'b-', label='Precision')
    ax.plot(epochs, recall, 'g-', label='Recall')
    ax.plot(epochs, f1, 'r-', label='F1 Score')
    ax.plot(epochs, accuracy, 'k-', label='Accuracy')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title('Constraint Prediction Performance Metrics')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_warm_start_metrics(mse: List[float], 
                           mae: List[float],
                           rel_error: List[float],
                           epochs: List[int],
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
    """
    Plot warm start prediction model metrics.
    
    Parameters:
    -----------
    mse, mae, rel_error : List[float]
        Lists of metric values
    epochs : List[int]
        List of epoch numbers
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, mse, 'b-', label='MSE')
    ax.plot(epochs, mae, 'g-', label='MAE')
    ax.plot(epochs, rel_error, 'r-', label='Relative Error')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.set_title('Warm Start Prediction Performance Metrics')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_solve_time_comparison(baseline_times: np.ndarray,
                              transformer_times: np.ndarray,
                              problem_sizes: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None,
                              show: bool = True,
                              log_scale: bool = False) -> None:
    """
    Plot comparison of solve times between baseline and transformer-enhanced approach.
    
    Parameters:
    -----------
    baseline_times : numpy.ndarray
        Array of solve times for baseline approach
    transformer_times : numpy.ndarray
        Array of solve times for transformer-enhanced approach
    problem_sizes : numpy.ndarray or None
        Array of problem sizes (e.g., number of variables or constraints)
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    log_scale : bool
        Whether to use log scale for the y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_sizes is not None:
        # Plot against problem sizes
        ax.plot(problem_sizes, baseline_times, 'bo-', label='OSQP Baseline')
        ax.plot(problem_sizes, transformer_times, 'ro-', label='TransformerMPC')
        ax.set_xlabel('Problem Size')
    else:
        # Plot histograms
        ax.hist(baseline_times, bins=30, alpha=0.5, label='OSQP Baseline')
        ax.hist(transformer_times, bins=30, alpha=0.5, label='TransformerMPC')
        ax.set_xlabel('Solve Time (seconds)')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_ylabel('Solve Time (seconds)' if problem_sizes is not None else 'Frequency')
    ax.set_title('Comparison of Solve Times: OSQP vs TransformerMPC')
    ax.legend()
    ax.grid(True)
    
    # Add speedup statistics
    mean_speedup = np.mean(baseline_times / transformer_times)
    median_speedup = np.median(baseline_times / transformer_times)
    max_speedup = np.max(baseline_times / transformer_times)
    
    stats_text = f"Mean Speedup: {mean_speedup:.2f}x\n"
    stats_text += f"Median Speedup: {median_speedup:.2f}x\n"
    stats_text += f"Max Speedup: {max_speedup:.2f}x"
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_solve_time_boxplot(baseline_times: np.ndarray,
                           transformer_times: np.ndarray,
                           constraint_only_times: Optional[np.ndarray] = None,
                           warmstart_only_times: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
    """
    Create boxplot comparison of solve times for different methods.
    
    Parameters:
    -----------
    baseline_times : numpy.ndarray
        Array of solve times for baseline approach
    transformer_times : numpy.ndarray
        Array of solve times for full transformer-enhanced approach
    constraint_only_times : numpy.ndarray or None
        Array of solve times using only constraint prediction
    warmstart_only_times : numpy.ndarray or None
        Array of solve times using only warm start prediction
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = []
    labels = []
    
    # Always include baseline and full transformer
    data.append(baseline_times)
    labels.append('OSQP Baseline')
    
    # Include constraint-only if provided
    if constraint_only_times is not None:
        data.append(constraint_only_times)
        labels.append('Constraint Prediction Only')
    
    # Include warmstart-only if provided
    if warmstart_only_times is not None:
        data.append(warmstart_only_times)
        labels.append('Warm Start Only')
    
    # Always include full transformer
    data.append(transformer_times)
    labels.append('Full TransformerMPC')
    
    # Create the boxplot
    ax.boxplot(data, labels=labels, showfliers=True)
    
    ax.set_ylabel('Solve Time (seconds)')
    ax.set_title('Comparison of Solve Times Across Methods')
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_active_constraints_histogram(active_percentages: np.ndarray,
                                     save_path: Optional[str] = None,
                                     show: bool = True) -> None:
    """
    Plot histogram of percentage of active constraints.
    
    Parameters:
    -----------
    active_percentages : numpy.ndarray
        Array of percentages of active constraints for each problem
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(active_percentages, bins=50, alpha=0.75)
    ax.axvline(np.mean(active_percentages), color='r', linestyle='--', 
              label=f'Mean: {np.mean(active_percentages):.2f}%')
    ax.axvline(np.median(active_percentages), color='g', linestyle='--', 
              label=f'Median: {np.median(active_percentages):.2f}%')
    
    ax.set_xlabel('Percentage of Active Constraints')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Active Constraints Percentage')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fallback_statistics(fallback_rates: Dict[str, float],
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
    """
    Plot statistics about fallback to full QP solve.
    
    Parameters:
    -----------
    fallback_rates : Dict[str, float]
        Dictionary of fallback rates for different methods
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(fallback_rates.keys())
    rates = list(fallback_rates.values())
    
    # Create bar plot
    ax.bar(methods, rates, color='skyblue')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Fallback Rate (%)')
    ax.set_title('Fallback Rates for Different Methods')
    ax.grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(rates):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_benchmarking_results(results: Dict[str, Dict[str, Union[float, np.ndarray]]],
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """
    Plot comprehensive benchmarking results.
    
    Parameters:
    -----------
    results : Dict
        Dictionary containing benchmarking results
    save_path : str or None
        Path to save the plot
    show : bool
        Whether to display the plot
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
    # Plot 1: Solve time comparison (boxplot)
    methods = list(results.keys())
    solve_times = [results[method]['solve_times'] for method in methods]
    
    axes[0, 0].boxplot(solve_times, labels=methods)
    axes[0, 0].set_ylabel('Solve Time (seconds)')
    axes[0, 0].set_title('Solve Times by Method')
    axes[0, 0].grid(True, axis='y')
    
    # Plot 2: Speedup factors
    baseline_method = methods[0]  # Assume first method is baseline
    speedups = {}
    
    for method in methods[1:]:
        speedup = results[baseline_method]['mean_solve_time'] / results[method]['mean_solve_time']
        speedups[method] = speedup
    
    speedup_methods = list(speedups.keys())
    speedup_values = list(speedups.values())
    
    axes[0, 1].bar(speedup_methods, speedup_values, color='green')
    axes[0, 1].set_ylabel('Speedup Factor (x)')
    axes[0, 1].set_title(f'Speedup Relative to {baseline_method}')
    axes[0, 1].grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(speedup_values):
        axes[0, 1].text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    # Plot 3: Fallback rates
    fallback_rates = {m: results[m].get('fallback_rate', 0) for m in methods[1:]}
    fallback_methods = list(fallback_rates.keys())
    fallback_values = list(fallback_rates.values())
    
    axes[1, 0].bar(fallback_methods, fallback_values, color='orange')
    axes[1, 0].set_ylabel('Fallback Rate (%)')
    axes[1, 0].set_title('Fallback Rates by Method')
    axes[1, 0].grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(fallback_values):
        axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Plot 4: Accuracy metrics for constraint predictor
    if 'constraint_predictor' in results and 'metrics' in results['constraint_predictor']:
        metrics = results['constraint_predictor']['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[1, 1].bar(metric_names, metric_values, color='purple')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Constraint Predictor Metrics')
        axes[1, 1].grid(True, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha='center')
    else:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()
