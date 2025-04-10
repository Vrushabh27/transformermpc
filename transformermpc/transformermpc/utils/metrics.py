"""
Metrics utility module for TransformerMPC.

This module provides functions for computing various metrics used to
evaluate the performance of the transformer models and the overall
solving pipeline.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_constraint_prediction_metrics(y_true: Union[np.ndarray, torch.Tensor],
                                         y_pred: Union[np.ndarray, torch.Tensor],
                                         threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics for constraint prediction.
    
    Parameters:
    -----------
    y_true : numpy.ndarray or torch.Tensor
        True binary labels
    y_pred : numpy.ndarray or torch.Tensor
        Predicted probability or scores
    threshold : float
        Threshold for converting probabilities to binary labels
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary containing precision, recall, F1 score, and accuracy
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Convert predictions to binary labels using threshold
    y_pred_binary = (y_pred > threshold).astype(np.int32)
    
    # Compute metrics
    precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def compute_warm_start_metrics(y_true: Union[np.ndarray, torch.Tensor],
                              y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """
    Compute regression metrics for warm start prediction.
    
    Parameters:
    -----------
    y_true : numpy.ndarray or torch.Tensor
        True solution values
    y_pred : numpy.ndarray or torch.Tensor
        Predicted solution values
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary containing MSE, MAE, and relative error
    """
    # Convert to numpy if tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Compute metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Compute relative error (avoid division by zero)
    denom = np.linalg.norm(y_true, axis=1)
    denom = np.where(denom < 1e-8, 1.0, denom)
    rel_error = np.mean(np.linalg.norm(y_true - y_pred, axis=1) / denom)
    
    return {
        'mse': mse,
        'mae': mae,
        'relative_error': rel_error
    }


def compute_active_constraint_stats(active_constraints: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute statistics about active constraints.
    
    Parameters:
    -----------
    active_constraints : List[numpy.ndarray]
        List of binary vectors indicating active constraints for each problem
        
    Returns:
    --------
    stats : Dict[str, float]
        Dictionary containing statistics about active constraints
    """
    # Calculate percentage of active constraints for each problem
    active_percentages = [np.mean(ac) * 100 for ac in active_constraints]
    
    # Compute statistics
    mean_percentage = np.mean(active_percentages)
    median_percentage = np.median(active_percentages)
    std_percentage = np.std(active_percentages)
    min_percentage = np.min(active_percentages)
    max_percentage = np.max(active_percentages)
    
    return {
        'mean_active_percentage': mean_percentage,
        'median_active_percentage': median_percentage,
        'std_active_percentage': std_percentage,
        'min_active_percentage': min_percentage,
        'max_active_percentage': max_percentage,
        'active_percentages': np.array(active_percentages)
    }


def compute_solve_time_metrics(baseline_times: np.ndarray,
                              transformer_times: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics comparing solve times.
    
    Parameters:
    -----------
    baseline_times : numpy.ndarray
        Array of solve times for baseline method
    transformer_times : numpy.ndarray
        Array of solve times for transformer-enhanced method
        
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary containing solve time comparison metrics
    """
    # Compute speedup ratios
    speedup_ratios = baseline_times / transformer_times
    
    # Compute statistics
    mean_speedup = np.mean(speedup_ratios)
    median_speedup = np.median(speedup_ratios)
    min_speedup = np.min(speedup_ratios)
    max_speedup = np.max(speedup_ratios)
    std_speedup = np.std(speedup_ratios)
    
    # Compute percentiles
    percentiles = np.percentile(speedup_ratios, [10, 25, 75, 90])
    
    # Compute time statistics
    mean_baseline_time = np.mean(baseline_times)
    mean_transformer_time = np.mean(transformer_times)
    median_baseline_time = np.median(baseline_times)
    median_transformer_time = np.median(transformer_times)
    
    return {
        'mean_speedup': mean_speedup,
        'median_speedup': median_speedup,
        'min_speedup': min_speedup,
        'max_speedup': max_speedup,
        'std_speedup': std_speedup,
        'p10_speedup': percentiles[0],
        'p25_speedup': percentiles[1],
        'p75_speedup': percentiles[2],
        'p90_speedup': percentiles[3],
        'mean_baseline_time': mean_baseline_time,
        'mean_transformer_time': mean_transformer_time,
        'median_baseline_time': median_baseline_time,
        'median_transformer_time': median_transformer_time,
        'speedup_ratios': speedup_ratios
    }


def compute_fallback_rate(fallback_flags: List[bool]) -> float:
    """
    Compute the rate of fallbacks to full QP solve.
    
    Parameters:
    -----------
    fallback_flags : List[bool]
        List of flags indicating whether fallback was used
        
    Returns:
    --------
    fallback_rate : float
        Percentage of problems that required fallback
    """
    return 100.0 * np.mean([1 if flag else 0 for flag in fallback_flags])


def compute_comprehensive_metrics(
        constraint_pred_true: np.ndarray,
        constraint_pred: np.ndarray,
        warmstart_true: np.ndarray,
        warmstart_pred: np.ndarray,
        baseline_times: np.ndarray,
        transformer_times: np.ndarray,
        fallback_flags: List[bool]
    ) -> Dict[str, Any]:
    """
    Compute comprehensive set of metrics for the entire pipeline.
    
    Parameters:
    -----------
    constraint_pred_true : numpy.ndarray
        True binary labels for constraint prediction
    constraint_pred : numpy.ndarray
        Predicted probability or scores for constraint prediction
    warmstart_true : numpy.ndarray
        True solution values for warm start prediction
    warmstart_pred : numpy.ndarray
        Predicted solution values for warm start prediction
    baseline_times : numpy.ndarray
        Array of solve times for baseline method
    transformer_times : numpy.ndarray
        Array of solve times for transformer-enhanced method
    fallback_flags : List[bool]
        List of flags indicating whether fallback was used
        
    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary containing all metrics
    """
    # Compute individual metrics
    constraint_metrics = compute_constraint_prediction_metrics(
        constraint_pred_true, constraint_pred)
    
    warmstart_metrics = compute_warm_start_metrics(
        warmstart_true, warmstart_pred)
    
    solve_time_metrics = compute_solve_time_metrics(
        baseline_times, transformer_times)
    
    fallback_rate = compute_fallback_rate(fallback_flags)
    
    # Combine all metrics
    metrics = {
        'constraint_prediction': constraint_metrics,
        'warm_start_prediction': warmstart_metrics,
        'solve_time': solve_time_metrics,
        'fallback_rate': fallback_rate
    }
    
    return metrics
