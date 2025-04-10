"""
Dataset module for TransformerMPC.

This module provides classes for creating, processing, and managing datasets
for training and evaluating the transformer models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict, Optional, Union, Any
import os
import pickle
import osqp
import scipy.sparse as sparse

from .qp_generator import QPProblem
from ..utils.osqp_wrapper import OSQPSolver


class QPDataset(Dataset):
    """
    Dataset class for QP problems.
    
    This class manages the datasets used for training and evaluating
    the transformer models. It handles data preprocessing, feature extraction,
    and creating input/target pairs for both transformer models.
    """
    
    def __init__(self, 
                 qp_problems: List[QPProblem],
                 precompute_solutions: bool = True,
                 max_constraints: Optional[int] = None,
                 feature_normalization: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize the QP dataset.
        
        Parameters:
        -----------
        qp_problems : List[QPProblem]
            List of QP problems
        precompute_solutions : bool
            Whether to precompute solutions and active constraints
        max_constraints : int or None
            Maximum number of constraints to consider (for padding)
        feature_normalization : bool
            Whether to normalize features
        cache_dir : str or None
            Directory to cache precomputed solutions
        """
        self.qp_problems = qp_problems
        self.precompute_solutions = precompute_solutions
        self.max_constraints = max_constraints or self._get_max_constraints()
        self.feature_normalization = feature_normalization
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Solver for computing solutions
        self.solver = OSQPSolver()
        
        # Precompute solutions if requested
        if precompute_solutions:
            self._precompute_all_solutions()
        
        # Compute feature statistics for normalization
        if feature_normalization:
            self._compute_feature_statistics()
            
    def _get_max_constraints(self) -> int:
        """
        Get maximum number of constraints across all problems.
        
        Returns:
        --------
        max_constraints : int
            Maximum number of constraints
        """
        return max(problem.n_constraints for problem in self.qp_problems)
            
    def _compute_feature_statistics(self) -> None:
        """
        Compute statistics for feature normalization.
        """
        # Extract raw features from each problem
        all_features = []
        for i in range(min(1000, len(self.qp_problems))):  # Use subset for efficiency
            features = self._extract_raw_features(self.qp_problems[i])
            all_features.append(features)
        
        # Concatenate features
        all_features = np.vstack(all_features)
        
        # Compute mean and standard deviation
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0)
        
        # Replace zeros in std to avoid division by zero
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using precomputed statistics.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Raw features
            
        Returns:
        --------
        normalized_features : numpy.ndarray
            Normalized features
        """
        if not self.feature_normalization:
            return features
        
        return (features - self.feature_mean) / self.feature_std
    
    def _extract_raw_features(self, problem: QPProblem) -> np.ndarray:
        """
        Extract raw features from a QP problem.
        
        Parameters:
        -----------
        problem : QPProblem
            QP problem instance
            
        Returns:
        --------
        features : numpy.ndarray
            Raw features
        """
        # Basic features: initial state and reference if available
        features = []
        
        if problem.initial_state is not None:
            features.append(problem.initial_state)
            
        if problem.reference is not None:
            features.append(problem.reference)
            
        # Add problem dimensions as features
        features.append(np.array([problem.n_variables, problem.n_constraints]))
        
        # Flatten and concatenate all features
        return np.concatenate([f.flatten() for f in features])
    
    def _precompute_all_solutions(self) -> None:
        """
        Precompute solutions and active constraints for all problems.
        """
        self.solutions = []
        self.active_constraints = []
        
        # Define cache file path if using caching
        cache_file = None
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, "qp_solutions_cache.pkl")
            
            # Load from cache if it exists
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.solutions = cache_data['solutions']
                    self.active_constraints = cache_data['active_constraints']
                    return
        
        # Solve all problems and identify active constraints
        for i, problem in enumerate(self.qp_problems):
            # Solve the QP problem
            solution = self.solver.solve(
                Q=problem.Q, 
                c=problem.c, 
                A=problem.A, 
                b=problem.b
            )
            
            # Store the solution
            self.solutions.append(solution)
            
            # Identify active constraints
            active = self._identify_active_constraints(problem, solution)
            self.active_constraints.append(active)
        
        # Save to cache if using caching
        if cache_file is not None:
            with open(cache_file, 'wb') as f:
                pickle.dump(
                    {
                        'solutions': self.solutions,
                        'active_constraints': self.active_constraints
                    }, 
                    f
                )
    
    def _identify_active_constraints(self, 
                                   problem: QPProblem, 
                                   solution: np.ndarray, 
                                   tol: float = 1e-6) -> np.ndarray:
        """
        Identify active constraints in a QP solution.
        
        Parameters:
        -----------
        problem : QPProblem
            QP problem instance
        solution : numpy.ndarray
            Solution vector
        tol : float
            Tolerance for identifying active constraints
            
        Returns:
        --------
        active : numpy.ndarray
            Binary vector indicating active constraints
        """
        # Compute constraint values: A * x - b
        constraint_values = problem.A @ solution - problem.b
        
        # Identify active constraints (those within tolerance of the boundary)
        active = np.abs(constraint_values) < tol
        
        # Pad or truncate to max_constraints
        if self.max_constraints is not None:
            if len(active) > self.max_constraints:
                active = active[:self.max_constraints]
            elif len(active) < self.max_constraints:
                padding = np.zeros(self.max_constraints - len(active), dtype=bool)
                active = np.concatenate([active, padding])
        
        return active.astype(np.float32)
    
    def __len__(self) -> int:
        """
        Get the number of problems in the dataset.
        
        Returns:
        --------
        length : int
            Number of problems
        """
        return len(self.qp_problems)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Parameters:
        -----------
        idx : int
            Index of the item
            
        Returns:
        --------
        item : Dict[str, torch.Tensor]
            Dictionary containing:
            - 'features': Input features for the models
            - 'active_constraints': Target for constraint predictor
            - 'solution': Target for warm start predictor
        """
        # Get the problem
        problem = self.qp_problems[idx]
        
        # Extract and normalize features
        features = self._extract_raw_features(problem)
        features = self._normalize_features(features)
        
        # Get solution and active constraints
        if self.precompute_solutions:
            solution = self.solutions[idx]
            active_constraints = self.active_constraints[idx]
        else:
            # Solve on-the-fly
            solution = self.solver.solve(
                Q=problem.Q, 
                c=problem.c, 
                A=problem.A, 
                b=problem.b
            )
            active_constraints = self._identify_active_constraints(problem, solution)
        
        # Convert to torch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        active_constraints_tensor = torch.tensor(active_constraints, dtype=torch.float32)
        solution_tensor = torch.tensor(solution, dtype=torch.float32)
        
        return {
            'features': features_tensor,
            'active_constraints': active_constraints_tensor,
            'solution': solution_tensor,
            'problem_idx': idx  # Include the problem index for reference
        }
    
    def get_problem(self, idx: int) -> QPProblem:
        """
        Get the original QP problem at the given index.
        
        Parameters:
        -----------
        idx : int
            Index of the problem
            
        Returns:
        --------
        problem : QPProblem
            QP problem instance
        """
        return self.qp_problems[idx]
    
    def get_dataloaders(self, 
                       batch_size: int = 32, 
                       val_split: float = 0.2,
                       test_split: float = 0.1, 
                       shuffle: bool = True,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.
        
        Parameters:
        -----------
        batch_size : int
            Batch size
        val_split : float
            Fraction of data to use for validation
        test_split : float
            Fraction of data to use for testing
        shuffle : bool
            Whether to shuffle the data
        num_workers : int
            Number of workers for data loading
            
        Returns:
        --------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        test_loader : DataLoader
            Test data loader
        """
        # Calculate dataset sizes
        dataset_size = len(self)
        test_size = int(dataset_size * test_split)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size - test_size
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(
            self, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def split(self, test_size: float = 0.2, seed: int = 42) -> Tuple['QPDataset', 'QPDataset']:
        """
        Split the dataset into training and test sets.
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        seed : int
            Random seed
            
        Returns:
        --------
        train_dataset : QPDataset
            Training dataset
        test_dataset : QPDataset
            Test dataset
        """
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle indices
        indices = np.arange(len(self.qp_problems))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        test_idx = int(len(indices) * (1 - test_size))
        
        # Split indices
        train_indices = indices[:test_idx]
        test_indices = indices[test_idx:]
        
        # Create new datasets
        train_problems = [self.qp_problems[i] for i in train_indices]
        test_problems = [self.qp_problems[i] for i in test_indices]
        
        # Create new dataset objects
        train_dataset = QPDataset(
            train_problems,
            precompute_solutions=self.precompute_solutions,
            max_constraints=self.max_constraints,
            feature_normalization=self.feature_normalization,
            cache_dir=self.cache_dir
        )
        
        test_dataset = QPDataset(
            test_problems,
            precompute_solutions=self.precompute_solutions,
            max_constraints=self.max_constraints,
            feature_normalization=self.feature_normalization,
            cache_dir=self.cache_dir
        )
        
        # Copy feature statistics to ensure consistent normalization
        if self.feature_normalization:
            test_dataset.feature_mean = self.feature_mean
            test_dataset.feature_std = self.feature_std
        
        return train_dataset, test_dataset
