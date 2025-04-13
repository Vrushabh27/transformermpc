"""
QP problem generator module.

This module provides classes for generating and representing QP problems
for training and testing the transformer models.
"""

import numpy as np
import pandas as pd
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union


@dataclass
class QPProblem:
    """
    A class representing a Quadratic Programming problem.
    
    A QP problem is defined as:
    minimize    0.5 * x^T Q x + c^T x
    subject to  A x <= b
    
    Attributes:
    -----------
    Q : numpy.ndarray
        Quadratic cost matrix (n x n)
    c : numpy.ndarray
        Linear cost vector (n)
    A : numpy.ndarray
        Constraint matrix (m x n)
    b : numpy.ndarray
        Constraint vector (m)
    initial_state : numpy.ndarray, optional
        Initial state for MPC problems
    reference : numpy.ndarray, optional
        Reference trajectory for MPC problems
    metadata : dict, optional
        Additional problem-specific information
    """
    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    initial_state: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return self.Q.shape[0]
    
    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        return self.A.shape[0]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "Q": self.Q,
            "c": self.c,
            "A": self.A,
            "b": self.b,
            "initial_state": self.initial_state,
            "reference": self.reference,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QPProblem':
        """Create QPProblem from dictionary."""
        return cls(**data)


class QPGenerator:
    """
    Generator class for creating QP problem instances.
    
    This class generates synthetic QP problems for training and testing
    the transformer models.
    """
    
    def __init__(self, 
                 state_dim: int = 4, 
                 input_dim: int = 2, 
                 horizon: int = 10,
                 num_samples: int = 20000,
                 seed: Optional[int] = None):
        """
        Initialize the QP generator.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space for MPC problems
        input_dim : int
            Dimension of input space for MPC problems
        horizon : int
            Time horizon for MPC problems
        num_samples : int
            Number of QP problems to generate
        seed : int or None
            Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.horizon = horizon
        self.num_samples = num_samples
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self) -> List[QPProblem]:
        """
        Generate QP problems.
        
        Returns:
        --------
        problems : List[QPProblem]
            List of generated QP problems
        """
        problems = []
        
        for i in range(self.num_samples):
            # Generate random system matrices for an MPC problem
            A_dyn, B_dyn = self._generate_dynamics()
            
            # Generate cost matrices
            Q_state = self._generate_state_cost()
            R_input = self._generate_input_cost()
            
            # Generate constraints
            state_constraints = self._generate_state_constraints()
            input_constraints = self._generate_input_constraints()
            
            # Generate random initial state and reference
            initial_state = np.random.randn(self.state_dim)
            reference = np.random.randn(self.horizon * self.state_dim)
            
            # Create the QP matrices for the MPC problem
            Q, c, A, b = self._create_mpc_matrices(
                A_dyn, B_dyn, Q_state, R_input, 
                state_constraints, input_constraints,
                initial_state, reference
            )
            
            # Create the QPProblem instance
            problem = QPProblem(
                Q=Q,
                c=c,
                A=A,
                b=b,
                initial_state=initial_state,
                reference=reference,
                metadata={
                    "type": "mpc",
                    "state_dim": self.state_dim,
                    "input_dim": self.input_dim,
                    "horizon": self.horizon,
                    "A_dynamics": A_dyn,
                    "B_dynamics": B_dyn
                }
            )
            
            problems.append(problem)
            
        return problems
    
    def _generate_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random discrete-time system dynamics matrices.
        
        Returns:
        --------
        A : numpy.ndarray
            State transition matrix (state_dim x state_dim)
        B : numpy.ndarray
            Input matrix (state_dim x input_dim)
        """
        # Generate a random discrete-time system
        A = np.random.randn(self.state_dim, self.state_dim)
        # Scale to make it stable
        eigenvalues, _ = np.linalg.eig(A)
        max_eig = np.max(np.abs(eigenvalues))
        A = A / (max_eig * 1.1)  # Scale to ensure stability
        
        B = np.random.randn(self.state_dim, self.input_dim)
        
        return A, B
    
    def _generate_state_cost(self) -> np.ndarray:
        """
        Generate state cost matrix.
        
        Returns:
        --------
        Q : numpy.ndarray
            State cost matrix (state_dim x state_dim)
        """
        Q_diag = np.abs(np.random.randn(self.state_dim))
        Q = np.diag(Q_diag)
        return Q
    
    def _generate_input_cost(self) -> np.ndarray:
        """
        Generate input cost matrix.
        
        Returns:
        --------
        R : numpy.ndarray
            Input cost matrix (input_dim x input_dim)
        """
        R_diag = np.abs(np.random.randn(self.input_dim))
        R = np.diag(R_diag)
        return R
    
    def _generate_state_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate state constraints.
        
        Returns:
        --------
        A_state : numpy.ndarray
            State constraint matrix (2*state_dim x state_dim)
        b_state : numpy.ndarray
            State constraint vector (2*state_dim)
        """
        # Simple box constraints on states
        A_state = np.vstack([np.eye(self.state_dim), -np.eye(self.state_dim)])
        
        # Random upper and lower bounds
        upper_bounds = np.abs(np.random.rand(self.state_dim) * 5 + 5)  # Random bounds between 5 and 10
        lower_bounds = np.abs(np.random.rand(self.state_dim) * 5 + 5)  # Random bounds between 5 and 10
        
        b_state = np.concatenate([upper_bounds, lower_bounds])
        
        return A_state, b_state
    
    def _generate_input_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate input constraints.
        
        Returns:
        --------
        A_input : numpy.ndarray
            Input constraint matrix (2*input_dim x input_dim)
        b_input : numpy.ndarray
            Input constraint vector (2*input_dim)
        """
        # Simple box constraints on inputs
        A_input = np.vstack([np.eye(self.input_dim), -np.eye(self.input_dim)])
        
        # Random upper and lower bounds
        upper_bounds = np.abs(np.random.rand(self.input_dim) * 2 + 1)  # Random bounds between 1 and 3
        lower_bounds = np.abs(np.random.rand(self.input_dim) * 2 + 1)  # Random bounds between 1 and 3
        
        b_input = np.concatenate([upper_bounds, lower_bounds])
        
        return A_input, b_input
    
    def _create_mpc_matrices(self, 
                            A_dyn: np.ndarray, 
                            B_dyn: np.ndarray, 
                            Q_state: np.ndarray, 
                            R_input: np.ndarray,
                            state_constraints: Tuple[np.ndarray, np.ndarray],
                            input_constraints: Tuple[np.ndarray, np.ndarray],
                            initial_state: np.ndarray,
                            reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create QP matrices for an MPC problem.
        
        Parameters:
        -----------
        A_dyn : numpy.ndarray
            State transition matrix
        B_dyn : numpy.ndarray
            Input matrix
        Q_state : numpy.ndarray
            State cost matrix
        R_input : numpy.ndarray
            Input cost matrix
        state_constraints : Tuple[numpy.ndarray, numpy.ndarray]
            State constraint matrices (A_state, b_state)
        input_constraints : Tuple[numpy.ndarray, numpy.ndarray]
            Input constraint matrices (A_input, b_input)
        initial_state : numpy.ndarray
            Initial state
        reference : numpy.ndarray
            Reference trajectory
            
        Returns:
        --------
        Q : numpy.ndarray
            QP cost matrix
        c : numpy.ndarray
            QP linear cost vector
        A : numpy.ndarray
            QP constraint matrix
        b : numpy.ndarray
            QP constraint vector
        """
        # Extract constraint matrices
        A_state, b_state = state_constraints
        A_input, b_input = input_constraints
        
        # Problem dimensions
        nx = self.state_dim
        nu = self.input_dim
        N = self.horizon
        
        # Total number of decision variables
        n_vars = N * nu
        
        # Construct the cost matrices
        Q_block = np.zeros((n_vars, n_vars))
        for i in range(N):
            idx = i * nu
            Q_block[idx:idx+nu, idx:idx+nu] = R_input
        
        # Compute the prediction matrices
        x_pred = [initial_state]
        for k in range(N):
            x_next = A_dyn @ x_pred[-1]
            x_pred.append(x_next)
        
        # Construct the linear cost term
        c = np.zeros(n_vars)
        
        # Construct constraint matrices
        # Initial state constraint is already handled in prediction
        
        # Input constraints for each time step
        A_in_list = []
        b_in_list = []
        
        for i in range(N):
            # Input constraints: A_input * u <= b_input
            A_i = np.zeros((A_input.shape[0], n_vars))
            idx = i * nu
            A_i[:, idx:idx+nu] = A_input
            A_in_list.append(A_i)
            b_in_list.append(b_input)
        
        # Combine all constraints
        A = np.vstack(A_in_list) if A_in_list else np.zeros((0, n_vars))
        b = np.concatenate(b_in_list) if b_in_list else np.zeros(0)
        
        return Q_block, c, A, b
    
    def save(self, problems: List[QPProblem], filepath: str) -> None:
        """
        Save generated problems to a file.
        
        Parameters:
        -----------
        problems : List[QPProblem]
            List of QP problems to save
        filepath : str
            Path to save the problems
        """
        # Convert problems to dictionaries
        data = [problem.to_dict() for problem in problems]
        
        # Save using numpy
        np.save(filepath, data, allow_pickle=True)
        
    @staticmethod
    def load(filepath: str) -> List[QPProblem]:
        """
        Load problems from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the problems from
            
        Returns:
        --------
        problems : List[QPProblem]
            List of loaded QP problems
        """
        # Load data from file
        data = np.load(filepath, allow_pickle=True)
        
        # Convert dictionaries to QPProblem instances
        problems = [QPProblem.from_dict(item) for item in data]
        
        return problems
