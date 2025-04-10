"""
OSQP solver wrapper module.

This module provides a wrapper for the OSQP solver to solve QP problems.
"""

import numpy as np
import osqp
import scipy.sparse as sparse
import time
from typing import Dict, Optional, Tuple, Union, List, Any

class OSQPSolver:
    """
    Wrapper class for the OSQP solver.
    
    This class provides a convenient interface to solve QP problems using OSQP.
    """
    
    def __init__(self, 
                 verbose: bool = False, 
                 max_iter: int = 4000,
                 eps_abs: float = 1e-6,
                 eps_rel: float = 1e-6,
                 polish: bool = True):
        """
        Initialize the OSQP solver wrapper.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print solver output
        max_iter : int
            Maximum number of iterations
        eps_abs : float
            Absolute tolerance
        eps_rel : float
            Relative tolerance
        polish : bool
            Whether to polish the solution
        """
        self.verbose = verbose
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.polish = polish
        
    def solve(self, 
              Q: np.ndarray, 
              c: np.ndarray, 
              A: Optional[np.ndarray] = None, 
              b: Optional[np.ndarray] = None,
              warm_start: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve a QP problem using OSQP.
        
        Parameters:
        -----------
        Q : numpy.ndarray
            Quadratic cost matrix (n x n)
        c : numpy.ndarray
            Linear cost vector (n)
        A : numpy.ndarray or None
            Constraint matrix (m x n)
        b : numpy.ndarray or None
            Constraint vector (m)
        warm_start : numpy.ndarray or None
            Warm start vector for the solver
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        """
        # Convert to sparse matrices for OSQP
        P = sparse.csc_matrix(Q)
        q = c
        
        # Check if we have constraints
        if A is not None and b is not None:
            A_sparse = sparse.csc_matrix(A)
            l = b  # Constraints are A*x <= b, so l <= A*x <= u, where l = -inf and u = b
            u = np.inf * np.ones(A.shape[0])
        else:
            # No constraints
            A_sparse = sparse.csc_matrix((0, Q.shape[0]))
            l = np.array([])
            u = np.array([])
        
        # Create the OSQP solver
        solver = osqp.OSQP()
        
        # Setup the problem
        solver.setup(P=P, q=q, A=A_sparse, l=-np.inf * np.ones_like(l), u=b,
                    verbose=self.verbose, max_iter=self.max_iter,
                    eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                    polish=self.polish)
        
        # Set warm start if provided
        if warm_start is not None:
            solver.warm_start(x=warm_start)
        
        # Solve the problem
        result = solver.solve()
        
        # Check if the solver was successful
        if result.info.status != 'solved':
            print(f"Warning: OSQP solver returned status {result.info.status}")
        
        # Return the solution
        return result.x
    
    def solve_with_time(self, 
                        Q: np.ndarray, 
                        c: np.ndarray, 
                        A: Optional[np.ndarray] = None, 
                        b: Optional[np.ndarray] = None,
                        warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Solve a QP problem using OSQP and return solution time.
        
        Parameters:
        -----------
        Same as solve method.
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        solve_time : float
            Solution time in seconds
        """
        # Measure the solution time
        start_time = time.time()
        solution = self.solve(Q, c, A, b, warm_start)
        solve_time = time.time() - start_time
        
        return solution, solve_time
    
    def solve_reduced(self,
                     Q: np.ndarray,
                     c: np.ndarray,
                     A: np.ndarray,
                     b: np.ndarray,
                     active_constraints: np.ndarray,
                     warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Solve a reduced QP problem with only active constraints.
        
        Parameters:
        -----------
        Q, c, A, b : Same as solve method
        active_constraints : numpy.ndarray
            Binary vector indicating which constraints are active
        warm_start : numpy.ndarray or None
            Warm start vector for the solver
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        is_feasible : bool
            Whether the solution is feasible for the original problem
        """
        # Get indices of active constraints
        active_indices = np.where(active_constraints > 0.5)[0]
        
        # Create reduced constraint matrices
        if len(active_indices) > 0:
            A_reduced = A[active_indices, :]
            b_reduced = b[active_indices]
        else:
            # No active constraints, solve unconstrained problem
            A_reduced = None
            b_reduced = None
        
        # Solve the reduced problem
        solution = self.solve(Q, c, A_reduced, b_reduced, warm_start)
        
        # Check if the solution satisfies the original constraints
        is_feasible = True
        if A is not None and b is not None:
            constraint_values = A @ solution - b
            is_feasible = np.all(constraint_values <= 1e-6)
        
        return solution, is_feasible
    
    def solve_reduced_with_time(self,
                              Q: np.ndarray,
                              c: np.ndarray,
                              A: np.ndarray,
                              b: np.ndarray,
                              active_constraints: np.ndarray,
                              warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, float]:
        """
        Solve a reduced QP problem with only active constraints and return solution time.
        
        Parameters:
        -----------
        Same as solve_reduced method.
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        is_feasible : bool
            Whether the solution is feasible for the original problem
        solve_time : float
            Solution time in seconds
        """
        # Measure the solution time
        start_time = time.time()
        solution, is_feasible = self.solve_reduced(Q, c, A, b, active_constraints, warm_start)
        solve_time = time.time() - start_time
        
        return solution, is_feasible, solve_time
    
    def solve_pipeline(self,
                     Q: np.ndarray,
                     c: np.ndarray,
                     A: np.ndarray,
                     b: np.ndarray,
                     active_constraints: np.ndarray,
                     warm_start: Optional[np.ndarray] = None,
                     fallback_on_violation: bool = True) -> Tuple[np.ndarray, float, bool]:
        """
        Solve a QP problem using the transformer-enhanced pipeline.
        
        This method first tries to solve the reduced problem with active constraints.
        If the solution isn't feasible for the original problem, it falls back to the full problem.
        
        Parameters:
        -----------
        Q, c, A, b : Same as solve method
        active_constraints : numpy.ndarray
            Binary vector indicating which constraints are active
        warm_start : numpy.ndarray or None
            Warm start vector for the solver
        fallback_on_violation : bool
            Whether to fall back to the full problem if constraints are violated
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        solve_time : float
            Solution time in seconds
        used_fallback : bool
            Whether the fallback solver was used
        """
        # Start timing
        start_time = time.time()
        
        # Try to solve the reduced problem
        solution, is_feasible = self.solve_reduced(Q, c, A, b, active_constraints, warm_start)
        
        # If the solution is not feasible and fallback is enabled, solve the full problem
        used_fallback = False
        if not is_feasible and fallback_on_violation:
            solution = self.solve(Q, c, A, b, warm_start)
            used_fallback = True
        
        # Compute total solution time
        solve_time = time.time() - start_time
        
        return solution, solve_time, used_fallback
