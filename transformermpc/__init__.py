"""
TransformerMPC: Accelerating Model Predictive Control via Neural Networks
====================================================================

TransformerMPC is a Python package that enhances the efficiency of solving
Quadratic Programming (QP) problems in Model Predictive Control (MPC)
using neural network models.

Authors: Vrushabh Zinage, Ahmed Khalil, Efstathios Bakolas
"""

__version__ = "0.1.6"

# Import core components for easy access
from .models.constraint_predictor import ConstraintPredictor
from .models.warm_start_predictor import WarmStartPredictor
from .data.qp_generator import QPGenerator, QPProblem
from .data.dataset import QPDataset
from .utils.osqp_wrapper import OSQPSolver
from .training.trainer import ModelTrainer

# Define the main solver class
class TransformerMPC:
    """
    TransformerMPC solver class that accelerates QP solving using transformer models.
    
    This class combines two transformer models:
    1. Constraint Predictor: Identifies inactive constraints
    2. Warm Start Predictor: Generates better initial points
    
    The combined pipeline significantly reduces computation time while maintaining solution quality.
    """
    
    def __init__(self, 
                 use_constraint_predictor=True, 
                 use_warm_start_predictor=True,
                 constraint_model_path=None,
                 warm_start_model_path=None,
                 fallback_on_violation=True):
        """
        Initialize the TransformerMPC solver.
        
        Parameters:
        -----------
        use_constraint_predictor : bool
            Whether to use the constraint predictor model
        use_warm_start_predictor : bool
            Whether to use the warm start predictor model
        constraint_model_path : str or None
            Path to a pre-trained constraint predictor model
        warm_start_model_path : str or None
            Path to a pre-trained warm start predictor model
        fallback_on_violation : bool
            Whether to fallback to full QP if constraints are violated
        """
        self.use_constraint_predictor = use_constraint_predictor
        self.use_warm_start_predictor = use_warm_start_predictor
        self.fallback_on_violation = fallback_on_violation
        
        # Initialize models if paths are provided
        if use_constraint_predictor:
            self.constraint_predictor = ConstraintPredictor.load(constraint_model_path)
        else:
            self.constraint_predictor = None
            
        if use_warm_start_predictor:
            self.warm_start_predictor = WarmStartPredictor.load(warm_start_model_path)
        else:
            self.warm_start_predictor = None
        
        # Initialize the OSQP solver
        self.solver = OSQPSolver()
        
    def solve(self, Q=None, c=None, A=None, b=None, qp_problem=None):
        """
        Solve a QP problem using the transformer-enhanced pipeline.
        
        Parameters:
        -----------
        Q : numpy.ndarray or None
            Quadratic cost matrix
        c : numpy.ndarray or None
            Linear cost vector
        A : numpy.ndarray or None
            Constraint matrix
        b : numpy.ndarray or None
            Constraint vector
        qp_problem : QPProblem or None
            QPProblem instance (alternative to specifying Q, c, A, b)
            
        Returns:
        --------
        solution : numpy.ndarray
            Optimal solution vector
        solve_time : float
            Computation time in seconds
        """
        import numpy as np
        import time
        
        # If qp_problem is provided, extract matrices from it
        if qp_problem is not None:
            Q = qp_problem.Q
            c = qp_problem.c
            A = qp_problem.A
            b = qp_problem.b
        
        # Check if all required matrices are provided
        if Q is None or c is None:
            raise ValueError("Q and c matrices must be provided")
        
        # Extract features for the QP problem
        n_vars = Q.shape[0]
        features = np.concatenate([
            Q.flatten(),
            c,
            A.flatten() if A is not None else np.zeros(0),
            b if b is not None else np.zeros(0)
        ])
        
        # Default values if no transformers are used
        active_constraints = np.ones(A.shape[0]) if A is not None else None
        warm_start = None
        
        # Predict active constraints if constraint predictor is enabled
        if self.use_constraint_predictor and self.constraint_predictor is not None and A is not None:
            try:
                active_constraints = self.constraint_predictor.predict(features)[0]
            except Exception as e:
                print(f"Warning: Constraint prediction failed: {e}")
                active_constraints = np.ones(A.shape[0])
        
        # Predict warm start if warm start predictor is enabled
        if self.use_warm_start_predictor and self.warm_start_predictor is not None:
            try:
                warm_start = self.warm_start_predictor.predict(features)[0]
            except Exception as e:
                print(f"Warning: Warm start prediction failed: {e}")
                warm_start = None
        
        # Solve the QP problem with the OSQP solver
        start_time = time.time()
        
        if A is not None and self.use_constraint_predictor:
            # Use transformer-enhanced pipeline
            solution, _, used_fallback = self.solver.solve_pipeline(
                Q=Q,
                c=c,
                A=A,
                b=b,
                active_constraints=active_constraints,
                warm_start=warm_start,
                fallback_on_violation=self.fallback_on_violation
            )
        else:
            # Use standard OSQP solver
            solution = self.solver.solve(
                Q=Q,
                c=c,
                A=A,
                b=b,
                warm_start=warm_start
            )
        
        solve_time = time.time() - start_time
        
        return solution, solve_time
    
    def solve_baseline(self, Q=None, c=None, A=None, b=None, qp_problem=None):
        """
        Solve a QP problem using standard OSQP without transformer enhancements.
        
        Parameters and returns same as solve().
        """
        import time
        
        # If qp_problem is provided, extract matrices from it
        if qp_problem is not None:
            Q = qp_problem.Q
            c = qp_problem.c
            A = qp_problem.A
            b = qp_problem.b
        
        # Check if all required matrices are provided
        if Q is None or c is None:
            raise ValueError("Q and c matrices must be provided")
            
        # Solve using standard OSQP
        start_time = time.time()
        solution, _ = self.solver.solve_with_time(Q, c, A, b)
        solve_time = time.time() - start_time
        
        return solution, solve_time
