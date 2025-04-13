import numpy as np
import time
from transformermpc import TransformerMPC

def main():
    print("Testing TransformerMPC solve method...")
    
    # Create a simple QP problem for testing
    n = 5  # dimension of the problem
    m = 3  # number of constraints
    
    # Create random matrices for the test problem
    np.random.seed(42)
    Q = np.random.rand(n, n)
    Q = Q.T @ Q  # Make Q positive definite
    c = np.random.rand(n)
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    
    # Create the TransformerMPC solver
    solver = TransformerMPC(
        use_constraint_predictor=False,
        use_warm_start_predictor=False
    )
    
    # Test the solve method
    start_time = time.time()
    x, solve_time = solver.solve(Q=Q, c=c, A=A, b=b)
    total_time = time.time() - start_time
    
    print(f"Solution vector: {x}")
    print(f"Solver time reported: {solve_time:.6f} seconds")
    print(f"Total time taken: {total_time:.6f} seconds")
    
    # Verify the solution with a simple objective function calculation
    objective = 0.5 * x.T @ Q @ x + c.T @ x
    print(f"Objective function value: {objective:.6f}")
    
    # Test the baseline solver
    baseline_start_time = time.time()
    x_baseline, baseline_time = solver.solve_baseline(Q=Q, c=c, A=A, b=b)
    baseline_total_time = time.time() - baseline_start_time
    
    print("\nBaseline solver results:")
    print(f"Solution vector: {x_baseline}")
    print(f"Solver time reported: {baseline_time:.6f} seconds")
    print(f"Total time taken: {baseline_total_time:.6f} seconds")
    
    # Calculate objective function for baseline solution
    baseline_objective = 0.5 * x_baseline.T @ Q @ x_baseline + c.T @ x_baseline
    print(f"Objective function value: {baseline_objective:.6f}")
    
    # Check if solutions are similar
    solution_diff = np.linalg.norm(x - x_baseline)
    print(f"\nNorm of solution difference: {solution_diff:.6f}")

if __name__ == "__main__":
    main() 