<p align="center">

  <h2 align="center">TransformerMPC: Accelerating Model Predictive Control via Transformers  [ICRA '25]</h2>
  <p align="center">
    <a href="https://vrushabh27.github.io/vrushabh_zinage/"><strong>Vrushabh Zinage</strong></a><sup>1</sup>
    ·
    <a href="https://github.com/itsahmedkhalil"><strong>Ahmed Khalil</strong></a><sup>1</sup>
   ·
    <a href="https://sites.utexas.edu/ebakolas/"><strong>Efstathios Bakolas</strong></a><sup>1</sup>
    
</p>


<p align="center">
    <sup>1</sup>University of Texas at Austin 
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2409.09573-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2409.09266) [![ProjectPage](https://img.shields.io/badge/Project_Page-TransformerMPC-blue)](https://transformer-mpc.github.io/)
  <div align="center"></div>
</p>

## Overview

TransformerMPC improves the computational efficiency of Model Predictive Control (MPC) problems using neural network models. It employs the following two prediction models:

1. **Constraint Predictor**: Identifies inactive constraints in MPC formulations
2. **Warm Start Predictor**: Generates better initial points for MPC solvers

By combining these models, TransformerMPC significantly reduces computation time while maintaining solution quality.

## Package Structure

The package is organized with a standard Python package structure:

```
transformermpc/
├── transformermpc/       # Core package module
│   ├── data/             # Data generation utilities
│   ├── models/           # Model implementations
│   ├── utils/            # Utility functions and metrics
│   ├── training/         # Training infrastructure
│   └── demo/             # Demo scripts
├── scripts/              # Demo and utility scripts
├── tests/                # Testing infrastructure
├── setup.py              # Package installation script
└── requirements.txt      # Dependencies
```

## Installation

Install directly from PyPI:

```bash
pip install transformermpc
```

Or install from source:

```bash
git clone https://github.com/vrushabh/transformermpc.git
cd transformermpc
pip install -e .
```

## Dependencies

- Python >= 3.7
- PyTorch >= 1.9.0
- OSQP >= 0.6.2
- NumPy, SciPy, and other scientific computing libraries
- Additional dependencies specified in requirements.txt

## Running the Demos

The package includes several demo scripts to showcase its capabilities:

### Boxplot Demo (Recommended)

```bash
python scripts/boxplot_demo.py
```

This script provides a visual comparison of different QP solving strategies using randomly generated problems, without requiring model training. It demonstrates the core concepts behind TransformerMPC by showing the performance impact of:
- Removing inactive constraints
- Using warm starts with different qualities
- Combining these strategies

The visualizations include boxplots, violin plots, and bar charts comparing solve times across different strategies.

### Simple Demo

```bash
python scripts/simple_demo.py
```

This script demonstrates the complete pipeline: generating QP problems, training models, and evaluating performance. After completion, it saves performance comparison plots in the `demo_results/results` directory.

### Verify Package Structure

To check that the package is installed correctly:

```bash
python scripts/verify_structure.py
```

### Customizing Demo Parameters

You can customize the boxplot demo by modifying parameters:

```bash
# Generate more problems with different dimensions
python scripts/boxplot_demo.py --num_samples 50 --state_dim 6 --input_dim 3 --horizon 10

# Save results to a custom directory 
python scripts/boxplot_demo.py --output_dir my_results
```

Similarly, for the simple demo:

```bash
# Generate QP problems with specific parameters
python scripts/simple_demo.py --num_samples 200 --state_dim 6 --input_dim 3 --horizon 10

# Customize training parameters
python scripts/simple_demo.py --epochs 20 --batch_size 32

# Use GPU for training if available
python scripts/simple_demo.py --use_gpu
```

## Usage in Projects

### Basic Example

```python
from transformermpc import TransformerMPC
import numpy as np

# Define your QP problem parameters
Q = np.array([[4.0, 1.0], [1.0, 2.0]])
c = np.array([1.0, 1.0])
A = np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
b = np.array([0.0, 0.0, -1.0, 2.0])

# Initialize the TransformerMPC solver
solver = TransformerMPC()

# Solve with model acceleration
solution, solve_time = solver.solve(Q, c, A, b)

print(f"Solution: {solution}")
print(f"Solve time: {solve_time} seconds")
```

### General Usage

```python
from transformermpc import TransformerMPC, QPProblem
import numpy as np

# Define your QP problem parameters
Q = np.array([[4.0, 1.0], [1.0, 2.0]])
c = np.array([1.0, 1.0])
A = np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0], [1.0, 1.0]])
b = np.array([0.0, 0.0, -1.0, 2.0])
initial_state = np.array([0.5, 0.5])  # Optional: initial state for MPC problems

# Create a QP problem instance
qp_problem = QPProblem(
    Q=Q,
    c=c,
    A=A,
    b=b,
    initial_state=initial_state  # Optional
)

# Initialize with custom settings
solver = TransformerMPC(
    use_constraint_predictor=True,
    use_warm_start_predictor=True,
    fallback_on_violation=True
)

# Solve the problem
solution, solve_time = solver.solve(qp_problem=qp_problem)
print(f"Solution: {solution}")
print(f"Solve time: {solve_time} seconds")

# Compare with baseline
baseline_solution, baseline_time = solver.solve_baseline(qp_problem=qp_problem)
print(f"Baseline time: {baseline_time} seconds")
```
## If you find our work useful, please cite us
```
@article{zinage2024transformermpc,
  title={TransformerMPC: Accelerating Model Predictive Control via Transformers},
  author={Zinage, Vrushabh and Khalil, Ahmed and Bakolas, Efstathios},
  journal={arXiv preprint arXiv:2409.09266},
  year={2024}
}
```

## License

This project is licensed under the MIT License. 
