# TransformerMPC

Accelerating Model Predictive Control via Transformers

## Overview

TransformerMPC is a Python package that enhances the efficiency of solving Quadratic Programming (QP) problems in Model Predictive Control (MPC) using transformer-based neural networks. It employs two specialized transformer models:

1. **Constraint Predictor**: Identifies inactive constraints in QP formulations
2. **Warm Start Predictor**: Generates better initial points for QP solvers

By combining these models, TransformerMPC significantly reduces computation time while maintaining solution quality.

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
- Transformers >= 4.15.0
- OSQP >= 0.6.2
- Additional dependencies specified in requirements.txt

## Running the Demo

The package includes a simplified demo script that demonstrates the complete workflow:

```bash
python simple_demo.py
```

This script performs the entire pipeline: generating QP problems, training models, and evaluating performance. After completion, it saves performance comparison plots in the `demo_results/results` directory.

### Additional Scripts

The package also includes these utility scripts:

1. **run_demo.py**: A wrapper script that executes the main demo module. Use it when working with the installed package:

   ```bash
   python run_demo.py
   ```

2. **test_benchmark.py**: For comprehensive performance evaluation with more metrics and visualizations:

   ```bash
   python test_benchmark.py
   ```
   
   This script focuses on benchmarking performance across multiple problems and generates detailed visualizations, including boxplots comparing solve times between the standard OSQP solver and transformer-enhanced methods.

All scripts save their results (including performance metrics and visualization plots) in the `demo_results/results` directory. The key visualizations include:

- **solve_time_comparison.png**: Line plot comparing baseline vs. transformer solve times
- **solve_time_boxplot.png**: Statistical distribution of solve times across different solver configurations

### Customizing Demo Parameters

You can customize the demo by modifying the following parameters:

#### Data Generation Parameters:
```bash
# Generate 5000 QP problems with state dimension 6, input dimension 3, and horizon 15
python simple_demo.py --num_samples 5000 --state_dim 6 --input_dim 3 --horizon 15
```

#### Training Parameters:
```bash
# Train the constraint predictor for 200 epochs and warm start predictor for 300 epochs
python simple_demo.py --cp_epochs 200 --ws_epochs 300 --batch_size 128
```

The number of epochs and samples significantly impact training time and model performance:
- `--cp_epochs`: Number of training epochs for the Constraint Predictor model
- `--ws_epochs`: Number of training epochs for the Warm Start Predictor model
- `--num_samples`: Number of QP problems to generate for training

Increasing these values will generally improve model accuracy but require more computation time. For quick experimentation, use lower values (e.g., 50-100 epochs, 1000-2000 samples). For production-quality models, consider higher values (300+ epochs, 5000+ samples).

#### Hardware Options:
```bash
# Use GPU for training if available
python simple_demo.py --use_gpu
```

#### Other Options:
```bash
# Skip training and use pre-trained models if available
python simple_demo.py --skip_training

# Specify custom output directory
python simple_demo.py --output_dir custom_results
```

### All Available Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_samples` | Number of QP problems to generate | 2000 |
| `--state_dim` | State dimension for MPC problems | 4 |
| `--input_dim` | Input dimension for MPC problems | 2 |
| `--horizon` | Time horizon for MPC problems | 10 |
| `--cp_epochs` | Epochs for constraint predictor training | 100 |
| `--ws_epochs` | Epochs for warm start predictor training | 100 |
| `--batch_size` | Batch size for training | 64 |
| `--test_size` | Fraction of data to use for testing | 0.2 |
| `--output_dir` | Directory to save results | "demo_results" |
| `--skip_training` | Skip training and use pretrained models | False |
| `--use_gpu` | Use GPU if available | False |

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
# Note: By default, this uses pre-trained models that come with the package.
# To train custom models with different epochs and samples, use the demo script
# or the training utilities as described in the "Customizing Demo Parameters" section.

# Solve with transformer acceleration
solution, solve_time = solver.solve(Q, c, A, b)

print(f"Solution: {solution}")
print(f"Solve time: {solve_time} seconds")
```

### Advanced Usage

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


## License

This project is licensed under the MIT License - see the LICENSE file for details. 