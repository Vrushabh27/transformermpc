# TransformerMPC

Accelerating Model Predictive Control via Transformers

## Overview

TransformerMPC is a Python package that enhances the efficiency of solving Quadratic Programming (QP) problems in Model Predictive Control (MPC) using transformer-based neural networks. It employs two specialized transformer models:

1. **Constraint Predictor**: Identifies inactive constraints in QP formulations
2. **Warm Start Predictor**: Generates better initial points for QP solvers

By combining these models, TransformerMPC significantly reduces computation time while maintaining solution quality.

## Citation

If you use this package, please cite:

```
Zinage, Vrushabh, Ahmed Khalil, and Efstathios Bakolas. "TransformerMPC: Accelerating Model Predictive Control via Transformers." Accepted to ICRA (2025).
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
- Transformers >= 4.15.0
- OSQP >= 0.6.2
- Additional dependencies specified in requirements.txt

## Usage

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

# Solve with transformer acceleration
solution, solve_time = solver.solve(Q, c, A, b)

print(f"Solution: {solution}")
print(f"Solve time: {solve_time} seconds")
```

### Advanced Usage

```python
from transformermpc import TransformerMPC, QPProblem

# Create a QP problem instance
qp_problem = QPProblem(
    Q=Q,
    c=c,
    A=A,
    b=b,
    initial_state=x0
)

# Initialize with custom settings
solver = TransformerMPC(
    use_constraint_predictor=True,
    use_warm_start_predictor=True,
    fallback_on_violation=True
)

# Solve the problem
solution = solver.solve(qp_problem)

# Compare with baseline
baseline_solution = solver.solve_baseline(qp_problem)
```

## Training Custom Models

### Generating Training Data

```python
from transformermpc.data import QPGenerator, QPDataset

# Generate QP problems
generator = QPGenerator(
    state_dim=4,
    horizon=10,
    num_samples=20000
)
qp_problems = generator.generate()

# Create dataset
dataset = QPDataset(qp_problems)
train_data, test_data = dataset.split(test_size=0.2)
```

### Training Models

```python
from transformermpc.training import ModelTrainer
from transformermpc.models import ConstraintPredictor, WarmStartPredictor

# Train constraint predictor
cp_trainer = ModelTrainer(
    model=ConstraintPredictor(),
    train_dataset=train_data,
    val_dataset=test_data,
    num_epochs=2000
)
cp_trainer.train(log_dir="logs/constraint_predictor")

# Train warm start predictor
ws_trainer = ModelTrainer(
    model=WarmStartPredictor(),
    train_dataset=train_data,
    val_dataset=test_data,
    num_epochs=2000
)
ws_trainer.train(log_dir="logs/warm_start_predictor")
```

## Demo

The package includes a comprehensive demo script that:

1. Generates 20,000 QP problems
2. Trains both transformer models for 2000 epochs
3. Tests the models on a separate test set
4. Compares computation time between standard OSQP and transformer-enhanced solving

Run the demo with:

```bash
python -m transformermpc.demo.demo
```

## Benchmarking Results

![Computation Time Comparison](./benchmarking_results.png)

The above graph shows the significant speed improvements achieved by TransformerMPC compared to standard OSQP solving. The transformer-enhanced pipeline typically achieves 2-5x faster computation times.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
