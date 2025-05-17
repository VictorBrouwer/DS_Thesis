# Repair Operators Evaluation Framework for PFSP

A modular framework for developing and evaluating different repair operators for the Permutation Flow Shop Problem (PFSP) using Adaptive Large Neighborhood Search (ALNS).

## Overview

This framework provides:

1. Base components for PFSP problem representation
2. Utilities for creating and visualizing solutions
3. A collection of destroy and repair operators 
4. An evaluation system to compare different repair operators

## Directory Structure

```
repair_operators_evaluation/
├── __init__.py
├── example.py                 # Example usage script
├── operators/                 # Repair and destroy operators
│   ├── __init__.py
│   ├── basic_operators.py     # Basic repair operators
│   └── destroy_operators.py   # Destroy operators
├── evaluation/                # Evaluation framework
│   ├── __init__.py
│   └── evaluator.py           # Methods to evaluate operators
└── utils/                     # Utility functions and classes
    ├── __init__.py
    ├── problem.py             # Problem representation
    └── visualization.py       # Visualization utilities
```

## How to Use

### Creating a New Repair Operator

1. Create a new file in the `operators/` directory or add to an existing one
2. Implement your repair operator following this pattern:

```python
def my_custom_repair(state, rng, **kwargs):
    """
    My custom repair operator.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters
        
    Returns:
        Repaired solution
    """
    # Your repair logic here
    
    return state
```

### Evaluating Repair Operators

To evaluate your repair operator against others:

```python
from repair_operators_evaluation.operators.basic_operators import greedy_repair
from repair_operators_evaluation.evaluation.evaluator import compare_repair_operators
from my_operators import my_custom_repair

# List of repair operators to compare
repair_operators = [
    greedy_repair,
    my_custom_repair
]

# Test instances
test_instances = [
    "path/to/instance1.txt",
    "path/to/instance2.txt"
]

# Run comparison
results, summary = compare_repair_operators(
    repair_operators, 
    test_instances,
    n_runs=3,      # 3 runs per instance
    iters=600      # 600 iterations per run
)
```

## Example

See `example.py` for a complete example of how to use the framework.

## Extending the Framework

- To add new destroy operators: add to `operators/destroy_operators.py`
- To add new repair operators: add to `operators/basic_operators.py` or create new files
- To customize evaluation: modify `evaluation/evaluator.py`

## References

This framework is based on the following papers and libraries:

1. T. Stützle and R. Ruiz (2018). "Iterated local search and variable neighborhood search"
2. N. Christofides, A. Mingozzi, and P. Toth (1979). "The vehicle routing problem"
3. The [alns](https://github.com/N-Wouda/ALNS) Python package 

## License

MIT License 