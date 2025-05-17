# ILS-Python: Prize Collecting TSP Solver

This folder contains Python implementations for solving the Prize Collecting Traveling Salesman Problem (PCTSP) using Iterated Local Search.

## Contents

- `run_pctsp_with_data.py`: Script to run PCTSP solver on a specific data instance
- `run_all_problem_sizes.py`: Script to process all problem sizes and generate statistics
- `debug_pctsp.py`: Script to debug solution quality calculations
- `inspect_pickle.py`: Script to inspect the structure of pickle data files

## Prerequisites

- Python 3.6+
- NumPy
- tabulate (for formatted output tables)

Install required packages:
```
pip install numpy tabulate
```

## Usage

### Running PCTSP solver on a specific instance

```bash
python run_pctsp_with_data.py ../pctsp_data/pctsp_20_20_instances.pkl -i 0 -v
```

Options:
- `-i/--instance`: Run for a specific instance index
- `-a/--all`: Run for all instances in the file
- `-v/--verbose`: Print detailed output
- `-s/--stochastic`: Use stochastic prize instead of deterministic
- `-r/--size-ratio`: Ratio of cities to use (default: 0.7)

### Processing all problem sizes

```bash
python run_all_problem_sizes.py
```

This will process all pickle files in the parent `pctsp_data` directory, run the PCTSP algorithm on each instance, and produce a summary table with statistics.

### Debugging solution quality

```bash
python debug_pctsp.py ../pctsp_data/pctsp_20_20_instances.pkl 0
```

This script provides a detailed breakdown of solution quality, showing travel costs, penalties, and prize collection.

### Inspecting pickle files

```bash
python inspect_pickle.py ../pctsp_data/pctsp_20_20_instances.pkl
```

This script inspects and displays the structure of the pickle data files.

## Problem Instances

Data files are located in the parent `pctsp_data` directory:

- `pctsp_20_20_instances.pkl`: 20 instances with 20 cities each
- `pctsp_50_20_instances.pkl`: 20 instances with 50 cities each
- `pctsp_100_20_instances.pkl`: 20 instances with 100 cities each

## Performance Results

| Problem Size | Average Objective | Average Time (seconds) |
|--------------|-------------------|------------------------|
| 20           | 4.60              | 0.10                   |
| 50           | 12.40             | 0.09                   |
| 100          | 23.59             | 0.10                   | 