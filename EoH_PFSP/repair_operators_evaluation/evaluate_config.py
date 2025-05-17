#!/usr/bin/env python
"""
Configuration-based evaluation script for repair operators.
Just edit the configuration section below and run this script.
"""

#################################################
# CONFIGURATION - EDIT THIS PART
#################################################

# Select the operators to evaluate
OPERATORS_TO_EVALUATE = [
    "greedy_repair",
    "regret_repair", 
    "greedy_repair_with_local_search",
    "insertion_priority_repair",
    "multi_phase_repair"
]

# Select the problem instances to solve
# Each path is relative to the repository root
PROBLEM_INSTANCES = [
    "../git_repo/PFSP/data/j50_m20/j50_m20_08.txt",
    # Add more instances as needed
    # "../git_repo/PFSP/data/j20_m5/j20_m5_01.txt",
    # "../git_repo/PFSP/data/j100_m10/j100_m10_01.txt"
]

# Number of runs per problem/operator combination
N_RUNS = 2

# Number of iterations for each run
ITERATIONS = 300

# Save results to CSV file
SAVE_TO_CSV = True
CSV_FILENAME = "repair_operator_results.csv"

# Print detailed results for each run
PRINT_DETAILED_RESULTS = True

#################################################
# DO NOT EDIT BELOW THIS LINE
#################################################

import sys
import os
import pandas as pd
import numpy as np
import time
from copy import deepcopy

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from repair_operators_evaluation package
import importlib
from repair_operators_evaluation.utils.problem import Data, NEH
from repair_operators_evaluation.operators.destroy_operators import (
    random_removal, adjacent_removal
)

# Import ALNS
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations


def evaluate_operator(repair_operator, data_file, n_runs=2, iters=300, seed=2345, detailed=True):
    """
    Evaluate a single repair operator on a single data file.
    """
    # Extract instance name from file path
    instance_name = data_file.split('/')[-1].split('.')[0]
    
    if detailed:
        print(f"\nEvaluating {repair_operator.__name__} on {instance_name}...")
    
    # Load data
    data = Data.from_file(data_file)
    
    # Store results
    gaps = []
    runtimes = []
    
    for run in range(n_runs):
        current_seed = seed + run
        
        if detailed:
            print(f"  Run {run+1}/{n_runs} (seed: {current_seed})")
        
        # Create initial solution
        init = NEH(data.processing_times, data)
        initial_obj = init.objective()
        
        # Setup ALNS
        alns = ALNS(np.random.default_rng(current_seed))
        
        # Add destroy operators
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(adjacent_removal)
        
        # Add repair operator
        alns.add_repair_operator(repair_operator)
        
        # Configure ALNS
        select = AlphaUCB(
            scores=[5, 2, 1, 0.5],
            alpha=0.05,
            num_destroy=len(alns.destroy_operators),
            num_repair=len(alns.repair_operators),
        )
        
        accept = SimulatedAnnealing.autofit(initial_obj, 0.05, 0.50, iters)
        stop = MaxIterations(iters)
        
        # Run ALNS
        start_time = time.time()
        result = alns.iterate(deepcopy(init), select, accept, stop)
        runtime = time.time() - start_time
        
        # Calculate metrics
        final_obj = result.best_state.objective()
        gap = 100 * (final_obj - data.bkv) / data.bkv
        
        gaps.append(gap)
        runtimes.append(runtime)
        
        if detailed:
            print(f"    Initial: {initial_obj}, Final: {final_obj}")
            print(f"    Gap: {gap:.2f}%, Runtime: {runtime:.2f}s")
    
    # Calculate average results
    avg_gap = np.mean(gaps)
    avg_runtime = np.mean(runtimes)
    
    if detailed:
        print(f"  Average gap: {avg_gap:.2f}%, Average runtime: {avg_runtime:.2f}s")
    
    return {
        'instance': instance_name,
        'operator': repair_operator.__name__,
        'avg_gap': avg_gap,
        'min_gap': min(gaps),
        'max_gap': max(gaps),
        'avg_runtime': avg_runtime
    }


def compare_operators(operators, data_files, n_runs=2, iters=300, detailed=True):
    """
    Compare multiple operators on multiple problem instances.
    """
    results = []
    
    for data_file in data_files:
        for operator in operators:
            result = evaluate_operator(
                operator, data_file, n_runs, iters, detailed=detailed
            )
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary by instance and operator
    summary = df.pivot_table(
        index='instance', 
        columns='operator', 
        values='avg_gap',
        aggfunc='mean'
    )
    
    print("\nAverage gap (%) by instance and operator:")
    print(summary)
    
    # Calculate overall performance
    operator_summary = df.groupby('operator').agg({
        'avg_gap': ['mean', 'min', 'max'],
        'avg_runtime': 'mean'
    })
    
    operator_summary.columns = ['avg_gap', 'min_gap', 'max_gap', 'avg_runtime']
    operator_summary = operator_summary.sort_values('avg_gap')
    
    print("\nOverall operator performance:")
    print(operator_summary)
    
    # Find best operator
    best_operator = operator_summary.index[0]
    best_gap = operator_summary.loc[best_operator, 'avg_gap']
    
    print(f"\nBest operator: {best_operator} with average gap: {best_gap:.2f}%")
    
    return df


if __name__ == "__main__":
    # Load operators dynamically
    operators = []
    
    # Import the operator modules
    basic_ops = importlib.import_module("repair_operators_evaluation.operators.basic_operators")
    custom_ops = importlib.import_module("repair_operators_evaluation.operators.custom_operators")
    
    # Get the operators
    for op_name in OPERATORS_TO_EVALUATE:
        if hasattr(basic_ops, op_name):
            operators.append(getattr(basic_ops, op_name))
        elif hasattr(custom_ops, op_name):
            operators.append(getattr(custom_ops, op_name))
        else:
            print(f"Warning: Operator '{op_name}' not found - skipping")
    
    if not operators:
        print("Error: No valid operators found to evaluate")
        sys.exit(1)
        
    if not PROBLEM_INSTANCES:
        print("Error: No problem instances specified")
        sys.exit(1)
    
    print(f"Evaluating {len(operators)} operators on {len(PROBLEM_INSTANCES)} problem instances")
    print(f"Each combination will be run {N_RUNS} times with {ITERATIONS} iterations")
    
    # Run the comparison
    results_df = compare_operators(
        operators, 
        PROBLEM_INSTANCES, 
        n_runs=N_RUNS, 
        iters=ITERATIONS,
        detailed=PRINT_DETAILED_RESULTS
    )
    
    # Save to CSV if requested
    if SAVE_TO_CSV:
        results_df.to_csv(CSV_FILENAME, index=False)
        print(f"\nResults saved to {CSV_FILENAME}") 