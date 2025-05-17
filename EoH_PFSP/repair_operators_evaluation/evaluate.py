#!/usr/bin/env python
"""
Simple evaluation script for repair operators.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from copy import deepcopy

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from repair_operators_evaluation package
from repair_operators_evaluation.operators.basic_operators import (
    greedy_repair, regret_repair, greedy_repair_with_local_search
)
from repair_operators_evaluation.operators.custom_operators import (
    insertion_priority_repair, multi_phase_repair
)
from repair_operators_evaluation.utils.problem import Data, NEH
from repair_operators_evaluation.operators.destroy_operators import (
    random_removal, adjacent_removal
)

# Import ALNS
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations

def evaluate_operator(repair_operator, data_file, n_runs=2, iters=300, seed=2345):
    """
    Evaluate a single repair operator on a single data file.
    """
    print(f"\nEvaluating {repair_operator.__name__} on {data_file.split('/')[-1]}...")
    
    # Load data
    data = Data.from_file(data_file)
    
    # Store results
    gaps = []
    runtimes = []
    
    for run in range(n_runs):
        current_seed = seed + run
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
        
        print(f"    Initial: {initial_obj}, Final: {final_obj}")
        print(f"    Gap: {gap:.2f}%, Runtime: {runtime:.2f}s")
    
    # Calculate average results
    avg_gap = np.mean(gaps)
    avg_runtime = np.mean(runtimes)
    
    print(f"  Average gap: {avg_gap:.2f}%, Average runtime: {avg_runtime:.2f}s")
    
    return {
        'operator': repair_operator.__name__,
        'avg_gap': avg_gap,
        'min_gap': min(gaps),
        'max_gap': max(gaps),
        'avg_runtime': avg_runtime
    }

def compare_operators(operators, data_file, n_runs=2, iters=300):
    """
    Compare multiple operators on a single data file.
    """
    results = []
    
    for operator in operators:
        result = evaluate_operator(operator, data_file, n_runs, iters)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\nSummary of results:")
    print(df)
    
    # Find best operator
    best_idx = df['avg_gap'].idxmin()
    best_operator = df.loc[best_idx, 'operator']
    best_gap = df.loc[best_idx, 'avg_gap']
    
    print(f"\nBest operator: {best_operator} with average gap: {best_gap:.2f}%")
    
    # Save to CSV
    df.to_csv("operator_comparison_results.csv", index=False)
    print("\nResults saved to operator_comparison_results.csv")
    
    return df

if __name__ == "__main__":
    # Define test instances
    test_file = "../git_repo/PFSP/data/j50_m20/j50_m20_08.txt"
    
    # Define operators to compare
    operators_to_compare = [
        greedy_repair,
        regret_repair,
        greedy_repair_with_local_search,
        insertion_priority_repair,
        multi_phase_repair
    ]
    
    # Run comparison
    compare_operators(operators_to_compare, test_file, n_runs=2, iters=300) 