"""
Evaluation framework for testing repair operators.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations, MaxRuntime

from repair_operators_evaluation.utils.problem import Data, NEH
from repair_operators_evaluation.operators.destroy_operators import (
    random_removal, adjacent_removal
)


def evaluate_repair_operator(repair_operator, data_files, destroy_operators=None, 
                         n_runs=3, iters=600, seed=2345):
    """
    Evaluates the performance of a repair operator across multiple instances and runs.
    
    Args:
        repair_operator: The repair operator function to evaluate
        data_files: List of data file paths to test on
        destroy_operators: List of destroy operators to use (defaults to [random_removal, adjacent_removal])
        n_runs: Number of runs per instance (with different seeds)
        iters: Number of iterations per run
        seed: Base random seed
    
    Returns:
        Dictionary with evaluation results
    """
    if destroy_operators is None:
        destroy_operators = [random_removal, adjacent_removal]
    
    results = {
        'instance': [],
        'run': [],
        'initial_obj': [],
        'final_obj': [],
        'gap_to_bkv': [],
        'runtime': [],
        'iterations': []
    }
    
    for data_file in data_files:
        # Extract instance name
        instance_name = data_file.split('/')[-1].split('.')[0]
        print(f"\nEvaluating on instance: {instance_name}")
        
        # Load data
        data = Data.from_file(data_file)
        
        for run in range(n_runs):
            current_seed = seed + run
            print(f"  Run {run+1}/{n_runs} (seed: {current_seed})")
            
            # Create initial solution
            init = NEH(data.processing_times, data)
            initial_obj = init.objective()
            
            # Setup ALNS
            alns = ALNS(np.random.default_rng(current_seed))
            
            # Add destroy operators
            for destroy_op in destroy_operators:
                alns.add_destroy_operator(destroy_op)
            
            # Add the repair operator being evaluated
            alns.add_repair_operator(repair_operator)
            
            # Configure ALNS parameters
            select = AlphaUCB(
                scores=[5, 2, 1, 0.5],
                alpha=0.05,
                num_destroy=len(alns.destroy_operators),
                num_repair=len(alns.repair_operators),
            )
            
            accept = SimulatedAnnealing.autofit(initial_obj, 0.05, 0.50, iters)
            stop = MaxIterations(iters)
            
            # Run ALNS and time it
            start_time = time.time()
            result = alns.iterate(deepcopy(init), select, accept, stop)
            runtime = time.time() - start_time
            
            # Calculate metrics
            final_obj = result.best_state.objective()
            gap = 100 * (final_obj - data.bkv) / data.bkv
            
            # Record results
            results['instance'].append(instance_name)
            results['run'].append(run+1)
            results['initial_obj'].append(initial_obj)
            results['final_obj'].append(final_obj)
            results['gap_to_bkv'].append(gap)
            results['runtime'].append(runtime)
            results['iterations'].append(iters)
            
            print(f"    Initial objective: {initial_obj}")
            print(f"    Final objective: {final_obj}")
            print(f"    Gap to BKV: {gap:.2f}%")
            print(f"    Runtime: {runtime:.2f}s")
    
    # Calculate summary statistics
    summary = {
        'avg_gap': np.mean(results['gap_to_bkv']),
        'min_gap': np.min(results['gap_to_bkv']),
        'max_gap': np.max(results['gap_to_bkv']),
        'avg_runtime': np.mean(results['runtime']),
        'repair_operator_name': repair_operator.__name__
    }
    
    print(f"\nSummary for {repair_operator.__name__}:")
    print(f"  Average gap to BKV: {summary['avg_gap']:.2f}%")
    print(f"  Min/Max gap to BKV: {summary['min_gap']:.2f}% / {summary['max_gap']:.2f}%")
    print(f"  Average runtime: {summary['avg_runtime']:.2f}s")
    
    return results, summary


def compare_repair_operators(repair_operators, data_files, n_runs=3, iters=600):
    """
    Compares multiple repair operators and visualizes their performance.
    
    Args:
        repair_operators: List of repair operator functions to compare
        data_files: List of data file paths to test on
        n_runs: Number of runs per instance
        iters: Number of iterations per run
    
    Returns:
        Dictionary with all evaluation results
    """
    all_results = {}
    summaries = []
    
    for repair_op in repair_operators:
        print(f"\n===== Evaluating {repair_op.__name__} =====")
        results, summary = evaluate_repair_operator(
            repair_op, data_files, n_runs=n_runs, iters=iters
        )
        all_results[repair_op.__name__] = results
        summaries.append(summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summaries)
    
    # Gap comparison visualization
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['repair_operator_name'], summary_df['avg_gap'])
    plt.ylabel('Average Gap to BKV (%)')
    plt.title('Performance Gap Comparison')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Runtime comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['repair_operator_name'], summary_df['avg_runtime'])
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Runtime Comparison')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nSummary Table:")
    print(summary_df[['repair_operator_name', 'avg_gap', 'min_gap', 'max_gap', 'avg_runtime']])
    
    return all_results, summary_df 