#!/usr/bin/env python
"""
Quick evaluation script that tests just one repair operator.
"""

import sys
import os
import time
from copy import deepcopy
import numpy as np

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

#################################################
# CONFIGURATION - EDIT THIS PART
#################################################

# The operator to evaluate (only one at a time for quick testing)
OPERATOR_NAME = "greedy_repair_with_local_search"

# The problem instance to solve (just one at a time)
PROBLEM_INSTANCE = "../git_repo/PFSP/data/j50_m20/j50_m20_08.txt"

# Number of runs
N_RUNS = 2

# Number of iterations per run
ITERATIONS = 300

#################################################
# DO NOT EDIT BELOW THIS LINE
#################################################

def quick_evaluate():
    """
    Quickly evaluate a single operator on a single problem instance.
    """
    print(f"Quick evaluation of {OPERATOR_NAME} on {PROBLEM_INSTANCE.split('/')[-1]}")
    print(f"Running {N_RUNS} times with {ITERATIONS} iterations each")
    
    # Import operator modules
    basic_ops = importlib.import_module("repair_operators_evaluation.operators.basic_operators")
    custom_ops = importlib.import_module("repair_operators_evaluation.operators.custom_operators")
    
    # Find the operator
    if hasattr(basic_ops, OPERATOR_NAME):
        operator = getattr(basic_ops, OPERATOR_NAME)
    elif hasattr(custom_ops, OPERATOR_NAME):
        operator = getattr(custom_ops, OPERATOR_NAME)
    else:
        print(f"Error: Operator '{OPERATOR_NAME}' not found")
        sys.exit(1)
    
    # Load problem instance
    try:
        data = Data.from_file(PROBLEM_INSTANCE)
        print(f"Problem: {data.n_jobs} jobs, {data.n_machines} machines, BKV: {data.bkv}")
    except Exception as e:
        print(f"Error loading problem instance: {e}")
        sys.exit(1)
    
    # Run the evaluation
    gaps = []
    runtimes = []
    objectives = []
    
    for run in range(N_RUNS):
        seed = 2345 + run
        print(f"\nRun {run+1}/{N_RUNS} (seed: {seed}):")
        
        # Create initial solution
        init = NEH(data.processing_times, data)
        initial_obj = init.objective()
        
        # Setup ALNS
        alns = ALNS(np.random.default_rng(seed))
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(adjacent_removal)
        alns.add_repair_operator(operator)
        
        # Configure ALNS
        select = AlphaUCB(
            scores=[5, 2, 1, 0.5],
            alpha=0.05,
            num_destroy=len(alns.destroy_operators),
            num_repair=len(alns.repair_operators),
        )
        
        accept = SimulatedAnnealing.autofit(initial_obj, 0.05, 0.50, ITERATIONS)
        stop = MaxIterations(ITERATIONS)
        
        # Run ALNS
        print(f"  Starting ALNS with initial objective: {initial_obj}")
        start_time = time.time()
        result = alns.iterate(deepcopy(init), select, accept, stop)
        runtime = time.time() - start_time
        
        # Calculate metrics
        final_obj = result.best_state.objective()
        gap = 100 * (final_obj - data.bkv) / data.bkv
        
        objectives.append(final_obj)
        gaps.append(gap)
        runtimes.append(runtime)
        
        print(f"  Final objective: {final_obj}")
        print(f"  Gap to BKV: {gap:.2f}%")
        print(f"  Runtime: {runtime:.2f} seconds")
    
    # Print summary
    print("\nSummary:")
    print(f"  Average objective: {np.mean(objectives):.2f}")
    print(f"  Average gap: {np.mean(gaps):.2f}%")
    print(f"  Min/Max gap: {min(gaps):.2f}% / {max(gaps):.2f}%")
    print(f"  Average runtime: {np.mean(runtimes):.2f} seconds")

if __name__ == "__main__":
    quick_evaluate() 