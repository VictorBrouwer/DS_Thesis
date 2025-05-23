#!/usr/bin/env python3
"""
Quick evaluation of EoH operator on a subset of instances
"""

import os
import json
import time
import pandas as pd
import numpy as np
from copy import deepcopy
import numpy.random as rnd

# Import PFSP components
import PFSP
from PFSP import (
    Data, Solution, NEH, compute_makespan,
    random_removal, adjacent_removal, 
    greedy_repair_then_local_search as greedy_repair
)

# Import ALNS components
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations

SEED = 2345
DATA = None

def load_best_eoh_operator():
    """Load the best EoH operator"""
    with open("demo_results/final_summary.json", 'r') as f:
        summary = json.load(f)
    
    best_individual = summary["final_population"][0]
    operator_code = best_individual["code"]
    
    namespace = {}
    exec(operator_code, globals(), namespace)
    return namespace["llm_repair"], best_individual

def create_simple_baseline():
    """Create a simple baseline operator"""
    def simple_greedy(state: Solution, rng, **kwargs) -> Solution:
        state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))
        while len(state.unassigned) != 0:
            job = state.unassigned.pop()
            state.opt_insert(job)
        return state
    return simple_greedy

def evaluate_operator(operator_func, data_file, iters=600):
    """Evaluate a single operator on a single instance"""
    # Load data
    data = Data.from_file(data_file)
    global DATA
    DATA = data
    PFSP.DATA = data
    
    # Create initial solution
    init = NEH(data.processing_times)
    
    # Setup ALNS
    alns = ALNS(rnd.default_rng(SEED))
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(adjacent_removal)
    alns.add_repair_operator(operator_func)
    
    # Configure ALNS
    select = AlphaUCB(
        scores=[5, 2, 1, 0.5],
        alpha=0.05,
        num_destroy=len(alns.destroy_operators),
        num_repair=len(alns.repair_operators),
    )
    accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, iters)
    stop = MaxIterations(iters)
    
    # Run ALNS
    start_time = time.time()
    result = alns.iterate(deepcopy(init), select, accept, stop)
    runtime = time.time() - start_time
    
    # Calculate results
    objective = result.best_state.objective()
    gap = 100 * (objective - data.bkv) / data.bkv
    
    return {
        'objective': objective,
        'gap': gap,
        'runtime': runtime,
        'bkv': data.bkv,
        'n_jobs': data.n_jobs,
        'n_machines': data.n_machines
    }

def main():
    print("🔬 Quick EoH Evaluation")
    print("=" * 40)
    
    # Load operators
    eoh_operator, eoh_info = load_best_eoh_operator()
    baseline_operator = create_simple_baseline()
    
    print(f"EoH Operator: {eoh_info['algorithm']}")
    print(f"Training Gap: {eoh_info['gap']:.2f}%")
    
    # Test instances - representative sample
    test_instances = [
        "data/j20_m5/j20_m5_01.txt",
        "data/j20_m5/j20_m5_02.txt", 
        "data/j20_m10/j20_m10_01.txt",
        "data/j20_m20/j20_m20_01.txt",
        "data/j50_m5/j50_m5_01.txt",
        "data/j50_m10/j50_m10_01.txt",
        "data/j50_m20/j50_m20_01.txt",
        "data/j100_m5/j100_m5_01.txt",
        "data/j100_m10/j100_m10_01.txt"
    ]
    
    results = []
    
    print(f"\nTesting on {len(test_instances)} instances...")
    
    for i, instance in enumerate(test_instances):
        instance_name = instance.split('/')[-1].split('.')[0]
        problem_type = instance.split('/')[-2]
        
        print(f"\n[{i+1}/{len(test_instances)}] {instance_name} ({problem_type})")
        
        # Test EoH operator
        print("  Testing EoH operator...")
        eoh_result = evaluate_operator(eoh_operator, instance)
        
        # Test baseline
        print("  Testing baseline...")
        baseline_result = evaluate_operator(baseline_operator, instance)
        
        # Store results
        result = {
            'instance': instance_name,
            'problem_type': problem_type,
            'size': f"{eoh_result['n_jobs']}x{eoh_result['n_machines']}",
            'n_jobs': eoh_result['n_jobs'],
            'n_machines': eoh_result['n_machines'],
            'bkv': eoh_result['bkv'],
            'eoh_objective': eoh_result['objective'],
            'eoh_gap': eoh_result['gap'],
            'eoh_time': eoh_result['runtime'],
            'baseline_objective': baseline_result['objective'],
            'baseline_gap': baseline_result['gap'],
            'baseline_time': baseline_result['runtime'],
            'improvement': baseline_result['gap'] - eoh_result['gap']
        }
        results.append(result)
        
        print(f"    EoH: {eoh_result['objective']} (gap: {eoh_result['gap']:.2f}%)")
        print(f"    Baseline: {baseline_result['objective']} (gap: {baseline_result['gap']:.2f}%)")
        print(f"    Improvement: {result['improvement']:.2f} pp")
    
    # Analysis
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("QUICK EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"EoH Average Gap: {df['eoh_gap'].mean():.2f}%")
    print(f"Baseline Average Gap: {df['baseline_gap'].mean():.2f}%")
    print(f"Average Improvement: {df['improvement'].mean():.2f} percentage points")
    print(f"EoH Win Rate: {(df['eoh_gap'] < df['baseline_gap']).mean()*100:.1f}%")
    
    print(f"\nBy Problem Size:")
    size_summary = df.groupby('n_jobs').agg({
        'eoh_gap': 'mean',
        'baseline_gap': 'mean', 
        'improvement': 'mean'
    }).round(2)
    print(size_summary)
    
    # Save results
    df.to_csv('quick_eoh_evaluation.csv', index=False)
    print(f"\n💾 Results saved to: quick_eoh_evaluation.csv")
    
    # Show detailed results
    print(f"\nDetailed Results:")
    print(df[['instance', 'size', 'eoh_gap', 'baseline_gap', 'improvement']].to_string(index=False, float_format='%.2f'))

if __name__ == "__main__":
    main() 