#!/usr/bin/env python3
"""
Evaluation of best EoH operator following evaluation.py structure
"""

import os
import json
import time
import pandas as pd
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

# Global variables
SEED = 2345
DATA = None

def load_best_eoh_operator():
    """Load the best operator from EoH results"""
    with open("demo_results/final_summary.json", 'r') as f:
        summary = json.load(f)
    
    best_individual = summary["final_population"][0]
    operator_code = best_individual["code"]
    
    # Create operator function
    namespace = {}
    exec(operator_code, globals(), namespace)
    return namespace["llm_repair"]

def create_baseline_repair():
    """Create baseline repair operator"""
    def baseline_repair(state: Solution, rng, **kwargs) -> Solution:
        """Baseline greedy repair - sorts by total processing time"""
        state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))
        while len(state.unassigned) != 0:
            job = state.unassigned.pop()
            state.opt_insert(job)
        return state
    return baseline_repair

def run_benchmark(data_files, approaches, seed=SEED, iters=600):
    """
    Benchmark different ALNS approaches on multiple problem instances.
    
    Args:
        data_files: List of paths to problem instance files
        approaches: Dictionary mapping approach names to lists of (destroy_ops, repair_ops)
        seed: Random seed for reproducibility
        iters: Number of iterations for each ALNS run
    
    Returns:
        Dictionary containing all benchmark results
    """
    results = {
        'instance_names': [],
        'instance_sizes': [],
        'problem_types': [],
        'n_jobs': [],
        'n_machines': [],
        'best_known_values': [],
    }
    
    # Initialize results dictionary for each approach
    for approach_name in approaches:
        results[f'{approach_name}_objectives'] = []
        results[f'{approach_name}_gaps'] = []
        results[f'{approach_name}_times'] = []
    
    for data_file in data_files:
        # Extract instance name from file path
        instance_name = data_file.split('/')[-1].split('.')[0]
        problem_type = data_file.split('/')[-2]
        print(f"Processing instance: {instance_name} (Type: {problem_type})")
        
        # Load data
        data = Data.from_file(data_file)
        global DATA
        DATA = data
        PFSP.DATA = data  # Also set in PFSP module
        
        results['instance_names'].append(instance_name)
        results['problem_types'].append(problem_type)
        results['instance_sizes'].append(f"{data.n_jobs}x{data.n_machines}")
        results['n_jobs'].append(data.n_jobs)
        results['n_machines'].append(data.n_machines)
        results['best_known_values'].append(data.bkv)
        
        # Create initial solution using NEH
        init = NEH(data.processing_times)
        
        # Run each approach
        for approach_name, (destroy_ops, repair_ops) in approaches.items():
            print(f"  Running {approach_name}...")
            
            # Setup ALNS
            alns = ALNS(rnd.default_rng(seed))
            
            # Add destroy operators
            for destroy_op in destroy_ops:
                alns.add_destroy_operator(destroy_op)
            
            # Add repair operators
            for repair_op in repair_ops:
                alns.add_repair_operator(repair_op)
            
            # Configure ALNS parameters
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
            
            # Record results
            objective = result.best_state.objective()
            gap = 100 * (objective - data.bkv) / data.bkv
            
            results[f'{approach_name}_objectives'].append(objective)
            results[f'{approach_name}_gaps'].append(gap)
            results[f'{approach_name}_times'].append(runtime)
            
            print(f"    Objective: {objective}, Gap: {gap:.2f}%, Time: {runtime:.2f}s")
    
    return results

def analyze_results(results):
    """
    Analyze the benchmark results and show average gaps by problem size.
    """
    approach_names = [name.replace('_objectives', '') for name in results.keys() 
                     if name.endswith('_objectives')]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Instance': results['instance_names'],
        'Problem_Type': results['problem_types'],
        'Size': results['instance_sizes'],
        'n_jobs': results['n_jobs'],
        'n_machines': results['n_machines'],
        'BKV': results['best_known_values']
    })
    
    for approach in approach_names:
        df[f'{approach}_Obj'] = results[f'{approach}_objectives']
        df[f'{approach}_Gap'] = results[f'{approach}_gaps']
        df[f'{approach}_Time'] = results[f'{approach}_times']
    
    print("\n" + "="*60)
    print("EoH OPERATOR EVALUATION RESULTS")
    print("="*60)
    
    # Overall performance summary
    print("\nOverall Performance:")
    for approach in approach_names:
        avg_gap = df[f'{approach}_Gap'].mean()
        print(f"  {approach}: {avg_gap:.2f}% average gap")
    
    # Average gap by problem size (number of jobs)
    print(f"\nAverage Gap by Problem Size:")
    print("-" * 40)
    
    size_summary = df.groupby('n_jobs').agg({
        **{f'{approach}_Gap': 'mean' for approach in approach_names},
        'BKV': 'mean'
    }).round(2)
    
    print(size_summary[[f'{approach}_Gap' for approach in approach_names]])
    
    # Overall improvement
    if len(approach_names) == 2:
        eoh_gaps = df[f'{approach_names[0]}_Gap']
        baseline_gaps = df[f'{approach_names[1]}_Gap']
        improvement = baseline_gaps.mean() - eoh_gaps.mean()
        win_rate = (eoh_gaps < baseline_gaps).mean() * 100
        
        print(f"\nComparison:")
        print(f"  Average improvement: {improvement:.2f} percentage points")
        print(f"  Win rate: {win_rate:.1f}% ({(eoh_gaps < baseline_gaps).sum()}/{len(eoh_gaps)} instances)")
    
    return df

if __name__ == "__main__":
    print("🔬 EoH Best Operator Evaluation")
    print("="*50)
    
    # Load operators
    try:
        eoh_operator = load_best_eoh_operator()
        baseline_operator = create_baseline_repair()
        print("✅ Loaded EoH operator and baseline")
    except Exception as e:
        print(f"❌ Error loading operators: {e}")
        exit(1)
    
    # Define the approaches to compare (following evaluation.py structure)
    approaches = {
        'EoH_Best': (
            [random_removal, adjacent_removal],  # destroy operators
            [eoh_operator]                       # repair operators
        ),
        'Baseline': (
            [random_removal, adjacent_removal],  # destroy operators
            [baseline_operator]                  # repair operators
        )
    }
    
    # Define test instances - representative sample from each size
    data_files = []
    
    # Small instances (j20) - 3 instances each
    for size in ["j20_m5", "j20_m10", "j20_m20"]:
        for i in [1, 2, 3]:  # First 3 instances
            data_files.append(f"data/{size}/{size}_{i:02d}.txt")
    
    # Medium instances (j50) - 3 instances each  
    for size in ["j50_m5", "j50_m10", "j50_m20"]:
        for i in [1, 2, 3]:  # First 3 instances
            data_files.append(f"data/{size}/{size}_{i:02d}.txt")
    
    # Large instances (j100) - 3 instances each
    for size in ["j100_m5", "j100_m10"]:
        for i in [1, 2, 3]:  # First 3 instances
            data_files.append(f"data/{size}/{size}_{i:02d}.txt")
    
    print(f"\nTesting on {len(data_files)} instances...")
    
    # Run the benchmark
    results = run_benchmark(data_files, approaches, seed=SEED, iters=600)
    
    # Analyze and display results
    results_df = analyze_results(results)
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"eoh_best_evaluation_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to: {csv_filename}")
    
    print("\n✅ Evaluation complete!") 