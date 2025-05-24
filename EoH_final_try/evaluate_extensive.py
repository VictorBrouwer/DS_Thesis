#!/usr/bin/env python3
"""
Extensive evaluation of best EoH operator on all instances
"""

import os
import json
import time
import pandas as pd
from copy import deepcopy
import numpy.random as rnd
import numpy as np

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
    """Load the best operator from extensive EoH results"""
    with open("extensive_results/final_summary.json", 'r') as f:
        summary = json.load(f)
    
    best_individual = summary["final_population"][0]
    operator_code = best_individual["code"]
    
    # Create operator function
    namespace = {}
    exec(operator_code, globals(), namespace)
    return namespace["llm_repair"], best_individual

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

def run_extensive_evaluation():
    """Run evaluation on all 80 Taillard instances"""
    
    print("🔬 Extensive Evaluation on All 80 Taillard Instances")
    print("=" * 60)
    
    # Load operators
    eoh_operator, eoh_info = load_best_eoh_operator()
    baseline_operator = create_baseline_repair()
    
    print(f"EoH Operator: {eoh_info['algorithm']}")
    print(f"Training Gap: {eoh_info['gap']:.2f}%")
    print()
    
    # All 80 Taillard instances
    data_files = []
    for size in [ "j100_m20", "j200_m10", "j200_m20"]:
        for i in range(1, 11):  # 10 instances each
            data_files.append(f"data/{size}/{size}_{i:02d}.txt")
    
    print(f"Testing on {len(data_files)} instances...")
    print("Estimated time: 30-45 minutes")
    print()
    
    # Store results
    results = {
        'instance': [],
        'problem_size': [],
        'n_jobs': [],
        'n_machines': [],
        'bkv': [],
        'eoh_objective': [],
        'eoh_gap': [],
        'eoh_time': [],
        'baseline_objective': [],
        'baseline_gap': [],
        'baseline_time': []
    }
    
    start_time = time.time()
    
    for i, data_file in enumerate(data_files):
        instance_name = data_file.split('/')[-1].split('.')[0]
        problem_type = data_file.split('/')[-2]
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(data_files)} instances completed ({elapsed/60:.1f} min elapsed)")
        
        # Load data
        try:
            data = Data.from_file(data_file)
            global DATA
            DATA = data
            PFSP.DATA = data
        except Exception as e:
            print(f"Error loading {data_file}: {e}")
            continue
        
        # Store instance info
        results['instance'].append(instance_name)
        results['problem_size'].append(f"{data.n_jobs}x{data.n_machines}")
        results['n_jobs'].append(data.n_jobs)
        results['n_machines'].append(data.n_machines)
        results['bkv'].append(data.bkv)
        
        # Create initial solution
        init = NEH(data.processing_times)
        
        # Test both operators
        for op_name, operator_func in [('eoh', eoh_operator), ('baseline', baseline_operator)]:
            try:
                # Setup ALNS
                alns = ALNS(rnd.default_rng(SEED))
                alns.add_destroy_operator(random_removal)
                alns.add_destroy_operator(adjacent_removal)
                alns.add_repair_operator(operator_func)
                
                # Configure ALNS
                select = AlphaUCB(
                    scores=[5, 2, 1, 0.5],
                    alpha=0.05,
                    num_destroy=2,
                    num_repair=1,
                )
                accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, 600)
                stop = MaxIterations(600)
                
                # Run ALNS
                op_start = time.time()
                result = alns.iterate(deepcopy(init), select, accept, stop)
                runtime = time.time() - op_start
                
                # Calculate results
                objective = result.best_state.objective()
                gap = 100 * (objective - data.bkv) / data.bkv
                
                results[f'{op_name}_objective'].append(objective)
                results[f'{op_name}_gap'].append(gap)
                results[f'{op_name}_time'].append(runtime)
                
            except Exception as e:
                print(f"Error with {op_name} on {instance_name}: {e}")
                # Add placeholder values
                results[f'{op_name}_objective'].append(data.bkv * 2)
                results[f'{op_name}_gap'].append(100.0)
                results[f'{op_name}_time'].append(0.0)
    
    total_time = time.time() - start_time
    print(f"\n✅ Evaluation completed in {total_time/60:.1f} minutes")
    
    return results

def analyze_extensive_results(results):
    """Analyze and display the extensive evaluation results"""
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXTENSIVE EVALUATION RESULTS")
    print("="*70)
    
    # Overall performance
    print(f"\nOverall Performance:")
    eoh_avg = df['eoh_gap'].mean()
    baseline_avg = df['baseline_gap'].mean()
    improvement = baseline_avg - eoh_avg
    win_rate = (df['eoh_gap'] < df['baseline_gap']).mean() * 100
    
    print(f"  EoH: {eoh_avg:.2f}% average gap")
    print(f"  Baseline: {baseline_avg:.2f}% average gap")
    print(f"  Improvement: {improvement:.2f} percentage points")
    print(f"  Win rate: {win_rate:.1f}% ({(df['eoh_gap'] < df['baseline_gap']).sum()}/{len(df)} instances)")
    
    # Results by problem size (number of jobs)
    print(f"\nAverage Gap by Problem Size:")
    print("-" * 50)
    
    size_results = df.groupby('n_jobs').agg({
        'eoh_gap': 'mean',
        'baseline_gap': 'mean',
        'eoh_time': 'mean',
        'baseline_time': 'mean'
    }).round(2)
    
    # Add improvement column
    size_results['improvement'] = size_results['baseline_gap'] - size_results['eoh_gap']
    
    print("Jobs | EoH Gap | Baseline Gap | Improvement | EoH Time | Baseline Time")
    print("-" * 70)
    for n_jobs, row in size_results.iterrows():
        print(f"{n_jobs:4d} | {row['eoh_gap']:7.2f} | {row['baseline_gap']:12.2f} | {row['improvement']:11.2f} | {row['eoh_time']:8.2f} | {row['baseline_time']:13.2f}")
    
    # Results by problem type (jobs x machines)
    print(f"\nAverage Gap by Problem Type:")
    print("-" * 50)
    
    type_results = df.groupby('problem_size').agg({
        'eoh_gap': 'mean',
        'baseline_gap': 'mean'
    }).round(2)
    type_results['improvement'] = type_results['baseline_gap'] - type_results['eoh_gap']
    
    print("Problem Size | EoH Gap | Baseline Gap | Improvement")
    print("-" * 50)
    for size, row in type_results.iterrows():
        print(f"{size:12s} | {row['eoh_gap']:7.2f} | {row['baseline_gap']:12.2f} | {row['improvement']:11.2f}")
    
    # Best and worst cases
    print(f"\nBest EoH Improvements:")
    df['improvement'] = df['baseline_gap'] - df['eoh_gap']
    best_improvements = df.nlargest(5, 'improvement')[['instance', 'improvement', 'eoh_gap', 'baseline_gap']]
    for _, row in best_improvements.iterrows():
        print(f"  {row['instance']}: {row['improvement']:+.2f} pp ({row['baseline_gap']:.2f}% → {row['eoh_gap']:.2f}%)")
    
    return df

def main():
    """Main function to run extensive evaluation"""
    
    # Check if extensive results exist
    if not os.path.exists("extensive_results/final_summary.json"):
        print("❌ Extensive results not found. Please run 'python run_extensive_eoh.py' first.")
        return
    
    # Run evaluation
    results = run_extensive_evaluation()
    
    # Analyze results
    results_df = analyze_extensive_results(results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"extensive_evaluation_results_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to: {csv_filename}")
    
    print("\n✅ Extensive evaluation complete!")

if __name__ == "__main__":
    main() 