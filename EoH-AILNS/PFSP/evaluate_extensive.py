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
import matplotlib.pyplot as plt

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

def run_single_instance_evaluation():
    """Run evaluation on single j100_m20 instance with plotting"""
    
    print("🔬 Single Instance Evaluation with Plotting")
    print("=" * 50)
    
    # Load operators
    eoh_operator, eoh_info = load_best_eoh_operator()
    baseline_operator = create_baseline_repair()
    
    print(f"EoH Operator: {eoh_info['algorithm']}")
    print(f"Training Gap: {eoh_info['gap']:.2f}%")
    print()
    
    # Single instance
    data_file = "data/j200_m10/j200_m10_01.txt"
    instance_name = data_file.split('/')[-1].split('.')[0]
    
    print(f"Testing on: {instance_name}")
    
    # Load data
    try:
        data = Data.from_file(data_file)
        global DATA
        DATA = data
        PFSP.DATA = data
    except Exception as e:
        print(f"❌ Error loading {data_file}: {e}")
        return
    
    print(f"📋 Instance: {data.n_jobs} jobs × {data.n_machines} machines | BKV: {data.bkv}")
    
    # Create initial solution
    init = NEH(data.processing_times)
    print(f"🚀 Initial solution objective: {init.objective()}")
    
    # Test both operators and store results for plotting
    results = {}
    
    for op_name, operator_func in [('EoH', eoh_operator), ('Baseline', baseline_operator)]:
        print(f"\n🔧 Testing {op_name} operator...")
        
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
        accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, 100)
        stop = MaxIterations(100)
        
        # Run ALNS
        start_time = time.time()
        result = alns.iterate(deepcopy(init), select, accept, stop)
        runtime = time.time() - start_time
        
        # Calculate results
        objective = result.best_state.objective()
        gap = 100 * (objective - data.bkv) / data.bkv
        
        print(f"✅ Final objective: {objective}")
        print(f"📊 Gap from BKV: {gap:.2f}%")
        print(f"⏱️  Runtime: {runtime:.1f}s")
        
        # Store result for plotting
        results[op_name] = result
    
    # Create comparison plot
    _, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both results
    colors = ['#2E86AB', '#A23B72']  # Blue for EoH, Red for Baseline
    for i, (op_name, result) in enumerate(results.items()):
        result.plot_objectives(ax=ax, color=colors[i], label=op_name)
    
    # Add BKV line
    ax.axhline(y=data.bkv, color='green', linestyle='--', alpha=0.7, label=f'BKV ({data.bkv})')
    
    # Customize plot
    ax.set_title(f'Objective Values Over Iterations - {instance_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value (Makespan)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add final results as text
    eoh_final = results['EoH'].best_state.objective()
    baseline_final = results['Baseline'].best_state.objective()
    eoh_gap = 100 * (eoh_final - data.bkv) / data.bkv
    baseline_gap = 100 * (baseline_final - data.bkv) / data.bkv
    
    textstr = f'Final Results:\nEoH: {eoh_final} ({eoh_gap:.2f}% gap)\nBaseline: {baseline_final} ({baseline_gap:.2f}% gap)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📈 Plot displayed showing objective evolution for both operators")
    print(f"🎯 EoH improvement over baseline: {baseline_gap - eoh_gap:.2f} percentage points")

def run_multiple_instances_evaluation():
    """Run evaluation on all instances of j100_m20"""
    
    print("🔬 Multiple Instances Evaluation")
    print("=" * 50)
    
    # Load operators
    eoh_operator, eoh_info = load_best_eoh_operator()
    baseline_operator = create_baseline_repair()
    
    print(f"EoH Operator: {eoh_info['algorithm']}")
    print(f"Training Gap: {eoh_info['gap']:.2f}%")
    print()
    
    # Define problem sizes to test
    problem_sizes = ["j100_m20"]
    
    # Store results for each problem size
    all_results = {}
    
    for problem_size in problem_sizes:
        print(f"\n🎯 Testing problem size: {problem_size}")
        print("-" * 40)
        
        # Results for this problem size
        size_results = {
            'eoh_gaps': [],
            'eoh_times': [],
            'baseline_gaps': [],
            'baseline_times': [],
            'instances': []
        }
        
        # Test all 10 instances
        for i in range(1, 2):
            data_file = f"data/{problem_size}/{problem_size}_{i:02d}.txt"
            instance_name = f"{problem_size}_{i:02d}"
            
            print(f"  📋 Instance {i}/10: {instance_name}", end=" ")
            
            # Load data
            try:
                data = Data.from_file(data_file)
                global DATA
                DATA = data
                PFSP.DATA = data
            except Exception as e:
                print(f"❌ Error loading {data_file}: {e}")
                continue
            
            print(f"({data.n_jobs}×{data.n_machines}, BKV: {data.bkv})")
            
            # Create initial solution
            init = NEH(data.processing_times)
            
            # Test both operators
            instance_results = {}
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
                    accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, 350)
                    stop = MaxIterations(350)
                    
                    # Run ALNS
                    start_time = time.time()
                    result = alns.iterate(deepcopy(init), select, accept, stop)
                    runtime = time.time() - start_time
                    
                    # Calculate results
                    objective = result.best_state.objective()
                    gap = 100 * (objective - data.bkv) / data.bkv
                    
                    instance_results[op_name] = {'gap': gap, 'time': runtime}
                    
                except Exception as e:
                    print(f"❌ Error with {op_name} on {instance_name}: {e}")
                    instance_results[op_name] = {'gap': 100.0, 'time': 0.0}
            
            # Store results
            if 'eoh' in instance_results and 'baseline' in instance_results:
                size_results['instances'].append(instance_name)
                size_results['eoh_gaps'].append(instance_results['eoh']['gap'])
                size_results['eoh_times'].append(instance_results['eoh']['time'])
                size_results['baseline_gaps'].append(instance_results['baseline']['gap'])
                size_results['baseline_times'].append(instance_results['baseline']['time'])
                
                # Show instance results
                eoh_gap = instance_results['eoh']['gap']
                baseline_gap = instance_results['baseline']['gap']
                improvement = baseline_gap - eoh_gap
                print(f"    🔧 EoH: {eoh_gap:.2f}% | Baseline: {baseline_gap:.2f}% | Improvement: {improvement:+.2f}pp")
        
        # Store results for this problem size
        all_results[problem_size] = size_results
    
    return all_results

def analyze_multiple_results(all_results):
    """Analyze and display results for multiple instances"""
    
    print("\n" + "="*70)
    print("MULTIPLE INSTANCES EVALUATION RESULTS")
    print("="*70)
    
    # Summary table
    print(f"\nSummary Results:")
    print("-" * 80)
    print(f"{'Problem':<12} {'Instances':<9} {'EoH Gap':<10} {'Baseline Gap':<12} {'Improvement':<12} {'EoH Time':<10} {'Baseline Time':<12}")
    print("-" * 80)
    
    for problem_size, results in all_results.items():
        if len(results['eoh_gaps']) == 0:
            continue
            
        # Calculate averages
        avg_eoh_gap = np.mean(results['eoh_gaps'])
        avg_baseline_gap = np.mean(results['baseline_gaps'])
        avg_improvement = avg_baseline_gap - avg_eoh_gap
        avg_eoh_time = np.mean(results['eoh_times'])
        avg_baseline_time = np.mean(results['baseline_times'])
        
        # Calculate win rate
        wins = sum(1 for i in range(len(results['eoh_gaps'])) 
                  if results['eoh_gaps'][i] < results['baseline_gaps'][i])
        win_rate = (wins / len(results['eoh_gaps'])) * 100
        
        print(f"{problem_size:<12} {len(results['instances']):<9} {avg_eoh_gap:<10.2f} {avg_baseline_gap:<12.2f} {avg_improvement:<12.2f} {avg_eoh_time:<10.1f} {avg_baseline_time:<12.1f}")
        
        # Detailed stats for this problem size
        print(f"\n📊 Detailed Stats for {problem_size}:")
        print(f"  🎯 Win Rate: {win_rate:.1f}% ({wins}/{len(results['instances'])} instances)")
        print(f"  📈 Gap Range - EoH: {min(results['eoh_gaps']):.2f}% to {max(results['eoh_gaps']):.2f}%")
        print(f"  📈 Gap Range - Baseline: {min(results['baseline_gaps']):.2f}% to {max(results['baseline_gaps']):.2f}%")
        print(f"  ⏱️  Time Range - EoH: {min(results['eoh_times']):.1f}s to {max(results['eoh_times']):.1f}s")
        print(f"  ⏱️  Time Range - Baseline: {min(results['baseline_times']):.1f}s to {max(results['baseline_times']):.1f}s")
    
    print("-" * 80)
    
    # Overall comparison
    all_eoh_gaps = []
    all_baseline_gaps = []
    all_eoh_times = []
    all_baseline_times = []
    total_instances = 0
    total_wins = 0
    
    for results in all_results.values():
        all_eoh_gaps.extend(results['eoh_gaps'])
        all_baseline_gaps.extend(results['baseline_gaps'])
        all_eoh_times.extend(results['eoh_times'])
        all_baseline_times.extend(results['baseline_times'])
        total_instances += len(results['instances'])
        total_wins += sum(1 for i in range(len(results['eoh_gaps'])) 
                         if results['eoh_gaps'][i] < results['baseline_gaps'][i])
    
    if total_instances > 0:
        overall_eoh_avg = np.mean(all_eoh_gaps)
        overall_baseline_avg = np.mean(all_baseline_gaps)
        overall_improvement = overall_baseline_avg - overall_eoh_avg
        overall_win_rate = (total_wins / total_instances) * 100
        overall_eoh_time = np.mean(all_eoh_times)
        overall_baseline_time = np.mean(all_baseline_times)
        
        print(f"\n🎯 OVERALL PERFORMANCE ACROSS ALL {total_instances} INSTANCES:")
        print(f"  EoH Average Gap: {overall_eoh_avg:.2f}%")
        print(f"  Baseline Average Gap: {overall_baseline_avg:.2f}%")
        print(f"  Average Improvement: {overall_improvement:.2f} percentage points")
        print(f"  Overall Win Rate: {overall_win_rate:.1f}% ({total_wins}/{total_instances})")
        print(f"  Average EoH Time: {overall_eoh_time:.1f}s")
        print(f"  Average Baseline Time: {overall_baseline_time:.1f}s")

def main():
    """Main function to run multiple instances evaluation"""
    
    # Check if extensive results exist
    if not os.path.exists("extensive_results/final_summary.json"):
        print("❌ Extensive results not found. Please run 'python run_extensive_eoh.py' first.")
        return
    
    # Run multiple instances evaluation
    results = run_multiple_instances_evaluation()
    
    # Analyze results
    analyze_multiple_results(results)
    
    print("\n✅ Multiple instances evaluation complete!")

if __name__ == "__main__":
    main() 