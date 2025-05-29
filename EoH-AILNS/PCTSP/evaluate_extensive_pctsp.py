#!/usr/bin/env python3
"""
Extensive evaluation of best EoH operator on all PCTSP instances
"""

import os
import json
import time
import pandas as pd
from copy import deepcopy
import numpy.random as rnd
import numpy as np

# Import PCTSP components
import PCTSP
from PCTSP import (
    PCTSPData, PCTSPSolution, construct_initial_solution, evaluate_operator,
    random_removal, worst_removal, greedy_repair, load_instances
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
    results_file = "eoh_pctsp_results/final_summary.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run the EoH framework first: python run_eoh_pctsp.py")
        return None, None
    
    with open(results_file, 'r') as f:
        summary = json.load(f)
    
    best_individual = summary["final_population"][0]
    operator_code = best_individual["code"]
    
    # Create operator function
    namespace = {}
    exec(operator_code, globals(), namespace)
    return namespace["llm_repair"], best_individual

def create_baseline_repair():
    """Create baseline repair operator"""
    def baseline_repair(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:
        """Baseline greedy repair - sorts by prize-to-penalty ratio"""
        if not state.unvisited:
            return state
        
        # Sort unvisited nodes by prize-to-penalty ratio
        ratios = []
        for node in state.unvisited:
            ratio = DATA.prizes[node] / (DATA.penalties[node] + 1e-6)
            ratios.append((node, ratio))
        
        # Sort by decreasing ratio (best first)
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Insert nodes until feasible or all inserted
        for node, _ in ratios:
            if node in state.unvisited:
                state.opt_insert(node)
                if state.is_feasible():
                    break
        
        # If still not feasible, insert remaining nodes
        while state.unvisited and not state.is_feasible():
            node = state.unvisited[0]
            state.opt_insert(node)
        
        return state
    return baseline_repair

def run_extensive_evaluation():
    """Run evaluation on all PCTSP instances"""
    
    print("🔬 Extensive Evaluation on All PCTSP Instances")
    print("=" * 60)
    
    # Load operators
    eoh_operator, eoh_info = load_best_eoh_operator()
    if eoh_operator is None:
        return None
        
    baseline_operator = create_baseline_repair()
    
    print(f"EoH Operator: {eoh_info['algorithm']}")
    print(f"Training Gap: {eoh_info['gap']:.2f}%")
    print(f"Training Objective: {eoh_info['objective']:.2f}")
    print()
    
    # Load all PCTSP instances
    all_instances = []
    problem_sizes = [20, 50, 100]
    
    for size in problem_sizes:
        instances = load_instances(size)
        if instances:
            all_instances.extend([(size, inst) for inst in instances])
            print(f"Loaded {len(instances)} instances of size {size}")
    
    if not all_instances:
        print("❌ No instances found!")
        return None
    
    print(f"\nTesting on {len(all_instances)} total instances...")
    print("Estimated time: 15-30 minutes")
    print()
    
    # Store results
    results = {
        'instance_id': [],
        'problem_size': [],
        'n_nodes': [],
        'total_prize_required': [],
        'eoh_objective': [],
        'eoh_gap_vs_baseline': [],
        'eoh_time': [],
        'eoh_feasible': [],
        'eoh_tour_length': [],
        'eoh_prize_collected': [],
        'baseline_objective': [],
        'baseline_time': [],
        'baseline_feasible': [],
        'baseline_tour_length': [],
        'baseline_prize_collected': []
    }
    
    start_time = time.time()
    
    for i, (size, instance) in enumerate(all_instances):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(all_instances)} instances completed ({elapsed/60:.1f} min elapsed)")
        
        # Set global data
        global DATA
        DATA = instance
        PCTSP.DATA = instance
        
        # Store instance info
        results['instance_id'].append(instance.instance_id)
        results['problem_size'].append(size)
        results['n_nodes'].append(instance.size)
        results['total_prize_required'].append(instance.total_prize)
        
        # Create initial solution
        init_solution = construct_initial_solution(use_greedy=True)
        
        # Test both operators
        for op_name, operator_func in [('eoh', eoh_operator), ('baseline', baseline_operator)]:
            try:
                # Setup ALNS
                alns = ALNS(rnd.default_rng(SEED))
                alns.add_destroy_operator(random_removal)
                alns.add_destroy_operator(worst_removal)
                alns.add_repair_operator(operator_func)
                
                # Configure ALNS
                select = AlphaUCB(
                    scores=[5, 2, 1, 0.5],
                    alpha=0.05,
                    num_destroy=2,
                    num_repair=1,
                )
                
                # Use the same temperature configuration as in the framework
                accept = SimulatedAnnealing.autofit(
                    init_obj=init_solution.objective(),
                    worse=0.20,  # Accept solutions up to 20% worse
                    accept_prob=0.80,  # 80% acceptance probability
                    num_iters=600,  # More iterations for extensive evaluation
                    method='exponential'
                )
                stop = MaxIterations(600)
                
                # Run ALNS
                op_start = time.time()
                result = alns.iterate(deepcopy(init_solution), select, accept, stop)
                runtime = time.time() - op_start
                
                # Calculate results
                best_solution = result.best_state
                objective = best_solution.objective()
                feasible = best_solution.is_feasible()
                tour_length = len(best_solution.tour)
                prize_collected = best_solution.total_prize()
                
                results[f'{op_name}_objective'].append(objective)
                results[f'{op_name}_time'].append(runtime)
                results[f'{op_name}_feasible'].append(feasible)
                results[f'{op_name}_tour_length'].append(tour_length)
                results[f'{op_name}_prize_collected'].append(prize_collected)
                
            except Exception as e:
                print(f"Error with {op_name} on instance {instance.instance_id}: {e}")
                # Add placeholder values
                results[f'{op_name}_objective'].append(init_solution.objective() * 2)
                results[f'{op_name}_time'].append(0.0)
                results[f'{op_name}_feasible'].append(False)
                results[f'{op_name}_tour_length'].append(0)
                results[f'{op_name}_prize_collected'].append(0.0)
    
    # Calculate gap vs baseline for EoH
    for i in range(len(results['eoh_objective'])):
        eoh_obj = results['eoh_objective'][i]
        baseline_obj = results['baseline_objective'][i]
        gap = 100 * (eoh_obj - baseline_obj) / baseline_obj if baseline_obj > 0 else 0
        results['eoh_gap_vs_baseline'].append(gap)
    
    total_time = time.time() - start_time
    print(f"\n✅ Evaluation completed in {total_time/60:.1f} minutes")
    
    return results

def analyze_extensive_results(results):
    """Analyze and display the extensive evaluation results"""
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXTENSIVE PCTSP EVALUATION RESULTS")
    print("="*70)
    
    # Overall performance
    print(f"\nOverall Performance:")
    eoh_avg = df['eoh_gap_vs_baseline'].mean()
    win_rate = (df['eoh_objective'] < df['baseline_objective']).mean() * 100
    feasible_rate_eoh = df['eoh_feasible'].mean() * 100
    feasible_rate_baseline = df['baseline_feasible'].mean() * 100
    
    print(f"  EoH vs Baseline: {eoh_avg:.2f}% average gap")
    print(f"  Win rate: {win_rate:.1f}% ({(df['eoh_objective'] < df['baseline_objective']).sum()}/{len(df)} instances)")
    print(f"  EoH feasible rate: {feasible_rate_eoh:.1f}%")
    print(f"  Baseline feasible rate: {feasible_rate_baseline:.1f}%")
    
    # Average objectives and times
    print(f"\nAverage Objectives:")
    print(f"  EoH: {df['eoh_objective'].mean():.2f}")
    print(f"  Baseline: {df['baseline_objective'].mean():.2f}")
    print(f"  Improvement: {df['baseline_objective'].mean() - df['eoh_objective'].mean():.2f}")
    
    print(f"\nAverage Runtime:")
    print(f"  EoH: {df['eoh_time'].mean():.2f}s")
    print(f"  Baseline: {df['baseline_time'].mean():.2f}s")
    
    # Results by problem size
    print(f"\nResults by Problem Size:")
    print("-" * 80)
    
    size_results = df.groupby('problem_size').agg({
        'eoh_objective': 'mean',
        'baseline_objective': 'mean',
        'eoh_gap_vs_baseline': 'mean',
        'eoh_time': 'mean',
        'baseline_time': 'mean',
        'eoh_feasible': 'mean',
        'baseline_feasible': 'mean'
    }).round(3)
    
    print("Size | EoH Obj | Base Obj | Gap(%) | EoH Time | Base Time | EoH Feas | Base Feas")
    print("-" * 80)
    for size, row in size_results.iterrows():
        print(f"{size:4d} | {row['eoh_objective']:7.2f} | {row['baseline_objective']:8.2f} | "
              f"{row['eoh_gap_vs_baseline']:6.2f} | {row['eoh_time']:8.2f} | {row['baseline_time']:9.2f} | "
              f"{row['eoh_feasible']:8.1%} | {row['baseline_feasible']:9.1%}")
    
    # Best improvements
    print(f"\nBest EoH Improvements (EoH better than baseline):")
    df['improvement'] = df['baseline_objective'] - df['eoh_objective']
    best_improvements = df[df['improvement'] > 0].nlargest(5, 'improvement')[
        ['instance_id', 'problem_size', 'improvement', 'eoh_objective', 'baseline_objective']
    ]
    
    if len(best_improvements) > 0:
        for _, row in best_improvements.iterrows():
            print(f"  Instance {row['instance_id']} (size {row['problem_size']}): "
                  f"{row['improvement']:+.2f} ({row['baseline_objective']:.2f} → {row['eoh_objective']:.2f})")
    else:
        print("  No instances where EoH outperformed baseline")
    
    # Worst cases
    print(f"\nWorst EoH Performance (baseline better than EoH):")
    worst_cases = df[df['improvement'] < 0].nsmallest(5, 'improvement')[
        ['instance_id', 'problem_size', 'improvement', 'eoh_objective', 'baseline_objective']
    ]
    
    if len(worst_cases) > 0:
        for _, row in worst_cases.iterrows():
            print(f"  Instance {row['instance_id']} (size {row['problem_size']}): "
                  f"{row['improvement']:+.2f} ({row['baseline_objective']:.2f} → {row['eoh_objective']:.2f})")
    else:
        print("  EoH outperformed baseline on all instances!")
    
    # Feasibility analysis
    print(f"\nFeasibility Analysis:")
    eoh_infeasible = df[~df['eoh_feasible']]
    baseline_infeasible = df[~df['baseline_feasible']]
    
    print(f"  EoH infeasible instances: {len(eoh_infeasible)}")
    print(f"  Baseline infeasible instances: {len(baseline_infeasible)}")
    
    if len(eoh_infeasible) > 0:
        print(f"  EoH infeasible by size: {eoh_infeasible['problem_size'].value_counts().to_dict()}")
    
    return df

def main():
    """Main function to run extensive evaluation"""
    
    # Check if EoH results exist
    if not os.path.exists("eoh_pctsp_results/final_summary.json"):
        print("❌ EoH results not found. Please run the EoH framework first:")
        print("   python run_eoh_pctsp.py --api_endpoint YOUR_API --api_key YOUR_KEY")
        return
    
    # Run evaluation
    results = run_extensive_evaluation()
    if results is None:
        return
    
    # Analyze results
    results_df = analyze_extensive_results(results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"extensive_pctsp_evaluation_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n💾 Results saved to: {csv_filename}")
    
    # Save summary statistics
    summary_stats = {
        'timestamp': timestamp,
        'total_instances': len(results_df),
        'problem_sizes': results_df['problem_size'].unique().tolist(),
        'overall_performance': {
            'eoh_avg_objective': float(results_df['eoh_objective'].mean()),
            'baseline_avg_objective': float(results_df['baseline_objective'].mean()),
            'avg_gap_vs_baseline': float(results_df['eoh_gap_vs_baseline'].mean()),
            'win_rate': float((results_df['eoh_objective'] < results_df['baseline_objective']).mean() * 100),
            'eoh_feasible_rate': float(results_df['eoh_feasible'].mean() * 100),
            'baseline_feasible_rate': float(results_df['baseline_feasible'].mean() * 100)
        },
        'by_problem_size': {}
    }
    
    for size in results_df['problem_size'].unique():
        size_data = results_df[results_df['problem_size'] == size]
        summary_stats['by_problem_size'][int(size)] = {
            'instances': len(size_data),
            'eoh_avg_objective': float(size_data['eoh_objective'].mean()),
            'baseline_avg_objective': float(size_data['baseline_objective'].mean()),
            'avg_gap': float(size_data['eoh_gap_vs_baseline'].mean()),
            'win_rate': float((size_data['eoh_objective'] < size_data['baseline_objective']).mean() * 100)
        }
    
    summary_filename = f"extensive_pctsp_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"💾 Summary saved to: {summary_filename}")
    
    print("\n✅ Extensive PCTSP evaluation complete!")

if __name__ == "__main__":
    main() 