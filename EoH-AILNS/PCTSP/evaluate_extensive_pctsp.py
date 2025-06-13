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
from alns.stop import MaxIterations, MaxRuntime

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



def run_extensive_evaluation():
    """Run evaluation on all PCTSP instances"""
    
    print("🔬 Extensive EoH Evaluation on All PCTSP Instances")
    print("=" * 60)
    
    # Load EoH operator
    eoh_operator, eoh_info = load_best_eoh_operator()
    if eoh_operator is None:
        return None
    
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
    
    print(f"\nTesting EoH operator on {len(all_instances)} total instances...")
    print("Estimated time: 10-20 minutes")
    print()
    
    # Store results
    results = {
        'instance_id': [],
        'problem_size': [],
        'n_nodes': [],
        'total_prize_required': [],
        'initial_objective': [],
        'eoh_objective': [],
        'eoh_improvement': [],
        'eoh_time': [],
        'eoh_feasible': [],
        'eoh_tour_length': [],
        'eoh_prize_collected': []
    }
    
    start_time = time.time()
    
    for i, (size, instance) in enumerate(all_instances):
        if i % 10 == 0 and i > 0:
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
        results['initial_objective'].append(init_solution.objective())
        
        print(f"\nInstance {instance.instance_id} (size {size}): Initial obj = {init_solution.objective():.2f}")
        
        # Test EoH operator
        try:
            # Setup ALNS
            alns = ALNS(rnd.default_rng(SEED))
            alns.add_destroy_operator(random_removal)
            alns.add_destroy_operator(worst_removal)
            alns.add_repair_operator(eoh_operator)
            
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
                num_iters=1000,  # High number of iterations (will be limited by time)
                method='exponential'
            )
            stop = MaxRuntime(60.0)  # 60 seconds time limit
            
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
            improvement = init_solution.objective() - objective
            
            # Store results
            results['eoh_objective'].append(objective)
            results['eoh_improvement'].append(improvement)
            results['eoh_time'].append(runtime)
            results['eoh_feasible'].append(feasible)
            results['eoh_tour_length'].append(tour_length)
            results['eoh_prize_collected'].append(prize_collected)
            
            print(f"  EoH RESULT: obj = {objective:.2f}, "
                  f"improvement = {improvement:+.2f}, "
                  f"feasible = {feasible}, time = {runtime:.1f}s")
            
        except Exception as e:
            print(f"  ERROR with EoH on instance {instance.instance_id}: {e}")
            # Add placeholder values
            objective = init_solution.objective() * 2
            improvement = init_solution.objective() - objective
            results['eoh_objective'].append(objective)
            results['eoh_improvement'].append(improvement)
            results['eoh_time'].append(0.0)
            results['eoh_feasible'].append(False)
            results['eoh_tour_length'].append(0)
            results['eoh_prize_collected'].append(0.0)
    
    total_time = time.time() - start_time
    
    # Show summary by problem size
    print(f"\n" + "="*60)
    print("EoH EVALUATION SUMMARY BY PROBLEM SIZE")
    print("="*60)
    
    df_temp = pd.DataFrame(results)
    for size in sorted(df_temp['problem_size'].unique()):
        size_data = df_temp[df_temp['problem_size'] == size]
        n_instances = len(size_data)
        improved_instances = (size_data['eoh_improvement'] > 0).sum()
        avg_improvement = size_data['eoh_improvement'].mean()
        avg_initial_obj = size_data['initial_objective'].mean()
        avg_eoh_obj = size_data['eoh_objective'].mean()
        feasible_rate = size_data['eoh_feasible'].mean()
        avg_time = size_data['eoh_time'].mean()
        
        print(f"Size {size:3d}: {n_instances:2d} instances, improved {improved_instances:2d}/{n_instances:2d} ({100*improved_instances/n_instances:4.1f}%), "
              f"avg improvement {avg_improvement:+6.2f}, "
              f"obj: {avg_initial_obj:.2f} → {avg_eoh_obj:.2f}, "
              f"feasible: {100*feasible_rate:4.1f}%, time: {avg_time:.1f}s")
    
    total_improved = (df_temp['eoh_improvement'] > 0).sum()
    overall_improvement = df_temp['eoh_improvement'].mean()
    overall_feasible = df_temp['eoh_feasible'].mean()
    print(f"\nOVERALL: {total_improved}/{len(df_temp)} instances improved ({100*total_improved/len(df_temp):.1f}%), "
          f"average improvement {overall_improvement:+.2f}, feasible rate {100*overall_feasible:.1f}%")
    
    print(f"\n✅ Evaluation completed in {total_time/60:.1f} minutes")
    
    return results

def analyze_extensive_results(results):
    """Analyze and display the EoH evaluation results"""
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("EXTENSIVE EoH PCTSP EVALUATION RESULTS")
    print("="*70)
    
    # Overall performance
    print(f"\nOverall EoH Performance:")
    avg_initial_obj = df['initial_objective'].mean()
    avg_eoh_obj = df['eoh_objective'].mean()
    avg_improvement = df['eoh_improvement'].mean()
    improved_instances = (df['eoh_improvement'] > 0).sum()
    feasible_rate = df['eoh_feasible'].mean() * 100
    avg_time = df['eoh_time'].mean()
    
    print(f"  Average initial objective: {avg_initial_obj:.2f}")
    print(f"  Average EoH objective: {avg_eoh_obj:.2f}")
    print(f"  Average improvement: {avg_improvement:+.2f}")
    print(f"  Instances improved: {improved_instances}/{len(df)} ({100*improved_instances/len(df):.1f}%)")
    print(f"  Feasible rate: {feasible_rate:.1f}%")
    print(f"  Average runtime: {avg_time:.1f}s")
    
    # Results by problem size
    print(f"\nResults by Problem Size:")
    print("-" * 90)
    
    size_results = df.groupby('problem_size').agg({
        'initial_objective': 'mean',
        'eoh_objective': 'mean',
        'eoh_improvement': 'mean',
        'eoh_time': 'mean',
        'eoh_feasible': 'mean'
    }).round(3)
    
    print("Size | Initial Obj | EoH Obj | Improvement | Time | Feasible % | Improved Instances")
    print("-" * 90)
    for size, row in size_results.iterrows():
        size_data = df[df['problem_size'] == size]
        improved_count = (size_data['eoh_improvement'] > 0).sum()
        total_count = len(size_data)
        print(f"{size:4d} | {row['initial_objective']:10.2f} | {row['eoh_objective']:7.2f} | "
              f"{row['eoh_improvement']:+10.2f} | {row['eoh_time']:4.1f} | {row['eoh_feasible']:9.1%} | "
              f"{improved_count:2d}/{total_count:2d} ({100*improved_count/total_count:4.1f}%)")
    
    # Best improvements
    print(f"\nBest EoH Improvements:")
    best_improvements = df[df['eoh_improvement'] > 0].nlargest(5, 'eoh_improvement')[
        ['instance_id', 'problem_size', 'initial_objective', 'eoh_objective', 'eoh_improvement']
    ]
    
    if len(best_improvements) > 0:
        for _, row in best_improvements.iterrows():
            print(f"  Instance {row['instance_id']} (size {row['problem_size']}): "
                  f"{row['eoh_improvement']:+.2f} ({row['initial_objective']:.2f} → {row['eoh_objective']:.2f})")
    else:
        print("  No instances with positive improvement")
    
    # Worst performances (largest degradations)
    print(f"\nWorst EoH Performances (largest degradations):")
    worst_cases = df[df['eoh_improvement'] < 0].nsmallest(5, 'eoh_improvement')[
        ['instance_id', 'problem_size', 'initial_objective', 'eoh_objective', 'eoh_improvement']
    ]
    
    if len(worst_cases) > 0:
        for _, row in worst_cases.iterrows():
            print(f"  Instance {row['instance_id']} (size {row['problem_size']}): "
                  f"{row['eoh_improvement']:+.2f} ({row['initial_objective']:.2f} → {row['eoh_objective']:.2f})")
    else:
        print("  No instances with negative improvement!")
    
    # Feasibility analysis
    print(f"\nFeasibility Analysis:")
    eoh_infeasible = df[~df['eoh_feasible']]
    
    print(f"  EoH infeasible instances: {len(eoh_infeasible)}")
    
    if len(eoh_infeasible) > 0:
        print(f"  Infeasible by size: {eoh_infeasible['problem_size'].value_counts().to_dict()}")
        print("  Sample infeasible instances:")
        sample_infeasible = eoh_infeasible.head(3)[['instance_id', 'problem_size', 'eoh_objective']]
        for _, row in sample_infeasible.iterrows():
            print(f"    Instance {row['instance_id']} (size {row['problem_size']}): obj = {row['eoh_objective']:.2f}")
    
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
            'initial_avg_objective': float(results_df['initial_objective'].mean()),
            'eoh_avg_objective': float(results_df['eoh_objective'].mean()),
            'avg_improvement': float(results_df['eoh_improvement'].mean()),
            'improvement_rate': float((results_df['eoh_improvement'] > 0).mean() * 100),
            'eoh_feasible_rate': float(results_df['eoh_feasible'].mean() * 100),
            'avg_runtime': float(results_df['eoh_time'].mean())
        },
        'by_problem_size': {}
    }
    
    for size in results_df['problem_size'].unique():
        size_data = results_df[results_df['problem_size'] == size]
        summary_stats['by_problem_size'][int(size)] = {
            'instances': len(size_data),
            'initial_avg_objective': float(size_data['initial_objective'].mean()),
            'eoh_avg_objective': float(size_data['eoh_objective'].mean()),
            'avg_improvement': float(size_data['eoh_improvement'].mean()),
            'improvement_rate': float((size_data['eoh_improvement'] > 0).mean() * 100),
            'feasible_rate': float(size_data['eoh_feasible'].mean() * 100)
        }
    
    summary_filename = f"extensive_pctsp_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"💾 Summary saved to: {summary_filename}")
    
    print("\n✅ Extensive EoH PCTSP evaluation complete!")

if __name__ == "__main__":
    main() 