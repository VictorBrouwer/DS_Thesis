#!/usr/bin/env python3
"""
Multi-size operator evaluation for PCTSP - evaluating 3 operators from different problem sizes
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
    PCTSPData, PCTSPSolution, construct_initial_solution,
    random_removal, adjacent_removal, greedy_repair, load_instances
)

# Import ALNS components
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxRuntime

# Global variables
SEED = 2345
DATA = None

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_three_operators():
    """Load the best operator from each problem size evolution"""
    
    operators = []
    operator_info = []
    
    # Load small operator (size 20)
    small_results_file = "eoh_pctsp_results/final_summary.json"
    if not os.path.exists(small_results_file):
        print(f"❌ Small operator results not found: {small_results_file}")
        return None, None
        
    with open(small_results_file, 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {"np": np, "DATA": None, "PCTSP": PCTSP}
    exec(best_individual["code"], globals(), namespace)
    small_operator = namespace["llm_repair"]
    operators.append(small_operator)
    operator_info.append(("Small (20)", best_individual))
    
    # Load medium operator (size 50)
    medium_results_file = "eoh_pctsp_results_50/final_summary.json"
    if not os.path.exists(medium_results_file):
        print(f"❌ Medium operator results not found: {medium_results_file}")
        print("Please run: python run_evolution_size_50.py")
        return None, None
        
    with open(medium_results_file, 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {"np": np, "DATA": None, "PCTSP": PCTSP}
    exec(best_individual["code"], globals(), namespace)
    medium_operator = namespace["llm_repair"]
    operators.append(medium_operator)
    operator_info.append(("Medium (50)", best_individual))
    
    # Load large operator (size 100)
    large_results_file = "eoh_pctsp_results_100/final_summary.json"
    if not os.path.exists(large_results_file):
        print(f"❌ Large operator results not found: {large_results_file}")
        print("Please run: python run_evolution_size_100.py")
        return None, None
        
    with open(large_results_file, 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {"np": np, "DATA": None, "PCTSP": PCTSP}
    exec(best_individual["code"], globals(), namespace)
    large_operator = namespace["llm_repair"]
    operators.append(large_operator)
    operator_info.append(("Large (100)", best_individual))
    
    print("✅ Loaded 3 operators: Small (20), Medium (50), Large (100)")
    for name, info in operator_info:
        print(f"  {name}: {info['algorithm']} - Training Gap: {info['gap']:.2f}%")
    
    return operators, operator_info

def run_single_test(instance, test_name, operators, operator_names):
    """Run ALNS on a single instance with 3 operators"""
    
    try:
        # Set global data
        global DATA
        DATA = instance
        PCTSP.DATA = instance
        
        print(f"\n🔬 {test_name}: Size {instance.size}, Total Prize: {instance.total_prize}")
        
        # Create initial solution
        init_solution = construct_initial_solution(use_greedy=True)
        init_obj = init_solution.objective()
        print(f"Initial objective: {init_obj:.2f}")
        
        # Setup ALNS with 3 operators
        alns = ALNS(rnd.default_rng(SEED))
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(adjacent_removal)
        
        for operator in operators:
            alns.add_repair_operator(operator)
        
        # Configure ALNS
        actual_repair_count = len(alns.repair_operators)
        select = AlphaUCB(
            scores=[5, 2, 1, 0.5],
            alpha=0.05,
            num_destroy=2,
            num_repair=actual_repair_count,
        )
        
        accept = SimulatedAnnealing.autofit(
            init_obj=init_obj,
            worse=0.20,
            accept_prob=0.80,
            num_iters=1000,
            method='exponential'
        )
        stop = MaxRuntime(60.0)  # 60 seconds per instance
        
        # Run ALNS
        start_time = time.time()
        result = alns.iterate(deepcopy(init_solution), select, accept, stop)
        runtime = time.time() - start_time
        
        # Results
        final_solution = result.best_state
        final_obj = final_solution.objective()
        improvement = init_obj - final_obj
        feasible = final_solution.is_feasible()
        tour_length = len(final_solution.tour)
        prize_collected = final_solution.total_prize()
        
        print(f"Final objective: {final_obj:.2f} | Improvement: {improvement:+.2f} | "
              f"Feasible: {feasible} | Time: {runtime:.1f}s")
        
        return {
            'test': test_name,
            'size': instance.size,
            'total_prize_required': instance.total_prize,
            'initial_objective': init_obj,
            'final_objective': final_obj,
            'improvement': improvement,
            'feasible': feasible,
            'tour_length': tour_length,
            'prize_collected': prize_collected,
            'runtime': runtime
        }
        
    except Exception as e:
        print(f"❌ Error in {test_name}: {e}")
        raise

def main():
    """Main function"""
    
    print("🧬 Multi-Size Operator Evaluation for PCTSP")
    print("=" * 60)
    
    # Load the 3 operators
    operators, operator_info = load_three_operators()
    if operators is None:
        return
    
    operator_names = [info[0] for info in operator_info]
    
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
        return
    
    print(f"\nTesting 3 operators on {len(all_instances)} instances:")
    print(f"Runtime: 60 seconds per instance | Total: ~{len(all_instances)} minutes")
    print()
    
    # Run tests on all instances
    all_results = []
    size_summaries = {}
    
    start_time = time.time()
    
    for i, (size, instance) in enumerate(all_instances):
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(all_instances)} instances completed ({elapsed/60:.1f} min elapsed)")
        
        test_name = f"size_{size}_inst_{instance.instance_id}"
        result = run_single_test(instance, test_name, operators, operator_names)
        all_results.append(result)
        
        # Update size summary
        if size not in size_summaries:
            size_summaries[size] = {
                'instances': 0,
                'initial_sum': 0,
                'final_sum': 0,
                'improve_sum': 0,
                'feasible_count': 0,
                'time_sum': 0
            }
        
        summary = size_summaries[size]
        summary['instances'] += 1
        summary['initial_sum'] += result['initial_objective']
        summary['final_sum'] += result['final_objective']
        summary['improve_sum'] += result['improvement']
        summary['feasible_count'] += int(result['feasible'])
        summary['time_sum'] += result['runtime']
    
    # Calculate final statistics
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY - MULTI-SIZE OPERATORS ON PCTSP")
    print("=" * 80)
    print("Operators tested:")
    for name, info in operator_info:
        print(f"  {name}: {info['algorithm']}")
    print()
    
    print("Size   Instances  Avg Initial  Avg Final    Avg Improve  Improve %  Feasible %   Avg Time ")
    print("-" * 90)
    
    total_instances = 0
    total_initial = 0
    total_final = 0
    total_improve = 0
    total_feasible = 0
    total_time = 0
    
    for size in sorted(size_summaries.keys()):
        summary = size_summaries[size]
        n = summary['instances']
        avg_initial = summary['initial_sum'] / n
        avg_final = summary['final_sum'] / n
        avg_improve = summary['improve_sum'] / n
        feasible_pct = (summary['feasible_count'] / n) * 100
        avg_time = summary['time_sum'] / n
        
        print(f"{size:<6} {n:<10} {avg_initial:>10.2f} {avg_final:>10.2f} {avg_improve:>10.2f} "
              f"{avg_improve/avg_initial*100:>9.1f} {feasible_pct:>9.1f} {avg_time:>9.1f}")
        
        total_instances += n
        total_initial += summary['initial_sum']
        total_final += summary['final_sum']
        total_improve += summary['improve_sum']
        total_feasible += summary['feasible_count']
        total_time += summary['time_sum']
    
    print("-" * 90)
    print(f"OVERALL {total_instances:<10} {total_initial/total_instances:>10.2f} "
          f"{total_final/total_instances:>10.2f} {total_improve/total_instances:>10.2f} "
          f"{total_improve/total_initial*100:>9.1f} {(total_feasible/total_instances*100):>9.1f} "
          f"{total_time/total_instances:>9.1f}")
    
    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = "multi_size_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save CSV
    df = pd.DataFrame(all_results)
    csv_file = os.path.join(results_dir, f"multi_size_pctsp_evaluation_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved to: {csv_file}")
    
    # Save summary JSON
    summary_data = {
        'timestamp': timestamp,
        'operators': [{'name': name, 'info': info} for name, info in operator_info],
        'size_summaries': size_summaries,
        'overall': {
            'instances': total_instances,
            'avg_initial': total_initial/total_instances,
            'avg_final': total_final/total_instances,
            'avg_improve': total_improve/total_instances,
            'improve_percent': total_improve/total_initial*100,
            'feasible_percent': total_feasible/total_instances*100,
            'avg_time': total_time/total_instances
        }
    }
    
    json_file = os.path.join(results_dir, f"multi_size_pctsp_evaluation_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    print(f"💾 Detailed results saved to: {json_file}")

if __name__ == "__main__":
    main() 