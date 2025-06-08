#!/usr/bin/env python3
"""
Simple multi-size operator evaluation - just like evaluate_extensive but with 3 operators
"""

import os
import json
import time
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
from alns.stop import MaxRuntime

# Global variables
SEED = 2345
DATA = None

def load_three_operators():
    """Load the best operator from each problem size evolution"""
    
    operators = []
    
    # Load small operator
    with open("extensive_results_small/final_summary.json", 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {}
    exec(best_individual["code"], globals(), namespace)
    small_operator = namespace["llm_repair"]
    
    # Load medium operator  
    with open("extensive_results_medium/final_summary.json", 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {}
    exec(best_individual["code"], globals(), namespace)
    medium_operator = namespace["llm_repair"]
    
    # Load large operator
    with open("extensive_results_large/final_summary.json", 'r') as f:
        summary = json.load(f)
    best_individual = summary["final_population"][0]
    namespace = {}
    exec(best_individual["code"], globals(), namespace)
    large_operator = namespace["llm_repair"]
    
    print("✅ Loaded 3 operators: Small, Medium, Large")
    return small_operator, medium_operator, large_operator

def run_single_test(data_file, test_name, small_op, medium_op, large_op):
    """Run ALNS on a single instance with 3 operators"""
    
    try:
        # Load data
        data = Data.from_file(data_file)
        global DATA
        DATA = data
        PFSP.DATA = data
        
        print(f"\n🔬 {test_name}: {data.n_jobs}×{data.n_machines}, BKV: {data.bkv}")
        
        # Create initial solution
        init = NEH(data.processing_times)
        init_gap = 100 * (init.objective() - data.bkv) / data.bkv
        print(f"Initial gap: {init_gap:.2f}%")
        
        # Setup ALNS with 3 operators
        alns = ALNS(rnd.default_rng(SEED))
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(adjacent_removal)
        
        alns.add_repair_operator(small_op)
        alns.add_repair_operator(medium_op)
        alns.add_repair_operator(large_op)
        
        # Configure ALNS
        actual_repair_count = len(alns.repair_operators)
        select = AlphaUCB(
            scores=[5, 2, 1, 0.5],
            alpha=0.05,
            num_destroy=2,
            num_repair=actual_repair_count,
        )
        accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, 100)
        stop = MaxRuntime(60)
        
        # Run ALNS
        start_time = time.time()
        result = alns.iterate(deepcopy(init), select, accept, stop)
        runtime = time.time() - start_time
        
        # Results
        final_obj = result.best_state.objective()
        final_gap = 100 * (final_obj - data.bkv) / data.bkv
        improvement = init_gap - final_gap
        
        print(f"Final gap: {final_gap:.2f}% | Improvement: {improvement:.2f}pp")
        
        return {
            'test': test_name,
            'size': f"{data.n_jobs}×{data.n_machines}",
            'bkv': data.bkv,
            'initial_gap': init_gap,
            'final_gap': final_gap,
            'improvement': improvement,
            'runtime': runtime
        }
        
    except Exception as e:
        print(f"❌ Error in {test_name}: {e}")
        raise

def main():
    """Main function"""
    
    print("🧬 Simple Multi-Size Operator Evaluation - All Instances")
    print("=" * 60)
    
    # Load the 3 operators
    small_op, medium_op, large_op = load_three_operators()
    
    # Problem sizes to test (all instances per size)
    problem_sizes = [
        "j20_m5", "j20_m10", "j20_m20",
        "j50_m5", "j50_m10", "j50_m20", 
        "j100_m5", "j100_m10", "j100_m20",
        "j200_m10", "j200_m20"
    ]
    
    print(f"Testing 3 operators on 11 problem sizes (10 instances each):")
    print(f"Runtime: 60 seconds per instance | Total: ~110 minutes")
    print()
    
    # Run tests on all instances
    all_results = []
    size_summaries = {}
    
    for problem_size in problem_sizes:
        print(f"\n🎯 Testing {problem_size}...")
        size_results = []
        
        # Test all 10 instances for this problem size
        for instance_num in range(1, 11):
            instance_file = f"data/{problem_size}/{problem_size}_{instance_num:02d}.txt"
            test_name = f"{problem_size}_{instance_num:02d}"
            
            try:
                result = run_single_test(instance_file, test_name, small_op, medium_op, large_op)
                all_results.append(result)
                size_results.append(result)
            except Exception as e:
                print(f"❌ Error on {test_name}: {e}")
        
        # Calculate averages for this problem size
        if size_results:
            avg_gap = sum(r['final_gap'] for r in size_results) / len(size_results)
            avg_improvement = sum(r['improvement'] for r in size_results) / len(size_results)
            avg_runtime = sum(r['runtime'] for r in size_results) / len(size_results)
            
            size_summary = {
                'problem_size': problem_size,
                'num_instances': len(size_results),
                'avg_final_gap': avg_gap,
                'avg_improvement': avg_improvement,
                'avg_runtime': avg_runtime,
                'detailed_results': size_results
            }
            size_summaries[problem_size] = size_summary
            
            print(f"✅ {problem_size}: Avg Gap = {avg_gap:.2f}%, Avg Improvement = {avg_improvement:.2f}pp")
    
    # Overall Summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL SUMMARY - AVERAGE GAPS BY PROBLEM SIZE")
    print(f"{'='*80}")
    
    print(f"{'Problem Size':<12} {'Instances':<10} {'Avg Gap (%)':<12} {'Avg Improve (pp)':<16}")
    print("-" * 60)
    
    for problem_size in problem_sizes:
        if problem_size in size_summaries:
            summary = size_summaries[problem_size]
            print(f"{summary['problem_size']:<12} {summary['num_instances']:<10} "
                  f"{summary['avg_final_gap']:<12.2f} {summary['avg_improvement']:<16.2f}")
    
    # Save detailed results
    os.makedirs("multi_size_results", exist_ok=True)
    
    # Save all individual results
    with open("multi_size_results/all_instances_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary by problem size
    with open("multi_size_results/size_summaries.json", 'w') as f:
        json.dump(size_summaries, f, indent=2)
    
    # Overall statistics
    if all_results:
        overall_avg_gap = sum(r['final_gap'] for r in all_results) / len(all_results)
        overall_avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
        total_instances = len(all_results)
        
        print("-" * 60)
        print(f"Overall Average Gap: {overall_avg_gap:.2f}%")
        print(f"Overall Average Improvement: {overall_avg_improvement:.2f} pp")
        print(f"Total Instances Tested: {total_instances}")
        print(f"\nResults saved to:")
        print(f"  - multi_size_results/all_instances_results.json (detailed)")
        print(f"  - multi_size_results/size_summaries.json (averages by size)")
    else:
        print("❌ No successful results to analyze")

if __name__ == "__main__":
    main() 