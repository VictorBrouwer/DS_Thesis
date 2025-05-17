"""
Extended example of using the repair operators evaluation framework 
with custom operators and parameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from repair_operators_evaluation.operators.basic_operators import (
    greedy_repair, greedy_repair_with_local_search
)
from repair_operators_evaluation.operators.custom_operators import (
    insertion_priority_repair, multi_phase_repair
)
from repair_operators_evaluation.evaluation.evaluator import (
    evaluate_repair_operator, compare_repair_operators
)


def compare_custom_operators():
    """
    Compare standard and custom repair operators.
    """
    # Define repair operators to test
    repair_operators = [
        greedy_repair,
        greedy_repair_with_local_search,
        insertion_priority_repair,
        multi_phase_repair
    ]
    
    # Define test instances
    test_instances = [
        "git_repo/PFSP/data/j50_m20/j50_m20_08.txt",
        # Add more instances as needed
    ]
    
    # Compare repair operators
    results, summary = compare_repair_operators(
        repair_operators, 
        test_instances,
        n_runs=2,    # 2 runs per instance
        iters=300    # 300 iterations per run
    )
    
    return results, summary


def tune_operator_parameters():
    """
    Demonstrate parameter tuning for the insertion_priority_repair operator.
    """
    # Define parameter combinations to test
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    
    # Test instance
    test_instance = ["git_repo/PFSP/data/j50_m20/j50_m20_08.txt"]
    
    # Store results for each parameter combination
    tuning_results = []
    
    for alpha in alpha_values:
        beta = 1.0 - alpha  # Make parameters sum to 1
        
        print(f"\n=== Testing parameters: alpha={alpha:.1f}, beta={beta:.1f} ===")
        
        # Custom kwargs to pass to the repair operator
        kwargs = {
            'alpha': alpha,
            'beta': beta
        }
        
        # Run evaluation
        results, summary = evaluate_repair_operator(
            lambda state, rng, **kw: insertion_priority_repair(
                state, rng, **{**kw, **kwargs}
            ),
            test_instance,
            n_runs=2,
            iters=300
        )
        
        # Store results
        tuning_results.append({
            'alpha': alpha,
            'beta': beta,
            'avg_gap': summary['avg_gap'],
            'avg_runtime': summary['avg_runtime']
        })
    
    # Create summary dataframe
    tuning_df = pd.DataFrame(tuning_results)
    
    # Plot parameter tuning results
    plt.figure(figsize=(10, 6))
    plt.plot(tuning_df['alpha'], tuning_df['avg_gap'], 'o-')
    plt.xlabel('Alpha Value (Weight for Processing Time Priority)')
    plt.ylabel('Average Gap to BKV (%)')
    plt.title('Parameter Tuning for Insertion Priority Repair')
    plt.grid(True)
    plt.xticks(alpha_values)
    plt.tight_layout()
    plt.show()
    
    # Find best parameters
    best_idx = tuning_df['avg_gap'].idxmin()
    best_alpha = tuning_df.loc[best_idx, 'alpha']
    best_beta = tuning_df.loc[best_idx, 'beta']
    best_gap = tuning_df.loc[best_idx, 'avg_gap']
    
    print(f"\nBest parameters: alpha={best_alpha:.1f}, beta={best_beta:.1f}")
    print(f"Best average gap: {best_gap:.2f}%")
    
    return tuning_df


if __name__ == "__main__":
    print("1. Comparing custom operators with standard operators...")
    results, summary = compare_custom_operators()
    
    print("\n\n2. Tuning parameters for insertion_priority_repair...")
    tuning_results = tune_operator_parameters() 