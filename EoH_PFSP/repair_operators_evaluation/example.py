"""
Example of using the repair operators evaluation framework.
"""

from repair_operators_evaluation.operators.basic_operators import (
    greedy_repair, regret_repair, greedy_repair_with_local_search
)
from repair_operators_evaluation.evaluation.evaluator import compare_repair_operators

def main():
    """
    Main function to demonstrate how to use the framework.
    """
    # Define repair operators to test
    repair_operators = [
        greedy_repair,
        regret_repair,
        greedy_repair_with_local_search
    ]
    
    # Define test instances
    test_instances = [
        "git_repo/PFSP/data/j50_m20/j50_m20_08.txt",
        # Add more instances as needed
    ]
    
    # Compare repair operators
    # Using reduced number of runs and iterations for faster results
    results, summary = compare_repair_operators(
        repair_operators, 
        test_instances,
        n_runs=2,    # 2 runs per instance
        iters=300    # 300 iterations per run
    )
    
    return results, summary


if __name__ == "__main__":
    main() 