# Grid Search for Optimal n_remove Parameter
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy.random as rnd

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# Set random seed for reproducibility
SEED = 2345
np.random.seed(SEED)

@dataclass
class PCTSPData:
    """Data class for a PCTSP instance"""
    instance_id: int
    size: int  # Number of nodes (not including depot)
    depot: np.ndarray  # Depot coordinates [x, y]
    locations: np.ndarray  # Node coordinates [n, 2]
    penalties: np.ndarray  # Penalties for skipping nodes
    prizes: np.ndarray  # Prizes for visiting nodes
    
    @property
    def total_prize(self):
        """The minimum required prize to collect"""
        return 1.0  # As per the problem definition (normalized)
    
    @classmethod
    def from_dict(cls, data_dict):
        """Create a PCTSPData object from a dictionary"""
        return cls(
            instance_id=data_dict['instance_id'],
            size=data_dict['size'],
            depot=data_dict['depot'],
            locations=data_dict['locations'],
            penalties=data_dict['penalties'],
            prizes=data_dict['deterministic_prize']
        )

class PCTSPSolution:
    """Represents a solution to the PCTSP problem"""
    def __init__(self, 
                 tour: List[int], 
                 unvisited: Optional[List[int]] = None):
        """
        Initialize a PCTSP solution
        
        Args:
            tour: List of node indices in the tour (does not include depot)
            unvisited: List of unvisited nodes (if None, will be determined based on the tour)
        """
        self.tour = tour
        if unvisited is None:
            self.unvisited = self._determine_unvisited()
        else:
            self.unvisited = unvisited

    def _determine_unvisited(self) -> List[int]:
        """Determine which nodes are unvisited based on the tour"""
        n = DATA.size
        unvisited = [i for i in range(n) if i not in self.tour]
        return unvisited
    
    def objective(self) -> float:
        """
        Calculate the objective value of the PCTSP solution
        
        Returns:
            Total cost = tour length + penalties for unvisited nodes
        """
        # Initialize tour with depot
        full_tour = [self.tour[i] for i in range(len(self.tour))]
        
        # Calculate tour length
        if not full_tour:
            # If no nodes are visited, the tour length is 0
            tour_length = 0
        else:
            # Distance from depot to first node
            tour_length = np.linalg.norm(DATA.depot - DATA.locations[full_tour[0]])
            
            # Distances between consecutive nodes
            for i in range(len(full_tour) - 1):
                tour_length += np.linalg.norm(
                    DATA.locations[full_tour[i]] - DATA.locations[full_tour[i+1]]
                )
            
            # Distance from last node back to depot
            tour_length += np.linalg.norm(DATA.locations[full_tour[-1]] - DATA.depot)
        
        # Calculate penalties for unvisited nodes
        penalty_cost = sum(DATA.penalties[i] for i in self.unvisited)
        
        return tour_length + penalty_cost
    
    def total_prize(self) -> float:
        """Calculate the total prize collected in the tour"""
        return sum(DATA.prizes[i] for i in self.tour)
    
    def is_feasible(self) -> bool:
        """Check if the solution satisfies the total prize constraint"""
        return self.total_prize() >= DATA.total_prize
    
    def insert(self, node: int, idx: int):
        """Insert a node at the specified index in the tour"""
        self.tour.insert(idx, node)
        if node in self.unvisited:
            self.unvisited.remove(node)
    
    def remove(self, node: int):
        """Remove a node from the tour"""
        self.tour.remove(node)
        self.unvisited.append(node)

def construct_initial_solution(use_greedy: bool = True) -> PCTSPSolution:
    """
    Construct an initial solution for the PCTSP
    
    Args:
        use_greedy: If True, use a greedy construction heuristic,
                    otherwise construct a random solution
    
    Returns:
        An initial PCTSP solution
    """
    n = DATA.size
    
    if use_greedy:
        # Calculate prize-to-penalty ratio for each node
        ratios = []
        for i in range(n):
            # Calculate distance from depot
            dist = np.linalg.norm(DATA.depot - DATA.locations[i])
            # Ratio of prize to (penalty + distance)
            ratio = DATA.prizes[i] / (DATA.penalties[i] + dist)
            ratios.append((i, ratio))
        
        # Sort nodes by decreasing ratio
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Start with an empty tour
        tour = []
        current_prize = 0
        
        # Add nodes until prize constraint is satisfied
        for i, _ in ratios:
            tour.append(i)
            current_prize += DATA.prizes[i]
            if current_prize >= DATA.total_prize:
                break
        
        # Reorder tour using nearest neighbor
        if tour:
            reordered_tour = [tour[0]]
            remaining = tour[1:]
            
            while remaining:
                current = reordered_tour[-1]
                dists = []
                for i, node in enumerate(remaining):
                    dist = np.linalg.norm(DATA.locations[current] - DATA.locations[node])
                    dists.append((i, dist))
                
                closest_idx, _ = min(dists, key=lambda x: x[1])
                reordered_tour.append(remaining[closest_idx])
                del remaining[closest_idx]
            
            tour = reordered_tour
        
        return PCTSPSolution(tour)
    else:
        # Random solution construction
        tour = []
        unvisited = list(range(n))
        current_prize = 0
        
        while current_prize < DATA.total_prize and unvisited:
            node = np.random.choice(unvisited)
            tour.append(node)
            unvisited.remove(node)
            current_prize += DATA.prizes[node]
        
        return PCTSPSolution(tour)

# =====================
# DESTROY OPERATORS
# =====================

def random_removal(solution, rng, n_remove=3, **kwargs):
    """
    Randomly removes n_remove nodes from the tour.
    Rationale: Diversifies search by random node removal.
    """
    destroyed = deepcopy(solution)
    if not destroyed.tour:
        return destroyed
    n_remove = min(n_remove, len(destroyed.tour))
    nodes_to_remove = rng.choice(destroyed.tour, size=n_remove, replace=False)
    for node in nodes_to_remove:
        destroyed.remove(node)
    return destroyed

def adjacent_removal(solution, rng, n_remove=3, **kwargs):
    """
    Removes n_remove adjacent nodes from the tour.
    Rationale: Removes spatially or sequentially connected nodes, 
    allowing for local restructuring of tour segments.
    """
    destroyed = deepcopy(solution)
    if not destroyed.tour or len(destroyed.tour) < n_remove:
        return destroyed
    
    # Randomly select a starting position in the tour
    max_start = len(destroyed.tour) - n_remove + 1
    start_pos = rng.integers(0, max_start)
    
    # Remove n_remove consecutive nodes from the tour
    nodes_to_remove = []
    for i in range(n_remove):
        if start_pos + i < len(destroyed.tour):
            nodes_to_remove.append(destroyed.tour[start_pos + i])
    
    # Check feasibility before removing
    temp_prize = destroyed.total_prize()
    for node in nodes_to_remove:
        temp_prize -= DATA.prizes[node]
    
    # Only remove if we can maintain some feasibility or the solution is minimal
    if temp_prize >= 0.5 * DATA.total_prize or len(destroyed.tour) <= n_remove:
        for node in nodes_to_remove:
            if node in destroyed.tour:
                destroyed.remove(node)
    else:
        # Remove fewer nodes to maintain feasibility
        nodes_removed = 0
        temp_prize = destroyed.total_prize()
        for node in nodes_to_remove:
            if temp_prize - DATA.prizes[node] >= 0.5 * DATA.total_prize:
                destroyed.remove(node)
                temp_prize -= DATA.prizes[node]
                nodes_removed += 1
                if nodes_removed >= n_remove:
                    break
    
    return destroyed

# =====================
# REPAIR OPERATORS
# =====================

def greedy_insertion(solution, rng, **kwargs):
    """
    Greedily inserts unvisited nodes with the best prize-to-cost ratio until feasible.
    Rationale: Inserts nodes with best prize-to-cost ratio.
    """
    repaired = deepcopy(solution)
    unvisited = repaired.unvisited.copy()
    while not repaired.is_feasible() and unvisited:
        best_node = None
        best_ratio = -float('inf')
        best_pos = 0
        for node in unvisited:
            for i in range(len(repaired.tour) + 1):
                temp = deepcopy(repaired)
                temp.insert(node, i)
                cost_increase = temp.objective() - repaired.objective()
                ratio = DATA.prizes[node] / (cost_increase + 1e-6)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_node = node
                    best_pos = i
        if best_node is not None:
            repaired.insert(best_node, best_pos)
            unvisited.remove(best_node)
        else:
            break
    return repaired

# =====================
# ALNS IMPLEMENTATION
# =====================

def run_alns(data_instance, 
             destroy_operators: List[Callable], 
             repair_operators: List[Callable], 
             n_iterations=1000, 
             seed=SEED):
    """
    Run the ALNS algorithm on a PCTSP instance with specified operators.
    
    Args:
        data_instance: The PCTSPData object for the instance.
        destroy_operators: A list of destroy operator functions.
        repair_operators: A list of repair operator functions.
        n_iterations: Maximum number of ALNS iterations.
        seed: Random seed for reproducibility.
    
    Returns:
        Tuple[PCTSPSolution, Dict]: The best solution found and statistics.
    """
    global DATA
    DATA = data_instance
    
    # Create initial solution
    initial_solution = construct_initial_solution(use_greedy=True)
    
    # Initialize ALNS with a random number generator
    rng = rnd.default_rng(seed)
    alns = ALNS(rng) 
    
    # Add destroy operators
    for op in destroy_operators:
        alns.add_destroy_operator(op)
    
    # Add repair operators
    for op in repair_operators:
        alns.add_repair_operator(op)
    
    # Set acceptance criterion (simulated annealing)
    accept = SimulatedAnnealing(
        start_temperature=20.0,
        end_temperature=0.01,
        step=0.1,
        method="exponential"
    )
    
    # Set weight adjustment (roulette wheel)
    num_destroy = len(destroy_operators)
    num_repair = len(repair_operators)
    
    # RouletteWheel expects scores in a specific format: [w11, w12, w21, w22]
    # Where wij is the weight for operator type i and weight type j
    # For simplicity, we'll use the same initial weight for all
    select = RouletteWheel(
        num_destroy=num_destroy,
        num_repair=num_repair,
        scores=[1.0, 1.0, 1.0, 1.0],  # Fixed: provide exactly 4 scores
        decay=0.8 
    )
    
    # Set stopping criterion
    stopping_criterion = MaxIterations(n_iterations)
    
    # Run the algorithm
    start_time = time.time()
    result = alns.iterate(
        initial_solution,
        select,
        accept,
        stopping_criterion
    )
    end_time = time.time()
    solution = result.best_state
    
    # Return the results 
    stats = {
        'objective_value': solution.objective(),
        'prize_collected': solution.total_prize(),
        'time': end_time - start_time,
        'feasible': solution.is_feasible()
    }
    
    return solution, stats

# =====================
# GRID SEARCH
# =====================

def run_grid_search(n_remove_values, num_instances=5, n_iterations=500, n_runs=3):
    """
    Run grid search for optimal n_remove parameter.
    
    Args:
        n_remove_values: List of n_remove values to test
        num_instances: Number of instances to test on
        n_iterations: Number of ALNS iterations per run
        n_runs: Number of runs per configuration for statistical significance
    
    Returns:
        DataFrame with results
    """
    
    # Load instances (using 20-node instances for faster testing)
    filename = "pctsp_data/pctsp_20_20_instances.pkl"
    try:
        with open(filename, 'rb') as f:
            all_instances = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Instance file not found at {filename}")
        return pd.DataFrame()
    
    # Use only the first num_instances
    instances = all_instances[:num_instances]
    
    results = []
    
    # Define the operators to test
    destroy_operators_configs = {
        'random_removal': [random_removal],
        'adjacent_removal': [adjacent_removal],
        'both': [random_removal, adjacent_removal]
    }
    
    repair_operators = [greedy_insertion]
    
    print("Starting Grid Search for Optimal n_remove Parameter")
    print("=" * 60)
    
    for config_name, destroy_ops in destroy_operators_configs.items():
        print(f"\nTesting configuration: {config_name}")
        print("-" * 40)
        
        for n_remove in n_remove_values:
            print(f"Testing n_remove = {n_remove}")
            
            config_results = []
            
            for instance_dict in instances:
                instance = PCTSPData.from_dict(instance_dict)
                
                # Run multiple times for statistical significance
                for run in range(n_runs):
                    # Create operators with current n_remove parameter
                    current_destroy_ops = []
                    for op in destroy_ops:
                        # Create a lambda that captures the current n_remove value
                        def make_operator(operator, n_rem):
                            return lambda sol, rng, **kwargs: operator(sol, rng, n_remove=n_rem, **kwargs)
                        current_destroy_ops.append(make_operator(op, n_remove))
                    
                    try:
                        solution, stats = run_alns(
                            instance, 
                            current_destroy_ops, 
                            repair_operators, 
                            n_iterations=n_iterations,
                            seed=SEED + run  # Different seed for each run
                        )
                        
                        result = {
                            'config': config_name,
                            'n_remove': n_remove,
                            'instance_id': instance.instance_id,
                            'run': run,
                            'objective_value': stats['objective_value'],
                            'prize_collected': stats['prize_collected'],
                            'time': stats['time'],
                            'feasible': stats['feasible'],
                            'tour_length': len(solution.tour)
                        }
                        
                        results.append(result)
                        config_results.append(stats['objective_value'])
                        
                    except Exception as e:
                        print(f"Error in run {run} for instance {instance.instance_id}: {e}")
                        continue
            
            # Print summary for this n_remove value
            if config_results:
                avg_obj = np.mean(config_results)
                std_obj = np.std(config_results)
                print(f"  Average objective: {avg_obj:.3f} ± {std_obj:.3f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def analyze_results(results_df):
    """
    Analyze and visualize the grid search results.
    
    Args:
        results_df: DataFrame with grid search results
    """
    if results_df.empty:
        print("No results to analyze.")
        return
    
    print("\nGrid Search Results Analysis")
    print("=" * 60)
    
    # Group by configuration and n_remove
    summary = results_df.groupby(['config', 'n_remove']).agg({
        'objective_value': ['mean', 'std', 'min'],
        'time': 'mean',
        'feasible': 'mean'
    }).round(3)
    
    print("\nSummary Statistics:")
    print(summary)
    
    # Find best configuration for each destroy operator set
    print("\nBest n_remove for each configuration:")
    for config in results_df['config'].unique():
        config_data = results_df[results_df['config'] == config]
        best_n_remove = config_data.groupby('n_remove')['objective_value'].mean().idxmin()
        best_obj = config_data.groupby('n_remove')['objective_value'].mean().min()
        print(f"  {config}: n_remove = {best_n_remove} (avg objective = {best_obj:.3f})")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot objective values by n_remove for each configuration
    for config in results_df['config'].unique():
        config_data = results_df[results_df['config'] == config]
        avg_by_n_remove = config_data.groupby('n_remove')['objective_value'].mean()
        std_by_n_remove = config_data.groupby('n_remove')['objective_value'].std()
        
        plt.errorbar(avg_by_n_remove.index, avg_by_n_remove.values, 
                    yerr=std_by_n_remove.values, 
                    marker='o', label=config, capsize=5)
    
    plt.xlabel('n_remove')
    plt.ylabel('Average Objective Value')
    plt.title('Grid Search Results: Objective Value vs n_remove')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap showing performance
    pivot_table = results_df.groupby(['config', 'n_remove'])['objective_value'].mean().unstack()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_table.values, cmap='viridis_r', aspect='auto')
    plt.colorbar(label='Average Objective Value')
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.xlabel('n_remove')
    plt.ylabel('Configuration')
    plt.title('Performance Heatmap: Lower is Better')
    
    # Add values to heatmap
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            plt.text(j, i, f'{pivot_table.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.show()

# =====================
# MAIN EXECUTION
# =====================

if __name__ == "__main__":
    # Define the range of n_remove values to test
    n_remove_values = [1, 2, 3, 4, 5, 6, 7]
    
    # Run the grid search
    print("Running grid search to find optimal n_remove parameter...")
    results = run_grid_search(
        n_remove_values=n_remove_values,
        num_instances=5,  # Test on 5 instances
        n_iterations=300,  # Reduced for faster testing
        n_runs=3  # 3 runs per configuration
    )
    
    # Save results
    results.to_csv('grid_search_n_remove_results.csv', index=False)
    print(f"\nResults saved to: grid_search_n_remove_results.csv")
    
    # Analyze and visualize results
    analyze_results(results)
    
    print("\nGrid search completed!") 