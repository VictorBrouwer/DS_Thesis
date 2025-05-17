#!/usr/bin/env python3
import pickle
import sys
import os
import numpy as np
import time
import glob
from tabulate import tabulate

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the package
from salesman.pctsp.model.pctsp import Pctsp
from salesman.pctsp.model import solution
from salesman.pctsp.algo import ilocal_search as ils

def compute_distance_matrix(locations, depot=None):
    """Compute Euclidean distance matrix between all locations"""
    # If depot is provided, add it at index 0
    if depot is not None:
        all_points = np.vstack([depot.reshape(1, -1), locations])
    else:
        all_points = locations
    
    n = len(all_points)
    dist_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate Euclidean distance and round to nearest integer
                # Scale by 100 for better precision
                dist = np.sqrt(np.sum((all_points[i] - all_points[j])**2))
                dist_matrix[i, j] = int(round(dist * 100))
    
    return dist_matrix

def load_pickle_data(pickle_file):
    """Load PCTSP data from pickle file"""
    with open(pickle_file, 'rb') as f:
        instances = pickle.load(f)
    return instances

def get_solution_details(solution):
    """Get detailed breakdown of solution quality"""
    pctsp = solution.pctsp
    route = solution.route
    size = solution.size
    
    # Calculate components
    prize_total = 0
    travel_cost = 0
    penalty_total = 0
    
    # Calculate prizes and travel costs for cities in the route
    for i in range(size):
        city = route[i]
        prize_total += pctsp.prize[city]
        
        if i > 0:
            prev_city = route[i-1]
            travel_cost += pctsp.cost[prev_city][city]
        
        # Add return cost to depot
        if i + 1 == size:
            travel_cost += pctsp.cost[city][0]
    
    # Calculate penalties for non-visited cities
    for i in range(size, len(route)):
        city = route[i]
        penalty_total += pctsp.penal[city]
    
    return {
        'prize_total': prize_total,
        'prize_min': pctsp.prize_min,
        'travel_cost': travel_cost,
        'penalty_total': penalty_total,
        'total_quality': travel_cost + penalty_total
    }

def run_pctsp_on_instance(instance, use_stochastic_prize=False, size_ratio=0.7):
    """Run PCTSP solver on a given instance"""
    pctsp = Pctsp()
    
    # Set a dummy file_name to avoid regex issues
    pctsp.file_name = f"problem_{instance['size']}_100_100_1000.pctsp"
    pctsp.type = 'A'  # Default type
    
    # Scale factor for better numerical stability
    scale_factor = 100
    
    # Compute distance matrix from locations
    dist_matrix = compute_distance_matrix(instance['locations'], instance['depot'])
    
    # Set prize and penalty
    if use_stochastic_prize:
        pctsp.prize = instance['stochastic_prize'] * scale_factor
    else:
        pctsp.prize = instance['deterministic_prize'] * scale_factor
    
    pctsp.penal = instance['penalties'] * scale_factor
    pctsp.cost = dist_matrix
    
    # Compute prize_min
    pctsp.prize_min = np.sum(pctsp.prize) * pctsp.sigma
    
    # Use size_ratio of cities
    size = int(len(pctsp.prize) * size_ratio)
    
    # Generate random solution
    start_time_random = time.time()
    s_random = solution.random(pctsp, size=size)
    random_time = time.time() - start_time_random
    
    # Get details of random solution
    random_details = get_solution_details(s_random)
    
    # Run improved local search
    start_time_ils = time.time()
    s_improved = ils.ilocal_search(s_random)
    ils_time = time.time() - start_time_ils
    
    # Get details of improved solution
    improved_details = get_solution_details(s_improved)
    
    # Return results with detailed quality breakdown
    return {
        'instance_id': instance['instance_id'],
        'instance_size': instance['size'],
        'random_solution': {
            'quality': s_random.quality,
            'size': s_random.size,
            'valid': s_random.is_valid(),
            'time': random_time,
            'travel_cost': random_details['travel_cost'],
            'penalty': random_details['penalty_total']
        },
        'improved_solution': {
            'quality': s_improved.quality,
            'size': s_improved.size,
            'valid': s_improved.is_valid(),
            'time': ils_time,
            'travel_cost': improved_details['travel_cost'],
            'penalty': improved_details['penalty_total']
        },
        'improvement_pct': (s_random.quality - s_improved.quality) / s_random.quality * 100 if s_random.quality != 0 else 0
    }

def process_pickle_file(pickle_file, use_stochastic_prize=False, size_ratio=0.7):
    """Process all instances in a pickle file"""
    print(f"\nProcessing {os.path.basename(pickle_file)}...")
    
    # Load instances
    instances = load_pickle_data(pickle_file)
    problem_size = instances[0]['size']
    
    results = []
    total_start_time = time.time()
    
    for i, instance in enumerate(instances):
        print(f"  Instance {i+1}/{len(instances)} (ID: {instance['instance_id']})...", end=' ', flush=True)
        
        try:
            result = run_pctsp_on_instance(
                instance,
                use_stochastic_prize=use_stochastic_prize,
                size_ratio=size_ratio
            )
            results.append(result)
            print("Done!")
        except Exception as e:
            print(f"Error: {e}")
    
    total_time = time.time() - total_start_time
    
    # Compute statistics
    valid_improved_solutions = [r for r in results if r['improved_solution']['valid']]
    improved_qualities = [r['improved_solution']['quality'] for r in valid_improved_solutions]
    improved_travel_costs = [r['improved_solution']['travel_cost'] for r in valid_improved_solutions]
    improved_penalties = [r['improved_solution']['penalty'] for r in valid_improved_solutions]
    improved_times = [r['improved_solution']['time'] for r in valid_improved_solutions]
    
    stats = {
        'problem_size': problem_size,
        'num_instances': len(instances),
        'num_valid_solutions': len(valid_improved_solutions),
        'avg_objective': np.mean(improved_qualities) if improved_qualities else 0,
        'avg_travel_cost': np.mean(improved_travel_costs) if improved_travel_costs else 0,
        'avg_penalty': np.mean(improved_penalties) if improved_penalties else 0,
        'min_objective': np.min(improved_qualities) if improved_qualities else 0,
        'max_objective': np.max(improved_qualities) if improved_qualities else 0,
        'avg_time': np.mean(improved_times) if improved_times else 0,
        'total_time': total_time
    }
    
    return stats

def main():
    # Get paths to pickle files - adjusted for ILS-python folder
    parent_data_dir = os.path.join("..", "pctsp_data")
    
    if os.path.exists(parent_data_dir):
        pickle_files = [os.path.join(parent_data_dir, f) for f in os.listdir(parent_data_dir) if f.endswith(".pkl")]
    else:
        # Try local directory or glob pattern as fallback
        pickle_files = glob.glob("../pctsp_data/pctsp_*_instances.pkl")
        if not pickle_files:
            pickle_files = glob.glob("pctsp_data/pctsp_*_instances.pkl")
    
    if not pickle_files:
        print("No pickle files found in pctsp_data/ directory")
        sys.exit(1)
    
    # Sort by problem size (lowest to highest)
    pickle_files.sort()
    
    # Process each file
    all_stats = []
    use_stochastic = False
    size_ratio = 0.7
    
    for pickle_file in pickle_files:
        stats = process_pickle_file(
            pickle_file,
            use_stochastic_prize=use_stochastic,
            size_ratio=size_ratio
        )
        all_stats.append(stats)
    
    # Print summary table
    print("\n" + "="*80)
    print("PCTSP SUMMARY BY PROBLEM SIZE")
    print("="*80)
    
    table_headers = ["Problem Size", "Instances", "Valid Solutions", "Avg Objective", "Avg Travel Cost", "Avg Penalty", "Min Objective", "Max Objective", "Avg Time (s)"]
    table_data = []
    
    for stats in all_stats:
        # Scale down by 100 to get more readable numbers
        scale_down = 100
        
        table_data.append([
            stats['problem_size'],
            stats['num_instances'],
            f"{stats['num_valid_solutions']}/{stats['num_instances']}",
            f"{stats['avg_objective']/scale_down:.2f}",
            f"{stats['avg_travel_cost']/scale_down:.2f}",
            f"{stats['avg_penalty']/scale_down:.2f}",
            f"{stats['min_objective']/scale_down:.2f}",
            f"{stats['max_objective']/scale_down:.2f}",
            f"{stats['avg_time']:.4f}"
        ])
    
    print(tabulate(table_data, headers=table_headers, tablefmt="grid"))
    print("="*80)
    
    # Also print in CSV format for easy copying to spreadsheets
    print("\nCSV Format:")
    print(",".join(table_headers))
    for row in table_data:
        print(",".join(str(cell) for cell in row))

if __name__ == "__main__":
    main() 