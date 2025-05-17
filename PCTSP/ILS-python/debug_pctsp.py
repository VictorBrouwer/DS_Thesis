#!/usr/bin/env python3
import pickle
import sys
import os
import numpy as np
import time

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
                dist = np.sqrt(np.sum((all_points[i] - all_points[j])**2))
                dist_matrix[i, j] = int(round(dist * 100))  # Scale up for better precision
    
    return dist_matrix

def load_pickle_data(pickle_file):
    """Load PCTSP data from pickle file"""
    with open(pickle_file, 'rb') as f:
        instances = pickle.load(f)
    return instances

def debug_solution_quality(solution):
    """Debug the components of solution quality"""
    pctsp = solution.pctsp
    route = solution.route
    size = solution.size
    
    # Recalculate the quality components
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
    
    # Total quality should be travel cost + penalty
    total_quality = travel_cost + penalty_total
    
    print(f"Prize total: {prize_total}")
    print(f"Prize minimum: {pctsp.prize_min}")
    print(f"Prize valid: {prize_total >= pctsp.prize_min}")
    print(f"Travel cost: {travel_cost}")
    print(f"Penalty total: {penalty_total}")
    print(f"Calculated quality: {total_quality}")
    print(f"Solution quality: {solution.quality}")
    
    return {
        'prize_total': prize_total,
        'prize_min': pctsp.prize_min,
        'is_valid': prize_total >= pctsp.prize_min,
        'travel_cost': travel_cost,
        'penalty_total': penalty_total,
        'calculated_quality': total_quality,
        'solution_quality': solution.quality
    }

def run_debug_on_instance(instance, pickle_file):
    """Run PCTSP solver on a given instance and debug solution quality"""
    print("\n" + "="*60)
    print(f"Debugging instance {instance['instance_id']} from {os.path.basename(pickle_file)}")
    print("="*60)
    
    pctsp = Pctsp()
    pctsp.file_name = f"problem_{instance['size']}_100_100_1000.pctsp"
    pctsp.type = 'A'
    
    # Scale up prizes and penalties
    scale_factor = 100
    
    # Extract data from instance
    dist_matrix = compute_distance_matrix(instance['locations'], instance['depot'])
    pctsp.prize = instance['deterministic_prize'] * scale_factor
    pctsp.penal = instance['penalties'] * scale_factor
    pctsp.cost = dist_matrix
    
    # Compute prize_min
    pctsp.prize_min = np.sum(pctsp.prize) * pctsp.sigma
    
    # Use 70% of cities
    size = int(len(pctsp.prize) * 0.7)
    
    print("\nProblem setup:")
    print(f"Problem size: {instance['size']} cities")
    print(f"Prize min threshold: {pctsp.prize_min:.2f}")
    print(f"Sum of all prizes: {np.sum(pctsp.prize):.2f}")
    print(f"Sum of all penalties: {np.sum(pctsp.penal):.2f}")
    print(f"Avg travel cost: {np.mean(dist_matrix[dist_matrix > 0]):.2f}")
    
    # Generate random solution
    print("\nGenerating random solution...")
    s_random = solution.random(pctsp, size=size)
    print("Random solution completed")
    
    print("\nRandom Solution Analysis:")
    random_debug = debug_solution_quality(s_random)
    
    # Run improved local search with debug
    print("\nRunning improved local search...")
    s_improved = ils.ilocal_search(s_random)
    print("Improved local search completed")
    
    print("\nImproved Solution Analysis:")
    improved_debug = debug_solution_quality(s_improved)
    
    # Examine quality improvement
    improvement = (random_debug['solution_quality'] - improved_debug['solution_quality']) / random_debug['solution_quality'] * 100
    
    print(f"\nImprovement: {improvement:.2f}%")
    print(f"Size change: From {s_random.size} to {s_improved.size} cities")
    
    return s_improved

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_pctsp.py <pickle_file> [instance_index]")
        print("Example: python debug_pctsp.py ../pctsp_data/pctsp_20_20_instances.pkl 0")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    
    # Adjust path if needed for ILS-python folder
    if not os.path.exists(pickle_file) and pickle_file.startswith("pctsp_data/"):
        parent_dir_pickle = os.path.join("..", pickle_file)
        if os.path.exists(parent_dir_pickle):
            pickle_file = parent_dir_pickle
    
    if not os.path.exists(pickle_file):
        print(f"Error: File {pickle_file} not found")
        print("Available pickle files:")
        parent_data_dir = os.path.join("..", "pctsp_data")
        if os.path.exists(parent_data_dir):
            for f in os.listdir(parent_data_dir):
                if f.endswith(".pkl"):
                    print(f"  - ../pctsp_data/{f}")
        sys.exit(1)
    
    instances = load_pickle_data(pickle_file)
    
    if len(sys.argv) >= 3:
        instance_idx = int(sys.argv[2])
        if 0 <= instance_idx < len(instances):
            run_debug_on_instance(instances[instance_idx], pickle_file)
        else:
            print(f"Instance index {instance_idx} out of range (0-{len(instances)-1})")
    else:
        # Run for first instance
        run_debug_on_instance(instances[0], pickle_file)

if __name__ == "__main__":
    main() 