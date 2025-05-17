#!/usr/bin/env python3
import pickle
import sys
import os
import numpy as np
import time
import argparse

# Add the parent directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the package
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
                dist_matrix[i, j] = int(round(dist * 100))
    
    return dist_matrix

def load_pickle_data(pickle_file):
    """Load PCTSP data from pickle file"""
    with open(pickle_file, 'rb') as f:
        instances = pickle.load(f)
    return instances

def run_pctsp_on_instance(instance, verbose=True, use_stochastic_prize=False, size_ratio=0.7):
    """Run PCTSP solver on a given instance"""
    pctsp = Pctsp()
    
    # Set a dummy file_name to avoid regex issues
    pctsp.file_name = f"problem_{instance['size']}_100_100_1000.pctsp"
    pctsp.type = 'A'  # Default type
    
    # Scale factor for better precision
    scale_factor = 100
    
    # Extract data from instance
    # Compute distance matrix from the locations
    dist_matrix = compute_distance_matrix(instance['locations'], instance['depot'])
    
    # Set prize and penalty
    if use_stochastic_prize:
        pctsp.prize = instance['stochastic_prize'] * scale_factor
    else:
        pctsp.prize = instance['deterministic_prize'] * scale_factor
    
    pctsp.penal = instance['penalties'] * scale_factor
    pctsp.cost = dist_matrix
    
    # Call setup without calling setup_type
    pctsp.prize_min = np.sum(pctsp.prize) * pctsp.sigma
    
    if verbose:
        print(f"Instance ID: {instance['instance_id']}")
        print(f"Instance Size: {instance['size']}")
        print(f"Prize min threshold: {pctsp.prize_min}")
        print(f"Using {'stochastic' if use_stochastic_prize else 'deterministic'} prize")
    
    # Use size_ratio of cities 
    size = int(len(pctsp.prize) * size_ratio)
    
    # Record time for random solution
    start_time_random = time.time()
    # Generate random solution
    s_random = solution.random(pctsp, size=size)
    random_time = time.time() - start_time_random
    
    if verbose:
        print(f"Random Solution Route: {s_random.route}")
        print(f"Size: {s_random.size}")
        print(f"Quality: {s_random.quality}")
        print(f"Valid: {s_random.is_valid()}")
        print(f"Time: {random_time:.4f} seconds")
        print("\n")
    
    # Record time for improved local search
    start_time_ils = time.time()
    # Run improved local search
    s_improved = ils.ilocal_search(s_random)
    ils_time = time.time() - start_time_ils
    
    if verbose:
        print(f"Improved Solution Route: {s_improved.route}")
        print(f"Size: {s_improved.size}")
        print(f"Quality: {s_improved.quality}")
        print(f"Valid: {s_improved.is_valid()}")
        print(f"Time: {ils_time:.4f} seconds")
        print(f"Quality improvement: {(s_random.quality - s_improved.quality) / s_random.quality * 100:.2f}%")
    
    results = {
        'instance_id': instance['instance_id'],
        'instance_size': instance['size'],
        'random_solution': {
            'quality': s_random.quality,
            'size': s_random.size,
            'valid': s_random.is_valid(),
            'time': random_time
        },
        'improved_solution': {
            'quality': s_improved.quality,
            'size': s_improved.size,
            'valid': s_improved.is_valid(),
            'time': ils_time
        },
        'improvement_pct': (s_random.quality - s_improved.quality) / s_random.quality * 100
    }
    
    return results

def run_all_instances(instances, verbose=False, use_stochastic_prize=False, size_ratio=0.7):
    """Run PCTSP solver on all instances and collect statistics"""
    results = []
    total_start_time = time.time()
    
    for i, instance in enumerate(instances):
        print(f"Processing instance {i+1}/{len(instances)} (ID: {instance['instance_id']})...", end=' ')
        sys.stdout.flush()
        
        try:
            result = run_pctsp_on_instance(
                instance, 
                verbose=verbose, 
                use_stochastic_prize=use_stochastic_prize,
                size_ratio=size_ratio
            )
            results.append(result)
            print("Done!")
        except Exception as e:
            print(f"Error: {e}")
    
    total_time = time.time() - total_start_time
    
    # Calculate statistics
    valid_random_solutions = [r for r in results if r['random_solution']['valid']]
    valid_improved_solutions = [r for r in results if r['improved_solution']['valid']]
    
    stats = {
        'total_instances': len(instances),
        'processed_instances': len(results),
        'valid_random_solutions': len(valid_random_solutions),
        'valid_improved_solutions': len(valid_improved_solutions),
        'avg_random_quality': np.mean([r['random_solution']['quality'] for r in results]) if results else 0,
        'avg_improved_quality': np.mean([r['improved_solution']['quality'] for r in results]) if results else 0,
        'avg_improvement_pct': np.mean([r['improvement_pct'] for r in results]) if results else 0,
        'avg_random_time': np.mean([r['random_solution']['time'] for r in results]) if results else 0,
        'avg_improved_time': np.mean([r['improved_solution']['time'] for r in results]) if results else 0,
        'total_time': total_time
    }
    
    return results, stats

def print_statistics(stats):
    """Print statistics in a nice format"""
    print("\n" + "="*50)
    print("PCTSP RESULTS SUMMARY")
    print("="*50)
    
    print(f"Total instances: {stats['total_instances']}")
    print(f"Processed instances: {stats['processed_instances']}")
    print(f"Valid random solutions: {stats['valid_random_solutions']} ({stats['valid_random_solutions']/stats['total_instances']*100:.1f}%)")
    print(f"Valid improved solutions: {stats['valid_improved_solutions']} ({stats['valid_improved_solutions']/stats['total_instances']*100:.1f}%)")
    
    print(f"\nAverage random solution quality: {stats['avg_random_quality']/100:.2f}")
    print(f"Average improved solution quality: {stats['avg_improved_quality']/100:.2f}")
    print(f"Average improvement: {stats['avg_improvement_pct']:.2f}%")
    
    print(f"\nAverage time for random solution: {stats['avg_random_time']:.4f} seconds")
    print(f"Average time for improved solution: {stats['avg_improved_time']:.4f} seconds")
    print(f"Total processing time: {stats['total_time']:.4f} seconds")
    print("="*50)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run PCTSP solver on pickle data instances')
    
    parser.add_argument('pickle_file', help='Path to the pickle file with PCTSP instances')
    parser.add_argument('-i', '--instance', type=int, help='Run for a specific instance index')
    parser.add_argument('-a', '--all', action='store_true', help='Run for all instances in the file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('-s', '--stochastic', action='store_true', help='Use stochastic prize instead of deterministic')
    parser.add_argument('-r', '--size-ratio', type=float, default=0.7, help='Ratio of cities to use (default: 0.7)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get path to pickle file - adjust for relative path within ILS-python folder
    pickle_file = args.pickle_file
    if not os.path.exists(pickle_file) and pickle_file.startswith("pctsp_data/"):
        parent_dir_pickle = os.path.join("..", pickle_file)
        if os.path.exists(parent_dir_pickle):
            pickle_file = parent_dir_pickle
    
    # Check if pickle file exists
    if not os.path.exists(pickle_file):
        print(f"Error: Pickle file '{pickle_file}' not found")
        print("Available pickle files:")
        parent_data_dir = os.path.join("..", "pctsp_data")
        if os.path.exists(parent_data_dir):
            for f in os.listdir(parent_data_dir):
                if f.endswith(".pkl"):
                    print(f"  - ../pctsp_data/{f}")
        else:
            print("  No data directory found")
        sys.exit(1)
    
    # Load instances
    instances = load_pickle_data(pickle_file)
    
    # Print number of instances
    print(f"Loaded {len(instances)} instances from {pickle_file}")
    
    if args.all:
        # Run for all instances
        results, stats = run_all_instances(
            instances, 
            verbose=args.verbose,
            use_stochastic_prize=args.stochastic,
            size_ratio=args.size_ratio
        )
        print_statistics(stats)
    elif args.instance is not None:
        # Run for specific instance
        if 0 <= args.instance < len(instances):
            print(f"Running PCTSP for instance {args.instance}")
            run_pctsp_on_instance(
                instances[args.instance], 
                verbose=True,
                use_stochastic_prize=args.stochastic,
                size_ratio=args.size_ratio
            )
        else:
            print(f"Instance index {args.instance} out of range (0-{len(instances)-1})")
    else:
        # Run for first instance
        print("Running PCTSP for instance 0")
        run_pctsp_on_instance(
            instances[0], 
            verbose=True,
            use_stochastic_prize=args.stochastic,
            size_ratio=args.size_ratio
        )

if __name__ == "__main__":
    main() 