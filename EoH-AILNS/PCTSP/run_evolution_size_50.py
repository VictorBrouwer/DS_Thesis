#!/usr/bin/env python3
"""
Run EoH-PCTSP Evolution for Problem Size 50
"""

import time
from eoh_pctsp import EoH_PCTSP

def main():
    """Run evolution for problem size 50"""
    
    print("🧬 EoH-PCTSP: Evolution for Problem Size 50")
    print("=" * 60)
    
    # Parameters for size 50 evolution
    pop_size = 5
    generations = 4
    problem_size = 50
    max_instances = 2
    output_dir = 'eoh_pctsp_results_50'
    
    print(f"Parameters:")
    print(f"  Population size: {pop_size}")
    print(f"  Generations: {generations}")
    print(f"  Problem size: {problem_size} nodes")
    print(f"  Max instances: {max_instances}")
    print(f"  Output directory: {output_dir}")
    print()
    
    estimated_time = pop_size * generations * 0.5
    print(f"Estimated runtime: {estimated_time:.1f} minutes")
    print()
    
    try:
        # Initialize and run EoH-PCTSP for size 50
        eoh = EoH_PCTSP(
            pop_size=pop_size,
            n_generations=generations,
            problem_size=problem_size,
            max_instances=max_instances,
            output_dir=output_dir
        )
        
        start_time = time.time()
        final_population = eoh.run()
        runtime = time.time() - start_time
        
        print(f"\n✅ EoH-PCTSP for size 50 completed successfully!")
        print(f"Total runtime: {runtime/60:.1f} minutes")
        print(f"Best operator:")
        print(f"  Algorithm: {final_population[0]['algorithm']}")
        print(f"  Objective: {final_population[0]['objective']:.2f}")
        print(f"  Gap: {final_population[0]['gap']:.2f}%")
        print(f"  Feasible: {final_population[0]['feasible']}")
        
        return final_population
        
    except Exception as e:
        print(f"❌ Error running EoH-PCTSP for size 50: {e}")
        raise

if __name__ == "__main__":
    main() 