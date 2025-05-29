#!/usr/bin/env python3
"""
Run EoH-PCTSP (Evolution of Heuristics for Price Collecting TSP)
"""

import argparse
import time
from eoh_pctsp import EoH_PCTSP

def main():
    """Main function to run EoH-PCTSP"""
    
    parser = argparse.ArgumentParser(description='Run Evolution of Heuristics for PCTSP')
    
    # EoH parameters
    parser.add_argument('--pop_size', type=int, default=4, 
                        help='Population size (default: 4)')
    parser.add_argument('--generations', type=int, default=3, 
                        help='Number of generations (default: 3)')
    parser.add_argument('--problem_size', type=int, default=20, choices=[20, 50, 100],
                        help='Problem size: 20, 50, or 100 nodes (default: 20)')
    parser.add_argument('--max_instances', type=int, default=2,
                        help='Maximum number of instances to use (default: 2)')
    parser.add_argument('--output_dir', type=str, default='eoh_pctsp_results',
                        help='Output directory (default: eoh_pctsp_results)')
    
    # LLM parameters
    parser.add_argument('--api_endpoint', type=str, default=None,
                        help='LLM API endpoint URL')
    parser.add_argument('--api_key', type=str, default=None,
                        help='LLM API key')
    parser.add_argument('--model_llm', type=str, default=None,
                        help='LLM model name')
    
    # Other parameters
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🧬 EoH-PCTSP: Evolution of Heuristics for Price Collecting TSP")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Population size: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Problem size: {args.problem_size} nodes")
    print(f"  Max instances: {args.max_instances}")
    print(f"  Output directory: {args.output_dir}")
    
    if args.api_endpoint:
        print(f"  LLM API endpoint: {args.api_endpoint}")
    if args.model_llm:
        print(f"  LLM model: {args.model_llm}")
    
    print()
    
    # Estimate runtime
    estimated_time = args.pop_size * args.generations * 0.5  # rough estimate in minutes
    print(f"Estimated runtime: {estimated_time:.1f} minutes")
    print()
    
    try:
        # Initialize and run EoH-PCTSP
        eoh = EoH_PCTSP(
            api_endpoint=args.api_endpoint,
            api_key=args.api_key,
            model_llm=args.model_llm,
            debug_mode=args.debug,
            pop_size=args.pop_size,
            n_generations=args.generations,
            problem_size=args.problem_size,
            max_instances=args.max_instances,
            output_dir=args.output_dir
        )
        
        start_time = time.time()
        final_population = eoh.run()
        runtime = time.time() - start_time
        
        print(f"\n✅ EoH-PCTSP completed successfully!")
        print(f"Total runtime: {runtime/60:.1f} minutes")
        print(f"Best operator:")
        print(f"  Algorithm: {final_population[0]['algorithm']}")
        print(f"  Objective: {final_population[0]['objective']:.2f}")
        print(f"  Gap: {final_population[0]['gap']:.2f}%")
        print(f"  Feasible: {final_population[0]['feasible']}")
        
        return final_population
        
    except Exception as e:
        print(f"❌ Error running EoH-PCTSP: {e}")
        raise

if __name__ == "__main__":
    main() 