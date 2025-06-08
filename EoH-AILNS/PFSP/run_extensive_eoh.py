#!/usr/bin/env python3
"""
Run extensive EoH with custom parameters on multiple problem sizes
"""

import time
import os
import json
from eoh_pfsp import EoH_PFSP

class CustomEoH_PFSP(EoH_PFSP):
    """EoH_PFSP with custom strategy selection (excluding m2)"""
    
    def evolve_population(self, population, generation):
        """Evolve the population for one generation (custom strategies)"""
        
        print(f"\n=== Evolution Generation {generation} ===")
        
        # Custom evolution strategies (excluding m2)
        strategies = [
            ('e1', 0.35),  # Create totally different algorithms
            ('e2', 0.35),  # Create algorithms motivated by existing ones  
            ('m1', 0.20),  # Modify existing algorithms
            ('m3', 0.10)   # Simplify for generalization
        ]
        
        new_individuals = []
        
        # Generate new individuals using different strategies
        for strategy, weight in strategies:
            n_offspring = max(1, int(self.pop_size * weight))
            
            for _ in range(n_offspring):
                try:
                    if strategy in ['e1', 'e2']:
                        # Use top 2 parents for evolution
                        parents = population[:2]
                        if strategy == 'e1':
                            code_str, algorithm_desc = self.evolution.e1(parents)
                        else:
                            code_str, algorithm_desc = self.evolution.e2(parents)
                    else:
                        # Use single best parent for mutation
                        parent = population[0]
                        if strategy == 'm1':
                            code_str, algorithm_desc = self.evolution.m1(parent)
                        else:  # m3
                            code_str, algorithm_desc = self.evolution.m3(parent)
                    
                    individual = self.create_operator_from_code(code_str, algorithm_desc)
                    individual['strategy'] = strategy
                    individual['generation'] = generation
                    new_individuals.append(individual)
                    
                except Exception as e:
                    print(f"Error in {strategy} evolution: {e}")
                    continue
        
        # Combine old and new populations
        combined_population = population + new_individuals
        
        # Select best individuals for next generation
        combined_population.sort(key=lambda x: x['objective'])
        next_population = combined_population[:self.pop_size]
        
        # Save population
        self._save_population(next_population, generation)
        
        print(f"\nGeneration {generation} results:")
        for i, ind in enumerate(next_population):
            print(f"  {i+1}. Objective: {ind['objective']}, Gap: {ind['gap']:.2f}%, Strategy: {ind.get('strategy', 'initial')}")
        
        return next_population

def run_single_evolution(problem_config, run_index):
    """Run evolution on a single problem instance"""
    
    data_file, problem_name, problem_description = problem_config
    output_dir = f"extensive_results_{problem_name}"
    
    print(f"\n{'='*80}")
    print(f"🧬 EVOLUTION RUN {run_index}/3: {problem_description}")
    print(f"{'='*80}")
    print(f"Instance: {data_file}")
    print(f"Output: {output_dir}/")
    print(f"Expected time: ~8-12 minutes")
    print()
    
    # Run EoH with custom parameters
    eoh = CustomEoH_PFSP(
        pop_size=5,
        n_generations=4,
        data_file=data_file,
        output_dir=output_dir
    )
    
    start_time = time.time()
    final_population = eoh.run()
    runtime = time.time() - start_time
    
    print(f"\n✅ Evolution completed for {problem_name}!")
    print(f"Runtime: {runtime/60:.1f} minutes")
    print(f"Best operator gap: {final_population[0]['gap']:.2f}%")
    print(f"Best operator algorithm: {final_population[0]['algorithm']}")
    
    return {
        'problem_name': problem_name,
        'problem_description': problem_description,
        'data_file': data_file,
        'output_dir': output_dir,
        'runtime_minutes': runtime/60,
        'best_gap': final_population[0]['gap'],
        'best_objective': final_population[0]['objective'],
        'final_population': final_population
    }

def combine_results(all_results):
    """Combine and analyze results from all 3 evolution runs"""
    
    print(f"\n{'='*80}")
    print(f"📊 COMBINED RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Create combined results directory
    combined_dir = "extensive_results_combined"
    os.makedirs(combined_dir, exist_ok=True)
    
    # Analysis summary
    total_time = sum(r['runtime_minutes'] for r in all_results)
    best_overall = min(all_results, key=lambda x: x['best_gap'])
    
    print(f"\nOverall Summary:")
    print(f"  Total runtime: {total_time:.1f} minutes")
    print(f"  Best overall gap: {best_overall['best_gap']:.2f}% ({best_overall['problem_name']})")
    print()
    
    # Detailed results
    print(f"Detailed Results by Problem Size:")
    print("-" * 70)
    print(f"{'Problem':<15} {'Best Gap':<10} {'Best Obj':<10} {'Runtime':<10} {'Operators'}")
    print("-" * 70)
    
    total_operators = 0
    for result in all_results:
        feasible_ops = sum(1 for op in result['final_population'] if op.get('feasible', False))
        total_operators += len(result['final_population'])
        
        print(f"{result['problem_name']:<15} {result['best_gap']:<10.2f} {result['best_objective']:<10} "
              f"{result['runtime_minutes']:<10.1f} {feasible_ops}/{len(result['final_population'])}")
    
    print("-" * 70)
    print(f"Total operators generated: {total_operators}")
    
    # Save combined summary
    combined_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_minutes': total_time,
        'total_operators': total_operators,
        'best_overall': {
            'problem': best_overall['problem_name'],
            'gap': best_overall['best_gap'],
            'objective': best_overall['best_objective']
        },
        'results_by_problem': all_results,
        'output_directories': [r['output_dir'] for r in all_results]
    }
    
    summary_file = os.path.join(combined_dir, "combined_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print(f"\n💾 Combined summary saved to: {summary_file}")
    
    # Instructions for next steps
    print(f"\n🎯 NEXT STEPS:")
    print(f"You now have 3 specialized operator sets:")
    for result in all_results:
        print(f"  - {result['output_dir']}/final_summary.json ({result['problem_description']})")
    
    print(f"\nTo evaluate operators from a specific problem size:")
    for result in all_results:
        print(f"  # For {result['problem_description']} operators:")
        print(f"  # Temporarily rename {result['output_dir']} to extensive_results")
        print(f"  # Then run: python evaluate_multi_operator.py")
        print()
    
    return combined_summary

def main():
    """Run extensive EoH on multiple problem sizes"""
    
    print("🧬 Running Extensive EoH on Multiple Problem Sizes")
    print("=" * 60)
    
    # Define problem configurations: (data_file, problem_name, description)
    problem_configs = [
        ("data/j20_m10/j20_m10_01.txt", "small", "Small Problems (20×10)"),
        ("data/j50_m20/j50_m20_01.txt", "medium", "Medium Problems (50×20)"),
        ("data/j100_m10/j100_m10_01.txt", "large", "Large Problems (100×10)")
    ]
    
    print("Evolution plan:")
    for i, (data_file, name, desc) in enumerate(problem_configs, 1):
        print(f"  {i}. {desc}: {data_file}")
    
    print(f"\nTotal estimated time: ~25-35 minutes")
    print("Each evolution will be saved to a separate directory.")
    print()
    
    proceed = input("Do you want to proceed? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Evolution cancelled.")
        return
    
    print(f"\n🚀 Starting multi-problem evolution...")
    
    # Run evolution on each problem
    all_results = []
    overall_start = time.time()
    
    for i, config in enumerate(problem_configs, 1):
        try:
            result = run_single_evolution(config, i)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error in evolution {i}: {e}")
            continue
    
    overall_time = time.time() - overall_start
    
    # Combine and analyze results
    if all_results:
        combined_summary = combine_results(all_results)
        
        print(f"\n🎉 ALL EVOLUTIONS COMPLETED!")
        print(f"Total time: {overall_time/60:.1f} minutes")
        print(f"Successfully generated {len(all_results)} operator sets")
    else:
        print(f"\n❌ No successful evolution runs completed")

if __name__ == "__main__":
    main() 