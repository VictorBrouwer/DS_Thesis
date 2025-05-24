#!/usr/bin/env python3
"""
Run extensive EoH with custom parameters (excluding m2 strategy)
"""

import time
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

def main():
    """Run extensive EoH with custom parameters"""
    
    print("🧬 Running Extensive EoH Generation (Custom)")
    print("=" * 60)
    print("Parameters:")
    print("  - Population size: 5")
    print("  - Generations: 4") 
    print("  - Strategies: e1, e2, m1, m3 (excluding m2)")
    print("  - Instance: j50_m20_01 (complex 50x20)")
    print("  - Output: extensive_results/")
    print()
    print("Estimated time: 8-12 minutes")
    print()
    
    # Run EoH with custom parameters
    eoh = CustomEoH_PFSP(
        pop_size=5,
        n_generations=4,
        data_file="data/j50_m20/j50_m20_01.txt",
        output_dir="extensive_results"
    )
    
    start_time = time.time()
    final_population = eoh.run()
    runtime = time.time() - start_time
    
    print(f"\n✅ Extensive EoH completed!")
    print(f"Runtime: {runtime/60:.1f} minutes")
    print(f"Best operator gap: {final_population[0]['gap']:.2f}%")
    print(f"Best operator algorithm: {final_population[0]['algorithm']}")
    
    return final_population

if __name__ == "__main__":
    main() 