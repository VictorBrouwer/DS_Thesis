import os
import json
import time
import re
import numpy as np
from copy import deepcopy
from typing import List, Dict, Any

from eoh_evolution_pfsp import Evolution
from llm_api import LLMInterface
from pfsp_prompts import GetPFSPPrompts
import PFSP
from PFSP import (
    Solution, compute_makespan, NEH, evaluate_operator,
    random_removal, adjacent_removal, greedy_repair_then_local_search
)

# Global DATA variable
DATA = None

class EoH_PFSP:
    """Evolution of Heuristics framework for PFSP repair operators"""
    
    def __init__(self, 
                 api_endpoint=None, 
                 api_key=None, 
                 model_llm=None,
                 debug_mode=True,
                 pop_size=4,
                 n_generations=3,
                 data_file="data/j20_m5/j20_m5_01.txt",
                 output_dir="eoh_results"):
        
        self.debug_mode = debug_mode
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.data_file = data_file
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "best"), exist_ok=True)
        
        # Initialize prompts
        self.prompts = GetPFSPPrompts()
        
        # Initialize evolution engine
        self.evolution = Evolution(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_LLM=model_llm,
            llm_use_local=False,
            llm_local_url=None,
            debug_mode=False,  # Disable interactive prompts for automated runs
            prompts=self.prompts
        )
        
        # Load problem data
        PFSP.DATA = PFSP.Data.from_file(self.data_file)
        
        # Also set the global DATA for this module
        global DATA
        DATA = PFSP.DATA
        
        # Create initial solution
        self.initial_solution = NEH(DATA.processing_times)
        
        print(f"Initialized EoH-PFSP for problem: {DATA.n_jobs} jobs, {DATA.n_machines} machines")
        print(f"Best known value: {DATA.bkv}")
        print(f"Initial solution objective: {self.initial_solution.objective()}")
        
    def create_operator_from_code(self, code_str: str, algorithm_desc: str) -> Dict[str, Any]:
        """Create an operator dictionary from LLM-generated code"""
        
        # Clean up the code (keep it simple)
        function_code = self._clean_code(code_str)
        
        # Create individual dictionary
        individual = {
            'algorithm': algorithm_desc,
            'code': function_code,
            'objective': None,
            'gap': None,
            'runtime': None,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'feasible': False
        }
        
        # Try to create and evaluate the operator (like original PFSP approach)
        try:
            # Create namespace using globals() like the original
            namespace = {}
            exec(function_code, globals(), namespace)
            
            if 'llm_repair' in namespace:
                operator_func = namespace['llm_repair']
                # Evaluate the operator
                evaluation = evaluate_operator(operator_func, deepcopy(self.initial_solution), self.data_file)
                
                individual.update({
                    'objective': int(evaluation['objective']),
                    'gap': float(evaluation['gap']),
                    'runtime': float(evaluation['runtime']),
                    'feasible': True
                })
                
                print(f"Created feasible operator - Objective: {evaluation['objective']}, Gap: {evaluation['gap']:.2f}%")
                
            else:
                print("Error: 'llm_repair' function not found in generated code")
                
        except Exception as e:
            print(f"Error creating operator: {e}")
            # Set default poor performance for infeasible operators
            individual.update({
                'objective': int(DATA.bkv * 2),
                'gap': 100.0,
                'runtime': 0.0,
                'feasible': False,
                'error': str(e)
            })
            
        return individual
    
    def _clean_code(self, code_str: str) -> str:
        """Clean and fix common issues in LLM-generated code (keep it simple like original)"""
        
        # Remove markdown backticks
        code_str = re.sub(r'^```python\n', '', code_str)
        code_str = re.sub(r'^```\n', '', code_str)
        code_str = re.sub(r'\n```$', '', code_str)
        
        # Fix common issues (like original)
        code_str = code_str.replace('random.shuffle', 'rng.shuffle')
        code_str = code_str.replace('random.choice', 'rng.choice')
        code_str = code_str.replace('random.randint', 'rng.integers')
        code_str = code_str.replace('random.random', 'rng.random')
        code_str = code_str.replace('calculate_makespan', 'compute_makespan')
        code_str = code_str.replace('np.random.', 'rng.')
        
        # Ensure the function signature is correct
        if 'def llm_repair(' not in code_str:
            code_str = re.sub(r'def \w+\(', 'def llm_repair(', code_str)
            
        return code_str
    
    def generate_initial_population(self) -> List[Dict[str, Any]]:
        """Generate initial population using i1 strategy"""
        
        print(f"\n=== Generating Initial Population (size: {self.pop_size}) ===")
        population = []
        
        for i in range(self.pop_size):
            print(f"\nGenerating individual {i+1}/{self.pop_size}")
            
            try:
                code_str, algorithm_desc = self.evolution.i1()
                individual = self.create_operator_from_code(code_str, algorithm_desc)
                population.append(individual)
                
            except Exception as e:
                print(f"Error generating individual {i+1}: {e}")
                # Create a fallback individual (like original)
                individual = {
                    'algorithm': f"Fallback operator {i+1}",
                    'code': self._get_fallback_code(),
                    'objective': int(DATA.bkv * 1.5),
                    'gap': 50.0,
                    'runtime': 0.0,
                    'feasible': False,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'error': str(e)
                }
                population.append(individual)
        
        # Sort population by objective (best first)
        population.sort(key=lambda x: x['objective'])
        
        # Save initial population
        self._save_population(population, 0)
        
        print(f"\nInitial population created:")
        for i, ind in enumerate(population):
            print(f"  {i+1}. Objective: {ind['objective']}, Gap: {ind['gap']:.2f}%, Feasible: {ind['feasible']}")
            
        return population
    
    def _get_fallback_code(self) -> str:
        """Get fallback repair operator code (simple and safe)"""
        return """def llm_repair(state, rng, **kwargs):
    # Fallback: simple greedy repair like original
    state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))
    while len(state.unassigned) != 0:
        job = state.unassigned.pop()
        state.opt_insert(job)
    return state"""
    
    def evolve_population(self, population: List[Dict[str, Any]], generation: int) -> List[Dict[str, Any]]:
        """Evolve the population for one generation"""
        
        print(f"\n=== Evolution Generation {generation} ===")
        
        # Evolution strategies and their selection probabilities
        strategies = [
            ('e1', 0.3),  # Create totally different algorithms
            ('e2', 0.3),  # Create algorithms motivated by existing ones
            ('m1', 0.2),  # Modify existing algorithms
            ('m2', 0.1),  # Change parameters
            ('m3', 0.1)   # Simplify for generalization
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
                        elif strategy == 'm2':
                            code_str, algorithm_desc = self.evolution.m2(parent)
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
    
    def _save_population(self, population: List[Dict[str, Any]], generation: int):
        """Save population to JSON file (with proper type conversion)"""
        
        filename = os.path.join(self.output_dir, "generations", f"generation_{generation}.json")
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        # Prepare data for saving
        save_data = {
            "generation": generation,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem_file": self.data_file,
            "best_known_value": int(DATA.bkv),
            "population_size": len(population),
            "population": convert_types(population)
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Also save the best individual separately
        best_filename = os.path.join(self.output_dir, "best", f"best_generation_{generation}.json")
        with open(best_filename, 'w') as f:
            json.dump(convert_types(population[0]), f, indent=2)
        
        print(f"Saved generation {generation} to {filename}")
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the complete EoH evolution process"""
        
        print("Starting EoH-PFSP Evolution...")
        start_time = time.time()
        
        # Generate initial population
        population = self.generate_initial_population()
        
        # Evolution loop
        for generation in range(1, self.n_generations + 1):
            population = self.evolve_population(population, generation)
            
            # Early stopping if we find a very good solution
            best_gap = population[0]['gap']
            if best_gap < 1.0:  # Less than 1% gap
                print(f"Early stopping: Found solution with {best_gap:.2f}% gap")
                break
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n=== Final Results ===")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Best solution objective: {population[0]['objective']}")
        print(f"Best solution gap: {population[0]['gap']:.2f}%")
        print(f"Best known value: {DATA.bkv}")
        
        # Save final summary (with type conversion)
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        summary = {
            "total_runtime": total_time,
            "generations": self.n_generations,
            "population_size": self.pop_size,
            "problem_file": self.data_file,
            "best_known_value": int(DATA.bkv),
            "best_objective": convert_types(population[0]['objective']),
            "best_gap": convert_types(population[0]['gap']),
            "initial_objective": convert_types(self.initial_solution.objective()),
            "final_population": convert_types(population)
        }
        
        summary_file = os.path.join(self.output_dir, "final_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved final summary to {summary_file}")
        
        return population

if __name__ == "__main__":
    # Example usage
    eoh = EoH_PFSP(
        debug_mode=True,
        pop_size=4,
        n_generations=3,
        data_file="data/j50_m20/j50_m20_01.txt"
    )
    
    final_population = eoh.run() 