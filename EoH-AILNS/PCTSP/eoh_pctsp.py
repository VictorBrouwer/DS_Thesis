import os
import json
import time
import re
import numpy as np
import sys
from copy import deepcopy
from typing import List, Dict, Any

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eoh_evolution_pctsp import Evolution
from interface_llm import InterfaceLLM
from pctsp_prompts import GetPCTSPPrompts
import PCTSP
from PCTSP import (
    PCTSPSolution, PCTSPData, construct_initial_solution, evaluate_operator,
    random_removal, worst_removal, greedy_repair, load_instances
)

# Global DATA variable
DATA = None

class EoH_PCTSP:
    """Evolution of Heuristics framework for PCTSP repair operators"""
    
    def __init__(self, 
                 api_endpoint=None, 
                 api_key=None, 
                 model_llm=None,
                 debug_mode=True,
                 pop_size=4,
                 n_generations=3,
                 problem_size=20,
                 max_instances=2,
                 output_dir="eoh_pctsp_results"):
        
        self.debug_mode = debug_mode
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.problem_size = problem_size
        self.max_instances = max_instances
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "best"), exist_ok=True)
        
        # Initialize prompts
        self.prompts = GetPCTSPPrompts()
        
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
        
        # Load problem instances (use only first max_instances)
        self.instances = load_instances(problem_size)[:max_instances]
        if not self.instances:
            raise ValueError(f"No instances found for problem size {problem_size}")
        
        print(f"Initialized EoH-PCTSP for {len(self.instances)} instances of size {problem_size}")
        print(f"Using instances: {[inst.instance_id for inst in self.instances]}")
        
        # Set the first instance as the primary one for evaluation
        self.primary_instance = self.instances[0]
        PCTSP.DATA = self.primary_instance
        
        # Also set the global DATA for this module
        global DATA
        DATA = self.primary_instance
        
        # Create initial solution for the primary instance
        self.initial_solution = construct_initial_solution(use_greedy=True)
        
        print(f"Primary instance: {self.primary_instance.size} nodes")
        print(f"Initial solution objective: {self.initial_solution.objective():.2f}")
        print(f"Initial solution feasible: {self.initial_solution.is_feasible()}")
        
    def create_operator_from_code(self, code_str: str, algorithm_desc: str) -> Dict[str, Any]:
        """Create an operator dictionary from LLM-generated code"""
        
        # Clean up the code
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
        
        # Try to create and evaluate the operator
        try:
            # Create namespace using globals() like the original
            namespace = {}
            exec(function_code, globals(), namespace)
            
            if 'llm_repair' in namespace:
                operator_func = namespace['llm_repair']
                
                # Evaluate the operator on the primary instance
                evaluation = evaluate_operator(operator_func, deepcopy(self.initial_solution), "primary_instance")
                
                individual.update({
                    'objective': float(evaluation['objective']),
                    'gap': float(evaluation['gap']),
                    'runtime': float(evaluation['runtime']),
                    'feasible': evaluation['feasible'],
                    'tour_length': evaluation.get('tour_length', 0),
                    'prize_collected': evaluation.get('prize_collected', 0)
                })
                
                print(f"Created operator - Objective: {evaluation['objective']:.2f}, Gap: {evaluation['gap']:.2f}%, Feasible: {evaluation['feasible']}")
                
            else:
                print("Error: 'llm_repair' function not found in generated code")
                
        except Exception as e:
            print(f"Error creating operator: {e}")
            # Set default poor performance for infeasible operators
            individual.update({
                'objective': float(self.initial_solution.objective() * 2),
                'gap': 100.0,
                'runtime': 0.0,
                'feasible': False,
                'error': str(e)
            })
            
        return individual
    
    def _clean_code(self, code_str: str) -> str:
        """Clean and fix common issues in LLM-generated code"""
        
        # Remove markdown backticks
        code_str = re.sub(r'^```python\n', '', code_str)
        code_str = re.sub(r'^```\n', '', code_str)
        code_str = re.sub(r'\n```$', '', code_str)
        
        # Fix common issues
        code_str = code_str.replace('random.shuffle', 'rng.shuffle')
        code_str = code_str.replace('random.choice', 'rng.choice')
        code_str = code_str.replace('random.randint', 'rng.integers')
        code_str = code_str.replace('random.random', 'rng.random')
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
                # Create a fallback individual
                individual = {
                    'algorithm': f"Fallback PCTSP operator {i+1}",
                    'code': self._get_fallback_code(),
                    'objective': float(self.initial_solution.objective() * 1.5),
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
            print(f"  {i+1}. Objective: {ind['objective']:.2f}, Gap: {ind['gap']:.2f}%, Feasible: {ind['feasible']}")
            
        return population
    
    def _get_fallback_code(self) -> str:
        """Get fallback repair operator code (simple and safe)"""
        return """def llm_repair(state, rng, **kwargs):
    # Fallback: simple greedy repair for PCTSP
    if not state.unvisited:
        return state
    
    # Sort unvisited nodes by prize-to-penalty ratio
    ratios = []
    for node in state.unvisited:
        ratio = DATA.prizes[node] / (DATA.penalties[node] + 1e-6)
        ratios.append((node, ratio))
    
    # Sort by decreasing ratio (best first)
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Insert nodes until feasible or all inserted
    for node, _ in ratios:
        if node in state.unvisited:
            state.opt_insert(node)
            if state.is_feasible():
                break
    
    # If still not feasible, insert remaining nodes
    while state.unvisited and not state.is_feasible():
        node = state.unvisited[0]
        state.opt_insert(node)
    
    return state"""
    
    def evolve_population(self, population: List[Dict[str, Any]], generation: int) -> List[Dict[str, Any]]:
        """Evolve the population for one generation"""
        
        print(f"\n=== Evolution Generation {generation} ===")
        
        # Evolution strategies (excluding m2 as requested)
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
            strategy = ind.get('strategy', 'initial')
            print(f"  {i+1}. Objective: {ind['objective']:.2f}, Gap: {ind['gap']:.2f}%, Strategy: {strategy}")
        
        return next_population
    
    def _save_population(self, population: List[Dict[str, Any]], generation: int):
        """Save population to JSON file"""
        
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert numpy types to native Python types
        population_copy = []
        for ind in population:
            ind_copy = {}
            for key, value in ind.items():
                ind_copy[key] = convert_types(value)
            population_copy.append(ind_copy)
        
        # Save generation file
        gen_file = os.path.join(self.output_dir, "generations", f"generation_{generation}.json")
        with open(gen_file, 'w') as f:
            json.dump(population_copy, f, indent=2)
        
        # Save best individual
        best_file = os.path.join(self.output_dir, "best", f"best_gen_{generation}.json")
        with open(best_file, 'w') as f:
            json.dump(population_copy[0], f, indent=2)
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the complete EoH process"""
        
        print(f"\n🧬 Starting EoH-PCTSP Evolution")
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Problem size: {self.problem_size}")
        print(f"Instances: {len(self.instances)}")
        
        start_time = time.time()
        
        # Generate initial population
        population = self.generate_initial_population()
        
        # Evolve for specified generations
        for generation in range(1, self.n_generations + 1):
            population = self.evolve_population(population, generation)
        
        runtime = time.time() - start_time
        
        # Save final summary
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        final_population = []
        for ind in population:
            ind_copy = {}
            for key, value in ind.items():
                ind_copy[key] = convert_types(value)
            final_population.append(ind_copy)
        
        summary = {
            'parameters': {
                'pop_size': self.pop_size,
                'n_generations': self.n_generations,
                'problem_size': self.problem_size,
                'max_instances': self.max_instances,
                'instances_used': [inst.instance_id for inst in self.instances]
            },
            'runtime': runtime,
            'final_population': final_population,
            'best_individual': final_population[0],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        summary_file = os.path.join(self.output_dir, "final_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ EoH-PCTSP Evolution completed!")
        print(f"Runtime: {runtime/60:.1f} minutes")
        print(f"Best operator objective: {population[0]['objective']:.2f}")
        print(f"Best operator gap: {population[0]['gap']:.2f}%")
        print(f"Results saved to: {self.output_dir}")
        
        return population 