#!/usr/bin/env python3
"""
Run extensive EoH with custom parameters on multiple problem sizes
"""

import time
import os
import json
import glob
from eoh_pfsp import EoH_PFSP
from PFSP import evaluate_operator_on_instances, Data, NEH
from copy import deepcopy

class CustomEoH_PFSP(EoH_PFSP):
    """EoH_PFSP with custom strategy selection (excluding m2) and custom evaluation on training data"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set up training instances based on problem size
        self.training_instances = self._get_training_instances()
        print(f"Training instances for evaluation: {self.training_instances}")
    
    def _get_training_instances(self):
        """Get 2 training instances based on the problem size inferred from data_file"""
        # Infer problem size from data_file path
        if "j20_m10" in self.data_file or "j20_m5" in self.data_file:
            problem_dir = "training_data/j20_m10"
        elif "j50_m20" in self.data_file:
            problem_dir = "training_data/j50_m20"
        elif "j100_m10" in self.data_file or "j100_m20" in self.data_file:
            problem_dir = "training_data/j100_m10"
        else:
            # Default to small instances if unsure
            problem_dir = "training_data/j20_m10"
            print(f"Warning: Could not infer problem size from {self.data_file}, using default: {problem_dir}")
        
        # Get all training instances from the directory
        training_files = sorted(glob.glob(f"{problem_dir}/*.txt"))
        
        if len(training_files) >= 2:
            return training_files[:2]  # Use first 2 instances
        elif len(training_files) == 1:
            # Use the same instance twice if only one available
            return training_files + training_files
        else:
            # Fallback to default if no training instances found
            print(f"Warning: No training instances found in {problem_dir}, falling back to test data")
            return [self.data_file]
    
    def create_operator_from_code(self, code_str: str, algorithm_desc: str):
        """Create an operator dictionary from LLM-generated code with custom training evaluation"""
        
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
            'feasible': False,
            'training_instances': self.training_instances
        }
        
        # Try to create and evaluate the operator on training instances
        try:
            # Create namespace with all necessary imports and globals
            from PFSP import Solution, compute_makespan, compute_completion_times, NEH, DATA
            import numpy as np
            
            print(f"  Setting up execution environment...")
            
            # Set up execution namespace with all necessary components
            exec_globals = globals().copy()
            exec_globals.update({
                'Solution': Solution,
                'compute_makespan': compute_makespan,
                'compute_completion_times': compute_completion_times,
                'NEH': NEH,
                'DATA': DATA,
                'np': np,
                'numpy': np,
                'math': __import__('math'),
                'heapq': __import__('heapq'),
                'random': __import__('random'),
                'collections': __import__('collections'),
                'itertools': __import__('itertools'),
                'operator': __import__('operator')
            })
            
            print(f"  Executing generated code...")
            namespace = {}
            exec(function_code, exec_globals, namespace)
            
            if 'llm_repair' in namespace:
                operator_func = namespace['llm_repair']
                print(f"  Successfully created llm_repair function, evaluating...")
                
                # Evaluate on training instances (2 instances, 30 seconds each)
                evaluation = self._evaluate_on_training_instances(operator_func)
                
                # Use sum of both instances for fitness (not average)
                individual.update({
                    'objective': int(evaluation['total_objective']),  # Sum of both instances (fitness)
                    'runtime': float(evaluation['total_runtime']),
                    'feasible': True,
                    'training_evaluation': evaluation
                })
                
                print(f"Created feasible operator - Total Objective: {evaluation['total_objective']:.0f} (fitness)")
                
            else:
                print("Error: 'llm_repair' function not found in generated code")
                available_funcs = [name for name in namespace.keys() if callable(namespace[name])]
                print(f"Available functions: {available_funcs}")
                
        except Exception as e:
            print(f"Error creating operator: {e}")
            print(f"Code being executed:\n{function_code[:200]}...")
            # Set default poor performance for infeasible operators
            # Use a high objective value (worse fitness)
            individual.update({
                'objective': 999999,  # High objective = poor fitness
                'runtime': 0.0,
                'feasible': False,
                'error': str(e)
            })
            
        return individual
    
    def _evaluate_on_training_instances(self, operator_func):
        """Evaluate operator on training instances (2 instances, 30 seconds each)"""
        from PFSP import evaluate_operator
        import PFSP
        
        total_objective = 0
        total_runtime = 0
        results = []
        
        for instance_path in self.training_instances:
            # Load the instance and set global DATA
            data = Data.from_file(instance_path)
            
            # Set the global DATA variable for this evaluation
            PFSP.DATA = data
            global DATA
            DATA = data
            
            # Create initial solution using NEH
            init = NEH(data.processing_times)
            
            # Evaluate the operator with 30 second time limit
            result = evaluate_operator(operator_func, deepcopy(init), instance_path, time_limit=30)
            results.append(result)
            
            total_objective += result["objective"]
            total_runtime += result["runtime"]
        
        n_instances = len(self.training_instances)
        return {
            "total_objective": total_objective,
            "average_objective": total_objective / n_instances,
            "total_runtime": total_runtime,
            "average_runtime": total_runtime / n_instances,
            "instance_results": results,
            "n_instances": n_instances
        }
    
    def _get_fallback_code(self):
        """Get fallback repair operator code (simple and safe)"""
        return """def llm_repair(state, rng, **kwargs):
    # Fallback: simple greedy repair
    state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))
    while len(state.unassigned) != 0:
        job = state.unassigned.pop()
        state.opt_insert(job)
    return state"""
    
    def _clean_code(self, code_str: str) -> str:
        """Clean and fix common issues in LLM-generated code"""
        import re
        
        # Remove markdown backticks
        code_str = re.sub(r'^```python\n', '', code_str)
        code_str = re.sub(r'^```\n', '', code_str)
        code_str = re.sub(r'\n```$', '', code_str)
        
        # Fix common issues
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
    
    def generate_initial_population(self):
        """Generate initial population using i1 strategy with training evaluation"""
        
        print(f"\n=== Generating Initial Population (size: {self.pop_size}) ===")
        print(f"Evaluating on training instances: {self.training_instances}")
        population = []
        
        for i in range(self.pop_size):
            print(f"\nGenerating individual {i+1}/{self.pop_size}")
            
            try:
                code_str, algorithm_desc = self.evolution.i1()
                individual = self.create_operator_from_code(code_str, algorithm_desc)
                population.append(individual)
                
            except Exception as e:
                print(f"Error generating individual {i+1}: {e}")
                # Create a fallback individual with poor fitness
                individual = {
                    'algorithm': f"Fallback operator {i+1}",
                    'code': self._get_fallback_code(),
                    'objective': 999999,  # High objective = poor fitness
                    'runtime': 0.0,
                    'feasible': False,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'error': str(e),
                    'training_instances': self.training_instances
                }
                population.append(individual)
        
        # Sort population by objective (best first)
        population.sort(key=lambda x: x['objective'])
        
        # Save initial population
        self._save_population(population, 0)
        
        print(f"\nInitial population created:")
        for i, ind in enumerate(population):
            print(f"  {i+1}. Objective: {ind['objective']}, Feasible: {ind['feasible']}")
            
        return population
    
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
            print(f"  {i+1}. Objective: {ind['objective']}, Strategy: {ind.get('strategy', 'initial')}")
        
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
    print(f"Expected time: ~8-12 minutes (using 2 training instances, 30s each)")
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
    print(f"Best operator objective: {final_population[0]['objective']}")
    print(f"Best operator algorithm: {final_population[0]['algorithm']}")
    
    return {
        'problem_name': problem_name,
        'problem_description': problem_description,
        'data_file': data_file,
        'output_dir': output_dir,
        'runtime_minutes': runtime/60,
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
    best_overall = min(all_results, key=lambda x: x['best_objective'])
    
    print(f"\nOverall Summary:")
    print(f"  Total runtime: {total_time:.1f} minutes")
    print(f"  Best overall objective: {best_overall['best_objective']} ({best_overall['problem_name']})")
    print()
    
    # Detailed results
    print(f"Detailed Results by Problem Size:")
    print("-" * 70)
    print(f"{'Problem':<15} {'Best Obj':<10} {'Runtime':<10} {'Operators'}")
    print("-" * 70)
    
    total_operators = 0
    for result in all_results:
        feasible_ops = sum(1 for op in result['final_population'] if op.get('feasible', False))
        total_operators += len(result['final_population'])
        
        print(f"{result['problem_name']:<15} {result['best_objective']:<10} "
              f"{result['runtime_minutes']:<10.1f} {feasible_ops}/{len(result['final_population'])}")
    
    print("-" * 70)
    print(f"Total operators generated: {total_operators}")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        import numpy as np
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
    
    # Save combined summary
    combined_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_minutes': total_time,
        'total_operators': total_operators,
        'best_overall': {
            'problem': best_overall['problem_name'],
            'objective': best_overall['best_objective']
        },
        'results_by_problem': all_results,
        'output_directories': [r['output_dir'] for r in all_results]
    }
    
    # Convert all types before saving
    combined_summary = convert_types(combined_summary)
    
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
    # Note: data_file is used for initialization but actual evaluation uses training_data/
    problem_configs = [
        ("data/j20_m10/j20_m10_01.txt", "small", "Small Problems (20×10)"),
        ("data/j50_m20/j50_m20_01.txt", "medium", "Medium Problems (50×20)"),
        ("data/j100_m10/j100_m10_01.txt", "large", "Large Problems (100×10)")
    ]
    
    print("Evolution plan (operators evaluated on 2 training instances, 30s each):")
    for i, (data_file, name, desc) in enumerate(problem_configs, 1):
        print(f"  {i}. {desc}: {data_file}")
    
    print(f"\nTotal estimated time: ~25-35 minutes")
    print("Each evolution will be saved to a separate directory.")
    print("📝 Note: Operators are evaluated on training data for better generalization.")
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