#!/usr/bin/env python3
"""
Demo script for EoH-PCTSP without requiring LLM API
"""

import time
import numpy as np
from PCTSP import PCTSPData, PCTSPSolution, construct_initial_solution, evaluate_operator, load_instances

def demo_pctsp_basic():
    """Demo basic PCTSP functionality"""
    
    print("🚀 EoH-PCTSP Demo: Basic Functionality")
    print("=" * 50)
    
    # Load instances
    print("Loading PCTSP instances...")
    instances = load_instances(problem_size=20)
    
    if not instances:
        print("❌ No instances found. Please check the data directory.")
        return
    
    print(f"✅ Loaded {len(instances)} instances")
    
    # Use first instance for demo
    instance = instances[0]
    print(f"\nUsing instance {instance.instance_id}:")
    print(f"  Size: {instance.size} nodes")
    print(f"  Required prize: {instance.total_prize}")
    print(f"  Depot: {instance.depot}")
    
    # Set global DATA
    import PCTSP
    PCTSP.DATA = instance
    
    # Create initial solution
    print("\nCreating initial solution...")
    initial_solution = construct_initial_solution(use_greedy=True)
    
    print(f"Initial solution:")
    print(f"  Tour: {initial_solution.tour}")
    print(f"  Tour length: {len(initial_solution.tour)} nodes")
    print(f"  Unvisited: {len(initial_solution.unvisited)} nodes")
    print(f"  Objective: {initial_solution.objective():.2f}")
    print(f"  Prize collected: {initial_solution.total_prize():.2f}")
    print(f"  Feasible: {initial_solution.is_feasible()}")
    
    return instance, initial_solution

def demo_repair_operators():
    """Demo repair operators"""
    
    print("\n🔧 Testing Repair Operators")
    print("=" * 50)
    
    instance, initial_solution = demo_pctsp_basic()
    
    # Create a damaged solution (remove some nodes)
    damaged_solution = PCTSPSolution(initial_solution.tour[:2])  # Keep only first 2 nodes
    print(f"\nDamaged solution:")
    print(f"  Tour: {damaged_solution.tour}")
    print(f"  Unvisited: {len(damaged_solution.unvisited)} nodes")
    print(f"  Objective: {damaged_solution.objective():.2f}")
    print(f"  Feasible: {damaged_solution.is_feasible()}")
    
    # Test greedy repair
    from PCTSP import greedy_repair
    import numpy.random as rnd
    
    print("\nApplying greedy repair...")
    rng = rnd.default_rng(2345)
    repaired_solution = greedy_repair(damaged_solution, rng)
    
    print(f"Repaired solution:")
    print(f"  Tour: {repaired_solution.tour}")
    print(f"  Tour length: {len(repaired_solution.tour)} nodes")
    print(f"  Unvisited: {len(repaired_solution.unvisited)} nodes")
    print(f"  Objective: {repaired_solution.objective():.2f}")
    print(f"  Prize collected: {repaired_solution.total_prize():.2f}")
    print(f"  Feasible: {repaired_solution.is_feasible()}")

def demo_mock_llm_operator():
    """Demo with a mock LLM-generated operator"""
    
    print("\n🤖 Testing Mock LLM Operator")
    print("=" * 50)
    
    instance, initial_solution = demo_pctsp_basic()
    
    # Create a mock LLM operator
    mock_operator_code = """def llm_repair(state, rng, **kwargs):
    # Mock LLM operator: distance-based greedy repair
    if not state.unvisited:
        return state
    
    # Calculate distances from depot for unvisited nodes
    distances = []
    for node in state.unvisited:
        dist = np.linalg.norm(DATA.depot - DATA.locations[node])
        distances.append((node, dist))
    
    # Sort by distance (closest first)
    distances.sort(key=lambda x: x[1])
    
    # Insert closest nodes first until feasible
    for node, _ in distances:
        if node in state.unvisited:
            state.opt_insert(node)
            if state.is_feasible():
                break
    
    # If still not feasible, insert remaining nodes by prize/penalty ratio
    if not state.is_feasible():
        ratios = []
        for node in state.unvisited:
            ratio = DATA.prizes[node] / (DATA.penalties[node] + 1e-6)
            ratios.append((node, ratio))
        
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        for node, _ in ratios:
            if node in state.unvisited:
                state.opt_insert(node)
                if state.is_feasible():
                    break
    
    return state"""
    
    print("Mock operator code:")
    print(mock_operator_code)
    
    # Execute the mock operator
    namespace = {}
    exec(mock_operator_code, globals(), namespace)
    mock_operator = namespace['llm_repair']
    
    # Evaluate the mock operator
    print("\nEvaluating mock operator...")
    evaluation = evaluate_operator(mock_operator, initial_solution, "mock_instance")
    
    print(f"Evaluation results:")
    print(f"  Objective: {evaluation['objective']:.2f}")
    print(f"  Gap: {evaluation['gap']:.2f}%")
    print(f"  Runtime: {evaluation['runtime']:.3f}s")
    print(f"  Feasible: {evaluation['feasible']}")
    print(f"  Tour length: {evaluation['tour_length']}")
    print(f"  Prize collected: {evaluation['prize_collected']:.2f}")

def demo_multiple_instances():
    """Demo with multiple instances"""
    
    print("\n📊 Testing Multiple Instances")
    print("=" * 50)
    
    instances = load_instances(problem_size=20)[:2]  # Use first 2 instances
    
    results = []
    
    for instance in instances:
        print(f"\nInstance {instance.instance_id}:")
        
        # Set global DATA
        import PCTSP
        PCTSP.DATA = instance
        
        # Create initial solution
        initial_solution = construct_initial_solution(use_greedy=True)
        
        print(f"  Initial objective: {initial_solution.objective():.2f}")
        print(f"  Initial feasible: {initial_solution.is_feasible()}")
        
        results.append({
            'instance_id': instance.instance_id,
            'size': instance.size,
            'initial_objective': initial_solution.objective(),
            'initial_feasible': initial_solution.is_feasible(),
            'tour_length': len(initial_solution.tour)
        })
    
    print(f"\nSummary of {len(results)} instances:")
    for result in results:
        print(f"  Instance {result['instance_id']}: "
              f"obj={result['initial_objective']:.2f}, "
              f"feasible={result['initial_feasible']}, "
              f"tour_len={result['tour_length']}")

def main():
    """Run all demos"""
    
    print("🧬 EoH-PCTSP Demo Suite")
    print("=" * 60)
    print("This demo tests the PCTSP framework without requiring LLM API")
    print()
    
    try:
        # Run demos
        demo_pctsp_basic()
        demo_repair_operators()
        demo_mock_llm_operator()
        demo_multiple_instances()
        
        print("\n✅ All demos completed successfully!")
        print("\nTo run the full EoH framework with LLM:")
        print("  python run_eoh_pctsp.py --api_endpoint YOUR_API --api_key YOUR_KEY")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 