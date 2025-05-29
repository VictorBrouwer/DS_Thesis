#!/usr/bin/env python3
"""
Simple test for the simplified EoH-PFSP framework
"""

import os
import json
from eoh_pfsp import EoH_PFSP

def test_simple_eoh():
    """Test the simplified EoH framework"""
    
    print("Testing simplified EoH-PFSP framework...")
    
    # Create EoH instance with small parameters for quick testing
    eoh = EoH_PFSP(
        debug_mode=True,
        pop_size=2,  # Small population
        n_generations=2,  # Few generations
        data_file="data/j20_m5/j20_m5_01.txt",
        output_dir="test_eoh_results"
    )
    
    print("\n1. Testing initial population generation...")
    
    # Test initial population generation
    try:
        population = eoh.generate_initial_population()
        print(f"✓ Generated initial population of size {len(population)}")
        
        # Verify population structure
        for i, ind in enumerate(population):
            required_keys = ['algorithm', 'code', 'objective', 'gap', 'runtime', 'feasible', 'timestamp']
            for key in required_keys:
                assert key in ind, f"Missing key '{key}' in individual {i}"
            print(f"  Individual {i+1}: Objective={ind['objective']}, Gap={ind['gap']:.2f}%, Feasible={ind['feasible']}")
            
    except Exception as e:
        print(f"✗ Error in initial population generation: {e}")
        return False
    
    print("\n2. Testing evolution...")
    
    # Test one evolution step
    try:
        evolved_pop = eoh.evolve_population(population, 1)
        print(f"✓ Evolution completed, population size: {len(evolved_pop)}")
        
    except Exception as e:
        print(f"✗ Error in evolution: {e}")
        return False
    
    print("\n3. Testing file saving...")
    
    # Check if files were created
    try:
        gen_files = os.listdir(os.path.join("test_eoh_results", "generations"))
        best_files = os.listdir(os.path.join("test_eoh_results", "best"))
        
        assert len(gen_files) >= 2, f"Expected at least 2 generation files, got {len(gen_files)}"
        assert len(best_files) >= 2, f"Expected at least 2 best files, got {len(best_files)}"
        
        print(f"✓ Created {len(gen_files)} generation files and {len(best_files)} best files")
        
        # Test loading a generation file
        with open(os.path.join("test_eoh_results", "generations", "generation_0.json"), 'r') as f:
            data = json.load(f)
            assert "generation" in data
            assert "population" in data
            assert len(data["population"]) == len(population)
            print("✓ Generation file format is correct")
            
    except Exception as e:
        print(f"✗ Error in file operations: {e}")
        return False
    
    print("\n4. Testing operator creation...")
    
    # Test creating an operator directly
    try:
        test_code = """def llm_repair(state, rng, **kwargs):
    # Simple test operator
    state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))
    while len(state.unassigned) != 0:
        job = state.unassigned.pop()
        state.opt_insert(job)
    return state"""
        
        individual = eoh.create_operator_from_code(test_code, "Test operator")
        assert individual['feasible'], "Test operator should be feasible"
        assert individual['objective'] is not None, "Test operator should have objective value"
        print(f"✓ Created test operator: Objective={individual['objective']}, Gap={individual['gap']:.2f}%")
        
    except Exception as e:
        print(f"✗ Error in operator creation: {e}")
        return False
    
    print("\n✓ All tests passed! Simplified EoH framework is working correctly.")
    return True

if __name__ == "__main__":
    success = test_simple_eoh()
    if success:
        print("\n🎉 Simplified EoH-PFSP framework test successful!")
    else:
        print("\n❌ Test failed.")
        exit(1) 