#!/usr/bin/env python3
"""
Simple demonstration of the EoH-PFSP framework

This script shows how easy it is to use the simplified EoH framework.
Just like the original PFSP code - simple, clear, and it works!
"""

from eoh_pfsp import EoH_PFSP

def main():
    print("🚀 EoH-PFSP Simple Demo")
    print("=" * 50)
    
    # Initialize the framework - just like the original PFSP!
    eoh = EoH_PFSP(
        debug_mode=True,
        pop_size=3,           # Small population for demo
        n_generations=2,      # Few generations for demo  
        data_file="data/j20_m5/j20_m5_01.txt",
        output_dir="demo_results"
    )
    
    print("\n🧬 Running Evolution of Heuristics...")
    
    # Run the complete evolution - that's it!
    final_population = eoh.run()
    
    print("\n📊 Final Results Summary:")
    print("-" * 30)
    
    best = final_population[0]
    print(f"Best Objective: {best['objective']}")
    print(f"Gap to BKV: {best['gap']:.2f}%")
    print(f"Feasible: {best['feasible']}")
    print(f"Runtime: {best['runtime']:.2f}s")
    
    print(f"\n🏆 Best Algorithm: {best['algorithm']}")
    
    print("\n📁 Results saved to 'demo_results/' directory")
    print("   - generation_X.json: Complete populations per generation")
    print("   - best/best_generation_X.json: Best individual per generation")
    print("   - final_summary.json: Complete run summary")
    
    print("\n✨ That's it! Simple and effective, just like the original PFSP code.")

if __name__ == "__main__":
    main() 