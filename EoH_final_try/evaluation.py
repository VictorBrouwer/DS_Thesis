def run_benchmark(data_files, approaches, seed=SEED, iters=8000):
    """
    Benchmark different ALNS approaches on multiple problem instances.
    
    Args:
        data_files: List of paths to problem instance files
        approaches: Dictionary mapping approach names to lists of (destroy_ops, repair_ops)
        seed: Random seed for reproducibility
        iters: Number of iterations for each ALNS run
    
    Returns:
        Dictionary containing all benchmark results
    """
    results = {
        'instance_names': [],
        'instance_sizes': [],
        'problem_types': [],
        'best_known_values': [],
    }
    
    # Initialize results dictionary for each approach
    for approach_name in approaches:
        results[f'{approach_name}_objectives'] = []
        results[f'{approach_name}_gaps'] = []
        results[f'{approach_name}_times'] = []
        results[f'{approach_name}_results'] = []  # Store the ALNS result objects
    
    for data_file in data_files:
        # Extract instance name from file path
        instance_name = data_file.split('/')[-1].split('.')[0]
        problem_type = data_file.split('/')[-2]
        print(f"\nProcessing instance: {instance_name} (Type: {problem_type})")
        
        # Load data
        data = Data.from_file(data_file)
        global DATA  # Use global DATA variable for the operators
        DATA = data
        
        results['instance_names'].append(instance_name)
        results['problem_types'].append(problem_type)
        results['instance_sizes'].append(f"{data.n_jobs}x{data.n_machines}")
        results['best_known_values'].append(data.bkv)
        
        # Create initial solution using NEH
        init = NEH(data.processing_times)
        
        # Run each approach
        for approach_name, (destroy_ops, repair_ops) in approaches.items():
            print(f"  Running {approach_name}...")
            
            # Setup ALNS
            alns = ALNS(rnd.default_rng(seed))
            
            # Add destroy operators
            for destroy_op in destroy_ops:
                alns.add_destroy_operator(destroy_op)
            
            # Add repair operators
            for repair_op in repair_ops:
                alns.add_repair_operator(repair_op)
            
            # Configure ALNS parameters
            select = AlphaUCB(
                scores=[5, 2, 1, 0.5],
                alpha=0.05,
                num_destroy=len(alns.destroy_operators),
                num_repair=len(alns.repair_operators),
            )
            accept = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, iters)
            stop = MaxIterations(iters)
            
            # Add time tracking
            time_stop = MaxRuntime(3600)  # 1 hour max runtime
            
            # Run ALNS
            start_time = time.time()
            result = alns.iterate(deepcopy(init), select, accept, stop)
            runtime = time.time() - start_time
            
            # Record results
            objective = result.best_state.objective()
            gap = 100 * (objective - data.bkv) / data.bkv
            
            results[f'{approach_name}_objectives'].append(objective)
            results[f'{approach_name}_gaps'].append(gap)
            results[f'{approach_name}_times'].append(runtime)
            results[f'{approach_name}_results'].append(result)  # Store the ALNS result object
            
            print(f"    Objective: {objective}, Gap: {gap:.2f}%, Time: {runtime:.2f}s")
    
    return results

def visualize_results(results):
    """
    Create visualizations to compare the approaches.
    """
    approach_names = [name.split('_')[0] for name in results.keys() 
                     if name.endswith('_objectives')]
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Instance': results['instance_names'],
        'Problem_Type': results['problem_types'],
        'Size': results['instance_sizes'],
        'BKV': results['best_known_values']
    })
    
    for approach in approach_names:
        df[f'{approach}_Obj'] = results[f'{approach}_objectives']
        df[f'{approach}_Gap'] = results[f'{approach}_gaps']
        df[f'{approach}_Time'] = results[f'{approach}_times']
    
    # 1. Problem type average gap comparison
    plt.figure(figsize=(14, 8))
    
    # Calculate average gap by problem type
    problem_type_avg = df.groupby('Problem_Type').apply(
        lambda x: pd.Series({
            f"{approach}_AvgGap": x[f"{approach}_Gap"].mean() 
            for approach in approach_names
        })
    ).reset_index()
    
    # Prepare data for bar chart
    problem_types = problem_type_avg['Problem_Type']
    x = np.arange(len(problem_types))
    width = 0.8 / len(approach_names)
    
    for i, approach in enumerate(approach_names):
        plt.bar(x + i*width - 0.4 + width/2, problem_type_avg[f'{approach}_AvgGap'], 
                width=width, label=approach)
    
    plt.xlabel('Problem Type')
    plt.ylabel('Average Gap to BKV (%)')
    plt.title('Average Performance Gap by Problem Type')
    plt.xticks(x, problem_types, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 2. Problem type average runtime comparison
    plt.figure(figsize=(14, 8))
    
    # Calculate average runtime by problem type
    problem_type_avg_time = df.groupby('Problem_Type').apply(
        lambda x: pd.Series({
            f"{approach}_AvgTime": x[f"{approach}_Time"].mean() 
            for approach in approach_names
        })
    ).reset_index()
    
    for i, approach in enumerate(approach_names):
        plt.bar(x + i*width - 0.4 + width/2, problem_type_avg_time[f'{approach}_AvgTime'], 
                width=width, label=approach)
    
    plt.xlabel('Problem Type')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Average Runtime by Problem Type')
    plt.xticks(x, problem_types, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 3. Summary table by problem type
    summary = pd.DataFrame({
        'Problem_Type': problem_types,
    })
    
    for approach in approach_names:
        summary[f'{approach}_AvgGap'] = problem_type_avg[f'{approach}_AvgGap']
    
    display(summary)
    
    # 4. Overall summary
    overall_summary = pd.DataFrame({
        'Approach': approach_names,
        'Avg Gap (%)': [df[f'{approach}_Gap'].mean() for approach in approach_names],
        'Best Gap (%)': [df[f'{approach}_Gap'].min() for approach in approach_names],
        'Worst Gap (%)': [df[f'{approach}_Gap'].max() for approach in approach_names],
        'Avg Time (s)': [df[f'{approach}_Time'].mean() for approach in approach_names],
    })
    
    display(overall_summary)
    
    # 5. Detailed results table
    display(df)
    
    return df

if __name__ == "__main__":
    import time
    import pandas as pd
    
    # Define the approaches to compare
    approaches = {
        'Baseline': (
            [random_removal, adjacent_removal],  # destroy operators
            [greedy_repair]    # repair operators
        ),
        'Gemini': (
            [
                random_removal, adjacent_removal
            ],  # destroy operators
            [optimal_insertion_repair
            ]  # repair operators
        )
    }
    
    # List all relevant Taillard instances - 10 instances for each problem type up to j100_m10
    data_files = []
    
    # j20_m5 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j20_m5/j20_m5_{i:02d}.txt")
        
    # j20_m10 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j20_m10/j20_m10_{i:02d}.txt")
        
    # j20_m20 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j20_m20/j20_m20_{i:02d}.txt")
        
    # j50_m5 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j50_m5/j50_m5_{i:02d}.txt")
        
    # j50_m10 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j50_m10/j50_m10_{i:02d}.txt")
        
    # j50_m20 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j50_m20/j50_m20_{i:02d}.txt")
        
    # j100_m5 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j100_m5/j100_m5_{i:02d}.txt")
        
    # j100_m10 (10 instances)
    for i in range(1, 11):
        data_files.append(f"data/j100_m10/j100_m10_{i:02d}.txt")
    
    # Run the benchmark
    results = run_benchmark(data_files, approaches, seed=SEED, iters=600)
    
    # Visualize and analyze the results
    results_df = visualize_results(results)
    
    # Save results to CSV
    results_df.to_csv('alns_operator_selection_results.csv', index=False)