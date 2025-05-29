import pandas as pd
import numpy as np
import glob
import os
import re

# Define the problem types we want to analyze (up to j100_m10)
problem_types = [
    'j20_m5', 'j20_m10', 'j20_m20',
    'j50_m5', 'j50_m10', 'j50_m20',
    'j100_m5', 'j100_m10'
]

# Function to extract problem type from instance name
def extract_problem_type(instance_name):
    if isinstance(instance_name, str):
        match = re.match(r'(j\d+_m\d+)_\d+', instance_name)
        if match:
            return match.group(1)
    return None

# Load results data
# First check if we have a combined file with all approaches
try:
    all_results = pd.read_csv('git_repo/PFSP/pfsp_all_results.csv')
    print("Found combined results file.")
except:
    # Otherwise, try to merge data from separate files
    claude_results = pd.read_csv('git_repo/PFSP/alns_operator_selection_results.csv')
    
    # Check if we have comparison results with all approaches
    try:
        comparison_results = pd.read_csv('git_repo/PFSP/alns_comparison_results.csv')
        print("Found comparison results file.")
        
        # We might need to merge these datasets
        # This is a placeholder as the actual merging would depend on data structure
        # all_results = pd.merge(claude_results, comparison_results, on='Instance', how='outer')
        
        # For now, we'll just use what we have
        all_results = claude_results
    except:
        print("Only Claude results found.")
        all_results = claude_results

# Extract problem type from instance name
all_results['Problem_Type'] = all_results['Instance'].apply(extract_problem_type)

# Filter for only the problem types we want
filtered_results = all_results[all_results['Problem_Type'].isin(problem_types)]

# Calculate average gap for each approach and problem type
average_gaps = {}

for problem_type in problem_types:
    problem_data = filtered_results[filtered_results['Problem_Type'] == problem_type]
    
    if len(problem_data) == 0:
        print(f"No data found for {problem_type}")
        continue
    
    # Check which approaches we have data for
    approaches = []
    if 'Claude_Gap' in problem_data.columns:
        approaches.append('Claude')
    if 'Baseline_Gap' in problem_data.columns:
        approaches.append('Baseline')
    if 'Gemini_Gap' in problem_data.columns:
        approaches.append('Gemini')
    
    problem_averages = {}
    for approach in approaches:
        gap_column = f"{approach}_Gap"
        if gap_column in problem_data.columns:
            avg_gap = problem_data[gap_column].mean()
            problem_averages[approach] = avg_gap
    
    average_gaps[problem_type] = problem_averages

# Create a pandas DataFrame for better display
results_df = pd.DataFrame.from_dict(average_gaps, orient='index')

# Fill NaN with "No data"
results_df = results_df.fillna("No data")

# Print the results
print("\nAverage Gap to Best-Known Solutions by Problem Type:")
print(results_df)

# Save to CSV
results_df.to_csv('average_gaps_to_bkv.csv')
print("\nResults saved to average_gaps_to_bkv.csv") 