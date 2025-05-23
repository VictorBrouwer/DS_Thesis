#!/usr/bin/env python3
"""
Visualize EoH evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('quick_eoh_evaluation.csv')

print("📊 EoH Evaluation Results Visualization")
print("=" * 50)

# 1. Overall performance comparison
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Average gaps by problem size
size_summary = df.groupby('n_jobs').agg({
    'eoh_gap': 'mean',
    'baseline_gap': 'mean'
}).round(2)

x = np.arange(len(size_summary.index))
width = 0.35

ax1.bar(x - width/2, size_summary['eoh_gap'], width, label='EoH', alpha=0.8)
ax1.bar(x + width/2, size_summary['baseline_gap'], width, label='Baseline', alpha=0.8)
ax1.set_xlabel('Number of Jobs')
ax1.set_ylabel('Average Gap to BKV (%)')
ax1.set_title('Average Performance by Problem Size')
ax1.set_xticks(x)
ax1.set_xticklabels(size_summary.index)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Individual instance comparison
ax2.scatter(df['baseline_gap'], df['eoh_gap'], alpha=0.7, s=60)
ax2.plot([0, df[['eoh_gap', 'baseline_gap']].max().max()], 
         [0, df[['eoh_gap', 'baseline_gap']].max().max()], 'r--', alpha=0.5)
ax2.set_xlabel('Baseline Gap (%)')
ax2.set_ylabel('EoH Gap (%)')
ax2.set_title('Instance-by-Instance Comparison')
ax2.grid(alpha=0.3)

# Add instance labels for points far from diagonal
for i, row in df.iterrows():
    if abs(row['eoh_gap'] - row['baseline_gap']) > 0.5:
        ax2.annotate(row['instance'], (row['baseline_gap'], row['eoh_gap']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

# Improvement by problem size
improvements = df.groupby('n_jobs')['improvement'].mean()
colors = ['green' if x > 0 else 'red' for x in improvements]
ax3.bar(improvements.index, improvements, color=colors, alpha=0.7)
ax3.set_xlabel('Number of Jobs')
ax3.set_ylabel('Average Improvement (percentage points)')
ax3.set_title('EoH Improvement over Baseline')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.grid(axis='y', alpha=0.3)

# Performance distribution
ax4.boxplot([df['eoh_gap'], df['baseline_gap']], labels=['EoH', 'Baseline'])
ax4.set_ylabel('Gap to BKV (%)')
ax4.set_title('Performance Distribution')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eoh_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("\n📈 Performance Analysis:")
print("-" * 30)

print(f"Overall Results:")
print(f"  EoH Average Gap: {df['eoh_gap'].mean():.2f}%")
print(f"  Baseline Average Gap: {df['baseline_gap'].mean():.2f}%") 
print(f"  Average Improvement: {df['improvement'].mean():.2f} percentage points")
print(f"  EoH Win Rate: {(df['eoh_gap'] < df['baseline_gap']).sum()}/{len(df)} ({(df['eoh_gap'] < df['baseline_gap']).mean()*100:.1f}%)")

print(f"\nBy Problem Size:")
for n_jobs, group in df.groupby('n_jobs'):
    avg_imp = group['improvement'].mean()
    win_rate = (group['eoh_gap'] < group['baseline_gap']).mean() * 100
    print(f"  {n_jobs} jobs: {avg_imp:+.2f} pp improvement, {win_rate:.0f}% win rate")

print(f"\nBest Improvements:")
best_improvements = df.nlargest(3, 'improvement')[['instance', 'improvement', 'eoh_gap', 'baseline_gap']]
for _, row in best_improvements.iterrows():
    print(f"  {row['instance']}: {row['improvement']:+.2f} pp ({row['baseline_gap']:.2f}% → {row['eoh_gap']:.2f}%)")

print(f"\nWorst Cases:")
worst_cases = df.nsmallest(3, 'improvement')[['instance', 'improvement', 'eoh_gap', 'baseline_gap']]
for _, row in worst_cases.iterrows():
    print(f"  {row['instance']}: {row['improvement']:+.2f} pp ({row['baseline_gap']:.2f}% → {row['eoh_gap']:.2f}%)")

# Summary statistics
print(f"\n📊 Summary Statistics:")
print("-" * 30)
print(f"Instances where EoH = Baseline: {(df['improvement'] == 0).sum()}")
print(f"Instances where EoH > Baseline: {(df['improvement'] > 0).sum()}")
print(f"Instances where EoH < Baseline: {(df['improvement'] < 0).sum()}")
print(f"Maximum improvement: {df['improvement'].max():.2f} pp")
print(f"Maximum degradation: {df['improvement'].min():.2f} pp")

print(f"\n✅ Results visualization saved as 'eoh_evaluation_results.png'") 