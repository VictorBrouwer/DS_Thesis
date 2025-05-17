"""
Visualization utilities for PFSP results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .problem import compute_completion_times, compute_makespan


def plot_gantt(schedule, data, name):
    """
    Plots a Gantt chart of the schedule for the permutation flow shop problem.
    """
    n_machines, n_jobs = data.processing_times.shape

    completion = compute_completion_times(schedule, data)
    start = completion - data.processing_times

    # Plot each job using its start and completion time
    cmap = plt.colormaps["rainbow"].resampled(n_jobs)
    machines, length, start_job, job_colors = zip(
        *[
            (i, data.processing_times[i, j], start[i, j], cmap(j - 1))
            for i in range(n_machines)
            for j in range(n_jobs)
        ]
    )

    _, ax = plt.subplots(1, figsize=(12, 6))
    ax.barh(machines, length, left=start_job, color=job_colors)

    ax.set_title(f"{name}\n Makespan: {compute_makespan(schedule, data)}")
    ax.set_ylabel(f"Machine")
    ax.set_xlabel(f"Completion time")
    ax.set_yticks(range(data.n_machines))
    ax.set_yticklabels(range(1, data.n_machines + 1))
    ax.invert_yaxis()

    plt.show()


def plot_comparison_bar(summary_df, metric='avg_gap', title=None, ylabel=None):
    """
    Plot a bar chart comparing repair operators based on a specified metric.
    
    Args:
        summary_df: DataFrame with summary statistics
        metric: Column name to use for comparison
        title: Plot title
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['repair_operator_name'], summary_df[metric])
    plt.ylabel(ylabel or metric)
    plt.title(title or f'Comparison of Repair Operators ({metric})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_multiple_metrics(summary_df, metrics=['avg_gap', 'avg_runtime'], 
                         titles=None, ylabels=None):
    """
    Plot multiple bar charts for different metrics.
    
    Args:
        summary_df: DataFrame with summary statistics
        metrics: List of metrics to plot
        titles: List of plot titles (same length as metrics)
        ylabels: List of y-axis labels (same length as metrics)
    """
    if titles is None:
        titles = [f'Comparison of Repair Operators ({m})' for m in metrics]
    
    if ylabels is None:
        ylabels = metrics
        
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        plt.bar(summary_df['repair_operator_name'], summary_df[metric])
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show() 