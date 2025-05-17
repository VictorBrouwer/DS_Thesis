"""
Custom repair operators for the PFSP.
This file demonstrates how to create new repair operators.
"""

from copy import deepcopy
import numpy as np


def insertion_priority_repair(state, rng, **kwargs):
    """
    Repair operator that uses a weighted combination of processing time
    and position-based priorities for insertion.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters that can include:
            - alpha: Weight for processing time priority (default: 0.6)
            - beta: Weight for position priority (default: 0.4)
        
    Returns:
        Repaired solution
    """
    # Get optional parameters with defaults
    alpha = kwargs.get('alpha', 0.6)
    beta = kwargs.get('beta', 0.4)
    
    destroyed = deepcopy(state)
    
    # Calculate priorities for each unassigned job
    job_priorities = []
    
    for job in destroyed.unassigned:
        # 1. Processing time priority - total processing time of the job
        proc_time_priority = sum(destroyed.data.processing_times[:, job])
        
        # 2. Position priority - estimate of best position (0 = first, 1 = last)
        # We'll use a simple heuristic based on average processing time per machine
        avg_time_per_machine = [
            destroyed.data.processing_times[m, job] for m in range(destroyed.data.n_machines)
        ]
        early_machines_avg = np.mean(avg_time_per_machine[:destroyed.data.n_machines//2])
        late_machines_avg = np.mean(avg_time_per_machine[destroyed.data.n_machines//2:])
        
        # If job processes faster in early machines, it should go earlier in schedule
        position_priority = early_machines_avg / (early_machines_avg + late_machines_avg)
        
        # Combined priority
        combined_priority = alpha * proc_time_priority + beta * (1 - position_priority)
        job_priorities.append((job, combined_priority))
    
    # Sort jobs by combined priority (highest first)
    job_priorities.sort(key=lambda x: -x[1])
    
    # Insert jobs in order of priority
    for job, _ in job_priorities:
        destroyed.opt_insert(job)
    
    return destroyed


def multi_phase_repair(state, rng, **kwargs):
    """
    A multi-phase repair operator that uses different strategies for 
    different subsets of the unassigned jobs.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters
        
    Returns:
        Repaired solution
    """
    from ..utils.problem import all_insert_cost
    
    destroyed = deepcopy(state)
    
    if not destroyed.unassigned:
        return destroyed
    
    # Sort unassigned jobs by total processing time
    jobs_by_proc_time = sorted(
        destroyed.unassigned, 
        key=lambda j: sum(destroyed.data.processing_times[:, j]), 
        reverse=True
    )
    
    # Phase 1: Insert a portion of longest jobs using greedy insertion
    n_greedy = max(1, len(jobs_by_proc_time) // 3)
    greedy_jobs = jobs_by_proc_time[:n_greedy]
    
    for job in greedy_jobs:
        destroyed.opt_insert(job)
        destroyed.unassigned.remove(job)
    
    # Phase 2: Insert a portion of the jobs using regret-based insertion
    remaining_jobs = len(destroyed.unassigned)
    n_regret = max(1, remaining_jobs // 2)
    
    jobs_regrets = []
    for job in destroyed.unassigned:
        idcs_costs = all_insert_cost(destroyed.schedule, job, destroyed.data)
        
        if len(idcs_costs) >= 2:
            best_cost = idcs_costs[0][1]
            second_best_cost = idcs_costs[1][1]
            regret = second_best_cost - best_cost
            jobs_regrets.append((job, regret))
        else:
            jobs_regrets.append((job, 0))
    
    jobs_regrets.sort(key=lambda x: -x[1])
    regret_jobs = [job for job, _ in jobs_regrets[:n_regret]]
    
    for job in regret_jobs:
        destroyed.opt_insert(job)
        destroyed.unassigned.remove(job)
    
    # Phase 3: Insert remaining jobs randomly
    remaining_jobs = destroyed.unassigned.copy()
    rng.shuffle(remaining_jobs)
    
    for job in remaining_jobs:
        destroyed.opt_insert(job)
        destroyed.unassigned.remove(job)
    
    return destroyed 