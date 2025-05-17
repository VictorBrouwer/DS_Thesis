"""
Destroy operators for the Permutation Flow Shop Problem.
"""

from copy import deepcopy


def random_removal(state, rng, n_remove=3):
    """
    Randomly remove a number jobs from the solution.
    
    Args:
        state: Solution to destroy
        rng: Random number generator
        n_remove: Number of jobs to remove
        
    Returns:
        Destroyed solution
    """
    destroyed = deepcopy(state)

    for job in rng.choice(state.data.n_jobs, n_remove, replace=False):
        if job in destroyed.schedule:  # Safety check
            destroyed.unassigned.append(job)
            destroyed.schedule.remove(job)

    return destroyed


def adjacent_removal(state, rng, n_remove=3):
    """
    Randomly remove a number adjacent jobs from the solution.
    
    Args:
        state: Solution to destroy
        rng: Random number generator
        n_remove: Number of adjacent jobs to remove
        
    Returns:
        Destroyed solution
    """
    destroyed = deepcopy(state)

    if len(state.schedule) <= n_remove:
        # Not enough jobs in the schedule, fall back to random removal
        for job in destroyed.schedule[:]:
            destroyed.unassigned.append(job)
            destroyed.schedule.remove(job)
        return destroyed

    start = rng.integers(len(state.schedule) - n_remove)
    jobs_to_remove = [state.schedule[start + idx] for idx in range(n_remove)]

    for job in jobs_to_remove:
        destroyed.unassigned.append(job)
        destroyed.schedule.remove(job)

    return destroyed


def worst_positions_removal(state, rng, n_remove=3):
    """
    Remove jobs that contribute the most to the makespan.
    
    Args:
        state: Solution to destroy
        rng: Random number generator
        n_remove: Number of jobs to remove
        
    Returns:
        Destroyed solution
    """
    from ..utils.problem import compute_makespan
    
    destroyed = deepcopy(state)
    
    # Calculate the contribution of each job to the makespan
    contributions = []
    for idx, job in enumerate(state.schedule):
        # Measure objective improvement when job is removed
        temp_schedule = state.schedule.copy()
        temp_schedule.remove(job)
        
        # Skip if removing would make schedule empty
        if not temp_schedule:
            continue
            
        original_makespan = compute_makespan(state.schedule, state.data)
        new_makespan = compute_makespan(temp_schedule, state.data)
        
        improvement = original_makespan - new_makespan
        contributions.append((job, improvement))
    
    # Sort jobs by contribution (highest improvement first)
    contributions.sort(key=lambda x: -x[1])
    
    # Remove the top n_remove jobs
    for job, _ in contributions[:n_remove]:
        destroyed.unassigned.append(job)
        destroyed.schedule.remove(job)
    
    return destroyed 