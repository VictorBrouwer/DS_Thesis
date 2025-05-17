"""
Basic repair operators for the Permutation Flow Shop Problem.
"""

from copy import deepcopy
import numpy.random as rnd


def greedy_repair(state, rng, **kwargs):
    """
    Greedily insert the unassigned jobs back into the schedule.
    The jobs are inserted in non-decreasing order of total processing times.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters
        
    Returns:
        Repaired solution
    """
    # Sort by total processing time
    state.unassigned.sort(key=lambda j: sum(state.data.processing_times[:, j]))

    while len(state.unassigned) != 0:
        job = state.unassigned.pop()  # largest total processing time first
        state.opt_insert(job)

    return state


def regret_repair(state, rng, **kwargs):
    """
    Repair operator that uses regret-based insertion.
    Jobs are inserted based on the difference (regret) between their 
    best and second best insertion positions.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters
        
    Returns:
        Repaired solution
    """
    from ..utils.problem import all_insert_cost
    
    destroyed = deepcopy(state)
    
    # Create a list of jobs and their regret values
    jobs_regrets = []
    for job in destroyed.unassigned:
        # Get all insertion positions and their costs
        idcs_costs = all_insert_cost(destroyed.schedule, job, destroyed.data)
        
        if len(idcs_costs) >= 2:
            # Calculate regret as difference between best and second best position
            best_cost = idcs_costs[0][1]
            second_best_cost = idcs_costs[1][1]
            regret = second_best_cost - best_cost
            jobs_regrets.append((job, regret))
        else:
            # If only one position is available, use a default regret value
            jobs_regrets.append((job, 0))
    
    # Sort by regret (highest regret first)
    jobs_regrets.sort(key=lambda x: -x[1])
    
    # Insert jobs in order of regret
    for job, _ in jobs_regrets:
        destroyed.opt_insert(job)
    
    return destroyed


def _local_search(solution, rng):
    """
    Improves the current solution in-place using the insertion neighborhood.
    A random job is selected and put in the best new position. This continues
    until relocating any of the jobs does not lead to an improving move.
    
    Args:
        solution: Solution to improve
        rng: Random number generator
    """
    improved = True

    while improved:
        improved = False
        current = solution.objective()

        for job in rng.choice(
            solution.schedule, len(solution.schedule), replace=False
        ):
            solution.remove(job)
            solution.opt_insert(job)

            if solution.objective() < current:
                improved = True
                current = solution.objective()
                break


def greedy_repair_with_local_search(state, rng, **kwargs):
    """
    Greedily insert the unassigned jobs back into the schedule, then
    apply local search to improve the solution.
    
    Args:
        state: The solution state to repair
        rng: Random number generator
        **kwargs: Additional parameters
        
    Returns:
        Repaired solution
    """
    # First do greedy repair
    state = greedy_repair(state, rng, **kwargs)
    
    # Then apply local search
    _local_search(state, rng)
    
    return state 