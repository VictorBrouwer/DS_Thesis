from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pandas as pd
import re
import json
import os

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations, MaxRuntime
from llm_api import LLMInterface

SEED = 2345

@dataclass
class Data:
    n_jobs: int
    n_machines: int
    bkv: int  # best known value
    processing_times: np.ndarray

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as fi:
            lines = fi.readlines()

            n_jobs, n_machines, _, bkv, _ = [
                int(num) for num in lines[1].split()
            ]
            processing_times = np.genfromtxt(lines[3:], dtype=int)

            return cls(n_jobs, n_machines, bkv, processing_times)

def compute_completion_times(schedule):
    """
    Compute the completion time for each job of the passed-in schedule.
    """
    completion = np.zeros(DATA.processing_times.shape, dtype=int)

    for idx, job in enumerate(schedule):
        for machine in range(DATA.n_machines):
            prev_job = completion[machine, schedule[idx - 1]] if idx > 0 else 0
            prev_machine = completion[machine - 1, job] if machine > 0 else 0
            processing = DATA.processing_times[machine, job]

            completion[machine, job] = max(prev_job, prev_machine) + processing

    return completion

def compute_makespan(schedule):
    """
    Returns the makespan, i.e., the maximum completion time.
    """
    return compute_completion_times(schedule)[-1, schedule[-1]]

class Solution:
    def __init__(
        self, schedule: List[int], unassigned: Optional[List[int]] = None
    ):
        self.schedule = schedule
        self.unassigned = unassigned if unassigned is not None else []

    def objective(self):
        return compute_makespan(self.schedule)

    def insert(self, job: int, idx: int):
        self.schedule.insert(idx, job)

    def opt_insert(self, job: int):
        """
        Optimally insert the job in the current schedule.
        """
        idcs_costs = all_insert_cost(self.schedule, job)
        idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
        self.insert(job, idx)

    def remove(self, job: int):
        self.schedule.remove(job)

def all_insert_cost(schedule: List[int], job: int) -> List[Tuple[int, float]]:
    """
    Computes all partial makespans when inserting a job in the schedule.
    O(nm) using Taillard's acceleration. Returns a list of tuples of the
    insertion index and the resulting makespan.

    [1] Taillard, E. (1990). Some efficient heuristic methods for the
    flow shop sequencing problem. European Journal of Operational Research,
    47(1), 65-74.
    """
    k = len(schedule) + 1
    m = DATA.processing_times.shape[0]
    p = DATA.processing_times

    # Earliest completion of schedule[j] on machine i before insertion
    e = np.zeros((m + 1, k))
    for j in range(k - 1):
        for i in range(m):
            e[i, j] = max(e[i, j - 1], e[i - 1, j]) + p[i, schedule[j]]

    # Duration between starting time and final makespan
    q = np.zeros((m + 1, k))
    for j in range(k - 2, -1, -1):
        for i in range(m - 1, -1, -1):
            q[i, j] = max(q[i + 1, j], q[i, j + 1]) + p[i, schedule[j]]

    # Earliest relative completion time
    f = np.zeros((m + 1, k))
    for l in range(k):
        for i in range(m):
            f[i, l] = max(f[i - 1, l], e[i, l - 1]) + p[i, job]

    # Partial makespan; drop the last (dummy) row of q
    M = np.max(f + q, axis=0)

    return [(idx, M[idx]) for idx in np.argsort(M)]

def random_removal(state: Solution, rng, n_remove=3) -> Solution:
    """
    Randomly remove a number jobs from the solution.
    """
    destroyed = deepcopy(state)

    for job in rng.choice(DATA.n_jobs, n_remove, replace=False):
        destroyed.unassigned.append(job)
        destroyed.schedule.remove(job)

    return destroyed

def adjacent_removal(state: Solution, rng, n_remove=3) -> Solution:
    """
    Randomly remove a number adjacent jobs from the solution.
    """
    destroyed = deepcopy(state)

    start = rng.integers(DATA.n_jobs - n_remove)
    jobs_to_remove = [state.schedule[start + idx] for idx in range(n_remove)]

    for job in jobs_to_remove:
        destroyed.unassigned.append(job)
        destroyed.schedule.remove(job)

    return destroyed

def greedy_repair(state: Solution, rng, **kwargs) -> Solution:
    """
    Greedily insert the unassigned jobs back into the schedule. The jobs are
    inserted in non-decreasing order of total processing times.
    """
    state.unassigned.sort(key=lambda j: sum(DATA.processing_times[:, j]))

    while len(state.unassigned) != 0:
        job = state.unassigned.pop()  # largest total processing time first
        state.opt_insert(job)

    return state

def local_search(solution: Solution, rng, **kwargs):
    """
    Improves the current solution in-place using the insertion neighborhood.
    A random job is selected and put in the best new position. This continues
    until relocating any of the jobs does not lead to an improving move.
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

def greedy_repair_then_local_search(state: Solution, rng, **kwargs):
    """
    Greedily insert the unassigned jobs back into the schedule (using NEH
    ordering). Apply local search afterwards.
    """
    state = greedy_repair(state, rng, **kwargs)
    local_search(state, rng, **kwargs)
    return state

def get_llm_repair_operator(save_to_json=True, load_from_json=True, json_file="llm_repair_operator.json"):
    """
    Get a repair operator from the LLM API.
    
    Args:
        save_to_json: Whether to save the LLM response to a JSON file
        load_from_json: Whether to try loading from a JSON file first
        json_file: Path to the JSON file
        
    Returns:
        The repair operator function code as a string
    """
    # Try to load from JSON if requested
    if load_from_json and os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded LLM response from {json_file}")
                return data["function_code"]
        except Exception as e:
            print(f"Error loading from JSON: {e}")
    
    # Otherwise, query the LLM
    llm = LLMInterface(debug_mode=True)
    
    prompt = """
    Generate a repair operator for the Permutation Flow Shop Problem (PFSP) using ALNS.
    
    The function signature should be:
    
    def llm_repair(state: Solution, rng, **kwargs) -> Solution:
        # Your implementation here
        return state
    
    You can use the global DATA object that has the following structure:
    - DATA.n_jobs: number of jobs
    - DATA.n_machines: number of machines
    - DATA.processing_times: numpy array of processing times [machine, job]
    
    The Solution class has:
    - schedule: list of jobs (integers)
    - unassigned: list of jobs that need to be inserted back
    - objective(): returns the makespan
    - insert(job, idx): inserts job at index idx
    - opt_insert(job): optimally inserts job at best position
    - remove(job): removes job from schedule
    
    You can also use these global functions:
    - compute_makespan(schedule): calculates the makespan of a schedule
    
    Important: Use the provided 'rng' parameter for any random operations, NOT the random module.
    The 'rng' parameter is a numpy.random.Generator object, so use methods like rng.choice() or rng.shuffle().
    
    Your operator should:
    1. Insert the unassigned jobs back into the solution in a smart way
    2. Try to minimize the makespan
    3. Be different from the greedy_repair and greedy_repair_then_local_search operators
    4. Return the modified solution
    
    Only provide the function code, no explanations.
    """
    
    response = llm.get_response(prompt)
    
    # Extract function code from response
    function_code = response
    # Try to extract just the function if there's additional text
    function_match = re.search(r'def llm_repair\(.*?return state(\n|$)', function_code, re.DOTALL)
    if function_match:
        function_code = function_match.group(0)
    
    # Clean up the code to remove markdown backticks if present
    function_code = re.sub(r'^```python\n', '', function_code)
    function_code = re.sub(r'^```\n', '', function_code)
    function_code = re.sub(r'\n```$', '', function_code)
    
    # Fix common issues
    function_code = function_code.replace('random.shuffle', 'rng.shuffle')
    function_code = function_code.replace('random.choice', 'rng.choice')
    function_code = function_code.replace('calculate_makespan', 'compute_makespan')
    
    # Save to JSON if requested
    if save_to_json:
        data = {
            "prompt": prompt,
            "raw_response": response,
            "function_code": function_code,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
                print(f"Saved LLM response to {json_file}")
        except Exception as e:
            print(f"Error saving to JSON: {e}")
    
    print("Generated repair operator:")
    print(function_code)
    
    return function_code

def create_llm_repair_operator(json_file="llm_repair_operator.json", force_new=True):
    """
    Create a repair operator function from the LLM-generated code.
    
    Args:
        json_file: Path to the JSON file to load/save
        force_new: Whether to force generating a new operator even if a JSON file exists
        
    Returns:
        A callable function that can be used as a repair operator
    """
    try:
        function_code = get_llm_repair_operator(
            save_to_json=True,
            load_from_json=not force_new,
            json_file=json_file
        )
        
        # Create a namespace for execution
        namespace = {}
        
        # Execute the function code in the namespace
        exec(function_code, globals(), namespace)
        
        # Return the function
        return namespace["llm_repair"]
    except Exception as e:
        print(f"Error creating LLM repair operator: {e}")
        print("Falling back to greedy_repair_then_local_search")
        return greedy_repair_then_local_search

def NEH(processing_times: np.ndarray) -> Solution:
    """
    Schedules jobs in decreasing order of the total processing times.

    [1] Nawaz, M., Enscore Jr, E. E., & Ham, I. (1983). A heuristic algorithm
    for the m-machine, n-job flow-shop sequencing problem. Omega, 11(1), 91-95.
    """
    largest_first = np.argsort(processing_times.sum(axis=0)).tolist()[::-1]
    solution = Solution([largest_first.pop(0)], [])

    for job in largest_first:
        solution.opt_insert(job)

    return solution

if __name__ == "__main__":
    # Load the data
    data_file = "data/j20_m5/j20_m5_01.txt"
    DATA = Data.from_file(data_file)
    
    print(f"Problem: {DATA.n_jobs} jobs, {DATA.n_machines} machines")
    print(f"Best known value: {DATA.bkv}")
    
    # Create initial solution using NEH
    init = NEH(DATA.processing_times)
    initial_obj = init.objective()
    initial_gap = 100 * (initial_obj - DATA.bkv) / DATA.bkv
    
    print(f"Initial solution objective: {initial_obj}")
    print(f"Initial gap to BKV: {initial_gap:.2f}%")
    
    # Run with LLM-generated repair operator
    print("\n=== Running with LLM-generated repair operator ===")
    llm_repair = create_llm_repair_operator(
        json_file="llm_repair_operator.json", 
        force_new=False  # Set to True to generate a new operator
    )
    
    alns_llm = ALNS(rnd.default_rng(SEED))
    alns_llm.add_destroy_operator(random_removal)
    alns_llm.add_destroy_operator(adjacent_removal)
    alns_llm.add_repair_operator(llm_repair)
    
    select_llm = AlphaUCB(
        scores=[5, 2, 1, 0.5],
        alpha=0.05,
        num_destroy=len(alns_llm.destroy_operators),
        num_repair=len(alns_llm.repair_operators),
    )
    
    ITERS = 600
    accept_llm = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, ITERS)
    stop_llm = MaxIterations(ITERS)
    
    start_time_llm = time.time()
    result_llm = alns_llm.iterate(deepcopy(init), select_llm, accept_llm, stop_llm)
    runtime_llm = time.time() - start_time_llm
    
    final_obj_llm = result_llm.best_state.objective()
    final_gap_llm = 100 * (final_obj_llm - DATA.bkv) / DATA.bkv
    
    print(f"LLM solution objective: {final_obj_llm}")
    print(f"LLM gap to BKV: {final_gap_llm:.2f}%")
    print(f"LLM runtime: {runtime_llm:.2f} seconds")
    
    # Run with original greedy_repair_then_local_search operator
    print("\n=== Running with original repair operator ===")
    alns_orig = ALNS(rnd.default_rng(SEED))
    alns_orig.add_destroy_operator(random_removal)
    alns_orig.add_destroy_operator(adjacent_removal)
    alns_orig.add_repair_operator(greedy_repair_then_local_search)
    
    select_orig = AlphaUCB(
        scores=[5, 2, 1, 0.5],
        alpha=0.05,
        num_destroy=len(alns_orig.destroy_operators),
        num_repair=len(alns_orig.repair_operators),
    )
    
    accept_orig = SimulatedAnnealing.autofit(init.objective(), 0.05, 0.50, ITERS)
    stop_orig = MaxIterations(ITERS)
    
    start_time_orig = time.time()
    result_orig = alns_orig.iterate(deepcopy(init), select_orig, accept_orig, stop_orig)
    runtime_orig = time.time() - start_time_orig
    
    final_obj_orig = result_orig.best_state.objective()
    final_gap_orig = 100 * (final_obj_orig - DATA.bkv) / DATA.bkv
    
    print(f"Original solution objective: {final_obj_orig}")
    print(f"Original gap to BKV: {final_gap_orig:.2f}%")
    print(f"Original runtime: {runtime_orig:.2f} seconds")
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Best known value: {DATA.bkv}")
    print(f"LLM solution: {final_obj_llm} (gap: {final_gap_llm:.2f}%, time: {runtime_llm:.2f}s)")
    print(f"Original solution: {final_obj_orig} (gap: {final_gap_orig:.2f}%, time: {runtime_orig:.2f}s)")
    