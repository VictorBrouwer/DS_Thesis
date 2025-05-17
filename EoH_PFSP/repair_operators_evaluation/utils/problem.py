"""
Problem representation for the Permutation Flow Shop Problem (PFSP).
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class Data:
    """
    Data class for a PFSP instance.
    """
    n_jobs: int
    n_machines: int
    bkv: int  # best known value
    processing_times: np.ndarray

    @classmethod
    def from_file(cls, path):
        """
        Load data from a file.
        """
        with open(path, "r") as fi:
            lines = fi.readlines()

            n_jobs, n_machines, _, bkv, _ = [
                int(num) for num in lines[1].split()
            ]
            processing_times = np.genfromtxt(lines[3:], dtype=int)

            return cls(n_jobs, n_machines, bkv, processing_times)


class Solution:
    """
    Solution representation for PFSP.
    """
    def __init__(
        self, schedule: List[int], unassigned: Optional[List[int]] = None, data=None
    ):
        self.schedule = schedule
        self.unassigned = unassigned if unassigned is not None else []
        self.data = data  # Store reference to problem data
        
    def objective(self):
        """
        Calculate the makespan of the solution.
        """
        return compute_makespan(self.schedule, self.data)

    def insert(self, job: int, idx: int):
        """
        Insert a job at a specific position.
        """
        self.schedule.insert(idx, job)

    def opt_insert(self, job: int):
        """
        Optimally insert the job in the current schedule.
        """
        idcs_costs = all_insert_cost(self.schedule, job, self.data)
        idx, _ = min(idcs_costs, key=lambda idx_cost: idx_cost[1])
        self.insert(job, idx)

    def remove(self, job: int):
        """
        Remove a job from the schedule.
        """
        self.schedule.remove(job)


def compute_completion_times(schedule, data):
    """
    Compute the completion time for each job of the passed-in schedule.
    """
    completion = np.zeros(data.processing_times.shape, dtype=int)

    for idx, job in enumerate(schedule):
        for machine in range(data.n_machines):
            prev_job = completion[machine, schedule[idx - 1]] if idx > 0 else 0
            prev_machine = completion[machine - 1, job] if machine > 0 else 0
            processing = data.processing_times[machine, job]

            completion[machine, job] = max(prev_job, prev_machine) + processing

    return completion


def compute_makespan(schedule, data):
    """
    Returns the makespan, i.e., the maximum completion time.
    """
    return compute_completion_times(schedule, data)[-1, schedule[-1]]


def all_insert_cost(schedule: List[int], job: int, data) -> List[Tuple[int, float]]:
    """
    Computes all partial makespans when inserting a job in the schedule.
    O(nm) using Taillard's acceleration. Returns a list of tuples of the
    insertion index and the resulting makespan.

    [1] Taillard, E. (1990). Some efficient heuristic methods for the
    flow shop sequencing problem. European Journal of Operational Research,
    47(1), 65-74.
    """
    k = len(schedule) + 1
    m = data.processing_times.shape[0]
    p = data.processing_times

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


def NEH(processing_times: np.ndarray, data) -> Solution:
    """
    Schedules jobs in decreasing order of the total processing times.

    [1] Nawaz, M., Enscore Jr, E. E., & Ham, I. (1983). A heuristic algorithm
    for the m-machine, n-job flow-shop sequencing problem. Omega, 11(1), 91-95.
    """
    largest_first = np.argsort(processing_times.sum(axis=0)).tolist()[::-1]
    solution = Solution([largest_first.pop(0)], [], data)

    for job in largest_first:
        solution.opt_insert(job)

    return solution 