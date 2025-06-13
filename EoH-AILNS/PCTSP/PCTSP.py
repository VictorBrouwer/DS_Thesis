import numpy as np
import pickle
import time
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy.random as rnd

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import AlphaUCB
from alns.stop import MaxIterations, MaxRuntime

# Global variables
SEED = 2345
DATA = None

@dataclass
class PCTSPData:
    """Data class for a PCTSP instance"""
    instance_id: int
    size: int  # Number of nodes (not including depot)
    depot: np.ndarray  # Depot coordinates [x, y]
    locations: np.ndarray  # Node coordinates [n, 2]
    penalties: np.ndarray  # Penalties for skipping nodes
    prizes: np.ndarray  # Prizes for visiting nodes
    
    @property
    def total_prize(self):
        """The minimum required prize to collect"""
        return 1.0  # As per the problem definition (normalized)
    
    @classmethod
    def from_dict(cls, data_dict):
        """Create a PCTSPData object from a dictionary"""
        return cls(
            instance_id=data_dict['instance_id'],
            size=data_dict['size'],
            depot=data_dict['depot'],
            locations=data_dict['locations'],
            penalties=data_dict['penalties'],
            prizes=data_dict['deterministic_prize']
        )

class PCTSPSolution:
    """Represents a solution to the PCTSP problem"""
    def __init__(self, 
                 tour: List[int], 
                 unvisited: Optional[List[int]] = None):
        """
        Initialize a PCTSP solution
        
        Args:
            tour: List of node indices in the tour (does not include depot)
            unvisited: List of unvisited nodes (if None, will be determined based on the tour)
        """
        self.tour = tour
        if unvisited is None:
            self.unvisited = self._determine_unvisited()
        else:
            self.unvisited = unvisited

    def _determine_unvisited(self) -> List[int]:
        """Determine which nodes are unvisited based on the tour"""
        n = DATA.size
        unvisited = [i for i in range(n) if i not in self.tour]
        return unvisited
    
    def objective(self) -> float:
        """
        Calculate the objective value of the PCTSP solution
        
        Returns:
            Total cost = tour length + penalties for unvisited nodes
        """
        # Initialize tour with depot
        full_tour = [self.tour[i] for i in range(len(self.tour))]
        
        # Calculate tour length
        if not full_tour:
            # If no nodes are visited, the tour length is 0
            tour_length = 0
        else:
            # Distance from depot to first node
            tour_length = np.linalg.norm(DATA.depot - DATA.locations[full_tour[0]])
            
            # Distances between consecutive nodes
            for i in range(len(full_tour) - 1):
                tour_length += np.linalg.norm(
                    DATA.locations[full_tour[i]] - DATA.locations[full_tour[i+1]]
                )
            
            # Distance from last node back to depot
            tour_length += np.linalg.norm(DATA.locations[full_tour[-1]] - DATA.depot)
        
        # Calculate penalties for unvisited nodes
        penalty_cost = sum(DATA.penalties[i] for i in self.unvisited)
        
        return tour_length + penalty_cost
    
    def total_prize(self) -> float:
        """Calculate the total prize collected in the tour"""
        return sum(DATA.prizes[i] for i in self.tour)
    
    def is_feasible(self) -> bool:
        """Check if the solution satisfies the total prize constraint"""
        return self.total_prize() >= DATA.total_prize
    
    def insert(self, node: int, idx: int):
        """Insert a node at the specified index in the tour"""
        self.tour.insert(idx, node)
        if node in self.unvisited:
            self.unvisited.remove(node)
    
    def remove(self, node: int):
        """Remove a node from the tour"""
        if node in self.tour:
            self.tour.remove(node)
            self.unvisited.append(node)
    
    def opt_insert(self, node: int):
        """Insert a node at the best position in the tour"""
        if not self.tour:
            self.insert(node, 0)
            return
        
        best_cost = float('inf')
        best_pos = 0
        
        # Try inserting at each position
        for pos in range(len(self.tour) + 1):
            # Create temporary tour
            temp_tour = self.tour.copy()
            temp_tour.insert(pos, node)
            
            # Calculate cost of this tour
            cost = self._calculate_tour_cost(temp_tour)
            
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        self.insert(node, best_pos)
    
    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate the cost of a given tour"""
        if not tour:
            return 0
        
        # Distance from depot to first node
        cost = np.linalg.norm(DATA.depot - DATA.locations[tour[0]])
        
        # Distances between consecutive nodes
        for i in range(len(tour) - 1):
            cost += np.linalg.norm(DATA.locations[tour[i]] - DATA.locations[tour[i+1]])
        
        # Distance from last node back to depot
        cost += np.linalg.norm(DATA.locations[tour[-1]] - DATA.depot)
        
        return cost

def construct_initial_solution(use_greedy: bool = True) -> PCTSPSolution:
    """
    Construct an initial solution for the PCTSP
    
    Args:
        use_greedy: If True, use a greedy construction heuristic,
                    otherwise construct a random solution
    
    Returns:
        An initial PCTSP solution
    """
    n = DATA.size
    
    if use_greedy:
        # Calculate prize-to-penalty ratio for each node
        ratios = []
        for i in range(n):
            # Calculate distance from depot
            dist = np.linalg.norm(DATA.depot - DATA.locations[i])
            # Ratio of prize to (penalty + distance)
            ratio = DATA.prizes[i] / (DATA.penalties[i] + dist + 1e-6)  # Add small epsilon to avoid division by zero
            ratios.append((i, ratio))
        
        # Sort nodes by decreasing ratio
        ratios.sort(key=lambda x: x[1], reverse=True)
        
        # Start with an empty tour
        tour = []
        current_prize = 0
        
        # Add nodes until prize constraint is satisfied
        for i, _ in ratios:
            tour.append(i)
            current_prize += DATA.prizes[i]
            if current_prize >= DATA.total_prize:
                break
        
        # Reorder tour using nearest neighbor
        if tour:
            reordered_tour = [tour[0]]
            remaining = tour[1:]
            
            while remaining:
                current = reordered_tour[-1]
                dists = []
                for i, node in enumerate(remaining):
                    dist = np.linalg.norm(DATA.locations[current] - DATA.locations[node])
                    dists.append((i, dist))
                
                closest_idx, _ = min(dists, key=lambda x: x[1])
                reordered_tour.append(remaining[closest_idx])
                del remaining[closest_idx]
            
            tour = reordered_tour
    else:
        # Randomly select nodes until prize constraint is satisfied
        available = list(range(n))
        np.random.shuffle(available)
        
        tour = []
        current_prize = 0
        
        for i in available:
            tour.append(i)
            current_prize += DATA.prizes[i]
            if current_prize >= DATA.total_prize:
                break
    
    # Create solution and ensure it's feasible
    solution = PCTSPSolution(tour)
    
    # If not feasible, add more nodes
    if not solution.is_feasible():
        available = solution.unvisited.copy()
        np.random.shuffle(available)
        
        while not solution.is_feasible() and available:
            node = available.pop(0)
            solution.insert(node, len(solution.tour))
    
    return solution

# Destroy operators
def random_removal(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:
    """Randomly remove nodes from the tour"""
    if not state.tour:
        return state
    
    # Remove 20-40% of nodes
    n_remove = max(1, min(len(state.tour), rng.integers(1, max(2, len(state.tour) // 2))))
    
    for _ in range(n_remove):
        if state.tour:
            node = rng.choice(state.tour)
            state.remove(node)
    
    return state

def adjacent_removal(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:
    """Remove a sequence of 3 adjacent nodes from the tour"""
    if not state.tour:
        return state
    
    # Fixed number of nodes to remove
    n_remove = 3
    
    # If tour is shorter than 3 nodes, remove all nodes
    if len(state.tour) <= n_remove:
        state.tour.clear()
        state.unvisited = list(range(DATA.size))
        return state
    
    # Select a random starting position
    start_pos = rng.integers(0, len(state.tour) - n_remove + 1)
    
    # Remove the sequence of adjacent nodes
    for _ in range(n_remove):
        if start_pos < len(state.tour):
            node = state.tour[start_pos]
            state.remove(node)
    
    return state

# Repair operators
def greedy_repair(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:
    """Greedy repair based on prize-to-penalty ratio"""
    # Sort unvisited nodes by prize-to-penalty ratio
    if not state.unvisited:
        return state
    
    ratios = []
    for node in state.unvisited:
        ratio = DATA.prizes[node] / (DATA.penalties[node] + 1e-6)
        ratios.append((node, ratio))
    
    # Sort by decreasing ratio (best first)
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    # Insert nodes until feasible or all inserted
    for node, _ in ratios:
        if node in state.unvisited:
            state.opt_insert(node)
            if state.is_feasible():
                break
    
    # If still not feasible, insert remaining nodes
    while state.unvisited and not state.is_feasible():
        node = state.unvisited[0]
        state.opt_insert(node)
    
    return state

def evaluate_operator(operator_func, initial_solution: PCTSPSolution, data_file: str) -> Dict[str, Any]:
    """
    Evaluate a repair operator using ALNS
    
    Args:
        operator_func: The repair operator function to evaluate
        initial_solution: Initial solution for the problem
        data_file: Path to the data file (for reference)
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Setup ALNS
        alns = ALNS(rnd.default_rng(SEED))
        
        # Add destroy operators
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(adjacent_removal)
        
        # Add the repair operator to evaluate
        alns.add_repair_operator(operator_func)
        
        # Configure ALNS
        select = AlphaUCB(
            scores=[5, 2, 1, 0.5],
            alpha=0.05,
            num_destroy=2,
            num_repair=1,
        )
        
        # Use a simple acceptance criterion
        accept = SimulatedAnnealing.autofit(
            init_obj=initial_solution.objective(),
            worse=0.20,  # Accept solutions up to 20% worse
            accept_prob=0.80,  # 80% acceptance probability
            num_iters=1000,  # High number of iterations (will be limited by time)
            method='exponential'
        )
        stop = MaxRuntime(30.0)  # 30 seconds time limit
        
        # Run ALNS
        start_time = time.time()
        result = alns.iterate(deepcopy(initial_solution), select, accept, stop)
        runtime = time.time() - start_time
        
        # Calculate results
        best_solution = result.best_state
        objective = best_solution.objective()
        
        # For PCTSP, we don't have a known best value, so we use the initial solution as reference
        initial_obj = initial_solution.objective()
        gap = 100 * (objective - initial_obj) / initial_obj if initial_obj > 0 else 0
        
        return {
            'objective': objective,
            'gap': gap,
            'runtime': runtime,
            'feasible': best_solution.is_feasible(),
            'tour_length': len(best_solution.tour),
            'prize_collected': best_solution.total_prize()
        }
        
    except Exception as e:
        print(f"Error in operator evaluation: {e}")
        return {
            'objective': initial_solution.objective() * 2,
            'gap': 100.0,
            'runtime': 0.0,
            'feasible': False,
            'tour_length': 0,
            'prize_collected': 0,
            'error': str(e)
        }

def load_instances(problem_size: int = 20) -> List[PCTSPData]:
    """Load PCTSP instances from pickle file"""
    filename = f"data/pctsp_{problem_size}_20_instances.pkl"
    try:
        with open(filename, 'rb') as f:
            instances_dict = pickle.load(f)
        
        instances = []
        for instance_dict in instances_dict:
            instances.append(PCTSPData.from_dict(instance_dict))
        
        return instances
    except FileNotFoundError:
        print(f"Error: Instance file not found at {filename}")
        return []

def load_training_instances(problem_size: int = 20) -> List[PCTSPData]:
    """Load PCTSP training instances from pickle file"""
    filename = f"training_data/pctsp{problem_size}_training.pkl"
    try:
        with open(filename, 'rb') as f:
            instances_tuples = pickle.load(f)
        
        instances = []
        for i, instance_tuple in enumerate(instances_tuples):
            # Unpack the tuple: (depot, locations, penalties, prizes, deterministic_prize)
            depot, locations, penalties, prizes, deterministic_prize = instance_tuple
            
            # Create PCTSPData object
            instance = PCTSPData(
                instance_id=i + 1,  # Start from 1
                size=len(locations),
                depot=np.array(depot),
                locations=np.array(locations),
                penalties=np.array(penalties),
                prizes=np.array(deterministic_prize)  # Use deterministic_prize as the main prizes
            )
            instances.append(instance)
        
        return instances
    except FileNotFoundError:
        print(f"Error: Training instance file not found at {filename}")
        return [] 