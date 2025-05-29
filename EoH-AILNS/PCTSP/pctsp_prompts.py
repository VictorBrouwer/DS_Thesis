class GetPCTSPPrompts:
    """PCTSP-specific prompts for Evolution of Heuristics"""
    
    def __init__(self):
        self.task_description = "Price Collecting Travelling Salesman Problem (PCTSP) repair operator"
        self.function_signature = "def llm_repair(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:"
        
    def get_task_description(self):
        return """
        Generate a repair operator for the Price Collecting Travelling Salesman Problem (PCTSP) using ALNS.
        
        The function signature should be:
        
        def llm_repair(state: PCTSPSolution, rng, **kwargs) -> PCTSPSolution:
            # Your implementation here
            return state
        
        You can use the global DATA object that has the following structure:
        - DATA.size: number of nodes (excluding depot)
        - DATA.depot: depot coordinates [x, y]
        - DATA.locations: node coordinates [n, 2] numpy array
        - DATA.penalties: penalties for skipping nodes [n] numpy array
        - DATA.prizes: prizes for visiting nodes [n] numpy array
        - DATA.total_prize: minimum required prize to collect (usually 1.0)
        
        The PCTSPSolution class has:
        - tour: list of visited nodes (integers, does not include depot)
        - unvisited: list of nodes that need to be inserted back
        - objective(): returns the total cost (tour length + penalties for unvisited nodes)
        - total_prize(): returns the total prize collected
        - is_feasible(): checks if the solution satisfies the prize constraint
        - insert(node, idx): inserts node at index idx in the tour
        - opt_insert(node): optimally inserts node at best position in the tour
        - remove(node): removes node from the tour
        
        You can also use these functions:
        - np.linalg.norm(point1 - point2): calculates Euclidean distance between two points
        
        Important: Use the provided 'rng' parameter for any random operations, NOT the random module.
        The 'rng' parameter is a numpy.random.Generator object, so use methods like rng.choice() or rng.shuffle().
        
        Your operator should:
        1. Insert the unvisited nodes back into the solution in a smart way
        2. Ensure the solution collects enough prize (at least DATA.total_prize)
        3. Try to minimize the total cost (tour length + penalties)
        4. Consider the trade-off between visiting nodes (tour cost) vs. skipping them (penalty cost)
        5. Return the modified solution
        
        Key PCTSP considerations:
        - Nodes with high prize-to-penalty ratios are generally good to visit
        - Nodes close to the current tour are cheaper to visit
        - The solution must collect at least DATA.total_prize worth of prizes
        - Unvisited nodes incur penalty costs
        
        Only provide the function code, no explanations.
        """
    
    def get_function_inputs(self):
        return [
            "state: PCTSPSolution object with tour and unvisited nodes",
            "rng: numpy random generator for random operations", 
            "**kwargs: additional keyword arguments"
        ]
    
    def get_function_outputs(self):
        return [
            "state: Modified PCTSPSolution object with nodes inserted into tour"
        ]

if __name__ == "__main__":
    prompts = GetPCTSPPrompts()
    print(prompts.get_task_description()) 