class GetPFSPPrompts:
    """PFSP-specific prompts for Evolution of Heuristics"""
    
    def __init__(self):
        self.task_description = "Permutation Flow Shop Problem (PFSP) repair operator"
        self.function_signature = "def llm_repair(state: Solution, rng, **kwargs) -> Solution:"
        
    def get_task_description(self):
        return """
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
        3. Return the modified solution
        
        Only provide the function code, no explanations.
        """
    
    def get_function_inputs(self):
        return [
            "state: Solution object with schedule and unassigned jobs",
            "rng: numpy random generator for random operations", 
            "**kwargs: additional keyword arguments"
        ]
    
    def get_function_outputs(self):
        return [
            "state: Modified Solution object with all jobs inserted into schedule"
        ]

if __name__ == "__main__":
    prompts = GetPFSPPrompts()
    print(prompts.get_task_description()) 