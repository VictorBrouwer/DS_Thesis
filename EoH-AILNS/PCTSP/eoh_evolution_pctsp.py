import re
import sys
import os
import signal
from functools import wraps
import time

# Add parent directory and PFSP directory to path to import interface_llm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'PFSP'))
from interface_llm import InterfaceLLM

def timeout(seconds):
    """Timeout decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            
            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class Evolution:
    """Evolution engine for PCTSP repair operators"""
    
    def __init__(self, api_endpoint=None, api_key=None, model_LLM=None, 
                 llm_use_local=False, llm_local_url=None, debug_mode=False, prompts=None):
        
        self.prompts = prompts
        self.debug_mode = debug_mode
        
        # Initialize LLM interface
        self.llm = InterfaceLLM(
            api_endpoint = api_endpoint or "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            api_key = api_key or "AIzaSyCbeYubub9_80nlO7lztX_X9L_HaBEAwoE",
            model_LLM=model_LLM or "gemini-2.0-flash",
            llm_use_local=llm_use_local,
            llm_local_url=llm_local_url,
            debug_mode=debug_mode
        )
        
    def _extract_code(self, response: str) -> str:
        """Extract function code from LLM response"""
        # Look for function definition
        function_match = re.search(r'def llm_repair\(.*?return state\s*(?:\n|$)', response, re.DOTALL)
        if function_match:
            return function_match.group(0).strip()
        
        # If no function found, clean up the response
        code = response.strip()
        
        # Remove markdown backticks
        code = re.sub(r'^```python\n', '', code)
        code = re.sub(r'^```\n', '', code) 
        code = re.sub(r'\n```$', '', code)
        
        return code
    
    @timeout(30)  # 30 second timeout
    def _get_llm_response(self, prompt):
        """Get response from LLM with timeout"""
        return self.llm.get_response(prompt)
    
    def i1(self):
        """Strategy i1: Generate initial repair operator from scratch"""
        
        prompt = self.prompts.get_task_description()
        
        print("Generating initial PCTSP repair operator...")
        if self.debug_mode:
            print(f"Prompt: {prompt}")
            
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = "Initial PCTSP repair operator generated from scratch"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            # Return a simple fallback operator
            fallback_code = """
def llm_repair(state, rng):
    # Simple fallback operator that removes and reinserts nodes randomly
    removed = []
    for _ in range(3):  # Remove 3 nodes
        if len(state.route) > 3:
            idx = rng.integers(0, len(state.route))
            removed.append(state.route.pop(idx))
    
    # Reinsert nodes at random positions
    for node in removed:
        if len(state.route) > 0:
            idx = rng.integers(0, len(state.route) + 1)
            state.route.insert(idx, node)
        else:
            state.route.append(node)
    
    return state
"""
            return fallback_code, "Fallback operator (timeout)"
    
    def e1(self, parents):
        """Strategy e1: Create totally different algorithms inspired by parents"""
        
        parent_codes = [p['code'] for p in parents[:2]]
        parent_algorithms = [p['algorithm'] for p in parents[:2]]
        
        prompt = f"""
        {self.prompts.get_task_description()}
        
        Create a completely different repair operator from these existing ones:
        
        Parent 1: {parent_algorithms[0]}
        {parent_codes[0]}
        
        Parent 2: {parent_algorithms[1]}  
        {parent_codes[1]}
        
        Generate a NEW PCTSP repair operator that uses completely different logic and approaches.
        Consider different strategies like:
        - Different node selection criteria (distance-based, prize-based, penalty-based)
        - Different insertion strategies (greedy, random, position-based)
        - Different feasibility handling approaches
        
        Only provide the function code, no explanations.
        """
        
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = "New PCTSP algorithm inspired by existing approaches but with different logic"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            return self._get_fallback_operator(), "Fallback operator (timeout)"
        
    def e2(self, parents):
        """Strategy e2: Create algorithms motivated by existing ones"""
        
        parent_codes = [p['code'] for p in parents[:2]]
        parent_algorithms = [p['algorithm'] for p in parents[:2]]
        
        prompt = f"""
        {self.prompts.get_task_description()}
        
        Combine and improve these PCTSP repair operators:
        
        Parent 1: {parent_algorithms[0]}
        {parent_codes[0]}
        
        Parent 2: {parent_algorithms[1]}
        {parent_codes[1]}
        
        Create a new PCTSP operator that combines the best ideas from both parents.
        Consider combining:
        - Node selection strategies from both parents
        - Insertion methods from both parents
        - Prize/penalty evaluation approaches
        
        Only provide the function code, no explanations.
        """
        
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = "Hybrid PCTSP algorithm combining ideas from parent operators"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            return self._get_fallback_operator(), "Fallback operator (timeout)"
    
    def m1(self, parent):
        """Strategy m1: Modify existing algorithm"""
        
        prompt = f"""
        {self.prompts.get_task_description()}
        
        Modify this existing PCTSP repair operator to improve its performance:
        
        Current operator: {parent['algorithm']}
        {parent['code']}
        
        Make significant modifications to improve the algorithm while keeping the core structure.
        Consider improvements like:
        - Better node selection criteria
        - Improved insertion strategies
        - Enhanced prize/penalty trade-off handling
        - More efficient feasibility checking
        
        Only provide the function code, no explanations.
        """
        
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = f"Modified version of: {parent['algorithm']}"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            return self._get_fallback_operator(), "Fallback operator (timeout)"
    
    def m2(self, parent):
        """Strategy m2: Change parameters/constants"""
        
        prompt = f"""
        {self.prompts.get_task_description()}
        
        Modify the parameters and constants in this PCTSP repair operator:
        
        Current operator: {parent['algorithm']}
        {parent['code']}
        
        Keep the same logic but change numerical parameters, thresholds, or constants to improve performance.
        Consider adjusting:
        - Prize/penalty ratio thresholds
        - Distance calculation weights
        - Selection probabilities
        - Insertion position preferences
        
        Only provide the function code, no explanations.
        """
        
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = f"Parameter-tuned version of: {parent['algorithm']}"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            return self._get_fallback_operator(), "Fallback operator (timeout)"
    
    def m3(self, parent):
        """Strategy m3: Simplify for generalization"""
        
        prompt = f"""
        {self.prompts.get_task_description()}
        
        Simplify this PCTSP repair operator to make it more general and robust:
        
        Current operator: {parent['algorithm']}
        {parent['code']}
        
        Remove complex logic and make it simpler while maintaining effectiveness.
        Focus on:
        - Simpler node selection rules
        - Straightforward insertion strategies
        - Basic but robust prize/penalty handling
        
        Only provide the function code, no explanations.
        """
        
        try:
            response = self._get_llm_response(prompt)
            code = self._extract_code(response)
            algorithm_desc = f"Simplified version of: {parent['algorithm']}"
            return code, algorithm_desc
        except TimeoutError:
            print("LLM response timed out after 30 seconds. Using fallback operator.")
            return self._get_fallback_operator(), "Fallback operator (timeout)"
    
    def _get_fallback_operator(self):
        """Return a simple fallback operator when LLM times out"""
        return """
def llm_repair(state, rng):
    # Simple fallback operator that removes and reinserts nodes randomly
    removed = []
    for _ in range(3):  # Remove 3 nodes
        if len(state.route) > 3:
            idx = rng.integers(0, len(state.route))
            removed.append(state.route.pop(idx))
    
    # Reinsert nodes at random positions
    for node in removed:
        if len(state.route) > 0:
            idx = rng.integers(0, len(state.route) + 1)
            state.route.insert(idx, node)
        else:
            state.route.append(node)
    
    return state
""" 