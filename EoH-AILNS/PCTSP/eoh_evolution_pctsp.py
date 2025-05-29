import re
import sys
import os

# Add parent directory to path to import interface_llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interface_llm import InterfaceLLM

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
    
    def i1(self):
        """Strategy i1: Generate initial repair operator from scratch"""
        
        prompt = self.prompts.get_task_description()
        
        print("Generating initial PCTSP repair operator...")
        if self.debug_mode:
            print(f"Prompt: {prompt}")
            
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = "Initial PCTSP repair operator generated from scratch"
        
        return code, algorithm_desc
    
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
        
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = "New PCTSP algorithm inspired by existing approaches but with different logic"
        
        return code, algorithm_desc
        
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
        
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = "Hybrid PCTSP algorithm combining ideas from parent operators"
        
        return code, algorithm_desc
    
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
        
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = f"Modified version of: {parent['algorithm']}"
        
        return code, algorithm_desc
    
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
        
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = f"Parameter-tuned version of: {parent['algorithm']}"
        
        return code, algorithm_desc
    
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
        - Reduced computational complexity
        
        Only provide the function code, no explanations.
        """
        
        response = self.llm.get_response(prompt)
        code = self._extract_code(response)
        
        algorithm_desc = f"Simplified version of: {parent['algorithm']}"
        
        return code, algorithm_desc 