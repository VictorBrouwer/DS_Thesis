from llm_api import LLMInterface

class InterfaceLLM:
    """Wrapper for LLMInterface to be compatible with EoH Evolution class"""
    
    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode):
        """
        Initialize the interface compatible with Evolution class expectations.
        
        Args:
            api_endpoint: API endpoint URL
            api_key: API key
            model_LLM: Model name
            llm_use_local: Whether to use local LLM (not used in our implementation)
            llm_local_url: Local LLM URL (not used in our implementation)
            debug_mode: Debug mode flag
        """
        self.llm_interface = LLMInterface(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model_name=model_LLM,
            debug_mode=debug_mode
        )
        
    def get_response(self, prompt_content):
        """
        Get response from LLM.
        
        Args:
            prompt_content: The prompt to send to the LLM
            
        Returns:
            LLM response as string
        """
        return self.llm_interface.get_response(prompt_content) 