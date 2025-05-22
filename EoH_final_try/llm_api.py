import http.client
import json
from urllib.parse import urlparse


class LLMInterface:
    def __init__(self, api_endpoint=None, api_key=None, model_name=None, debug_mode=False):
        """
        Initialize the LLM interface for generating repair operators.
        
        Args:
            api_endpoint: URL for the API (full URL for Gemini, host for OpenAI-style)
            api_key: API key for authentication
            model_name: Name of the model to use
            debug_mode: Whether to print debug information
        """
        # Default values for Google Gemini API
        self.api_endpoint = api_endpoint or "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.api_key = api_key or "AIzaSyCbeYubub9_80nlO7lztX_X9L_HaBEAwoE"
        self.model_name = model_name or "gemini-2.0-flash"
        self.debug_mode = debug_mode
        self.n_trial = 5

        # Parse endpoint for different API formats
        parsed = urlparse(self.api_endpoint)
        if parsed.scheme and parsed.netloc:
            self.host = parsed.netloc
            self.path = parsed.path
        else:
            self.host = self.api_endpoint
            self.path = None

    def get_response(self, prompt_content):
        """
        Get a response from the LLM API based on the provided prompt.
        
        Args:
            prompt_content: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        # Determine payload and headers based on API type
        if "generativelanguage.googleapis.com" in self.host:
            # Gemini API format
            payload = json.dumps({
                "contents": [
                    {"parts": [{"text": prompt_content}]}
                ]
            })
            path = self.path or "/v1beta/models/gemini-2.0-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }
        elif "openai" in self.host:
            # OpenAI API format
            payload = json.dumps({
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.7,
            })
            path = "/v1/chat/completions"
            headers = {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            }
        else:
            # Generic format for other APIs
            payload = json.dumps({
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
            })
            path = "/v1/chat/completions"
            headers = {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            }

        # Try to get a response multiple times
        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                if self.debug_mode:
                    print(f"Failed to get response after {self.n_trial} attempts")
                return response
            
            try:
                conn = http.client.HTTPSConnection(self.host)
                conn.request("POST", path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                
                if self.debug_mode:
                    print(f"API Response: {json_data}")
                
                if "generativelanguage.googleapis.com" in self.host:
                    # Gemini response parsing
                    response = json_data["candidates"][0]["content"]["parts"][0]["text"]
                elif "openai" in self.host or "v1/chat/completions" in path:
                    # OpenAI-style response parsing
                    response = json_data["choices"][0]["message"]["content"]
                else:
                    # Generic fallback
                    response = str(json_data)
                
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API call ({n_trial}/{self.n_trial}): {e}")
                continue

        return response 