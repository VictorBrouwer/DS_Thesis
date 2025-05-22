import http.client
import json
from urllib.parse import urlparse


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        # Accepts full URL for Gemini, or just host for OpenAI-style
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

        # Parse endpoint for Gemini
        parsed = urlparse(api_endpoint)
        if parsed.scheme and parsed.netloc:
            self.host = parsed.netloc
            self.path = parsed.path
        else:
            self.host = api_endpoint
            self.path = None

    def get_response(self, prompt_content):
        # Gemini expects a different payload and path
        if "generativelanguage.googleapis.com" in self.host:
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
        else:
            # OpenAI-style fallback
            payload = json.dumps({
                "model": self.model_LLM,
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
            })
            path = "/v1/chat/completions"
            headers = {
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                "Content-Type": "application/json",
                "x-api2d-no-cache": "1",
            }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.host)
                conn.request("POST", path, payload, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                if "generativelanguage.googleapis.com" in self.host:
                    # Gemini response parsing
                    response = json_data["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    # OpenAI-style response parsing
                    response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print("Error in API. Restarting the process...", e)
                continue

        return response