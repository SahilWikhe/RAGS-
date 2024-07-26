import requests
import json
import time
from typing import List, Dict

class OllamaLlamaModel:
    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "max_tokens": max_length
        }
        
        start_time = time.time()
        full_response = ""
        try:
            print(f"Sending request to Ollama API for model {self.model_name}")
            with requests.post(self.api_url, json=data, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if 'response' in json_response:
                            full_response += json_response['response']
                            print(json_response['response'], end='', flush=True)
                        if 'done' in json_response and json_response['done']:
                            break
                    if time.time() - start_time > 30:
                        print("\nResponse generation timed out after 30 seconds.")
                        break
            
            print(f"\nReceived response from Ollama API in {time.time() - start_time:.2f} seconds")
            return full_response.strip()
        except requests.exceptions.RequestException as e:
            print(f"\nRequest exception: {str(e)}")
            return f"Error: {str(e)}"