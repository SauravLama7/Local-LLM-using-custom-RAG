import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(prompt, model = "llama3.2:1b"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                yield data["response"]