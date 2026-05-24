import requests
import json
import base64
from pathlib import Path

# Ollama server API
OLLAMA_URL = "http://localhost:11434/api/generate"

# Model which supports vison
VISION_MODELS = {"ministral-3:3b"}

# Encode Image
def encode_image(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string for Ollama."""
    return base64.b64encode(image_bytes).decode("utf-8")

def generate(prompt, model="llama3.2:1b", image_bytes: bytes | None = None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    # Attach image only if model supports vision
    if image_bytes and model in VISION_MODELS:
        payload["images"] = [encode_image(image_bytes)]

    response = requests.post(OLLAMA_URL, json=payload, stream=True)

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                yield data["response"]