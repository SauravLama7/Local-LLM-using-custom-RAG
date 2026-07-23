import requests
import json
import base64
from io import BytesIO

# Ollama server API
OLLAMA_URL = "http://localhost:11434/api/generate"

# Model which supports vision
VISION_MODELS = {"ministral-3:3b"}

def convert_to_png(image_bytes: bytes) -> bytes:
    """Convert any image format to PNG for Ollama compatibility."""
    from PIL import Image
    img    = Image.open(BytesIO(image_bytes))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def encode_image(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string for Ollama."""
    # Convert to PNG first to ensure compatibility
    try:
        png_bytes = convert_to_png(image_bytes)
        print(f"🖼️ Converted to PNG — size: {len(png_bytes)} bytes")
        return base64.b64encode(png_bytes).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Image conversion failed: {e} — sending raw")
        return base64.b64encode(image_bytes).decode("utf-8")

def generate(prompt, model="qwen2.5:3b", image_bytes: bytes | None = None):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "num_ctx": 3048,
            "num_gpu": 99,
        }
    }

    # Attach image only if model supports vision
    if image_bytes and model in VISION_MODELS:
        payload["images"] = [encode_image(image_bytes)]
        print(f"🖼️ Image attached — size: {len(image_bytes)} bytes")
    else:
        print(f"⚠️ Image NOT attached — image provided: {image_bytes is not None}, vision model: {model in VISION_MODELS}")

    print(f"📤 Sending to Ollama — model: {model}")

    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    print(f"📥 Ollama status: {response.status_code}")

    token_count = 0
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "error" in data:
                print(f"❌ Ollama error: {data['error']}")
            if "response" in data:
                token_count += 1
                yield data["response"]

    print(f"📊 Total tokens yielded: {token_count}")