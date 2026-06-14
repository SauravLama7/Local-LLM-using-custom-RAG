import json
import os

BASE_DIR = "memory/chats"

def get_chat_path(username: str, model_name: str) -> str:
    safe_user  = username.replace(":", "_").replace(" ", "_")
    safe_model = model_name.replace(":", "_")
    return f"{BASE_DIR}/{safe_user}_{safe_model}"

def load_chat(username: str, model_name: str) -> list:
    path = get_chat_path(username, model_name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_chat(username: str, model_name: str, messages: list):
    os.makedirs(BASE_DIR, exist_ok=True)
    path = get_chat_path(username, model_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)