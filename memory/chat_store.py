import json
import os

BASE_DIR = "memory/chats"

def get_chat_path(model_name):
    safe_name = model_name.replace(":","_")
    return f"{BASE_DIR}/{safe_name}"

def load_chat(model_name):

    path = get_chat_path(model_name)

    if not os.path.exists(path):
        return[]
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_chat(model_name, message):
    os.makedirs(BASE_DIR,exist_ok=True)

    path = get_chat_path(model_name)

    with open(path, "w", encoding = "utf-8") as f:
        json.dump(message, f, indent=2, ensure_ascii=False)