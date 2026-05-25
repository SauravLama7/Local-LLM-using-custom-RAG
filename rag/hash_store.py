import hashlib
import json
import os

HASH_PATH = "chroma_db/file_hashes.json"

# Hashing files
def hash_file(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Loading hashes
def load_hashes() -> dict:
    """Load saved hashes from disk."""
    if not os.path.exists(HASH_PATH):
        return{}
    with open(HASH_PATH, "r") as f:
        return json.load(f)
    
# Save Hashes
def save_hashes(hashes: dict):
    """Persist hashes to disk."""
    os.makedirs(os.path.dirname(HASH_PATH), exist_ok= True)
    with open(HASH_PATH, "w") as f:
        json.dump(hashes, f, indent = 2)

# for changed files
def get_changed_files(filepaths: list[str]) -> tuple[list[str],dict]:
    """
    Compare current file hashes against saved ones.
    Returns (changed_files, updated_hash_dict)
    """
    saved = load_hashes()
    updated = saved.copy()
    changed = []

    for filepath in filepaths:
        current_hash = hash_file(filepath)
        if saved.get(filepath) != current_hash:
            changed.append(filepath)
            updated[filepath] =  current_hash
        else:
            print(f"  ⏭️  Unchanged, skipping: {os.path.basename(filepath)}")
    
    return changed, updated
