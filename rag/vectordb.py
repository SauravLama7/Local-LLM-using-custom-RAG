import chromadb

_client = None
_collection = None

def get_client():
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(path="chroma_db")

    return _client

def get_collection():
    global _collection

    if _collection is not None:
        return _collection
    client = get_client()

    _collection = client.get_or_create_collection("docs")

    return _collection

def reset_collection():
    client = get_client()

    try:
        client.delete_collection("docs")
    except:
        pass
    
