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

def delete_by_source(filename: str):
    """Delete all chunks from a specific source file."""
    collection = get_collection()

    # Find all chunks from this source
    results = collection.get(
        where = {"source": filename},
        include = ["documents"]
    )

    ids = results["ids"]

    if ids:
        collection.delete(ids = ids)
        print(f"🗑️ Deleted {len(ids)} chunks from {filename}")
    else: 
        print(f"⚠️ No chunks found for {filename}")
