import os
import uuid
from pathlib import Path

from rag.embedding import embed
from rag.vectordb import get_collection, reset_collection

# Config
DATA_PATH = "data/raw"
CHUNK_SIZE = 500
OVERLAP = 100

# Text Chunking
def chunk_text(text,chunk_size = CHUNK_SIZE, overlap = OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# Load Files
def load_files(path):
    texts = []

    for file in Path(path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append((file.name,f.read()))

    return texts

# Ingest Pipeline
def ingest():
    reset_collection()
    collection = get_collection()
    files = load_files(DATA_PATH)

    all_docs = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for filename, text in files:
        chunks = chunk_text(text)
        embeddings = embed(chunks)

        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())

            all_docs.append(chunk)
            all_embeddings.append(embeddings[i])
            all_ids.append(doc_id)
            all_metadata.append({
                "source": filename,
                "chunk_id": i
            })
# Store in chromaDB    
    collection.add(
        documents = all_docs,
        embeddings = all_embeddings,
        ids = all_ids,
        metadatas = all_metadata
    )

    print(f"✅ Ingested {len(all_docs)} chunks from {len(files)} files")

# Run

if __name__ == "__main__":
    ingest()
