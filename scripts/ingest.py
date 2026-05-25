from pathlib import Path
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.bm25_store import save_bm25
from rag.embedding import embed
from rag.vectordb import get_collection

# Config
DATA_PATH = "data/raw"
CHUNK_SIZE = 500
OVERLAP = 100

# PDF Reader
def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"

    return text


# Load Files
def load_files(path):
    texts = []

    for file in Path(path).glob("*"):
        print(f"Loading:{file.name}")
        
        if file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                texts.append((file.name,f.read()))

        elif file.suffix.lower() == ".pdf":
            text = read_pdf(file)
            texts.append((file.name,text))

    return texts


# Chunking(Langchain)
splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = OVERLAP
)

def chunk_text(text):
    return splitter.split_text(text)


# Ingest Pipeline
def ingest():
    collection = get_collection()
    files = load_files(DATA_PATH)

    all_docs = []
    all_embeddings = []
    all_ids = []
    all_metadata = []

    for filename, text in files:
        if not text.strip():
            print(f"Skipping empty files:{filename}")
            continue
        
        chunks = chunk_text(text)
        embeddings = embed(chunks)


        # Safety check
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Embedding mismatch in {filename}:"
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}_{i}"

            all_docs.append(chunk)
            all_embeddings.append(embeddings[i])
            all_ids.append(doc_id)
            all_metadata.append({
                "source": filename,
                "chunk_id": i
            })
# Store in chromaDB    
    collection.upsert(
        documents = all_docs,
        embeddings = all_embeddings,
        ids = all_ids,
        metadatas = all_metadata
    )

    save_bm25(all_docs)

    print(f"✅ Ingested {len(all_docs)} chunks from {len(files)} files")

# Run

if __name__ == "__main__":
    ingest()