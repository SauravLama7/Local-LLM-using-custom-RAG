# 🧠 Local LLM RAG Assistant

A fully local, privacy-first document Q&A system built with Ollama, ChromaDB, Streamlit and a hybrid RAG pipeline. No data leaves your machine.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Default Credentials](#default-credentials)

---

## ✨ Features

### 🔍 Retrieval Pipeline
- **Hybrid search** — combines vector search (ChromaDB) and BM25 keyword search
- **Reciprocal Rank Fusion (RRF)** — merges results from both search methods
- **CrossEncoder reranking** — re-scores retrieved chunks for better precision
- **Parallel retrieval** — BM25 and vector search run simultaneously
- **File hashing** — skips unchanged files on re-ingest
- **Role-based document filtering** — retrieval respects user permissions

### 📄 Document Support
- PDF 
- TXT
- CSV
- DOCX (Word documents)
- File-type aware chunking per document type

### 🤖 Agent System
- **Tool routing** — decides between `rag_search`, `direct_answer`, and `clarify`
- **Self-correction** — rephrases and retries queries with low confidence scores
- **Hallucination guard** — embedding similarity check to verify answers against context
- **Live status indicators** — shows what the agent is doing in real time
- **Vision support** — image analysis via `ministral-3:3b`(can add your vision supported model)

### 💬 Chat Experience
- Streaming responses (token by token)
- Per-user chat history per model
- Inline citations with `[1]` `[2]` markers
- Source badges with expandable chunk previews
- Image upload and analysis
- Multi-document filter (select specific documents to search)
- Pagination for long chat histories

### 🔐 Auth & Access Control
- Login / logout with SQLite user database
- Guest mode with query limit (10 per session) and message length limit(Additionally, Guests cannto access certain document, cant add/delete document
  and no persistant chat history)
- Role-based document access (keywords matched to filenames)
- Per-user chat history saved separately per model
- Session-based auth gate

### 🛠️ Admin Panel (admin role only)
- Add / delete users
- View and manage all users with roles
- Audit log with filtering by user and pagination
- Role permissions manager (add/remove keywords per role, create new roles)

### 📊 Evaluation
- RAGAS evaluation framework integration
- Tests both models on same test cases
- Measures faithfulness, answer relevancy, context precision, context recall
- Side-by-side model comparison with winner per metric

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Agent (decide_tool)
    │
    ├── clarify       → ask user for more info
    ├── direct_answer → answer without RAG
    └── rag_search
            │
            ▼
    Query Embedding (Embedding model)
            │
    ┌───────┴───────┐
    │               │
Vector Search    BM25 Search
(ChromaDB)      (rank-bm25)
    │               │
    └───────┬───────┘
            │
        RRF Fusion
            │
        CrossEncoder Rerank
        (Reranker model)
            │
        Build Prompt + Context
            │
        Ollama LLM Generate
            │
        Hallucination Check
            │
        Stream Response + Citations
```

---

## 💻 Requirements

### Hardware(Depends on model used)
- **Minimum:** 8GB RAM, any modern CPU
- **Recommended:** 16GB+ VRAM GPU (RTX 3080+), 32GB RAM, modern CPU

### Software
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

---

## 🚀 Installation

### 1 — Clone the repository

```bash
git clone <your-repo-url>
cd Local-LLM
```

### 2 — Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Install PyTorch with CUDA (for GPU support)

Check your CUDA version first:
```bash
nvidia-smi
```

Then install matching PyTorch (replace cu128 with your version):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU detection:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 5 — Install and start Ollama

Download from [ollama.com](https://ollama.com) and pull models:

These are defaults if you choose your own model change MODEL_OPTION in app.py and prompt in prompt.py

```bash
ollama pull qwen2.5:3b
ollama pull ministral-3:3b
```

Optional larger models (requires 8GB+ VRAM):
```bash
ollama pull qwen2.5:14b
ollama pull llama3.1:8b
```

### 6 — Initialize the database

```bash
python -c "from auth.db import init_db; init_db(); print('DB initialized')"
```

### 7 — Add your documents

Place your documents in `data/raw/`:
```
data/raw/
 ├── company_policy.pdf
 ├── employee_records.csv
 ├── technical_docs.docx
 └── notes.txt
```

### 8 — Ingest documents

Optionally you can also ingest from the system

```bash
python scripts/ingest.py
```
or

```bash
python -m scripts.ingest
```
---

## ⚙️ Configuration

### Embedding model (`rag/embedding.py`)
```python
# Default — Lighter option
SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# good quality, fast
SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
```

### Reranker (`rag/reranker.py`)
```python
# Default
CrossEncoder("BAAI/bge-reranker-base", device="cpu")

# Better option
CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
```

### Retrieval settings (`rag/retriever.py`)
```python
def retrieve_docs(query, k=7, rerank_k=4, ...):
# k         — number of candidates from vector + BM25 search
# rerank_k  — number of chunks passed to LLM after reranking
```

### LLM settings (`rag/llm.py`)
```python
"options": {
    "num_ctx":    3048,   # context window size
    "num_gpu":    99,     # layers on GPU (99 = max)
    "keep_alive": "30m"   # keep model loaded
}
```

### Chunk settings (`scripts/ingest.py`)
```python
CHUNK_SIZE = 500   # characters per chunk
OVERLAP    = 100   # overlap between chunks
```

### Guest limits (`app.py`)
```python
GUEST_QUERY_LIMIT = 10   # max queries per guest session
# message length limit = 500 characters
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

For faster startup (disables file watcher):
```bash
streamlit run app.py --server.fileWatcherType none
```

---

## 📖 Usage Guide

### Logging In
- Use credentials from [Default Credentials](#default-credentials)
- Or click **Continue as Guest** for limited access

### Asking Questions
- Type your question in the chat input
- The agent automatically decides whether to search documents or answer directly
- Citations appear below each answer with expandable chunk previews

### Uploading Documents (logged-in users only)
1. In the sidebar, click **Add New Documents**
2. Upload PDF, TXT, CSV, or DOCX files
3. Click **⚡ Ingest Documents**
4. Documents are available immediately

### Filtering Documents
- Use **🔎 Filter Source** multiselect to search within specific documents
- Leave empty to search all accessible documents

### Image Analysis (Vision support model only)
1. Select `ministral-3:3b` model
2. Upload image in **🖼️ Image Input** section
3. Type your question about the image
4. The model will analyze and respond

### Admin Panel
Accessible only to `admin` role:
- **👥 Manage Users** — add/delete users, assign roles
- **📋 Audit Log** — view all queries with filters
- **🔐 Role Permissions** — manage document access per role

### Changing Password
Available to all logged-in users:
- Click **🔑 Change Password** in sidebar
- Enter current password and new password

---

## 📊 Evaluation

### Setup
```bash
pip install ragas==0.1.9 datasets langchain==0.2.16 langchain-core==0.2.38 langchain-community==0.2.16 langchain-ollama==0.1.3
ollama pull nomic-embed-text
```

### Add test cases
Edit `eval/test_cases.json`:
```json
[
    {
        "question": "Who is the CEO?",
        "ground_truth": "The CEO is..."
    }
]
```

### Run evaluation
```bash
cd eval
python run_eval.py
```

Results saved to `eval/results_comparison.json`.

### Metrics explained

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer stick to retrieved context? |
| Answer Relevancy | Is the answer relevant to the question? |
| Context Precision | Are retrieved chunks actually useful? |
| Context Recall | Did retrieval find all necessary information? |

---

## 📁 Project Structure

```
Local-LLM/
│
├── app.py                    # Main Streamlit application
│
├── rag/
│   ├── agent.py              # Tool routing (decide_tool)
│   ├── agent_runner.py       # Agent loop with self-correction
│   ├── citation_renderer.py  # Citation popup UI
│   ├── embedding.py          # Sentence embedding model
│   ├── fusion.py             # Reciprocal Rank Fusion
│   ├── hallucination_guard.py# Answer grounding check
│   ├── llm.py                # Ollama LLM interface
│   ├── prompts.py            # System prompts per model
│   ├── rag_chain.py          # Prompt building + retrieval orchestration
│   ├── reranker.py           # CrossEncoder reranker
│   ├── retriever.py          # Hybrid retrieval pipeline
│   └── vectordb.py           # ChromaDB interface
│
├── auth/
│   ├── db.py                 # SQLite auth + audit log
│   └── login_page.py         # Login UI
│
├── memory/
│   ├── chat_store.py         # Per-user chat persistence
│   └── chats/                # Chat history files
│
├── scripts/
│   └── ingest.py             # Document ingestion pipeline
│
├── eval/
│   ├── ragas_eval.py         # RAGAS evaluation setup
│   ├── run_eval.py           # Evaluation runner
│   ├── test_cases.json       # Test questions + ground truth
│   └── results_comparison.json # Evaluation results
│
├── data/
│   └── raw/                  # Place documents here
│
├── chroma_db/                # ChromaDB vector store (auto-created)
│   └── bm25_index.pkl        # BM25 index (auto-created)
│
├── auth/
│   └── users.db              # SQLite database (auto-created)
│
└── requirements.txt
```

---

## 🔑 Default Credentials

| Username | Password | Role | Document Access |
|---|---|---|---|
| admin | admin123 | admin | All documents |
| bipin | bipin123 | hr | HR, Employee, Policy, Leave |
| rohan | rohan123 | dev | Technical, Engineering, Dev |
| — | — | guest | Public/General docs only (10 query limit) |

> ⚠️ Change default passwords immediately after first login using the **🔑 Change Password** feature in the sidebar.

---

## 🔐 Role-Based Document Access

Access is controlled by keyword matching in filenames:

```
Role: hr
Keywords: HR, Employee, Policy, Leave

→ Nexora_HR_Employee_Table.pdf    ✅ accessible
→ Nexora_Budget_Plan_2026.pdf     ❌ not accessible
```

Add new keywords via the **Admin Panel → Role Permissions** UI.

---

## 🐛 Common Issues

**Ollama not responding**
```bash
ollama serve   # start Ollama server
ollama ps      # check loaded models
```

**ChromaDB dimension mismatch**
Occurs when switching embedding models. Reset and re-ingest:
```bash
# Windows
rmdir /s /q chroma_db
python scripts/ingest.py
```

**BM25 index not found**
```bash
python scripts/ingest.py
```

**Slow responses**
- Check `ollama ps` — model should show `100% GPU`
- Ensure `keep_alive` is set in `llm.py`
- Add pre-warm call on app startup