# LearnMate AI

LearnMate AI is a compact educational RAG application for PDF, DOCX, TXT, and YouTube transcript sources. It uses BGE embeddings with Qdrant dense search, a rebuildable BM25 lexical index, and Reciprocal Rank Fusion (RRF). FastAPI owns ingestion, retrieval, memory, and streaming; Gradio is only the presentation layer.

## Architecture

`Gradio → FastAPI → MongoDB session memory → dense Qdrant + BM25 → RRF → Ollama`

Qdrant stores the chunk text and metadata, so the in-memory BM25 index is rebuilt from persistent Qdrant payloads whenever the API starts and after each ingestion. Each chunk includes document ID, SHA-256 document hash, chunk ID, source metadata, upload time, and token count.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

For local Qdrant storage, leave `QDRANT_URL` empty in `.env`. The embedding model is loaded from the local Hugging Face cache by default (`EMBEDDING_LOCAL_FILES_ONLY=true`), avoiding an unnecessary startup network request.

Start MongoDB with Docker Desktop if it is not already running:

```bash
docker compose up -d mongodb
```

Then start Ollama and the application:

```bash
ollama serve
ollama pull llama3.1:8b
uvicorn api:app --host 127.0.0.1 --port 8000
python app.py
```

Open FastAPI documentation at `http://127.0.0.1:8000/docs` and Gradio at `http://127.0.0.1:7860`.

## API

- `POST /upload` — multipart PDF, DOCX, or TXT ingestion. Exact-content duplicates return `409`.
- `POST /youtube` — transcript ingestion for `youtube.com/watch`, `youtu.be`, and `youtube.com/live` URLs.
- `POST /query` — JSON `{ "question": "...", "session_id": "optional" }`; returns answer, sources, and session ID.
- `POST /query/stream` — same request body; streams plain-text tokens and returns the session ID in `X-Session-ID`.
- `GET /history/{session_id}` — returns the configured window of persisted messages.
- `GET /health` — reports whether RAG and MongoDB initialized successfully.

MongoDB is required for query endpoints so the application never silently pretends that persistent memory works. Configure `MEMORY_WINDOW` to bound the context passed to the model.

## Retrieval and benchmark

`DENSE_TOP_K`, `BM25_TOP_K`, `TOP_K`, and `RRF_K` are configurable. RRF uses ranks rather than directly comparing BM25 and cosine scores. Run the fixed 38-query/43-target benchmark with `python metrics.py`.

It preserves the dense Recall@5 benchmark and reports the same methodology for hybrid retrieval, including average/P95 latency and absolute/relative recall change. Results are only reported when the model and existing collection are accessible; this repository does not hard-code benchmark outcomes.

## Docker

```bash
copy .env.example .env
docker compose up --build
```

Compose starts FastAPI, Gradio, Qdrant, and MongoDB with named data volumes. Ollama intentionally remains a host service by default; set `OLLAMA_HOST` in `.env` (default: `http://host.docker.internal:11434`). Pull the configured model on the host before asking a question.

## Tests and limitations

Run `python -m unittest -v test_core` for service-free unit tests covering sentence chunking, duplicate hashing, YouTube URL parsing, and RRF behavior. Database, streaming, and Docker tests require the corresponding external services and are integration checks.

The system is text-and-transcript multimodal, not image-understanding multimodal. Retrieval quality depends on source content and the locally available embedding model; there is no reranker or OCR for scanned PDFs.
