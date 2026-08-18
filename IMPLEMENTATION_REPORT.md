# Implementation Report

## 1. Executive Summary

The existing dense Qdrant RAG project was extended rather than rewritten. It now has a persistent-rebuildable BM25 index, rank-based hybrid RRF retrieval, MongoDB session memory, non-streaming and streaming FastAPI query endpoints, an API-consuming Gradio UI, centralized environment settings, and Docker assets.

## 2. Before vs After

| Feature | Before | After |
|---|---|---|
| Dense retrieval | Qdrant only | Preserved as `retrieve_dense` |
| BM25 | Not implemented | `rank-bm25` index from Qdrant payloads |
| Hybrid RRF | Not implemented | Rank-only RRF in `RAGEngine.retrieve` |
| MongoDB memory | Config names only | Persisted per-message records and bounded history |
| Session management | None | Optional/generated `session_id` returned by API |
| Streaming | Engine generator only | FastAPI `StreamingResponse` and Gradio consumption |
| FastAPI | Basic endpoints | Validation, error status codes, history endpoint |
| Gradio | Called normal query only | Calls streaming FastAPI endpoint |
| Docker | Claimed, no assets | Dockerfile and Compose services/volumes |

## 3. Architecture

`User → Gradio → FastAPI → MongoDB history → dense Qdrant + BM25 → RRF → context prompt → host Ollama → StreamingResponse → Gradio`

Qdrant retains chunks and their metadata. On process start and after document ingestion, the engine scrolls the collection and rebuilds BM25 from those same records. Consequently a restart does not orphan lexical retrieval.

## 4. File-by-file Changes

| File | Change and responsibility |
|---|---|
| `config.py` | Environment-driven application settings and retrieval/memory controls. |
| `rag.py` | Sentence chunking, SHA-256 duplicate guard, Qdrant, BM25, RRF, prompts, and Ollama calls. |
| `memory.py` | MongoDB message persistence and bounded chronological history. |
| `models.py` | Search result with a stable Qdrant point ID for fusion. |
| `api.py` | HTTP validation, ingestion, querying, streaming, session handling, and safe errors. |
| `app.py` | Basic Gradio presentation layer that talks only to FastAPI. |
| `utils.py` | Extra validation for YouTube URL parsing. |
| `metrics.py` | Adds the identical-ground-truth hybrid Recall@5 and latency measurement. |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Container deployment assets. |
| `.env.example`, `.gitignore` | Safe configuration template and local-data/secret exclusion. |
| `test_core.py` | Service-free unit checks. |
| `README.md` | Actual behavior and operating instructions. |

## 5. Hybrid Retrieval

Dense retrieval embeds the question with BGE and queries Qdrant. BM25 tokenizes the same chunk text and returns lexical rankings. RRF uses `1 / (RRF_K + rank)` for each rank list; it never compares cosine similarity with BM25 scores. `DENSE_TOP_K`, `BM25_TOP_K`, `TOP_K`, and `RRF_K` are configurable.

## 6. MongoDB Memory

The `messages` collection stores one document per message: `session_id`, `role`, `content`, UTC `timestamp`, and assistant source metadata. Its compound `(session_id, timestamp)` index supports recent-history reads. `MEMORY_WINDOW` caps messages supplied to the prompt. Query endpoints fail with HTTP 503 if MongoDB cannot be reached; persistence is never silently substituted with memory-only state.

## 7. Streaming

`/query/stream` retrieves context first, opens an Ollama streaming chat, and yields each content token via `StreamingResponse`. The API sends `X-Session-ID`; after a completed stream, the assembled assistant message and sources are persisted. Gradio iterates the HTTP response and updates the answer incrementally. First-token latency can be measured by `metrics.py` only when Ollama is available.

## 8. Docker

Compose provides `api`, `gradio`, `qdrant`, and `mongodb`. Named `qdrant_data` and `mongodb_data` volumes preserve infrastructure data. Compose reads secrets/configuration from `.env`; no credentials are included. Ollama intentionally remains on the host, reachable from containers at `host.docker.internal` by default.

## 9. Testing

| Check | Result |
|---|---|
| `python -m unittest -v test_core` | Passed: 3 tests (chunk/hash, YouTube URL forms, hybrid RRF) |
| Python compile check | Passed for modified Python files |
| Hybrid benchmark | Not measured: environment could not obtain the embedding model |
| FastAPI/MongoDB/Ollama integration | Not run: external services unavailable |
| Docker compose/build | Not run: Docker CLI is not installed |

## 10. Benchmark Results

The previously supplied baseline remains preserved: 38 queries, 43 targets, dense Recall@5 **79.07%** (34/43), average **35.50 ms**, median **35.46 ms**, P95 **41.76 ms**. These are historical baseline values, not a rerun in this environment.

Hybrid Recall@5, improvement, and hybrid latency are **not measured**. The benchmark was run, but model loading failed because the environment could not access the embedding model. `metrics.py` now produces the required comparable values as soon as the existing model/cache and collection are accessible; no hybrid values were fabricated.

## 11. Resume Claims Verification

| Resume claim | Implemented? | Evidence |
|---|---|---|
| Multimodal documents + YouTube transcripts | Yes | `utils.py`, `api.py` |
| Qdrant dense vector search | Yes | `rag.py` |
| BM25 and hybrid RRF | Yes | `rag.py` |
| FastAPI and Gradio | Yes | `api.py`, `app.py` |
| Dockerized deployment | Assets created; unverified locally | Docker files |
| MongoDB persistent memory/session tracking | Yes, service-dependent | `memory.py`, `api.py` |
| Sentence chunking/SHA-256/metadata | Yes | `rag.py` |
| Streaming LLM responses | Yes, Ollama-dependent | `api.py`, `app.py` |
| Modular RAG architecture | Yes | Separately scoped modules |

## 12. Known Limitations

BM25 is rebuilt at startup, so very large collections increase cold-start time. MongoDB and Ollama are mandatory for answering. YouTube availability depends on captions and the upstream service. Scanned PDFs need OCR. There is no learned reranker.

## 13. Interview Explanation

### 30 seconds

LearnMate indexes educational documents and YouTube captions into sentence-aware chunks. It combines Qdrant semantic retrieval and BM25 keyword retrieval with RRF, gives the retrieved context plus recent MongoDB-backed conversation memory to Ollama, and streams the answer through FastAPI to Gradio.

### 1 minute

Each upload is SHA-256 checked, chunked on sentence boundaries, embedded with BGE, and stored in Qdrant with source metadata. BM25 is rebuilt from persisted Qdrant payloads, avoiding a separate unsynchronized lexical store. At query time dense and lexical candidates are independently ranked, then RRF fuses their ranks. MongoDB stores a bounded session history so follow-up questions can use prior turns. The UI stays thin: it only calls FastAPI, including its streaming endpoint.

### 3 minutes

The retrieval design addresses complementary failures: embeddings find paraphrases, while BM25 captures exact terms such as technical names and values. Rank fusion avoids invalid score normalization between cosine similarity and BM25. The full source payload remains in Qdrant, allowing BM25 to be reproducibly reconstructed at restart. The API treats memory as operational infrastructure: if MongoDB is down, it returns a clear 503 instead of falsely claiming persistence. Streaming starts with retrieval, forwards Ollama chunks immediately, then persists the final assistant text. Compose isolates Qdrant and MongoDB behind persistent volumes while allowing host Ollama during development.

## 14. Interview Questions

1. **Why hybrid search?** Dense search handles semantic paraphrases; BM25 handles exact vocabulary.
2. **Why RRF?** It fuses relative ranks without treating unrelated score scales as comparable.
3. **What does `RRF_K` do?** It dampens the contribution of lower ranks.
4. **Where is BM25 data stored?** Its source chunks are persisted as Qdrant payloads and rebuilt in memory.
5. **Why Qdrant?** It provides vector indexing plus payload filtering/storage for dense retrieval.
6. **Why not store vectors in MongoDB?** Qdrant is specialized for vector similarity; MongoDB owns conversation records.
7. **How are duplicate uploads detected?** SHA-256 of the extracted text is checked in Qdrant payloads.
8. **Why sentence-aware chunks?** They preserve meaning better than arbitrary character splits.
9. **What is Recall@5?** Relevant ground-truth chunks returned in the top five divided by all relevant targets.
10. **Why compare against the same ground truth?** It makes dense versus hybrid recall comparison fair.
11. **What metadata is indexed?** IDs, hash, source name/type, time, chunk text, and token count.
12. **What is session memory?** Recent MongoDB messages for one session ID added to the next prompt.
13. **Why bound memory?** It controls prompt size and limits irrelevant old context.
14. **What if MongoDB is down?** Query endpoints return a clear 503.
15. **What if Ollama is down?** Generation endpoints return a clear 503.
16. **How does streaming work?** Ollama iterator → FastAPI `StreamingResponse` → Gradio HTTP iteration.
17. **When is streamed text persisted?** After the streaming generator finishes with received content.
18. **How is a session ID created?** The API generates UUID when clients omit it.
19. **Which YouTube URLs work?** Watch, short `youtu.be`, and live URLs; transcript availability still governs ingestion.
20. **Why host Ollama outside Compose?** It avoids duplicating a local model runtime while containers use `host.docker.internal`.

## 15. Tradeoffs

Dense retrieval has semantic strength but may miss rare strings; BM25 has the opposite tradeoff. Qdrant is used for vectors and MongoDB for chronological session documents. MongoDB offers durable queryable history whereas Redis would favor transient low-latency state. RRF is cheap and robust; a reranker could improve precision but adds latency/model complexity. Local Ollama avoids hosted API cost and data egress but needs local resources. Smaller chunks improve pinpointing while larger chunks preserve context.

## 16. Future Improvements

Add OCR for scans, metadata/source filters in the UI, a learned reranker, evaluation datasets for answer faithfulness, multilingual embeddings, and authenticated user accounts.
