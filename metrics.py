"""Standalone, non-destructive benchmark for the current LearnMate AI codebase.

This script deliberately imports and calls the project's RAGEngine methods.  It
never writes to the production ``learnmate`` collection; ingestion and duplicate
checks use a short-lived, uniquely named Qdrant collection instead.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LINE = "=" * 76
RETRIEVAL_QUERIES = (
    "What is deep learning?",
    "What are neural networks?",
    "How does RAG improve factual accuracy?",
    "What are transformer models?",
    "How does personalized learning work?",
)
# Manually curated from the payloads already present in the production
# ``learnmate`` collection on 2026-08-14, before running this evaluation.
# A target is identified by the same (document_id, chunk_id) pair returned in
# SearchResult.payload by RAGEngine.  The questions are paraphrased where
# practical and the targets were selected from source content, not rankings.
DENSE_RECALL_EVALUATION = (
    {
        "query": "Which AI technique uses neural networks?",
        "relevant": (("80ae5417-95d0-4d44-aef9-fc8969d57c16", 0),),
    },
    {
        "query": "In the source's AI taxonomy, where does machine learning fit?",
        "relevant": (("80ae5417-95d0-4d44-aef9-fc8969d57c16", 0),),
    },
    {
        "query": "What effect does retrieval-augmented generation have on factual accuracy?",
        "relevant": (("80ae5417-95d0-4d44-aef9-fc8969d57c16", 0),),
    },
    {
        "query": "What mechanism do transformer models rely on?",
        "relevant": (("80ae5417-95d0-4d44-aef9-fc8969d57c16", 0),),
    },
    {
        "query": "What does a Type I civilization harness according to the Kardashev discussion?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 5),),
    },
    {
        "query": "Beyond energy use, what environmental capabilities would a Type I society possess?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 6),),
    },
    {
        "query": "What property is used to rank civilizations on the Kardashev scale?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 336),),
    },
    {
        "query": "How does a stellar Type II civilization differ from a planetary Type I civilization?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 337),),
    },
    {
        "query": "How could lunar settlers cultivate food indoors despite the Moon's environment?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 82),),
    },
    {
        "query": "Why are domes suitable structures for habitats with pressurized air?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 86),),
    },
    {
        "query": "Explain the powder-bed process proposed for making construction material from lunar regolith.",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 72),),
    },
    {
        "query": "What makes a small air leak especially dangerous for a lunar habitat?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 71),),
    },
    {
        "query": "How much energy can a single large X-class solar flare release?",
        "relevant": (
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 142),
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 143),
        ),
    },
    {
        "query": "Why would a Mercury settlement need protection from radiation and solar activity?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 56),),
    },
    {
        "query": "What observation showed how Jupiter responds to a huge impact?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 133),),
    },
    {
        "query": "Why would nuclear weapons have little lasting effect on Jupiter?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 134),),
    },
    {
        "query": "What megastructure could let a Type II civilization use the Sun's power?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 294),),
    },
    {
        "query": "Describe how a Dyson sphere is intended to capture a star's energy.",
        "relevant": (
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 191),
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 345),
        ),
    },
    {
        "query": "If a Dyson sphere blocked starlight, what signal could still reveal it?",
        "relevant": (
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 260),
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 261),
        ),
    },
    {
        "query": "How is a Dyson swarm different from one solid shell around a star?",
        "relevant": (
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 40),
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 430),
        ),
    },
    {
        "query": "What astronomical observation could suggest Dyson spheres around several stars?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 269),),
    },
    {
        "query": "What are von Neumann probes proposed to do in interstellar expansion?",
        "relevant": (
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 48),
            ("31bf9020-7e52-4edf-8781-53ab43d71541", 49),
        ),
    },
    {
        "query": "Which nearby system is described as the first proper destination beyond our solar system?",
        "relevant": (("a47166b7-ab99-4d8a-935c-35f77b3a7cc9", 21),),
    },
    {
        "query": "Why does the transcript rule out faster-than-light travel for now?",
        "relevant": (("a47166b7-ab99-4d8a-935c-35f77b3a7cc9", 8),),
    },
    {
        "query": "Why does being in a star's habitable zone not guarantee a world is suitable for humans?",
        "relevant": (("a47166b7-ab99-4d8a-935c-35f77b3a7cc9", 28),),
    },
    {
        "query": "How quickly could the fictional fast ship reach Alpha Centauri?",
        "relevant": (("a47166b7-ab99-4d8a-935c-35f77b3a7cc9", 10),),
    },
    {
        "query": "What makes Uranus and Neptune difficult targets for resource extraction?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 185),),
    },
    {
        "query": "What conditions make Trappist-1e comparatively hospitable, despite an important danger?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 220),),
    },
    {
        "query": "What continuing environmental risk remains even after establishing settlements on Trappist-1e?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 223),),
    },
    {
        "query": "Why would settlers on K2-18b need a floating habitat?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 226),),
    },
    {
        "query": "What atmospheric change might prevent Kepler-62f from staying frozen?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 234),),
    },
    {
        "query": "What ability characterizes the hypothetical Type V civilization?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 285),),
    },
    {
        "query": "What scale of control distinguishes Type VI from Type VII civilization in the discussion?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 318),),
    },
    {
        "query": "Which propulsion technologies are mentioned as possible engines for future starships?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 358),),
    },
    {
        "query": "What does the Breakthrough Starshot program aim to develop?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 359),),
    },
    {
        "query": "Why is reaching Earth orbit compared with paying for a body made of solid gold?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 367),),
    },
    {
        "query": "What medical capabilities can brain implants provide to amputees?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 278),),
    },
    {
        "query": "According to the discussion, why might humans and AI agents need to change to survive in space?",
        "relevant": (("31bf9020-7e52-4edf-8781-53ab43d71541", 433),),
    },
)
RECALL_BENCHMARK_SUMMARY: dict[str, float | int] | None = None
HYBRID_RECALL_BENCHMARK_SUMMARY: dict[str, float | int] | None = None
TEMPORARY_COLLECTION_PATHS: list[Path] = []
SAMPLE_TEXT = " ".join(
    [
        "Deep learning is a branch of machine learning that uses neural networks with multiple layers.",
        "Each layer transforms information so a model can learn patterns from examples.",
        "Transformer models use attention to weigh the most relevant parts of an input sequence.",
        "Retrieval augmented generation supplies relevant source passages to a language model before it answers.",
        "In education, personalized learning adapts explanations and practice to a learner's progress.",
    ]
    * 12
)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def seconds(value: float) -> str:
    return f"{value:.2f} s"


def percentile_95(values: list[float]) -> float:
    """Linear-interpolated p95, without requiring NumPy."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def section(number: int, title: str) -> None:
    print(f"\n[{number}] {title}")


def item(label: str, value: Any) -> None:
    print(f"{label:<25}: {value}")


def source_text() -> str:
    """Read project Python sources only to describe features actually present."""
    contents = []
    for path in Path(__file__).resolve().parent.glob("*.py"):
        if path.name != Path(__file__).name:
            contents.append(path.read_text(encoding="utf-8", errors="replace").lower())
    return "\n".join(contents)


def supported_inputs() -> list[tuple[str, bool]]:
    # Source inspection is intentional: support is determined by utils.py's
    # implemented extractors even if an unrelated runtime dependency is absent.
    utils_source = (Path(__file__).resolve().parent / "utils.py").read_text(
        encoding="utf-8", errors="replace"
    )
    return [
        ("PDF", "def extract_pdf" in utils_source),
        ("DOCX", "def extract_docx" in utils_source),
        ("TXT", "def extract_txt" in utils_source),
        ("YouTube Transcript", "def extract_youtube_transcript" in utils_source),
    ]


def api_request(url: str, payload: bytes | None = None) -> None:
    headers = {"Content-Type": "application/json"} if payload else {}
    request = Request(url, data=payload, headers=headers, method="POST" if payload else "GET")
    with urlopen(request, timeout=120) as response:
        response.read()


def unavailable_service_error(error: Exception) -> bool:
    """Recognize expected Ollama/http-client availability failures without hiding bugs."""
    module = type(error).__module__
    return isinstance(error, (ConnectionError, OSError, RuntimeError, TimeoutError)) or module.startswith(
        ("ollama", "httpx", "httpcore")
    )


def benchmark_api() -> None:
    section(7, "FASTAPI")
    root = "http://127.0.0.1:8000/"
    try:
        api_request(root)
    except (URLError, TimeoutError, ConnectionError, OSError):
        print("FastAPI benchmark: SKIPPED - API not running")
        return
    except HTTPError as error:
        print(f"FastAPI benchmark: SKIPPED - API returned HTTP {error.code}")
        return

    get_times = []
    for _ in range(5):
        start = time.perf_counter()
        api_request(root)
        get_times.append(time.perf_counter() - start)
    item("GET / samples", len(get_times))
    item("GET / average", ms(statistics.mean(get_times)))
    item("GET / median", ms(statistics.median(get_times)))
    item("GET / P95", ms(percentile_95(get_times)))

    # /query is read-only from the API's perspective, but it may be slow because
    # it calls Ollama.  One measured request prevents an API benchmark from
    # repeatedly consuming model resources.
    try:
        start = time.perf_counter()
        api_request(root + "query", b'{"question":"What is deep learning?"}')
        query_time = time.perf_counter() - start
        item("POST /query samples", 1)
        item("POST /query latency", ms(query_time))
    except (URLError, TimeoutError, ConnectionError, OSError):
        print("POST /query: SKIPPED - API became unavailable")
    except HTTPError as error:
        print(f"POST /query: SKIPPED - API returned HTTP {error.code}")


def benchmark_llm(engine: Any) -> None:
    section(6, "LLM GENERATION")
    from config import LLM_MODEL

    query = "What is deep learning?"
    try:
        start = time.perf_counter()
        results = engine.retrieve(query)
        engine.build_prompt(query, results)
        retrieval_prompt_time = time.perf_counter() - start

        start = time.perf_counter()
        engine.generate_answer(query)
        total_time = time.perf_counter() - start

        item("Model", LLM_MODEL)
        item("Retrieval + prompt time", ms(retrieval_prompt_time))
        item("Total LLM response time", seconds(total_time))
        item("End-to-end query latency", seconds(total_time))
    except Exception as error:
        if unavailable_service_error(error):
            print("LLM benchmark: SKIPPED - Ollama/model unavailable")
            return
        raise

    try:
        start = time.perf_counter()
        first_chunk_time = None
        chunk_count = 0
        for chunk in engine.stream_answer(query):
            if first_chunk_time is None and chunk:
                first_chunk_time = time.perf_counter() - start
            if chunk:
                chunk_count += 1
        streaming_time = time.perf_counter() - start
        item("First streaming chunk", ms(first_chunk_time) if first_chunk_time is not None else "not received")
        item("Streaming total time", seconds(streaming_time))
        item("Non-empty stream chunks", chunk_count)
    except Exception as error:
        if unavailable_service_error(error):
            print("Streaming benchmark: SKIPPED - Ollama/model unavailable")
        else:
            raise


def benchmark_dense_recall_at_5(engine: Any, client: Any, collection_name: str) -> bool:
    """Measure item-level Recall@5 against manually verified collection chunks."""
    section("4.1", "DENSE RETRIEVAL RECALL@5")
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Recall@5: NOT MEASURED — unable to inspect indexed payloads ({error})")
        return False

    payload_by_target = {
        (point.payload.get("document_id"), point.payload.get("chunk_id")): point.payload
        for point in points
        if point.payload and point.payload.get("document_id") is not None and point.payload.get("chunk_id") is not None
    }
    available_targets = set(payload_by_target)
    required_targets = {
        target for evaluation in DENSE_RECALL_EVALUATION for target in evaluation["relevant"]
    }
    missing_targets = required_targets - available_targets
    if missing_targets:
        print("Recall@5: NOT MEASURED — insufficient ground-truth data")
        print(f"Missing verified chunk identifiers: {sorted(missing_targets)}")
        return False

    total_relevant = 0
    retrieved_relevant = 0
    successful_queries = 0
    query_recalls: list[float] = []
    latencies: list[float] = []
    print("Ground truth uses manually verified (document_id, chunk_id) payload pairs.")
    print("Targets were selected from indexed text before retrieval rankings were run.")
    for index, evaluation in enumerate(DENSE_RECALL_EVALUATION, start=1):
        relevant = set(evaluation["relevant"])
        try:
            start = time.perf_counter()
            results = engine.retrieve(evaluation["query"], top_k=5)
            latencies.append(time.perf_counter() - start)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Recall@5: NOT MEASURED — dense retrieval failed ({error})")
            return False
        returned = {
            (result.payload.get("document_id"), result.payload.get("chunk_id"))
            for result in results
        }
        matches = relevant & returned
        query_recall = len(matches) / len(relevant)
        total_relevant += len(relevant)
        retrieved_relevant += len(matches)
        successful_queries += bool(matches)
        query_recalls.append(query_recall)

        print("-" * 66)
        print(f"Query #{index}")
        item("Question", evaluation["query"])
        print("Ground Truth:")
        for doc_id, chunk_id in sorted(relevant):
            payload = payload_by_target[(doc_id, chunk_id)]
            print(f"  source_name={payload.get('source_name')}")
            print(f"  document_id={doc_id}, chunk_id={chunk_id}")
        print("Retrieved Top 5:")
        if not results:
            print("  no results returned")
        for rank, result in enumerate(results, start=1):
            payload = result.payload
            print(
                f"  {rank}. source_name={payload.get('source_name')}; "
                f"document_id={payload.get('document_id')}; "
                f"chunk_id={payload.get('chunk_id')}; score={result.score:.4f}"
            )
        print("Relevant Retrieved:")
        if matches:
            for doc_id, chunk_id in sorted(matches):
                print(f"  document_id={doc_id}, chunk_id={chunk_id}")
        else:
            print("  none")
        item("Query Recall@5", f"{query_recall * 100:.2f}% ({len(matches)}/{len(relevant)})")

    recall = retrieved_relevant / total_relevant if total_relevant else 0.0
    query_level_recall = statistics.mean(query_recalls) if query_recalls else 0.0
    query_success_rate = successful_queries / len(DENSE_RECALL_EVALUATION) if DENSE_RECALL_EVALUATION else 0.0
    print("-" * 66)
    item("Evaluation queries", len(DENSE_RECALL_EVALUATION))
    item("Total ground-truth chunks", total_relevant)
    item("Relevant chunks retrieved @5", retrieved_relevant)
    item("Overall Dense Recall@5", f"{recall * 100:.2f}%")
    item("Queries with relevant result", f"{successful_queries}/{len(DENSE_RECALL_EVALUATION)}")
    item("Query-level Recall@5", f"{query_level_recall * 100:.2f}%")
    item("Query-level success rate", f"{query_success_rate * 100:.2f}%")
    item("Minimum query recall", f"{min(query_recalls) * 100:.2f}%")
    item("Maximum query recall", f"{max(query_recalls) * 100:.2f}%")
    item("Mean query recall", f"{query_level_recall * 100:.2f}%")
    item("Average retrieval latency", ms(statistics.mean(latencies)))
    item("Median retrieval latency", ms(statistics.median(latencies)))
    item("P95 retrieval latency", ms(percentile_95(latencies)))
    print("Previous benchmark       : 5 queries, 100.00% Recall@5")
    print(
        f"New benchmark            : {len(DENSE_RECALL_EVALUATION)} queries, "
        f"{recall * 100:.2f}% overall Dense Retrieval Recall@5"
    )
    print("The new benchmark is more representative because it contains paraphrased and")
    print("harder questions across the indexed documents and educational transcripts.")
    global RECALL_BENCHMARK_SUMMARY
    RECALL_BENCHMARK_SUMMARY = {
        "recall": recall,
        "queries": len(DENSE_RECALL_EVALUATION),
        "average_latency": statistics.mean(latencies),
        "p95_latency": percentile_95(latencies),
    }
    return True


def benchmark_hybrid_recall_at_5(engine: Any, client: Any, collection_name: str) -> bool:
    """Run the identical 38-query ground truth against Dense + BM25 + RRF."""
    section("4.2", "HYBRID DENSE + BM25 + RRF RECALL@5")
    try:
        engine.rebuild_bm25_index()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Hybrid Recall@5: NOT MEASURED — unable to build BM25 ({error})")
        return False
    total = found = 0
    latencies: list[float] = []
    for evaluation in DENSE_RECALL_EVALUATION:
        try:
            start = time.perf_counter()
            results = engine.retrieve(evaluation["query"], top_k=5)
            latencies.append(time.perf_counter() - start)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Hybrid Recall@5: NOT MEASURED — retrieval failed ({error})")
            return False
        relevant = set(evaluation["relevant"])
        returned = {(r.payload.get("document_id"), r.payload.get("chunk_id")) for r in results}
        total += len(relevant)
        found += len(relevant & returned)
    recall = found / total if total else 0.0
    item("Evaluation queries", len(DENSE_RECALL_EVALUATION))
    item("Total ground-truth chunks", total)
    item("Relevant chunks retrieved @5", found)
    item("Overall Hybrid Recall@5", f"{recall * 100:.2f}%")
    item("Average retrieval latency", ms(statistics.mean(latencies)))
    item("P95 retrieval latency", ms(percentile_95(latencies)))
    global HYBRID_RECALL_BENCHMARK_SUMMARY
    HYBRID_RECALL_BENCHMARK_SUMMARY = {"recall": recall, "average_latency": statistics.mean(latencies), "p95_latency": percentile_95(latencies)}
    return True


def cleanup_temporary_collection_paths() -> None:
    """Remove only exact local-Qdrant directories created by this benchmark run."""
    for temporary_path in TEMPORARY_COLLECTION_PATHS:
        try:
            shutil.rmtree(temporary_path)
        except FileNotFoundError:
            pass
        except OSError as error:
            print(f"Temporary benchmark directory cleanup: SKIPPED - {error}")


def main() -> None:
    print(LINE)
    print("                 LEARNMATE AI PERFORMANCE REPORT")
    print(LINE)

    section(11, "SYSTEM INFORMATION")
    item("Python", sys.version.split()[0])
    item("OS", f"{platform.system()} {platform.release()}")
    item("CPU", platform.processor() or "unavailable")
    item("PyTorch", version("torch"))
    item("Sentence Transformers", version("sentence-transformers"))
    item("Qdrant Client", version("qdrant-client"))

    try:
        import torch
        item("CUDA available", "YES" if torch.cuda.is_available() else "NO")
    except ImportError:
        item("CUDA available", "SKIPPED - PyTorch unavailable")

    try:
        from config import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from rag import RAGEngine
    except ImportError as error:
        print(f"\nCore benchmarks: SKIPPED - missing dependency ({error.name})")
        finish_without_core()
        return

    # This is the actual model constructor called by RAGEngine.__init__, timed
    # independently so Qdrant connection time is not mislabeled as model load.
    section(1, "EMBEDDING MODEL")
    try:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        model_load_time = time.perf_counter() - start
        dimension = embedding_model.get_embedding_dimension()
        item("Model", EMBEDDING_MODEL)
        item("Embedding dimension", dimension)
        item("Model load time", seconds(model_load_time))
    except (OSError, RuntimeError, ConnectionError) as error:
        print(f"Embedding benchmark: SKIPPED - model unavailable ({error})")
        finish_without_core()
        return

    # Construct an instance without __init__: its real methods are used below,
    # while this avoids any chance of __init__ creating the production collection.
    try:
        client = QdrantClient(path=QDRANT_PATH)
        collection_names = {c.name for c in client.get_collections().collections}
    except (OSError, RuntimeError, ValueError) as error:
        print(f"\nQdrant benchmarks: SKIPPED - database unavailable ({error})")
        finish_without_core()
        return

    engine = RAGEngine.__new__(RAGEngine)
    engine.embedding_model = embedding_model
    engine.client = client

    section(2, "CHUNKING")
    start = time.perf_counter()
    chunks = engine.chunk_text(SAMPLE_TEXT)
    chunk_time = time.perf_counter() - start
    lengths = [len(chunk) for chunk in chunks]
    item("Input characters", len(SAMPLE_TEXT))
    item("Chunks generated", len(chunks))
    item("Average chunk length", f"{statistics.mean(lengths):.1f}")
    item("Minimum chunk length", min(lengths))
    item("Maximum chunk length", max(lengths))
    item("Chunking time", ms(chunk_time))

    temporary_collection = f"learnmate_metrics_{uuid.uuid4().hex}"
    try:
        client.create_collection(
            collection_name=temporary_collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        # add_document resolves COLLECTION_NAME from rag.py at call time.  This
        # temporary substitution makes the unmodified implementation target only
        # the disposable collection, and is restored even when a benchmark fails.
        import rag as rag_module
        production_collection = rag_module.COLLECTION_NAME
        rag_module.COLLECTION_NAME = temporary_collection
        try:
            section(3, "DOCUMENT INGESTION")
            start = time.perf_counter()
            ingestion_chunks = engine.chunk_text(SAMPLE_TEXT)
            embedding_model.encode(ingestion_chunks, normalize_embeddings=True)
            embedding_time = time.perf_counter() - start

            start = time.perf_counter()
            document_id = engine.add_document(SAMPLE_TEXT, "metrics_sample.txt", "benchmark")
            ingestion_time = time.perf_counter() - start
            indexed_count = len(ingestion_chunks) if document_id else 0
            item("Chunks indexed", indexed_count)
            item("Embeddings generated", len(ingestion_chunks))
            item("Embedding generation time", seconds(embedding_time))
            item("Total ingestion/indexing", seconds(ingestion_time))
            item("Throughput", f"{indexed_count / ingestion_time:.2f} chunks/s" if ingestion_time else "n/a")
            item("Approx. documents/sec", f"{1 / ingestion_time:.2f}" if ingestion_time else "n/a")

            section(5, "DUPLICATE DETECTION")
            hash_start = time.perf_counter()
            for _ in range(1000):
                engine.generate_document_hash(SAMPLE_TEXT)
            hash_time = (time.perf_counter() - hash_start) / 1000
            document_hash = engine.generate_document_hash(SAMPLE_TEXT)
            checks = []
            duplicate_detected = False
            for _ in range(5):
                check_start = time.perf_counter()
                duplicate_detected = engine.document_exists(document_hash)
                checks.append(time.perf_counter() - check_start)
            item("SHA-256 hash time (avg)", ms(hash_time))
            item("Duplicate check (median)", ms(statistics.median(checks)))
            item("Duplicate detected", "YES" if duplicate_detected else "NO")
        finally:
            rag_module.COLLECTION_NAME = production_collection
    except (OSError, RuntimeError, ValueError) as error:
        print(f"\nIngestion/duplicate benchmark: SKIPPED - temporary Qdrant collection unavailable ({error})")
    finally:
        try:
            client.delete_collection(temporary_collection)
        except (OSError, RuntimeError, ValueError):
            print("Temporary benchmark collection cleanup: SKIPPED - could not delete collection")
        # qdrant-client's local mode can retain an empty storage folder after a
        # successful delete_collection().  Defer its removal until client.close()
        # releases Windows' storage.sqlite handle.
        temporary_path = Path(QDRANT_PATH).resolve() / "collection" / temporary_collection
        TEMPORARY_COLLECTION_PATHS.append(temporary_path)

    section(4, "DENSE RETRIEVAL (warm)")
    if COLLECTION_NAME not in collection_names:
        print("Dense retrieval benchmark: SKIPPED - production collection does not exist")
    else:
        try:
            engine.retrieve(RETRIEVAL_QUERIES[0])  # warm-up; excluded from measurements
            latencies: list[float] = []
            scores: list[float] = []
            top_one_scores: list[float] = []
            result_counts: list[int] = []
            for query in RETRIEVAL_QUERIES:
                for _ in range(3):
                    start = time.perf_counter()
                    results = engine.retrieve(query)
                    latencies.append(time.perf_counter() - start)
                    result_counts.append(len(results))
                    if results:
                        top_one_scores.append(results[0].score)
                        scores.extend(result.score for result in results)
            item("Queries tested", len(RETRIEVAL_QUERIES))
            item("Measured retrieval runs", len(latencies))
            item("Average latency", ms(statistics.mean(latencies)))
            item("Median latency", ms(statistics.median(latencies)))
            item("Minimum latency", ms(min(latencies)))
            item("Maximum latency", ms(max(latencies)))
            item("P95 latency", ms(percentile_95(latencies)))
            item("Results returned (avg)", f"{statistics.mean(result_counts):.2f}")
            item("Top-1 score (avg)", f"{statistics.mean(top_one_scores):.4f}" if top_one_scores else "SKIPPED - no results")
            item("Average top-k score", f"{statistics.mean(scores):.4f}" if scores else "SKIPPED - no results")
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Dense retrieval benchmark: SKIPPED - Qdrant query failed ({error})")

    section(8, "VECTOR DATABASE")
    if COLLECTION_NAME not in collection_names:
        print("Vector database statistics: SKIPPED - production collection does not exist")
    else:
        try:
            info = client.get_collection(COLLECTION_NAME)
            item("Collection", COLLECTION_NAME)
            item("Indexed vectors", getattr(info, "points_count", "unavailable"))
            item("Vector dimension", dimension)
            item("Collection status", getattr(info, "status", "unavailable"))
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Vector database statistics: SKIPPED - unable to inspect collection ({error})")

    recall_measured = benchmark_dense_recall_at_5(engine, client, COLLECTION_NAME)
    hybrid_measured = benchmark_hybrid_recall_at_5(engine, client, COLLECTION_NAME) if recall_measured else False
    if recall_measured and hybrid_measured:
        dense = RECALL_BENCHMARK_SUMMARY["recall"]
        hybrid = HYBRID_RECALL_BENCHMARK_SUMMARY["recall"]
        section("4.3", "HYBRID IMPROVEMENT")
        item("Absolute Recall Improvement", f"{(hybrid - dense) * 100:.2f} percentage points")
        item("Relative Recall Improvement", f"{((hybrid - dense) / dense * 100) if dense else 0:.2f}%")
    benchmark_llm(engine)
    benchmark_api()
    client.close()
    cleanup_temporary_collection_paths()
    finish_without_core(recall_measured)


def finish_without_core(recall_measured: bool = False) -> None:
    section(9, "SUPPORTED INPUTS")
    for name, available in supported_inputs():
        item(name, "YES" if available else "NO")

    code = source_text()
    section(10, "NOT CURRENTLY BENCHMARKED")
    absent = [
        ("BM25 performance", "bm25" not in code, "NOT IMPLEMENTED IN CURRENT CODE"),
        ("Hybrid retrieval", "hybrid" not in code, "NOT IMPLEMENTED IN CURRENT CODE"),
        ("RRF performance", "reciprocal rank fusion" not in code and "rrf" not in code, "NOT IMPLEMENTED IN CURRENT CODE"),
        ("MongoDB memory", "mongoclient" not in code and "pymongo" not in code, "NOT IMPLEMENTED IN CURRENT CODE"),
        ("Docker deployment", not Path("Dockerfile").exists() and not Path("docker-compose.yml").exists(), "NOT BENCHMARKED"),
        ("Cloud deployment", True, "NOT BENCHMARKED"),
        ("Recall@K", not recall_measured, "NO LABELED GROUND-TRUTH DATASET"),
        ("Precision@K", True, "NO LABELED GROUND-TRUTH DATASET"),
        ("Hallucination reduction", True, "NO CONTROLLED EVALUATION DATASET"),
    ]
    for name, is_absent, reason in absent:
        if is_absent:
            item(name, reason)

    section(12, "RESUME-SAFE METRICS")
    if recall_measured and RECALL_BENCHMARK_SUMMARY is not None:
        item("Dense Retrieval Recall@5", f"{RECALL_BENCHMARK_SUMMARY['recall'] * 100:.2f}%")
        item("Evaluation Queries", RECALL_BENCHMARK_SUMMARY["queries"])
        item("Average Retrieval", ms(float(RECALL_BENCHMARK_SUMMARY["average_latency"])))
        item("P95 Retrieval", ms(float(RECALL_BENCHMARK_SUMMARY["p95_latency"])))
        print(
            "Recall@5 was calculated against manually verified ground-truth "
            "document/chunk identifiers from the indexed educational resources."
        )
        if HYBRID_RECALL_BENCHMARK_SUMMARY is not None:
            dense = float(RECALL_BENCHMARK_SUMMARY["recall"])
            hybrid = float(HYBRID_RECALL_BENCHMARK_SUMMARY["recall"])
            item("Hybrid Retrieval Recall@5", f"{hybrid * 100:.2f}%")
            item("Hybrid Average Retrieval", ms(float(HYBRID_RECALL_BENCHMARK_SUMMARY["average_latency"])))
            item("Hybrid P95 Retrieval", ms(float(HYBRID_RECALL_BENCHMARK_SUMMARY["p95_latency"])))
            item("Absolute Improvement", f"{(hybrid-dense)*100:.2f} percentage points")
            item("Relative Improvement", f"{((hybrid-dense)/dense*100) if dense else 0:.2f}%")
    else:
        print("Recall metrics are available only when the collection and embedding model can be measured.")
    print(f"\n{LINE}\n                         BENCHMARK COMPLETE\n{LINE}")


if __name__ == "__main__":
    main()
