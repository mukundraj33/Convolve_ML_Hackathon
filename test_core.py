"""Unit tests that do not require Qdrant, MongoDB, Ollama, or model downloads."""
import unittest
from rag import RAGEngine
from models import SearchResult
from utils import get_video_id


class CoreTests(unittest.TestCase):
    def test_sentence_chunking_and_hash_are_deterministic(self):
        engine = RAGEngine.__new__(RAGEngine)
        chunks = engine.chunk_text("First sentence. Second sentence. Third sentence.")
        self.assertTrue(chunks and "First sentence." in chunks[0])
        self.assertEqual(engine.generate_document_hash("same"), engine.generate_document_hash("same"))

    def test_youtube_url_forms_and_invalid_url(self):
        self.assertEqual(get_video_id("https://www.youtube.com/watch?v=abc123"), "abc123")
        self.assertEqual(get_video_id("https://youtu.be/abc123"), "abc123")
        self.assertEqual(get_video_id("https://youtube.com/live/abc123"), "abc123")
        with self.assertRaises(ValueError): get_video_id("https://example.com/watch?v=abc123")

    def test_hybrid_rrf_combines_dense_and_lexical_rankings(self):
        engine = RAGEngine.__new__(RAGEngine)
        a = SearchResult(0.9, {"document_id": "d", "chunk_id": 1}, "a")
        b = SearchResult(0.8, {"document_id": "d", "chunk_id": 2}, "b")
        engine.retrieve_dense = lambda query: [a, b]
        engine.retrieve_bm25 = lambda query: [b, a]
        results = engine.retrieve("question", top_k=2)
        self.assertEqual({result.point_id for result in results}, {"a", "b"})
        self.assertTrue(all(result.score > 0 for result in results))
