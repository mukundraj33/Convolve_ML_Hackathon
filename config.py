"""Centralized, environment-driven settings for LearnMate AI."""
from os import getenv
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL = getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
COLLECTION_NAME = getenv("QDRANT_COLLECTION", "learnmate")
QDRANT_URL = getenv("QDRANT_URL")
QDRANT_PATH = getenv("QDRANT_PATH", "./database/qdrant")
MONGODB_URI = getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = getenv("MONGODB_DATABASE", "learnmate")
MONGODB_COLLECTION = getenv("MONGODB_COLLECTION", "messages")
CHUNK_SIZE = int(getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(getenv("CHUNK_OVERLAP", "100"))
DENSE_TOP_K = int(getenv("DENSE_TOP_K", "10"))
BM25_TOP_K = int(getenv("BM25_TOP_K", "10"))
TOP_K = int(getenv("TOP_K", "5"))
RRF_K = int(getenv("RRF_K", "60"))
MEMORY_WINDOW = int(getenv("MEMORY_WINDOW", "10"))
# Compatibility for existing scripts.
MONGO_URI, DATABASE_NAME = MONGODB_URI, MONGODB_DATABASE
