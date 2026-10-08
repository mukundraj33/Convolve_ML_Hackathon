"""Environment-driven settings for LearnMate AI."""
from os import getenv
from dotenv import load_dotenv

load_dotenv()

def env_bool(name: str, default: bool) -> bool:
    return getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}

API_HOST = getenv("API_HOST", "127.0.0.1")
API_PORT = int(getenv("API_PORT", "8000"))
LLM_MODEL = getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_PREDICT = int(getenv("OLLAMA_NUM_PREDICT", "256"))
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
# Avoids a Hugging Face network check when a model is already cached locally.
# Set false on a provisioned server that may download the model.
EMBEDDING_LOCAL_FILES_ONLY = env_bool("EMBEDDING_LOCAL_FILES_ONLY", True)

COLLECTION_NAME = getenv("QDRANT_COLLECTION", "learnmate")
QDRANT_URL = getenv("QDRANT_URL", "").strip() or None
QDRANT_PATH = getenv("QDRANT_PATH", "./database/qdrant")

MONGODB_URI = getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = getenv("MONGODB_DATABASE", "learnmate")
MONGODB_COLLECTION = getenv("MONGODB_COLLECTION", "messages")
MONGODB_SERVER_SELECTION_TIMEOUT_MS = int(getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000"))

CHUNK_SIZE = int(getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(getenv("CHUNK_OVERLAP", "100"))
DENSE_TOP_K = int(getenv("DENSE_TOP_K", "10"))
BM25_TOP_K = int(getenv("BM25_TOP_K", "10"))
TOP_K = int(getenv("TOP_K", "5"))
RRF_K = int(getenv("RRF_K", "60"))
MEMORY_WINDOW = int(getenv("MEMORY_WINDOW", "10"))
