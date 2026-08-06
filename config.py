"""
config.py

All project configuration lives here.
Changing a model or path requires editing only this file.
"""

# --------------------------
# Models
# --------------------------

LLM_MODEL = "llama3.1:8b"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --------------------------
# Qdrant
# --------------------------

COLLECTION_NAME = "learnmate"

QDRANT_PATH = "./database/qdrant"

# --------------------------
# MongoDB
# --------------------------

MONGO_URI = "mongodb://localhost:27017"

DATABASE_NAME = "learnmate"

COLLECTION_DOCUMENTS = "documents"

COLLECTION_HISTORY = "history"

# --------------------------
# Chunking
# --------------------------

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100