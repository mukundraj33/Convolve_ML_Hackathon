from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Model loaded successfully!")

text = """
Artificial Intelligence is transforming education by enabling
personalized learning experiences.
"""

embedding = model.encode(text)

print("\nEmbedding Dimension:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])