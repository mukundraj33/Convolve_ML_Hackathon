"""
rag.py

Core Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities:
- Load embedding model
- Initialize Qdrant
- Chunk documents
- Generate embeddings
- Store vectors
"""

from typing import List
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


from rank_bm25 import BM25Okapi
import ollama

from utils import (
    extract_text,
    extract_youtube_transcript,
)

class RAGPipeline:
    def __init__(self):

        print("Loading Embedding Model...")

        self.embedding_model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )

        print("Embedding Model Loaded")

        self.collection_name = "learnmate"

        self.qdrant = QdrantClient(":memory:")

        self._create_collection()

        
                # BM25 storage
        self.bm25 = None
        self.bm25_chunks = []

    def _create_collection(self):

        collections = self.qdrant.get_collections().collections

        names = [c.name for c in collections]

        if self.collection_name not in names:

            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )

    def chunk_text(self, text: str,
               chunk_size: int = 500,
               overlap: int = 100):

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunks.append(text[start:end])

            start += chunk_size - overlap

        return chunks

    def embed_chunks(self, chunks: List[str]):

        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True,
        )

        return embeddings

    def add_document(self, text: str, source: str):

        chunks = self.chunk_text(text)

        embeddings = self.embed_chunks(chunks)

        points = []

        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk_id": idx,
                    },
                )
            )

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        # ---------- BM25 ----------

        self.bm25_chunks.extend(chunks)

        tokenized = [doc.split() for doc in self.bm25_chunks]

        self.bm25 = BM25Okapi(tokenized)

        print(f"Stored {len(points)} chunks.")

    def dense_search(
        self,
        query: str,
        top_k: int = 5,
    ):

        query_vector = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        return results.points
    
    
    def sparse_search(
        self,
        query: str,
        top_k: int = 5,
    ):

        if self.bm25 is None:
            return []

        scores = self.bm25.get_scores(query.split())

        ranked = sorted(
            zip(self.bm25_chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
    
    
    def hybrid_search(
    self,
    query: str,
    top_k: int = 5,
):

        dense_results = self.dense_search(query, top_k)

        sparse_results = self.sparse_search(query, top_k)

        chunks = []

        seen = set()

        # Dense retrieval

        for item in dense_results:

            text = item.payload["text"]

            if text not in seen:

                seen.add(text)

                chunks.append(text)

        # Sparse retrieval

        for text, _ in sparse_results:

            if text not in seen:

                seen.add(text)

                chunks.append(text)

        return chunks[:top_k]
    
    
    
    def build_prompt(
    self,
    query,
    retrieved_chunks,
):

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
    You are an AI learning assistant.

    Answer ONLY from the provided context.

    If the answer is not present,
    say:

    "I cannot answer using the uploaded resources."

    Context:

    {context}

    Question:

    {query}

    Answer:
    """

        return prompt
    
    
    def stream_answer(
    self,
    query,
):

        retrieved_chunks = self.hybrid_search(query)

        prompt = self.build_prompt(
            query,
            retrieved_chunks,
        )

        stream = ollama.chat(

            model="llama3.1:8b",

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],

            stream=True,
        )

        for chunk in stream:

            yield chunk["message"]["content"]
            
            
    def ingest_file(self, file_path: str):

        text = extract_text(file_path)

        self.add_document(
            text=text,
            source=file_path,
        )


    def ingest_youtube(self, url: str):

        transcript = extract_youtube_transcript(url)

        self.add_document(
            text=transcript,
            source=url,
        )