from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from datetime import datetime
import uuid

from config import *
import re
import hashlib
from models import SearchResult
import ollama
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)



class RAGEngine:

    def __init__(self):

        print("Loading Embedding Model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding Model Loaded")

        print("Connecting Qdrant...")

        self.client = QdrantClient(
            path=QDRANT_PATH
        )

        self.initialize_collection()

        print("RAG Engine Ready")

    def initialize_collection(self):

        collections = self.client.get_collections().collections

        names = [c.name for c in collections]

        if COLLECTION_NAME not in names:

            self.client.create_collection(

                collection_name=COLLECTION_NAME,

                vectors_config=VectorParams(

                    size=self.embedding_model.get_embedding_dimension(),

                    distance=Distance.COSINE,

                ),
            )

    

    def chunk_text(self, text: str):

        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []

        current_chunk = ""

        for sentence in sentences:

            if len(current_chunk) + len(sentence) < CHUNK_SIZE:

                current_chunk += sentence + " "

            else:

                chunks.append(current_chunk.strip())

                overlap = current_chunk[-CHUNK_OVERLAP:]

                current_chunk = overlap + " " + sentence

        if current_chunk:

            chunks.append(current_chunk.strip())

        return chunks
    
    def add_document(
    self,
    text: str,
    source_name: str,
    source_type: str,
):

        print(f"\nProcessing {source_name}")

        document_id = str(uuid.uuid4())

        document_hash = self.generate_document_hash(text)

        if self.document_exists(document_hash):

            print("Document already indexed.")

            return None

        chunks = self.chunk_text(text)

        print(f"Generated {len(chunks)} chunks")

        embeddings = self.embedding_model.encode(
            chunks,
            normalize_embeddings=True,
        )

        points = []

        uploaded_at = datetime.utcnow().isoformat()

        for idx, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            payload = {

                "document_id": document_id,

                "document_hash": document_hash,

                "chunk_id": idx,

                "text": chunk,

                "source_name": source_name,

                "source_type": source_type,

                "uploaded_at": uploaded_at,

                "token_count": len(chunk.split()),

            }

            points.append(

                PointStruct(

                    id=str(uuid.uuid4()),

                    vector=embedding.tolist(),

                    payload=payload,

                )

            )

        self.client.upsert(

            collection_name=COLLECTION_NAME,

            points=points,

        )

        print(f"Stored {len(points)} vectors.")

        return document_id
    
    def generate_document_hash(self, text: str):

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        
    def document_exists(self, document_hash: str):

        results = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_hash",
                        match=MatchValue(
                            value=document_hash
                        ),
                    )
                ]
            ),
            limit=1,
        )

        return len(results[0]) > 0
    
    def retrieve(
    self,
    query: str,
    top_k: int = 5,
):

        query_vector = self.embedding_model.encode(

            query,

            normalize_embeddings=True,

        ).tolist()

        results = self.client.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            limit=top_k,

        )

        retrieved_chunks = []

        for point in results.points:

            retrieved_chunks.append(

                SearchResult(

                    score=point.score,

                    payload=point.payload,

                )

            )

        return retrieved_chunks
    
    def build_prompt(
    self,
    query: str,
    search_results,
):

        context = []

        for result in search_results:

            payload = result.payload

            context.append(

                f"""
    Source : {payload['source_name']}
    Type   : {payload['source_type']}
    Chunk  : {payload['chunk_id']}

    Content:
    {payload['text']}
    """
            )

        context = "\n\n".join(context)

        prompt = f"""
    You are LearnMate AI.

    You are an educational assistant.

    Rules:

    1. Answer ONLY from the supplied context.

    2. If the answer is not present,
    say

    "I could not find the answer in the uploaded resources."

    3. Never make up information.

    4. Keep answers concise.

    5. Never generate citations or page numbers.

    6. Never invent document names.

    7. The application will attach citations automatically.

    =========================
    Retrieved Context
    =========================

    {context}

    =========================
    Question
    =========================

    {query}

    =========================
    Answer
    =========================
    """

        return prompt
    
    
    def generate_answer(
    self,
    query: str,
):

        search_results = self.retrieve(query)

        prompt = self.build_prompt(
            query,
            search_results,
        )

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[

                {

                    "role": "user",

                    "content": prompt,

                }

            ],

        )

        answer = response["message"]["content"]

        citations = []

        seen = set()

        for result in search_results:

            payload = result.payload

            key = (
                payload["source_name"],
                payload["chunk_id"],
            )

            if key not in seen:

                seen.add(key)

                citations.append(

                    f"- {payload['source_name']} "
                    f"(Chunk {payload['chunk_id']})"

                )

        answer += "\n\nSources\n"

        answer += "\n".join(citations)

        return answer, search_results
    
    def stream_answer(
    self,
    query: str,
):

        search_results = self.retrieve(query)

        prompt = self.build_prompt(
            query,
            search_results,
        )

        stream = ollama.chat(

            model=LLM_MODEL,

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