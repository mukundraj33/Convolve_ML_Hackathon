"""Qdrant dense search + persistent-rebuildable BM25 + RRF generation engine."""
from datetime import datetime, timezone
import hashlib, logging, re, uuid
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from config import (BM25_TOP_K, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DENSE_TOP_K,
                    EMBEDDING_LOCAL_FILES_ONLY, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_HOST, OLLAMA_NUM_PREDICT,
                    QDRANT_PATH, QDRANT_URL, RRF_K, TOP_K)
from models import SearchResult

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        logger.info("Loading embedding model %s (local_files_only=%s)", EMBEDDING_MODEL, EMBEDDING_LOCAL_FILES_ONLY)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=EMBEDDING_LOCAL_FILES_ONLY)
        self.client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)
        self.initialize_collection(); self._bm25 = None; self._bm25_chunks = []; self.rebuild_bm25_index()
        logger.info("RAG engine ready with %s BM25 chunks", len(self._bm25_chunks))
    def initialize_collection(self):
        if COLLECTION_NAME not in {c.name for c in self.client.get_collections().collections}:
            self.client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=self.embedding_model.get_embedding_dimension(), distance=Distance.COSINE))
    def chunk_text(self, text):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]; chunks=[]; current=""
        for sentence in sentences:
            if current and len(current)+len(sentence)+1 > CHUNK_SIZE:
                chunks.append(current.strip()); current=current[-CHUNK_OVERLAP:]+" "+sentence
            else: current += (" " if current else "")+sentence
        return chunks + ([current.strip()] if current.strip() else [])
    def add_document(self, text, source_name, source_type):
        if not text or not text.strip(): raise ValueError("No extractable text was found.")
        document_hash=self.generate_document_hash(text)
        if self.document_exists(document_hash): return None
        chunks=self.chunk_text(text)
        if not chunks: raise ValueError("No sentence chunks could be created.")
        document_id=str(uuid.uuid4()); uploaded_at=datetime.now(timezone.utc).isoformat(); embeddings=self.embedding_model.encode(chunks, normalize_embeddings=True); points=[]
        for i,(chunk,vector) in enumerate(zip(chunks,embeddings)):
            payload={"document_id":document_id,"document_hash":document_hash,"chunk_id":i,"text":chunk,"source_name":source_name,"source_type":source_type,"uploaded_at":uploaded_at,"token_count":len(chunk.split())}
            points.append(PointStruct(id=str(uuid.uuid4()),vector=vector.tolist(),payload=payload))
        self.client.upsert(COLLECTION_NAME,points=points,wait=True); self.rebuild_bm25_index(); return document_id
    @staticmethod
    def generate_document_hash(text): return hashlib.sha256(text.encode("utf-8")).hexdigest()
    def document_exists(self, document_hash):
        points,_=self.client.scroll(COLLECTION_NAME,scroll_filter=Filter(must=[FieldCondition(key="document_hash",match=MatchValue(value=document_hash))]),limit=1); return bool(points)
    def _all_payloads(self):
        records=[]; offset=None
        while True:
            points,offset=self.client.scroll(COLLECTION_NAME,offset=offset,limit=256,with_payload=True,with_vectors=False)
            records.extend({**p.payload,"_point_id":str(p.id)} for p in points if p.payload and p.payload.get("text"))
            if offset is None: return records
    @staticmethod
    def _tokenize(text): return re.findall(r"\b\w+\b",text.lower())
    def rebuild_bm25_index(self):
        self._bm25_chunks=self._all_payloads(); tokens=[self._tokenize(c["text"]) for c in self._bm25_chunks]; self._bm25=BM25Okapi(tokens) if tokens else None
    def retrieve_dense(self, query, top_k=DENSE_TOP_K):
        vector=self.embedding_model.encode(query,normalize_embeddings=True).tolist(); points=self.client.query_points(COLLECTION_NAME,query=vector,limit=top_k).points
        return [SearchResult(p.score,p.payload,str(p.id)) for p in points]
    def retrieve_bm25(self, query, top_k=BM25_TOP_K):
        if self._bm25 is None:return []
        scores=self._bm25.get_scores(self._tokenize(query)); indices=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:top_k]
        return [SearchResult(float(scores[i]),{k:v for k,v in self._bm25_chunks[i].items() if k!="_point_id"},self._bm25_chunks[i]["_point_id"]) for i in indices if scores[i]>0]
    def retrieve(self, query, top_k=TOP_K):
        fused={}
        for ranking in (self.retrieve_dense(query),self.retrieve_bm25(query)):
            for rank,result in enumerate(ranking,1):
                key=result.point_id or f"{result.payload.get('document_id')}:{result.payload.get('chunk_id')}"; score,_=fused.get(key,(0,result)); fused[key]=(score+1/(RRF_K+rank),result)
        return [SearchResult(score,result.payload,result.point_id) for score,result in sorted(fused.values(),key=lambda x:x[0],reverse=True)[:top_k]]
    def build_prompt(self,query,search_results,history=None):
        context="\n\n".join(f"Source: {r.payload['source_name']} (chunk {r.payload['chunk_id']})\n{r.payload['text']}" for r in search_results); conversation="\n".join(f"{m['role']}: {m['content']}" for m in (history or []))
        return f"You are LearnMate AI, a concise educational assistant. Answer only from retrieved context. If it lacks the answer, say: I could not find the answer in the uploaded resources.\n\nRecent conversation:\n{conversation}\n\nRetrieved context:\n{context}\n\nQuestion: {query}\nAnswer:"
    def _chat(self,prompt,stream=False):
        return ollama.Client(host=OLLAMA_HOST).chat(
            model=LLM_MODEL,
            messages=[{"role":"user","content":prompt}],
            stream=stream,
            options={"num_predict": OLLAMA_NUM_PREDICT},
        )
    @staticmethod
    def sources(results):
        seen=set(); output=[]
        for r in results:
            p=r.payload; key=(p["source_name"],p["chunk_id"])
            if key not in seen: seen.add(key); output.append({"source_name":key[0],"chunk_id":key[1],"source_type":p["source_type"]})
        return output
    def generate_answer(self,query,history=None):
        results=self.retrieve(query); return self._chat(self.build_prompt(query,results,history))["message"]["content"],results
    def stream_answer(self,query,history=None):
        results=self.retrieve(query)
        for chunk in self._chat(self.build_prompt(query,results,history),stream=True): yield chunk["message"]["content"]
