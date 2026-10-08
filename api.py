"""FastAPI transport layer for the LearnMate RAG engine."""
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from config import MEMORY_WINDOW
from memory import ConversationMemory, MemoryUnavailableError
from rag import RAGEngine
from utils import (TranscriptUnavailableError, YoutubeServiceError, extract_text,
                   extract_youtube_transcript)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
UPLOAD_DIR = Path("uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = None
    app.state.memory = None
    app.state.rag_error = None
    app.state.memory_error = None
    UPLOAD_DIR.mkdir(exist_ok=True)
    try:
        app.state.rag = RAGEngine()
    except Exception as error:
        app.state.rag_error = str(error)
        logger.exception("RAG initialization failed")
    try:
        app.state.memory = ConversationMemory()
    except MemoryUnavailableError as error:
        app.state.memory_error = str(error)
        logger.warning("MongoDB initialization failed: %s", error)
    yield
    if app.state.memory:
        app.state.memory.close()
    if app.state.rag:
        app.state.rag.client.close()


app = FastAPI(title="LearnMate AI", version="2.1", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty.")
        return value


class YoutubeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def get_rag(request: Request) -> RAGEngine:
    rag = request.app.state.rag
    if rag is None:
        detail = "RAG is unavailable. Check the embedding model, Qdrant settings, and API logs."
        raise HTTPException(status_code=503, detail=detail)
    return rag


def get_memory(request: Request) -> ConversationMemory:
    memory = request.app.state.memory
    if memory is None:
        raise HTTPException(status_code=503, detail="MongoDB memory is unavailable. Start MongoDB and check MONGODB_URI.")
    return memory


def get_session_id(value: str | None) -> str:
    return value.strip() if value and value.strip() else str(uuid.uuid4())


def read_history(memory: ConversationMemory, session_id: str) -> list[dict]:
    try:
        return memory.history(session_id, MEMORY_WINDOW)
    except MemoryUnavailableError as error:
        logger.exception("MongoDB history read failed")
        raise HTTPException(status_code=503, detail=str(error)) from error


def save_message(memory: ConversationMemory, session_id: str, role: str, content: str, sources: list[dict] | None = None) -> None:
    try:
        memory.add_message(session_id, role, content, sources)
    except MemoryUnavailableError as error:
        logger.exception("MongoDB message write failed")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/")
def home():
    return {"message": "LearnMate AI API Running"}


@app.get("/health")
def health(request: Request):
    return {"rag_ready": request.app.state.rag is not None, "memory_ready": request.app.state.memory is not None,
            "rag_error": request.app.state.rag_error, "memory_error": request.app.state.memory_error}


@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(415, "Unsupported file type. Use PDF, DOCX, or TXT.")
    filename = Path(file.filename).name
    filepath = UPLOAD_DIR / filename
    with filepath.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        document_id = get_rag(request).add_document(extract_text(str(filepath)), filename, "document")
    except ValueError as error:
        raise HTTPException(422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Document indexing failed for %s", filename)
        raise HTTPException(503, detail="Document indexing failed; check Qdrant and the embedding model.") from error
    if document_id is None:
        raise HTTPException(409, "Duplicate document: identical content is already indexed.")
    return {"status": "success", "filename": filename, "document_id": document_id}


@app.post("/youtube")
def upload_youtube(request: Request, payload: YoutubeRequest):
    try:
        document_id = get_rag(request).add_document(extract_youtube_transcript(payload.url), payload.url, "youtube")
    except ValueError as error:
        raise HTTPException(400, detail=str(error)) from error
    except TranscriptUnavailableError as error:
        raise HTTPException(422, detail=str(error)) from error
    except YoutubeServiceError as error:
        raise HTTPException(503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("YouTube ingestion failed")
        raise HTTPException(400, detail="Transcript unavailable or YouTube URL could not be processed.") from error
    if document_id is None:
        raise HTTPException(409, "Duplicate document: this transcript is already indexed.")
    return {"status": "success", "document_id": document_id}


@app.post("/query")
def ask_question(request: Request, payload: QueryRequest):
    session_id = get_session_id(payload.session_id)
    memory = get_memory(request)
    history = read_history(memory, session_id)
    try:
        answer, results = get_rag(request).generate_answer(payload.question.strip(), history)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Query generation failed")
        raise HTTPException(503, "Query failed; check Qdrant, Ollama, and the embedding model.") from error
    sources = get_rag(request).sources(results)
    save_message(memory, session_id, "user", payload.question.strip())
    save_message(memory, session_id, "assistant", answer, sources)
    return {"answer": answer, "sources": sources, "session_id": session_id}


@app.post("/query/stream")
def stream_question(request: Request, payload: QueryRequest):
    session_id = get_session_id(payload.session_id)
    memory = get_memory(request)
    history = read_history(memory, session_id)
    rag = get_rag(request)
    try:
        results = rag.retrieve(payload.question.strip())
        stream = rag._chat(rag.build_prompt(payload.question.strip(), results, history), stream=True)
    except Exception as error:
        logger.exception("Streaming query setup failed")
        raise HTTPException(503, "Streaming query failed; check Qdrant, Ollama, and the embedding model.") from error
    sources = rag.sources(results)
    save_message(memory, session_id, "user", payload.question.strip())

    def generate():
        pieces: list[str] = []
        try:
            for item in stream:
                token = item["message"]["content"]
                pieces.append(token)
                yield token
        except Exception:
            logger.exception("Ollama stream failed")
            raise
        finally:
            if pieces:
                try:
                    memory.add_message(session_id, "assistant", "".join(pieces), sources)
                except MemoryUnavailableError:
                    logger.exception("Could not persist streamed answer")

    return StreamingResponse(generate(), media_type="text/plain", headers={"X-Session-ID": session_id})


@app.get("/history/{session_id}")
def history(request: Request, session_id: str):
    if not session_id.strip():
        raise HTTPException(422, "Invalid session ID.")
    return {"session_id": session_id, "messages": read_history(get_memory(request), session_id)}
