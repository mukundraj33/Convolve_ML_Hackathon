"""FastAPI transport layer; all retrieval and generation stays in RAGEngine."""
import os, shutil, uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rag import RAGEngine
from memory import ConversationMemory, MemoryUnavailableError
from config import MEMORY_WINDOW
from utils import extract_text, extract_youtube_transcript

app=FastAPI(title="LearnMate AI",version="2.0")
rag=RAGEngine(); UPLOAD_DIR="uploads"; os.makedirs(UPLOAD_DIR,exist_ok=True)
class QueryRequest(BaseModel):
    question:str=Field(min_length=1,max_length=4000)
    session_id:str|None=Field(default=None,max_length=128)
class YoutubeRequest(BaseModel): url:str=Field(min_length=1,max_length=2048)
def memory():
    try:return ConversationMemory()
    except MemoryUnavailableError as error: raise HTTPException(503,detail=str(error))
def session(value): return value or str(uuid.uuid4())

@app.get("/")
def home(): return {"message":"LearnMate AI API Running"}
@app.post("/upload")
async def upload_document(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf",".docx",".txt"}: raise HTTPException(415,"Unsupported file type. Use PDF, DOCX, or TXT.")
    filename=Path(file.filename).name; filepath=os.path.join(UPLOAD_DIR,filename)
    with open(filepath,"wb") as buffer: shutil.copyfileobj(file.file,buffer)
    try: document_id=rag.add_document(extract_text(filepath),filename,"document")
    except ValueError as error: raise HTTPException(422,detail=str(error))
    except Exception as error: raise HTTPException(503,detail="Qdrant or embedding service is unavailable.") from error
    if document_id is None: raise HTTPException(409,"Duplicate document: identical content is already indexed.")
    return {"status":"success","filename":filename,"document_id":document_id}
@app.post("/youtube")
def upload_youtube(request:YoutubeRequest):
    try: document_id=rag.add_document(extract_youtube_transcript(request.url),request.url,"youtube")
    except ValueError as error: raise HTTPException(400,detail=str(error))
    except Exception as error: raise HTTPException(400,detail="Transcript unavailable or YouTube URL could not be processed.") from error
    if document_id is None: raise HTTPException(409,"Duplicate document: this transcript is already indexed.")
    return {"status":"success","document_id":document_id}
@app.post("/query")
def ask_question(request:QueryRequest):
    session_id=session(request.session_id); store=memory(); history=store.history(session_id,MEMORY_WINDOW)
    try: answer,results=rag.generate_answer(request.question.strip(),history)
    except Exception as error: raise HTTPException(503,"Ollama, Qdrant, or the embedding model is unavailable.") from error
    sources=rag.sources(results); store.add_message(session_id,"user",request.question.strip()); store.add_message(session_id,"assistant",answer,sources)
    return {"answer":answer,"sources":sources,"session_id":session_id}
@app.post("/query/stream")
def stream_question(request:QueryRequest):
    session_id=session(request.session_id); store=memory(); history=store.history(session_id,MEMORY_WINDOW)
    try:
        results=rag.retrieve(request.question.strip()); stream=rag._chat(rag.build_prompt(request.question.strip(),results,history),stream=True)
    except Exception as error: raise HTTPException(503,"Ollama, Qdrant, or the embedding model is unavailable.") from error
    sources=rag.sources(results); store.add_message(session_id,"user",request.question.strip())
    def generate():
        pieces=[]
        try:
            for item in stream:
                token=item["message"]["content"]; pieces.append(token); yield token
        finally:
            if pieces: store.add_message(session_id,"assistant","".join(pieces),sources)
    return StreamingResponse(generate(),media_type="text/plain",headers={"X-Session-ID":session_id})
@app.get("/history/{session_id}")
def history(session_id:str):
    if not session_id.strip(): raise HTTPException(422,"Invalid session ID.")
    return {"session_id":session_id,"messages":memory().history(session_id,MEMORY_WINDOW)}
