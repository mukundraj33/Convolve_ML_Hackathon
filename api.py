from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from rag import RAGEngine
from utils import (
    extract_text,
    extract_youtube_transcript,
)

app = FastAPI(
    title="LearnMate AI",
    version="1.0"
)

rag = RAGEngine()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):

    question: str


class YoutubeRequest(BaseModel):

    url: str


@app.get("/")
def home():

    return {

        "message": "LearnMate AI API Running"

    }


@app.post("/upload")

async def upload_document(

    file: UploadFile = File(...)

):

    filepath = os.path.join(

        UPLOAD_DIR,

        file.filename,

    )

    with open(filepath, "wb") as buffer:

        shutil.copyfileobj(

            file.file,

            buffer,

        )

    text = extract_text(filepath)

    rag.add_document(

        text=text,

        source_name=file.filename,

        source_type="document",

    )

    return {

        "status": "success",

        "filename": file.filename,

    }



@app.post("/youtube")
def upload_youtube(request: YoutubeRequest):
    try:
        transcript = extract_youtube_transcript(request.url)

        rag.add_document(
            text=transcript,
            source_name=request.url,
            source_type="youtube",
        )

        return {
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@app.post("/query")

def ask_question(

    request: QueryRequest,

):

    answer, sources = rag.generate_answer(

        request.question

    )

    return {

        "answer": answer

    }