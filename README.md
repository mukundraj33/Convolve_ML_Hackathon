# 📚 LearnMate AI
### Multimodal Hybrid Retrieval-Augmented Generation (RAG) System for Personalized Educational Assistance

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Gradio](https://img.shields.io/badge/Gradio-Frontend-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-red)
![MongoDB](https://img.shields.io/badge/MongoDB-Conversation%20Memory-success)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.1-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

# 🚀 Overview

LearnMate AI is a **production-ready Multimodal Hybrid Retrieval-Augmented Generation (RAG)** platform designed to improve educational accessibility by providing **personalized learning assistance** from both **documents** and **YouTube lectures**.

The system combines **semantic retrieval**, **keyword retrieval**, **persistent conversational memory**, and **streaming Large Language Model inference** to answer educational questions with source citations while recommending the most relevant learning resources.

Unlike conventional chatbots, LearnMate AI grounds every response using retrieved educational content, significantly reducing hallucinations and improving factual accuracy.

---

# ✨ Features

## 📄 Multimodal Knowledge Base

- PDF Upload
- DOCX Upload
- TXT Upload
- YouTube Transcript Ingestion
- Automatic Metadata Extraction

---

## 🧠 Hybrid Retrieval

- Dense Vector Search (BGE Embeddings)
- BM25 Keyword Retrieval
- Reciprocal Rank Fusion (RRF)
- Metadata-aware Retrieval
- Source Filtering

---

## 🤖 Large Language Model

- Local Llama 3.1 using Ollama
- Prompt Engineering
- Streaming Responses
- Automatic Source Citations
- Hallucination Reduction

---

## 💬 Conversational AI

- Persistent Chat Sessions
- MongoDB Conversation Memory
- Context-aware Follow-up Questions
- User-specific Learning History
- Multi-turn Conversation Support

---

## 🎓 Personalized Learning

- Resource Recommendation
- Relevant PDF Suggestions
- Related YouTube Videos
- Topic-wise Recommendations

---

## ⚡ Backend

- FastAPI REST APIs
- Swagger Documentation
- Modular Service Architecture
- Input Validation
- Error Handling

---

## 🎨 Frontend

- Gradio Chat Interface
- ChatGPT-style Conversation UI
- Live Streaming Responses
- Document Upload
- YouTube Upload

---

## ☁ Deployment

- Docker
- Docker Compose
- Environment Variables
- Persistent Volumes
- Production Ready

---

# 🏗 System Architecture

```
                   User

                     │

               Gradio UI

                     │

              HTTP Requests

                     │

                FastAPI API

                     │

      ┌──────────────┼───────────────┐
      ▼                              ▼

 MongoDB                     RAG Engine

(Chat Memory)

                                     │

       ┌──────────────┬──────────────┬──────────────┐

       ▼              ▼              ▼

   Embeddings      Hybrid Search     Ollama

(BGE Model)      (Qdrant + BM25)   (Llama 3.1)

       │

       ▼

 Document Store

(PDF, DOCX, TXT, YouTube)
```

---

# 🔄 Workflow

```
Upload Document

↓

Text Extraction

↓

Sentence Chunking

↓

Embedding Generation

↓

Hybrid Indexing

↓

Qdrant + BM25

↓

User Query

↓

Hybrid Retrieval

↓

Conversation Context (MongoDB)

↓

Prompt Construction

↓

Llama 3.1

↓

Streaming Response

↓

Answer + Citations + Recommended Resources
```

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Backend | FastAPI |
| Frontend | Gradio |
| LLM | Llama 3.1 (Ollama) |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| Vector Database | Qdrant |
| Conversation Memory | MongoDB |
| Hybrid Search | BM25 + Dense Retrieval |
| Containerization | Docker |
| API Documentation | Swagger UI |
| Document Processing | PyPDF2, python-docx |
| Transcript Extraction | youtube-transcript-api |

---

# 📂 Supported Input Formats

| Type | Supported |
|-------|-----------|
| PDF | ✅ |
| DOCX | ✅ |
| TXT | ✅ |
| YouTube Videos | ✅ |
| YouTube Live | ✅ |
| YouTube Shorts | ✅ |
| Embedded Videos | ✅ |

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/LearnMate-AI.git

cd LearnMate-AI
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download

https://ollama.com

Pull Model

```bash
ollama pull llama3.1:8b
```

Run

```bash
ollama serve
```

---

## Start MongoDB

```bash
docker compose up mongodb
```

---

## Start Qdrant

```bash
docker compose up qdrant
```

---

## Start Backend

```bash
uvicorn api:app --reload
```

---

## Start Frontend

```bash
python app.py
```

---

# 🐳 Docker

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Services

- FastAPI
- Gradio
- MongoDB
- Qdrant
- Ollama

---

# 📡 API Endpoints

## Upload Document

```
POST /upload
```

Uploads educational documents.

---

## Upload YouTube

```
POST /youtube
```

Indexes YouTube transcript.

---

## Ask Question

```
POST /query
```

Returns

- Answer
- Sources
- Recommended Resources

---

## Get Chat History

```
GET /history/{session_id}
```

---

## Delete Session

```
DELETE /history/{session_id}
```

---

# 📊 Retrieval Pipeline

```
Question

↓

Embedding

↓

Dense Search

↓

BM25 Search

↓

Reciprocal Rank Fusion

↓

Top-k Retrieval

↓

Prompt Engineering

↓

Llama 3.1

↓

Streaming Response
```

---

# 💾 MongoDB Collections

```
chat_sessions

messages

user_preferences

feedback
```

---

# 📚 Metadata Stored

Each indexed chunk stores

```json
{
    "document_id":"",
    "document_hash":"",
    "chunk_id":0,
    "source_name":"",
    "source_type":"",
    "uploaded_at":"",
    "token_count":120
}
```

---

# 🧪 Engineering Highlights

- Sentence-aware Chunking
- SHA-256 Duplicate Detection
- Metadata-aware Retrieval
- Automatic Source Citation
- Hybrid Retrieval (Dense + BM25)
- Persistent Conversation Memory
- Streaming Token Generation
- Modular SOLID Architecture
- Dockerized Deployment
- Production-ready REST APIs

---

# 📈 Future Improvements

- OCR for scanned PDFs
- Whisper Speech-to-Text
- Multilingual Support
- Image Retrieval
- Vision-Language Models
- User Authentication
- Role-based Access
- Cloud Deployment (AWS/GCP/Azure)

---

# 👨‍💻 Author

**Mukund Raj**

Final Year Undergraduate  
Indian Institute of Technology Bombay

GitHub: https://github.com/mukundraj33

LinkedIn: https://linkedin.com/in/mukundrajiitb

Portfolio: https://mukundraj.com

---
