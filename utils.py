"""
utils.py

Utility functions for extracting text from different document types.

Supported formats:
- PDF
- DOCX
- TXT
- YouTube Transcript
"""

from pathlib import Path

from pypdf import PdfReader
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


# ----------------------------
# PDF
# ----------------------------

def extract_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text.strip()


# ----------------------------
# DOCX
# ----------------------------

def extract_docx(file_path: str) -> str:
    """Extract text from DOCX."""

    doc = Document(file_path)

    paragraphs = []

    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)

    return "\n".join(paragraphs)


# ----------------------------
# TXT
# ----------------------------

def extract_txt(file_path: str) -> str:
    """Extract text from TXT."""

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# YouTube Transcript
# ----------------------------

def get_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    """

    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path[1:]

    if parsed.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        return parse_qs(parsed.query)["v"][0]

    raise ValueError("Invalid YouTube URL")


def extract_youtube_transcript(url: str) -> str:
    """
    Download transcript from YouTube.
    """

    video_id = get_video_id(url)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    full_text = ""

    for chunk in transcript:
        full_text += chunk["text"] + " "

    return full_text.strip()


# ----------------------------
# Dispatcher
# ----------------------------

def extract_text(file_path: str) -> str:
    """
    Automatically detect file type and extract text.
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    elif extension == ".txt":
        return extract_txt(file_path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )