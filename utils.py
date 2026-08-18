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


from youtube_transcript_api import (
    YouTubeTranscriptApi
)
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)

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

from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


from urllib.parse import urlparse, parse_qs

from urllib.parse import urlparse, parse_qs


def get_video_id(url: str):

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Unsupported YouTube URL")

    hostname = parsed.hostname.lower() if parsed.hostname else ""

    # youtube.com/watch?v=...
    if hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    ):

        params = parse_qs(parsed.query)

        if "v" in params:
            return params["v"][0]

        path = parsed.path.strip("/")

        # youtube.com/embed/<id>
        if path.startswith("embed/"):
            return path.split("/")[1]

        # youtube.com/shorts/<id>
        if path.startswith("shorts/"):
            return path.split("/")[1]

        # youtube.com/live/<id>
        if path.startswith("live/"):
            return path.split("/")[1]

    # youtu.be/<id>
    elif hostname == "youtu.be":

        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    raise ValueError("Unsupported YouTube URL")

def extract_youtube_transcript(url: str):

    video_id = get_video_id(url)

    try:

        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        text = ""

        for item in transcript:
            text += item.text + " "

        return text

    except TranscriptsDisabled:
        raise Exception(
            "This YouTube video has subtitles disabled."
        )

    except NoTranscriptFound:
        raise Exception(
            "No transcript available for this video."
        )

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
