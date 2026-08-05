import gradio as gr

from rag import RAGPipeline

rag = RAGPipeline()


# -----------------------------
# Upload File
# -----------------------------
def upload_document(file):

    if file is None:
        return "Please upload a document."

    rag.ingest_file(file.name)

    return "Document uploaded successfully."


# -----------------------------
# Upload YouTube
# -----------------------------
def upload_youtube(url):

    if not url.strip():
        return "Enter a YouTube URL."

    rag.ingest_youtube(url)

    return "Transcript added successfully."


# -----------------------------
# Ask Question
# -----------------------------
def ask_question(query):

    if not query.strip():
        return ""

    answer = ""

    for token in rag.stream_answer(query):

        answer += token

        yield answer


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="LearnMate AI") as demo:

    gr.Markdown(
        """
# LearnMate AI

### Multimodal Learning Assistant

Supported Sources

- PDF

- DOCX

- TXT

- YouTube Lecture Transcript
"""
    )

    with gr.Tab("Upload Document"):

        file_input = gr.File()

        upload_btn = gr.Button("Upload")

        upload_output = gr.Textbox(label="Status")

        upload_btn.click(
            upload_document,
            inputs=file_input,
            outputs=upload_output,
        )

    with gr.Tab("YouTube"):

        youtube_url = gr.Textbox(
            label="YouTube URL"
        )

        youtube_btn = gr.Button(
            "Add Transcript"
        )

        youtube_output = gr.Textbox(
            label="Status"
        )

        youtube_btn.click(
            upload_youtube,
            inputs=youtube_url,
            outputs=youtube_output,
        )

    with gr.Tab("Ask Questions"):

        question = gr.Textbox(
            label="Question",
            lines=2,
        )

        ask_btn = gr.Button(
            "Ask"
        )

        answer = gr.Textbox(
            label="Answer",
            lines=18,
        )

        ask_btn.click(
            ask_question,
            inputs=question,
            outputs=answer,
        )


demo.launch()