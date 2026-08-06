import gradio as gr
import requests

API = "http://127.0.0.1:8000"


def upload_document(file):

    if file is None:

        return "Please choose a file."

    with open(file.name, "rb") as f:

        response = requests.post(

            f"{API}/upload",

            files={

                "file": (

                    file.name.split("/")[-1],

                    f,

                )

            },

        )

    if response.status_code == 200:

        return "✅ Document uploaded successfully."

    return response.text


def upload_youtube(url):

    response = requests.post(

        f"{API}/youtube",

        json={

            "url": url

        },

    )

    if response.status_code == 200:

        return "✅ Transcript indexed."

    return response.text


def ask_question(question):

    response = requests.post(

        f"{API}/query",

        json={

            "question": question

        },

    )

    if response.status_code != 200:

        return response.text

    return response.json()["answer"]



with gr.Blocks(

    title="LearnMate AI"

) as demo:

    gr.Markdown(

        "# 📚 LearnMate AI"

    )

    gr.Markdown(

        "### Multimodal Educational Assistant"

    )

    with gr.Tab("📄 Documents"):

        file = gr.File(

            label="Upload PDF / DOCX / TXT"

        )

        upload_btn = gr.Button(

            "Upload"

        )

        upload_status = gr.Textbox(

            label="Status"

        )

        upload_btn.click(

            upload_document,

            inputs=file,

            outputs=upload_status,

        )

    with gr.Tab("🎥 YouTube"):

        youtube = gr.Textbox(

            label="YouTube URL"

        )

        youtube_btn = gr.Button(

            "Ingest Video"

        )

        youtube_status = gr.Textbox(

            label="Status"

        )

        youtube_btn.click(

            upload_youtube,

            inputs=youtube,

            outputs=youtube_status,

        )

    with gr.Tab("💬 Ask"):

        question = gr.Textbox(

            lines=3,

            label="Question"

        )

        ask_btn = gr.Button(

            "Ask"

        )

        answer = gr.Textbox(

            lines=15,

            label="Answer"

        )

        ask_btn.click(

            ask_question,

            inputs=question,

            outputs=answer,

        )

demo.launch()