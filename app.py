"""Small Gradio presentation layer that calls FastAPI only."""
import os, requests, gradio as gr
API=os.getenv("API_URL","http://127.0.0.1:8000")
def upload_document(file):
    if file is None:return "Please choose a file."
    with open(file.name,"rb") as handle: response=requests.post(f"{API}/upload",files={"file":(os.path.basename(file.name),handle)},timeout=120)
    return "Document uploaded successfully." if response.ok else response.text
def upload_youtube(url):
    response=requests.post(f"{API}/youtube",json={"url":url},timeout=120); return "Transcript indexed." if response.ok else response.text
def ask(question,session_id):
    response=requests.post(f"{API}/query/stream",json={"question":question,"session_id":session_id or None},stream=True,timeout=180)
    if not response.ok: yield response.text,session_id; return
    active=response.headers.get("X-Session-ID",session_id); answer=""
    for token in response.iter_content(chunk_size=None,decode_unicode=True): answer+=token; yield answer,active
with gr.Blocks(title="LearnMate AI") as demo:
    gr.Markdown("# LearnMate AI\nMultimodal Hybrid RAG for personalized learning")
    with gr.Row():
        file=gr.File(label="Upload PDF / DOCX / TXT"); upload=gr.Button("Upload"); status=gr.Textbox(label="Status")
    upload.click(upload_document,file,status)
    with gr.Row():
        url=gr.Textbox(label="YouTube URL"); ingest=gr.Button("Ingest"); yt_status=gr.Textbox(label="Status")
    ingest.click(upload_youtube,url,yt_status)
    session_id=gr.Textbox(label="Session ID (leave blank to start a session)")
    question=gr.Textbox(label="Question",lines=3); ask_button=gr.Button("Ask"); answer=gr.Textbox(label="Streaming answer",lines=15)
    ask_button.click(ask,[question,session_id],[answer,session_id])
if __name__=="__main__": demo.launch(server_name="0.0.0.0")
