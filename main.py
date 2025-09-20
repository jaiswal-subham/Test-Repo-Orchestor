# app/main.py
import logging
import os
import io
import uuid
import json
from typing import Tuple, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import fitz  # PyMuPDF

from dotenv import load_dotenv
load_dotenv(override=True)

from orchestrator import run_orchestrator   # ensure this file is on path

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("main")

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", 28000))

app = FastAPI(title="Orchestrator API (no-gradio)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

DOC_STORE = {}

class UploadResponse(BaseModel):
    doc_id: str
    name: str
    pages: int
    file_url: str
    message: str

class ChatRequest(BaseModel):
    message: str
    doc_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    state: dict

class EmailRequest(BaseModel):
    to: str
    subject: str
    conversation: str


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=stream)
    pages = doc.page_count
    text_pages = []
    for i in range(pages):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text_pages.append(text)
    full_text = "\n\n".join([f"--- Page {i+1} ---\n{p}" for i,p in enumerate(text_pages)])
    return full_text, pages

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    doc_id = str(uuid.uuid4())
    filename = f"{doc_id}.pdf"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(contents)

    try:
        full_text, pages = extract_text_from_pdf_bytes(contents)
    except Exception as e:
        full_text, pages = "", 0
        logger.exception("PDF extraction failed for %s: %s", filename, e)

    DOC_STORE[doc_id] = {
        "name": file.filename,
        "pages": pages,
        "text": full_text,
        "file_path": path,
    }

    file_url = f"http://localhost:8000/static/{filename}"
    return UploadResponse(
        doc_id=doc_id,
        name=file.filename,
        pages=pages,
        file_url=file_url,
        message="Uploaded, saved, and text extracted (if possible).",
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must be provided")

    user_msg = {"role": "user", "content": req.message}

    pdf_text = None
    if req.doc_id:
        entry = DOC_STORE.get(req.doc_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Document id not found")
        pdf_text = entry.get("text", "")

    try:
        result = run_orchestrator([user_msg], doc_text=pdf_text)
    except Exception as e:
        logger.exception("Orchestrator run failed")
        raise HTTPException(status_code=500, detail=str(e))

    final_reply = ""
    messages = result.get("messages") or []
    if not messages:
        final_reply = ""
    else:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            final_reply = last_msg.content
        elif isinstance(last_msg, dict):
            final_reply = last_msg.get("content", "")
        else:
            final_reply = str(last_msg)

    # Ensure reply is string
    if not isinstance(final_reply, str):
        try:
            final_reply = json.dumps(final_reply)
        except Exception:
            final_reply = str(final_reply)

    return ChatResponse(reply=final_reply, state=result)

@app.post("/send-email")
async def send_email(req: EmailRequest):
    return {
        "status": "ok",
        "message": "Dummy send: email not actually delivered in this mock app.",
        "payload": {
            "to": req.to,
            "subject": req.subject,
            "conversation": req.conversation[:10000],
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
