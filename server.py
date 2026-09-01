import os
import sys
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_all
from core.vector_store import load_vector_store
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
from utils.rate_limiter import get_redis_client

app = FastAPI(title="Meeting Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = "outputs/history.json"
_active_rag_chain = None


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    os.makedirs("outputs", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


class ProcessRequest(BaseModel):
    url: str
    language: str = "english"


class QuestionRequest(BaseModel):
    question: str


@app.get("/api/status")
def get_status():
    r = get_redis_client()
    return {
        "status": "online",
        "redis": "connected" if r else "offline",
    }


@app.get("/api/documents")
def get_documents():
    return load_history()


def _execute_pipeline(source: str, language: str = "english", original_filename: str = None):
    global _active_rag_chain
    # 1. Process audio
    chunks = process_input(source)
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to extract audio chunks.")

    # 2. Transcribe
    transcript = transcribe_all(chunks, language=language)
    if not transcript:
        raise HTTPException(status_code=500, detail="Transcription produced empty text.")

    # 3. Summarize & Title
    title = generate_title(transcript)
    summary_text = summarize(transcript)

    # 4. Extract Insights
    insights = extract_all(transcript)

    # 5. Build RAG Index
    _active_rag_chain = build_rag_chain(transcript)

    # Save to outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    with open("outputs/meeting_notes.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n## Summary\n{summary_text}\n\n## Action Items\n{insights['action_items']}\n\n## Decisions\n{insights['key_decisions']}\n\n## Questions\n{insights['questions']}\n")

    doc_id = str(uuid.uuid4())[:8]
    fname = original_filename or (title if title else os.path.basename(source))
    if not fname.endswith((".wav", ".mp3", ".mp4", ".pdf", ".txt")):
        fname = f"{fname[:30]}.mp3"

    doc = {
        "id": doc_id,
        "name": fname,
        "title": title,
        "status": "Indexed",
        "added": "Today",
        "size": f"{len(chunks)} Chunk(s)",
        "summary": summary_text,
        "action_items": insights.get("action_items", ""),
        "key_decisions": insights.get("key_decisions", ""),
        "questions": insights.get("questions", ""),
        "transcript": transcript,
        "source": source,
    }

    history = load_history()
    # Prepend new doc
    history.insert(0, doc)
    save_history(history)

    return doc


@app.post("/api/process")
def process_video(req: ProcessRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL or file path cannot be empty.")
    try:
        doc = _execute_pipeline(req.url.strip(), language=req.language)
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), language: str = Form("english")):
    try:
        os.makedirs("downloades", exist_ok=True)
        temp_path = os.path.join("downloades", file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc = _execute_pipeline(temp_path, language=language, original_filename=file.filename)
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
def ask_rag(req: QuestionRequest):
    global _active_rag_chain
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if _active_rag_chain is None:
            if os.path.exists("vector_db"):
                _active_rag_chain = load_rag_chain()
            else:
                raise HTTPException(status_code=400, detail="No transcript indexed yet. Please process a video or document first.")

        answer = ask_question(_active_rag_chain, req.question.strip())
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
