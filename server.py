import json
import logging
import os
import shutil
import time
import traceback
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import our core engine
from main import MeetingIntelligencePipeline

app = FastAPI(title="Meeting Intelligence API - Stage 4")

# Initialize the Meeting Intelligence Engine
print("Initializing the Meeting Intelligence Engine...")
pipeline = MeetingIntelligencePipeline()
print("Engine Online.")

# Mount the static directory to serve the Vanilla Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Add CORS for complete frontend decoupling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    meeting_id: Optional[str] = None
    stream: bool = True

active_ingestions = {}

def process_audio(file_path: str, meeting_id: str):
    active_ingestions[meeting_id] = "processing"
    try:
        pipeline.ingest(file_path, meeting_id=meeting_id)
        active_ingestions[meeting_id] = "done"
    except Exception as e:
        active_ingestions[meeting_id] = f"error: {str(e)}"

@app.post("/api/upload")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    os.makedirs("data/audio", exist_ok=True)
    
    # Generate ID based on filename
    base_name = "".join(c for c in file.filename if c.isalnum() or c in " ._-").strip()
    meeting_id = base_name.rsplit('.', 1)[0]
    
    # Ensure uniqueness if it already exists
    existing_ids = [m["meeting_id"] for m in pipeline.list_meetings()]
    if meeting_id in existing_ids:
        meeting_id = f"{meeting_id}_{int(time.time())}"
        
    file_path = f"data/audio/{meeting_id}_{int(time.time())}.wav"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_audio, file_path, meeting_id)
    return {"message": "Upload successful", "meeting_id": meeting_id}

@app.get("/api/status")
async def get_status():
    return {"active": active_ingestions}

@app.get("/api/meetings")
async def get_meetings():
    """Returns all ingested meetings stored in the database."""
    meetings = pipeline.list_meetings()
    return {"meetings": meetings}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handles ChatGPT style inputs. 
    Global mode uses vector search; Meeting-specific mode uses full transcript analysis.
    """
    if request.meeting_id:
        # Full transcript detailed RAG (Deep Insights)
        try:
            chunks = pipeline.store.get_meeting_chunks(request.meeting_id)
            if not chunks:
                return {"error": f"Meeting {request.meeting_id} not found."}
            
            report = pipeline.insights_gen.generate_single_meeting_insights(
                meeting_id=request.meeting_id,
                chunks=chunks, 
                custom_prompt=request.query
            )
            return {"answer": report, "sources": [request.meeting_id], "confidence": 1.0}
        except Exception as e:
            return {"error": str(e)}
        
    else:
        # Generic Vector Search RAG
        try:
            result = pipeline.query(request.query)
            return {"answer": result["answer"], "sources": result["sources"], "confidence": result["confidence"]}
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
