from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
from typing import List, Optional, Dict, Any

from services.pdf_parser import extract_text_from_pdf, count_tokens, get_pdf_page_count
from services.ai_service import generate_questions_ai, extract_questions_ai
from services.job_service import create_job, update_job, get_job

app = FastAPI(title="Question Import Trial API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

async def background_process_questions(
    job_id: str, 
    content: bytes, 
    is_pdf: bool, 
    mode: str, # 'generate' or 'extract'
    **kwargs
):
    try:
        update_job(job_id, "processing")
        text = extract_text_from_pdf(content, force=True) if is_pdf else safe_decode(content)
        
        if mode == 'generate':
            questions = generate_questions_ai(
                text, 
                kwargs.get('mcq_count', 5), 
                kwargs.get('essay_count', 0), 
                kwargs.get('difficulty', 'Medium')
            )
        else:
            questions = extract_questions_ai(text)
            
        update_job(job_id, "completed", result=questions)
    except Exception as e:
        update_job(job_id, "failed", error=str(e))

@app.post("/api/import/sheet")
async def import_sheet(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV or Excel allowed.")
    
    return {
        "message": "Sheet processed",
        "questions": [
            {"id": 101, "type": "mcq", "title": "What is 2+2?", "options": ["3", "4", "5"], "correct": 1, "difficulty": "Easy"},
            {"id": 102, "type": "mcq", "title": "Which is a JS framework?", "options": ["React", "Django", "Flask"], "correct": 0, "difficulty": "Medium"}
        ]
    }

def safe_decode(content: bytes) -> str:
    """Try to decode bytes with common encodings."""
    # Try common encodings, including UTF-16 which powershell uses
    for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode('utf-8', errors='ignore')

@app.post("/api/import/generate")
async def generate_questions(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mcq_count: int = Form(5),
    essay_count: int = Form(0),
    difficulty: str = Form("Medium"),
    background: bool = Form(False)
):
    print(f"Generate Request: file={file.filename}, mcq={mcq_count}, essay={essay_count}, bg={background}")
    content = await file.read()
    is_pdf = file.filename.endswith('.pdf')
    
    if is_pdf:
        page_count = get_pdf_page_count(content)
        if page_count > 10 and not background:
            return {
                "status": "large_file",
                "page_count": page_count,
                "message": "Large file detected. Please confirm background processing."
            }

    if background:
        job_id = create_job()
        background_tasks.add_task(
            background_process_questions, 
            job_id, content, is_pdf, 'generate', 
            mcq_count=mcq_count, essay_count=essay_count, difficulty=difficulty
        )
        return {"status": "queued", "job_id": job_id}

    # Synchronous path for small files
    try:
        text = extract_text_from_pdf(content) if is_pdf else safe_decode(content)
        print(f"Extracted/Decoded text length: {len(text)}")
        questions = generate_questions_ai(text, mcq_count, essay_count, difficulty)
        return {"status": "completed", "questions": questions}
    except Exception as e:
        print(f"Error in generate_questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import/extract")
async def extract_questions(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background: bool = Form(False)
):
    print(f"Extract Request: file={file.filename}, bg={background}")
    content = await file.read()
    is_pdf = file.filename.endswith('.pdf')
    
    if is_pdf:
        page_count = get_pdf_page_count(content)
        if page_count > 10 and not background:
            return {
                "status": "large_file",
                "page_count": page_count,
                "message": "Large file detected. Please confirm background processing."
            }

    if background:
        job_id = create_job()
        background_tasks.add_task(background_process_questions, job_id, content, is_pdf, 'extract')
        return {"status": "queued", "job_id": job_id}

    # Synchronous path
    try:
        text = extract_text_from_pdf(content) if is_pdf else safe_decode(content)
        print(f"Extracted/Decoded text length: {len(text)}")
        questions = extract_questions_ai(text)
        return {"status": "completed", "questions": questions}
    except Exception as e:
        print(f"Error in extract_questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
