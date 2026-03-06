import uuid
from typing import Dict, Any, Optional

# In-memory storage for jobs (use Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "result": None,
        "error": None
    }
    return job_id

def update_job(job_id: str, status: str, result: Any = None, error: str = None):
    if job_id in jobs:
        jobs[job_id]["status"] = status
        if result is not None:
            jobs[job_id]["result"] = result
        if error is not None:
            jobs[job_id]["error"] = error

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return jobs.get(job_id)
