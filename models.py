from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Data Models ---

class RelevantCriteria(BaseModel):
    criteria: str

class RelevanceResult(BaseModel):
    chain_of_thought: str = Field(..., description="Step-by-step reasoning process evaluating the fit")
    relevanceScore: int = Field(..., description="Score from 0 to 100")
    summary: str = Field(..., description="Short summary for the card")
    criteria_matched: List[str] = Field(..., description="List of criteria matched (e.g. 'Project Type', 'Tech Stack')")

class KeyFile(BaseModel):
    path: str
    reason: str

class KeyFilesResult(BaseModel):
    thought_process: str = Field(..., description="Explanation of how the key files were selected")
    files: List[KeyFile]

class AuditPoint(BaseModel):
    title: str
    description: str
    severity: str = Field(..., description="high, medium, or low")
    reasoning: str = Field(..., description="Why this issue is important and how it affects the system")
    evidence_snippet: str = Field(..., description="Verbatim code snippet proving this finding")
    file_path: str = Field(..., description="The name or path of the file where the evidence was found")

class AuditReport(BaseModel):
    items: List[AuditPoint]

class QuestionItem(BaseModel):
    question: str = "N/A"
    context: str = "N/A"
    reference_answer: str = "N/A"
    difficulty: str = Field("intermediate", description="beginner, intermediate, or expert")
    source_file: str = Field("N/A", description="The path of the file from the repo that triggered this question")
    selection_reason: str = Field("N/A", description="Why this specific code part/point was chosen for the question")
    jd_relation: str = Field("N/A", description="How this point relates to the Job Description requirements")

class InterviewQuestions(BaseModel):
    questions: List[QuestionItem]

class FileNode(BaseModel):
    path: str
    type: str
    size: Optional[int] = 0
    url: Optional[str] = None

class JDPillar(BaseModel):
    pillar_name: str = Field(..., description="A specific requirement from the JD (e.g., 'React Experience')")
    description: str = Field(..., description="Description of this hiring mandate")
    evidence_found: str = Field(..., description="Specific evidence from the candidate's repos matching this pillar")
    is_satisfied: bool = Field(..., description="Whether the candidate shows significant evidence for this pillar")
    top_repos: List[str] = Field(..., description="Names of 2-3 repos that best prove this skill")

class PillarSearchReport(BaseModel):
    hiring_rubric_summary: str = Field(..., description="The criteria extracted from the JD for evaluating this profile")
    pillars: List[JDPillar]
    unrelated_repos: List[str] = Field(..., description="Repositories that do not match any JD pillars")

class RepoStructure(BaseModel):
    owner: str
    repo: str
    files: List[FileNode]

class RepoSummary(BaseModel):
    name: str
    description: Optional[str] = ""
    language: Optional[str] = ""
    topics: List[str] = []
    size: int = 0
    updated_at: str
    url: str
    default_branch: str
    heuristic_score: int = 0

class AnalysisResult(BaseModel):
    relevance: RelevanceResult
    key_files: KeyFilesResult
    audit: List[AuditPoint]
    questions: List[str]

# --- Fail-Safe Registry ---
# Provides guaranteed default instances when LLM output is corrupted

FAIL_SAFE_DEFAULTS = {
    AuditReport: lambda: AuditReport(items=[]),
    PillarSearchReport: lambda: PillarSearchReport(
        hiring_rubric_summary="Analysis error", 
        pillars=[], 
        unrelated_repos=[]
    ),
    RelevanceResult: lambda: RelevanceResult(
        chain_of_thought="JSON Fail", 
        relevanceScore=0, 
        summary="Error", 
        criteria_matched=[]
    ),
    KeyFilesResult: lambda: KeyFilesResult(
        thought_process="JSON Fail", 
        files=[]
    ),
    InterviewQuestions: lambda: InterviewQuestions(
        questions=[]
    )
}
