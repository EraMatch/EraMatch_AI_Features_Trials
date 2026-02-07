import os
import json
import requests
import ollama
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
    question: str
    context: str
    reference_answer: str
    difficulty: str = Field(..., description="beginner, intermediate, or expert")
    source_file: str = Field(..., description="The path of the file from the repo that triggered this question")
    selection_reason: str = Field(..., description="Why this specific code part/point was chosen for the question")
    jd_relation: str = Field(..., description="How this point relates to the Job Description requirements")

class InterviewQuestions(BaseModel):
    questions: List[QuestionItem]

class FileNode(BaseModel):
    path: str
    type: str
    size: Optional[int] = 0
    url: Optional[str] = None

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

class AnalysisResult(BaseModel):
    relevance: RelevanceResult
    key_files: KeyFilesResult
    audit: List[AuditPoint]
    questions: List[str]

# --- GitHub Service ---

GITHUB_API_BASE = 'https://api.github.com'
IGNORED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.lock', '.json', '.toml', '.yml', '.yaml', '.xml', '.csv'}
IGNORED_DIRS = {'node_modules', 'dist', 'build', '.git', '__pycache__', 'venv', 'env', '.idea', '.vscode'}

def parse_github_url(url: str):
    try:
        parts = url.rstrip('/').split('/')
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    except:
        pass
    return None, None

def fetch_user_repos(username: str, token: str = "") -> List[RepoSummary]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f"{GITHUB_API_BASE}/users/{username}/repos?sort=updated&per_page=15"
    resp = requests.get(url, headers=headers)
    
    if not resp.ok:
        raise Exception(f"Failed to fetch user repos: {resp.status_code} {resp.text}")
    
    repos = []
    for r in resp.json():
        if r.get('fork', False): 
            continue
            
        repos.append(RepoSummary(
            name=r['name'],
            description=r.get('description') or "",
            language=r.get('language') or "Unknown",
            topics=r.get('topics', []),
            size=r.get('size', 0),
            updated_at=r['updated_at'],
            url=r['html_url'],
            default_branch=r.get('default_branch', 'main')
        ))
    return repos

def rank_repos_by_heuristics(repos: List[RepoSummary], jd_text: str) -> List[RepoSummary]:
    """Fast, metadata-based ranking to prioritize candidates for LLM analysis."""
    jd_lower = jd_text.lower()
    
    def score_repo(repo: RepoSummary):
        score = 0
        
        # 1. Language Match (High Weight)
        if repo.language and repo.language.lower() in jd_lower:
            score += 40
            
        # 2. Topic/Name Keyword Match (High Weight)
        keywords = repo.topics + repo.name.replace('-', ' ').replace('_', ' ').split()
        for kw in keywords:
            if len(kw) > 2 and kw.lower() in jd_lower:
                score += 20
                break # Cap keyword bonus per repo for metadata stage
        
        # 3. Maturity Check (Size/Health)
        if repo.size > 500: # Over ~500KB suggests more than just one file
            score += 15
        elif repo.size > 50:
            score += 5
            
        # 4. Notebook Detection (Data/AI Roles)
        if "notebook" in jd_lower or "data" in jd_lower:
            # Check topics or name for notebook indicators
            if any(term in repo.name.lower() or term in " ".join(repo.topics).lower() for term in ["notebook", "jupyter", "analysis", "rag"]):
                score += 20
                
        # 5. Recency (Tie breaker)
        # (Already sorted by updated_at from GitHub API, so implicit in original order)
        
        return score

    # Sort repos by our heuristic score in descending order
    return sorted(repos, key=score_repo, reverse=True)

def fetch_file_content(file_node: FileNode, token: str = "") -> str:
    headers = {'Accept': 'application/vnd.github.v3.raw'}
    if token:
        headers['Authorization'] = f'token {token}'
        
    try:
        resp = requests.get(file_node.url, headers=headers)
        if not resp.ok:
            raise Exception(f"Failed to fetch file content: {file_node.path}")
        
        text = resp.text
        if file_node.path.endswith('.ipynb'):
            try:
                import json
                notebook = json.loads(text)
                content = []
                for i, cell in enumerate(notebook.get('cells', [])):
                    cell_type = cell.get('cell_type', '')
                    source = "".join(cell.get('source', []))
                    if cell_type == 'markdown':
                        content.append(f"\n[CELL {i+1} - MARKDOWN]\n{source}\n")
                    elif cell_type == 'code':
                        content.append(f"\n[CELL {i+1} - CODE]\n{source}\n")
                return "".join(content)
            except Exception as e:
                return f"# Error parsing notebook: {e}\n\nRaw Content:\n{text[:1000]}"
                
        return text
    except Exception as e:
        print(f"Error fetching file {file_node.path}: {e}")
        return ""

def fetch_repo_structure(username: str, repo: str, token: str = "") -> tuple[RepoStructure, str]:
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'
        
    try:
        base_url = f"{GITHUB_API_BASE}/repos/{username}/{repo}/git/trees/main?recursive=1"
        resp = requests.get(base_url, headers=headers)
        if resp.status_code == 404:
            base_url = f"{GITHUB_API_BASE}/repos/{username}/{repo}/git/trees/master?recursive=1"
            resp = requests.get(base_url, headers=headers)
            
        if not resp.ok:
             return RepoStructure(owner=username, repo=repo, files=[]), "Failed to fetch structure"

        data = resp.json()
        files = []
        readme_url = None
        main_notebook_url = None
        
        for item in data.get('tree', []):
            if item['type'] == 'blob':
                if any(ignored in item['path'] for ignored in IGNORED_DIRS):
                    continue
                if any(item['path'].endswith(ext) for ext in IGNORED_EXTENSIONS):
                    continue
                    
                f = FileNode(
                    path=item['path'],
                    type='blob',
                    size=item.get('size', 0),
                    url=item.get('url')
                )
                files.append(f)
                
                path_lower = f.path.lower()
                if path_lower == 'readme.md':
                    readme_url = f.url
                
                if not main_notebook_url and f.path.endswith('.ipynb'):
                    main_notebook_url = f.url
                    
        description_text = ""
        def simple_fetch(url):
            h = headers.copy()
            h['Accept'] = 'application/vnd.github.v3.raw'
            return requests.get(url, headers=h).text

        if readme_url:
            try:
                description_text = simple_fetch(readme_url)[:4000]
            except: pass
        
        if not description_text and main_notebook_url:
            try:
                nb_content = fetch_file_content(FileNode(path="dummy.ipynb", type="blob", url=main_notebook_url), token)
                description_text = "No README found. Notebook Context:\n" + nb_content[:2000]
            except: pass
            
        return RepoStructure(owner=username, repo=repo, files=files), description_text

    except Exception as e:
        print(f"Error: {e}")
        return RepoStructure(owner=username, repo=repo, files=[]), ""

# --- LLM Services ---

def load_prompt(prompt_name: str, **kwargs) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{prompt_name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
            formatted = template.format(**kwargs)
            if not formatted.strip():
                raise ValueError(f"Prompt {prompt_name} resulted in an empty string")
            return formatted
    except Exception as e:
        error_msg = f"Error loading prompt {prompt_name}: {e}"
        print(error_msg)
        raise Exception(error_msg)

def query_ollama(model: str, prompt: str, schema_class=None, host: str = 'http://127.0.0.1:11434', options: Dict[str, Any] = None) -> Any:
    if options is None:
        options = {"temperature": 0}
    
    try:
        client = ollama.Client(host=host)
        response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}], options=options)
        content = response['message']['content']
        
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        elif "<think>" in content: 
            content = content.split("<think>")[-1].strip()

        import re
        
        # 2. Extract JSON block (Aggressive Extraction)
        # Search for the widest possible block starting with { or [ and ending with } or ]
        json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        
        if json_match:
            json_content = json_match.group(1)
        else:
            # Fallback to old behavior if regex fails but braces exist
            json_content = content
            if "{" in json_content:
                start = json_content.find("{")
                end = json_content.rfind("}")
                if start != -1 and end != -1:
                    json_content = json_content[start:end+1]
        
        # 3. Handle list-root JSON if schema expects an object
        # If the model returned a raw list [...] but we expect an object {"questions": [...]}, wrap it
        if json_content.strip().startswith("[") and schema_class and hasattr(schema_class, 'model_fields'):
            first_field = list(schema_class.model_fields.keys())[0]
            json_content = f'{{"{first_field}": {json_content}}}'

        if schema_class:
            try:
                return schema_class.model_validate_json(json_content)
            except Exception as ve:
                # Last-ditch: clean common JSON errors (trailing commas, etc.)
                try:
                    import json
                    cleaned_json = re.sub(r',\s*([\]}])', r'\1', json_content)
                    return schema_class.model_validate_json(cleaned_json)
                except:
                    snippet = json_content[:200] + "..." if len(json_content) > 200 else json_content
                    raise Exception(f"JSON Validation Error: {str(ve)}\nAttempted to parse: {snippet}")
        return json_content
    except Exception as e:
        raise Exception(f"Ollama Error ({model}): {str(e)}")

def pull_ollama_model(model: str, host: str = 'http://127.0.0.1:11434'):
    try:
        client = ollama.Client(host=host)
        for progress in client.pull(model, stream=True):
            pass
        return True
    except Exception as e:
        raise Exception(f"Failed to pull {model}: {str(e)}")

def check_relevance(jd: str, file_list: str, readme_content: str, model: str, ollama_host: str) -> RelevanceResult:
    prompt = load_prompt("relevance", jd=jd[:1500], readme_content=readme_content[:2000], file_list=file_list[:3000])
    return query_ollama(model, prompt, RelevanceResult, host=ollama_host)

def identify_key_files(file_list: str, readme_content: str, model: str, ollama_host: str) -> KeyFilesResult:
    prompt = load_prompt("key_files", readme_content=readme_content[:1000], file_list=file_list[:5000])
    return query_ollama(model, prompt, KeyFilesResult, host=ollama_host)

def perform_deep_audit(code_context: str, model: str, ollama_host: str) -> List[AuditPoint]:
    prompt = load_prompt("audit", code_context=code_context[:30000])
    result = query_ollama(model, prompt, AuditReport, host=ollama_host)
    return verify_audit_findings(result.items, code_context)

def verify_audit_findings(findings: List[AuditPoint], code_context: str) -> List[AuditPoint]:
    verified = []
    context_collapsed = "".join(code_context.split())
    for f in findings:
        snippet = f.evidence_snippet.strip()
        if not snippet: continue
        if snippet in code_context:
            verified.append(f)
        else:
            snippet_collapsed = "".join(snippet.split())
            if snippet_collapsed in context_collapsed:
                verified.append(f)
    return verified

def synthesize_questions(audit_report: List[AuditPoint], model: str, ollama_host: str) -> InterviewQuestions:
    # Truncate evidence snippets to prevent the prompt from becoming too large/distracting for the synthesizer
    summarized_points = []
    for a in audit_report:
        snippet = (a.evidence_snippet[:200] + "...") if len(a.evidence_snippet) > 200 else a.evidence_snippet
        summarized_points.append(f"- Finding: {a.title}\n  Concept: {a.description}\n  Source File: {a.file_path}\n  Evidence (Truncated): {snippet}")
    
    audit_summary = "\n".join(summarized_points)
    prompt = load_prompt("synthesis", audit_summary=audit_summary)
    return query_ollama(model, prompt, InterviewQuestions, host=ollama_host)
