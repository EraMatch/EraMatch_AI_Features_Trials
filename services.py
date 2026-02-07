import os
import json
import requests
import ollama
import re
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# --- Data Models ---

from models import (
    RelevantCriteria, RelevanceResult, KeyFile, KeyFilesResult,
    AuditPoint, AuditReport, QuestionItem, InterviewQuestions,
    FileNode, JDPillar, PillarSearchReport, RepoStructure,
    RepoSummary, AnalysisResult, FAIL_SAFE_DEFAULTS
)

# --- GitHub Service ---

GITHUB_API_BASE = 'https://api.github.com'
IGNORED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.lock', '.json', '.toml', '.yml', '.yaml', '.xml', '.csv', '.log', '.bin'}
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
    
    # Increase to 100 to get a full profile view
    url = f"{GITHUB_API_BASE}/users/{username}/repos?sort=updated&per_page=100"
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

def categorize_profile(repos: List[RepoSummary], jd_text: str, model: str, ollama_host: str) -> PillarSearchReport:
    """Analyze entire profile to extract hiring pillars and map relevant repos."""
    repo_list_str = "\n".join([f"- {r.name}: {r.description[:500]} (Lang: {r.language}, Topics: {r.topics})" for r in repos])
    prompt = load_prompt("categorization", repo_list=repo_list_str, jd=jd_text[:2000])
    return query_ollama(model, prompt, PillarSearchReport, host=ollama_host)

def categorize_profile_parallel(repos: List[RepoSummary], jd_text: str, model: str, ollama_host: str) -> PillarSearchReport:
    """ULTIMATE TURBO: Split repos into chunks and analyze pillars in parallel."""
    # Split 30 repos into 3 chunks of 10
    chunk_size = 10
    chunks = [repos[i:i + chunk_size] for i in range(0, len(repos), chunk_size)]
    
    reports: List[PillarSearchReport] = []
    
    def process_chunk(chunk_repos):
        repo_list_str = "\n".join([f"- {r.name}: {r.description[:500]} (Lang: {r.language}, Topics: {r.topics})" for r in chunk_repos])
        # Force extreme conciseness in the prompt for speed
        prompt = load_prompt("categorization", repo_list=repo_list_str, jd=jd_text[:2000])
        return query_ollama(model, prompt, PillarSearchReport, host=ollama_host)

    with ThreadPoolExecutor(max_workers=3) as executor:
        reports = list(executor.map(process_chunk, chunks))
    
    # Merge reports
    merged_report = PillarSearchReport(
        hiring_rubric_summary=reports[0].hiring_rubric_summary if reports else "Analysis complete.",
        pillars=[],
        unrelated_repos=[]
    )
    
    seen_pillars = {}
    for r in reports:
        for p in r.pillars:
            if p.pillar_name not in seen_pillars:
                seen_pillars[p.pillar_name] = p
                merged_report.pillars.append(p)
            else:
                # Merge top_repos for existing pillars
                existing = seen_pillars[p.pillar_name]
                existing.top_repos = list(set(existing.top_repos + p.top_repos))
                existing.is_satisfied = existing.is_satisfied or p.is_satisfied
        
        merged_report.unrelated_repos = list(set(merged_report.unrelated_repos + r.unrelated_repos))
        
    return merged_report

def rank_repos_by_heuristics(repos: List[RepoSummary], jd_text: str, target_repos: List[str] = None) -> List[RepoSummary]:
    """Fast, metadata-based ranking with Domain Multipliers and Intrigue Bonuses."""
    jd_lower = jd_text.lower()
    target_repos = target_repos or []
    
    def score_repo(repo: RepoSummary):
        base_score = 0
        
        # 1. Language Match (Core Requirement)
        target_lang_match = False
        if repo.language and repo.language.lower() in jd_lower:
            base_score += 30
            target_lang_match = True
            
        # 2. Topic/Name Keyword Match
        keywords = repo.topics + repo.name.replace('-', ' ').replace('_', ' ').split()
        for kw in keywords:
            if len(kw) > 2 and kw.lower() in jd_lower:
                base_score += 20
                break
        
        # 3. Maturity Check (Size/Health)
        is_large = repo.size > 1000 # > 1MB
        if repo.size > 500:
            base_score += 15
        elif repo.size > 50:
            base_score += 5
            
        # 4. Notebook Detection
        if "notebook" in jd_lower or "data" in jd_lower:
            if any(term in repo.name.lower() or term in " ".join(repo.topics).lower() for term in ["notebook", "jupyter", "analysis", "rag"]):
                base_score += 20
        
        final_score = base_score
        
        # 5. SANITY THRESHOLD & DOMAIN BOOST
        # Only boost if the repo has some base-level relevance (avoids boosting junk/empty matches)
        if repo.name in target_repos and base_score > 10:
            final_score += 40
        
        # 6. INTRIGUE BONUS (Safe Net)
        # If it's a large repo in the right language but LLM missed it or it has no description
        if repo.name not in target_repos and target_lang_match and is_large:
            final_score += 20
                
        repo.heuristic_score = final_score
        return final_score

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
        
        # 1. Strip markdown if present
        json_content = content
        if "```json" in content:
            json_content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content:
            # Finding the largest block
            blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
            if blocks:
                json_content = max(blocks, key=len)
        
        # 2. Extract JSON block (Regex Search)
        # Search for the widest possible block starting with { or [ and ending with } or ]
        json_match = re.search(r'(\{.*\}|\[.*\])', json_content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            # Second attempt: check original content if markdown split was too aggressive
            json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
        
        # 3. Last-ditch: fallback if regex fails but braces exist
        if "{" in json_content:
            start = json_content.find("{")
            end = json_content.rfind("}")
            if start != -1 and end != -1:
                json_content = json_content[start:end+1]
        
        # 4. Handle list-root JSON if schema expects an object
        if json_content.strip().startswith("[") and schema_class and hasattr(schema_class, 'model_fields'):
            first_field = list(schema_class.model_fields.keys())[0]
            json_content = f'{{"{first_field}": {json_content}}}'

        if schema_class:
            try:
                # ULTIMATE TURBO: Strip any markdown code blocks if the model ignored instructions
                if "```json" in content:
                    json_content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_content = content.split("```")[1].split("```")[0].strip()
                else:
                    json_content = content.strip()

                return schema_class.model_validate_json(json_content)
            except Exception as ve:
                print(f"\n--- DEBUG: LLM OUTPUT ({model}) ---\n{content}\n------------------------\n")
                
                # REPAIR PHASE: Try standard cleaning
                try:
                    cleaned_json = re.sub(r',\s*([\]}])', r'\1', json_content)
                    return schema_class.model_validate_json(cleaned_json)
                except:
                    # PANIC PHASE: If it's a critical structural failure, return a safe default
                    # from the FAIL_SAFE_DEFAULTS registry to keep the engine running.
                    if schema_class in FAIL_SAFE_DEFAULTS:
                        print(f"FAILS SAFE: Returning default factory instance for {schema_class.__name__} due to JSON corruption.")
                        return FAIL_SAFE_DEFAULTS[schema_class]()
                    
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
    # TURBO MODE: Reduce context window to 12k to speed up local inference significantly
    prompt = load_prompt("audit", code_context=code_context[:12000])
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
