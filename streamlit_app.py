import streamlit as st
import os
import time
from concurrent.futures import ThreadPoolExecutor
from services import (
    parse_github_url, 
    fetch_user_repos,
    categorize_profile_parallel,
    rank_repos_by_heuristics,
    fetch_repo_structure, 
    fetch_file_content,
    check_relevance,
    identify_key_files,
    perform_deep_audit,
    synthesize_questions,
    pull_ollama_model
)
from models import AnalysisResult, PillarSearchReport

# Page Config
st.set_page_config(
    page_title="RecruitAI 2.0 - Agentic Scanner",
    page_icon="[Scanner]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional White/Green Theme
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "static", "css", "styles.css")
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

load_css()

# Initialize session state for models if not present
if "model_filter" not in st.session_state: st.session_state.model_filter = "kimi-k2.5:cloud"
if "model_map" not in st.session_state: st.session_state.model_map = "kimi-k2.5:cloud"
if "model_audit" not in st.session_state: st.session_state.model_audit = "deepseek-v3.1:671b-cloud"
if "model_synth" not in st.session_state: st.session_state.model_synth = "deepseek-v3.1:671b-cloud"

# Sidebar
with st.sidebar:
    st.title("RecruitAI 2.0")
    st.caption("Agentic User Scanner")
    
    st.subheader("Configuration")
    gh_token = st.text_input("GitHub Token (Optional)", type="password", help="Higher rate limits")
    ollama_host = st.text_input("Ollama Host", value="http://127.0.0.1:11434")

    st.subheader("Model Selection")

    st.session_state.model_filter = st.text_input("Filter Model", value=st.session_state.model_filter)
    st.session_state.model_map = st.text_input("Mapper Model", value=st.session_state.model_map)
    st.session_state.model_audit = st.text_input("Audit Model", value=st.session_state.model_audit)
    st.session_state.model_synth = st.text_input("Synth Model", value=st.session_state.model_synth)

    col_pull, col_local = st.columns(2)
    
    with col_pull:
        pull_btn = st.button("Pull All Models", use_container_width=True)
    
    with col_local:
        if st.button("Switch to Local", help="Auto-select optimized local models", use_container_width=True):
            st.session_state.model_filter = "qwen2.5-coder:1.5b"
            st.session_state.model_map = "llama3.2"
            st.session_state.model_audit = "qwen2.5-coder:7b"
            st.session_state.model_synth = "deepseek-r1:1.5b"
            st.rerun()

    if pull_btn:
        with st.status("Pulling models on remote host...", expanded=True) as s:
            models = [st.session_state.model_filter, st.session_state.model_map, st.session_state.model_audit, st.session_state.model_synth]
            models = list(set(models)) # Deduplicate
            
            for m in models:
                s.write(f"Pulling **{m}**...")
                try:
                    pull_ollama_model(m, ollama_host)
                    s.write(f"Checked: {m}")
                except Exception as e:
                    s.write(f"Failed: {m}: {e}")
            s.update(label="Ollama models verified", state="complete")

# Main Content
st.title("Candidate Deep Scan")
st.markdown("Enter a GitHub username to automatically scan their portfolio, select the most relevant repo, and generate interview questions.")

col1, col2 = st.columns([1, 1])

with col1:
    jd_text = st.text_area("Job Description", height=200, placeholder="Paste JD here...")

with col2:
    username = st.text_input("GitHub Username", placeholder="e.g. antirez")
    st.caption("We will scan their recent public repositories.")
    run_btn = st.button("Start Agentic Scan", type="primary", use_container_width=True)

# Results Area
if run_btn:
    if not jd_text or not username:
        st.error("Please provide both Job Description and Username.")
        st.stop()

    client = None

    # --- PHASE 1: SCANNING (TOURNAMENT) ---
    st.subheader(f"Analyzing {username}'s Portfolio")
    
    timings = {}
    tournament_container = st.container()
    
    with st.status("Scanning Repositories...", expanded=True) as status:
        start_tournament = time.perf_counter()
        status.write("Fetching user repositories...")
        try:
            repos = fetch_user_repos(username, gh_token)
            status.write(f"Found {len(repos)} public repositories.")
        except Exception as e:
            st.error(str(e))
            st.stop()
            
        # --- PHASE 1: JD-DRIVEN PILLAR SEARCH ---
        status.write("Pre-filtering candidates & analyzing JD mandates...")
        start_fingerprint = time.perf_counter()
        
        try:
            # Trial 2c: Pre-filter top 50
            pre_filtered_repos = rank_repos_by_heuristics(repos, jd_text)[:50]
            
            pillar_report = categorize_profile(pre_filtered_repos, jd_text, st.session_state.model_filter, ollama_host)
            end_fingerprint = time.perf_counter()
            timings["PillarSearch"] = end_fingerprint - start_fingerprint
            
            st.markdown(f"""
            <div style="background:#f0f7ff; border:1px solid #cce3ff; border-radius:8px; padding:16px; margin-bottom:20px; border-left: 5px solid #0969da;">
                <h4 style="margin:0; color:#1f2328; font-size:1.1em;">Target Hiring Rubric</h4>
                <p style="margin:8px 0; color:#424a53; font-size:0.95em;">{pillar_report.hiring_rubric_summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Explore Pillar Alignment & Evidence", expanded=False):
                for pillar in pillar_report.pillars:
                    status_icon = "Match Found" if pillar.is_satisfied else "Gap Identified"
                    status_color = "#1a7f37" if pillar.is_satisfied else "#9a6700"
                    
                    st.markdown(f"### <span style='color:{status_color};'>{status_icon}: {pillar.pillar_name}</span>", unsafe_allow_html=True)
                    st.caption(pillar.description)
                    st.info(f"**Evidence:** {pillar.evidence_found}")
                    if pillar.top_repos:
                        st.markdown(f"**Key Repositories:** {', '.join([f'`{r}`' for r in pillar.top_repos])}")
                
                if pillar_report.unrelated_repos:
                    st.markdown("**Other/Unrelated Projects:**")
                    st.caption(", ".join(pillar_report.unrelated_repos))

            # Extract names for Domain Boost
            target_repos_names = []
            for pillar in pillar_report.pillars:
                if pillar.is_satisfied:
                    target_repos_names.extend(pillar.top_repos)

            # PHASE 1b: THE HYBRID HUB
            repos_to_scout = rank_repos_by_heuristics(repos, jd_text, target_repos_names)[:6]
            
            status.write(f"Scouting {len(repos_to_scout)} repositories...")

        except Exception as e:
            status.write(f"Pillar search fail: {str(e)}. Using heuristics...")
            pillar_report = PillarSearchReport(
                hiring_rubric_summary="Pillar search failed. Falling back to simple heuristics.",
                pillars=[],
                unrelated_repos=[r.name for r in repos]
            )
            repos_to_scout = rank_repos_by_heuristics(repos, jd_text)[:6]
            target_repos_names = []

        # --- PHASE 2: COMPETITIVE TOURNAMENT ---
        status.write("Starting Competitive Tournament (Trial 2c Style)...")
        start_tournament = time.perf_counter()
        
        best_repo = None
        best_score = -1
        best_repo_structure = None
        best_repo_readme = None
        best_repo_result = None
        
        # Concurrency Tuning
        is_cloud_model = ":cloud" in st.session_state.model_filter.lower()
        workers = min(6, len(repos_to_scout)) if is_cloud_model else min(3, len(repos_to_scout))
        
        from concurrent.futures import as_completed
        
        progress_bar = st.progress(0)
        
        def scout_task(repo, filter_model):
            try:
                structure, readme_content = fetch_repo_structure(username, repo.name, gh_token)
                file_list_str = "\n".join([f.path for f in structure.files])[:5000]
                result = check_relevance(jd_text, file_list_str, readme_content, filter_model, ollama_host)
        batch_size = 6 if ":cloud" in st.session_state.model_filter.lower() else 3
        progress_bar = st.progress(0)
        
        for i in range(0, len(repos_to_scout), batch_size):
            batch = repos_to_scout[i:i + batch_size]
            status.write(f"Analyzing batch {i//batch_size + 1} ({len(batch)} repos)...")
            
            def process_repo(repo):
                try:
                    struct, readme = fetch_repo_structure(username, repo.name, github_token)
                    res = check_relevance(jd_text, str([f.path for f in struct.files])[:5000], readme, st.session_state.model_filter, ollama_host)
                    return repo, struct, readme, res, None
                except Exception as ex:
                    return repo, None, None, None, str(ex)

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = list(executor.map(process_repo, batch))
            
            for repo, struct, readme, res, err in results:
                if err:
                    status.write(f"Skipped {repo.name}: {err}")
                    continue
                    
                status.write(f"Evaluated **{repo.name}**: {res.relevanceScore}%")
                
                with tournament_container:
                    color = "#238636" if res.relevanceScore > 70 else "#9a6700" if res.relevanceScore > 40 else "#656d76"
                    boost_badge = '<span style="background:#fff8c5; color:#735c0f; border:1px solid #d4a72c; padding:2px 8px; border-radius:12px; font-size:0.75em; margin-right:4px; font-weight:bold;">DOMAIN BOOST</span>' if repo.name in target_repos_names else ""
                    st.markdown(f'<div class="repo-card" style="border-left: 5px solid {color}"><strong>{repo.name} {boost_badge}</strong> <br>Score: {res.relevanceScore}</div>', unsafe_allow_html=True)
                
                if res.relevanceScore > best_score:
                    best_score = res.relevanceScore
                    best_repo = repo
                    best_repo_structure = struct
                    best_repo_readme = readme
                    best_repo_result = res
                
                if best_score >= 80:
                    break
            
            progress_bar.progress(min(1.0, (i + batch_size) / len(repos_to_scout)))
            if best_score >= 80:
                status.write(f"Superiority Exit: Found {best_repo.name} ({best_score}%)")
                break
            
        if not best_repo or best_score < 40:
            status.update(label="No relevant repositories found.", state="error")
            st.error("No repositories matched the JD significantly.")
            st.stop()
            
        end_tournament = time.perf_counter()
        timings["Tournament"] = end_tournament - start_tournament
        status.write(f"Winner: {best_repo.name} ({best_score}%)")
        status.update(label="Tournament Complete", state="complete", expanded=False)

    # --- PHASE 2: DEEP ANALYSIS (WINNER) ---
    st.divider()
    st.subheader(f"Analyzing Winner: {best_repo.name}")
    
    with st.status("Deep Analysis in Progress...", expanded=True) as status:
        # TIMING: Start Deep Analysis
        start_mapping = time.perf_counter()
        
        # MAP
        status.write(f"Identifying key files with {st.session_state.model_map}...")
        
        # HYPER-TURBO: Optimize file list truncation
        # Prioritize technical engineering files to stay under token limits and speed up LLM
        tech_exts = {'.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp', '.rb', '.c', '.h', '.cs', '.php', '.sql'}
        all_files = best_repo_structure.files
        prioritized_files = [f for f in all_files if any(f.path.endswith(ext) for ext in tech_exts)]
        if not prioritized_files: prioritized_files = all_files # Fallback
        
        # Take the top 100 most "important-looking" deep files
        file_list_str = "\n".join([f.path for f in prioritized_files[:100]])
        
        key_files_result = identify_key_files(file_list_str, best_repo_readme, st.session_state.model_map, ollama_host)
        
        if not key_files_result.files:
             st.error("Could not identify key files.")
             st.stop()
             
        end_mapping = time.perf_counter()
        timings["Mapping"] = end_mapping - start_mapping
             
        with st.expander("Key File Selection Logic", expanded=True):
            st.info(key_files_result.thought_process)
             
        # HYPER-TURBO: Parallelize file content fetching
        full_code_context = ""
        status.write(f"Fetching content for {len(key_files_result.files)} key files in parallel...")
        
        def fetch_task(kf):
            node = next((f for f in best_repo_structure.files if f.path == kf.path), None)
            if not node:
                 node = next((f for f in best_repo_structure.files if kf.path in f.path), None)
            
            if node:
                content = fetch_file_content(node, gh_token)
                return f"\n\n--- FILE: {node.path} ---\n{content}\n", node.path, kf.reason
            return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            fetch_results = list(executor.map(fetch_task, key_files_result.files))
            
        for res in fetch_results:
            if res:
                context_chunk, path, reason = res
                full_code_context += context_chunk
                st.markdown(f"- Found `{path}` ({reason})")
        
        # AUDIT
        start_audit = time.perf_counter()
        status.write(f"Auditing code with {st.session_state.model_audit}...")
        audit_report = perform_deep_audit(full_code_context, st.session_state.model_audit, ollama_host)
        end_audit = time.perf_counter()
        timings["Audit"] = end_audit - start_audit
        status.write(f"Found {len(audit_report)} verifiable audit points.")
        
        # SYNTHESIS
        start_synth = time.perf_counter()
        status.write(f"Synthesizing questions with {st.session_state.model_synth}...")
        questions = synthesize_questions(audit_report, st.session_state.model_synth, ollama_host)
        end_synth = time.perf_counter()
        timings["Synthesis"] = end_synth - start_synth
        
        status.update(label="Deep Analysis Complete", state="complete", expanded=False)

    # Performance Summary
    st.info(f"Analysis Timeline: " + " | ".join([f"{k}: {v:.1f}s" for k, v in timings.items()]))

    # --- DISPLAY FINAL RESULTS ---
    
    # 1. Winner Context & Export
    st.success(f"**Winner Selected: {best_repo.name} ({best_score}%)**")
    with st.expander("Why this repository was chosen exactly", expanded=False):
        st.info(best_repo_result.chain_of_thought)

    # Prepare JSON Export Package
    import json
    export_package = {
        "candidate_username": username,
        "job_description_context": jd_text,
        "hiring_rubric": pillar_report.model_dump(),
        "selected_repo": {
            "name": best_repo.name,
            "url": best_repo.url,
            "relevance_score": best_score,
            "selection_reasoning": best_repo_result.chain_of_thought,
            "criteria_matched": best_repo_result.criteria_matched
        },
        "interview_questions": [q.model_dump() for q in questions.questions]
    }
    
    st.download_button(
        label="Download ACE Interview Package (JSON)",
        data=json.dumps(export_package, indent=2),
        file_name=f"{username}_recruiting_package.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()
    
    col_audit, col_quest = st.columns(2)

    with col_audit:
        st.subheader("Technical Audit")
        for point in audit_report:
            severity_color = "#d1242f" if point.severity == "high" else "#ce5019" if point.severity == "medium" else "#0969da"
            with st.expander(f"**{point.title}**", expanded=True):
                    st.markdown(f"<span style='color:{severity_color}; font-weight:bold; font-size:0.8em; text-transform:uppercase;'>{point.severity}</span>", unsafe_allow_html=True)
                    st.write(point.description)
                    
                    st.markdown(f"<strong style='color:#57606a; font-size:0.8em;'>LOGIC:</strong> <span style='color:#424a53; font-size:0.9em;'>{point.reasoning}</span>", unsafe_allow_html=True)
                    
                    if point.evidence_snippet:
                        st.markdown(f"""
                        <div style="background:#f6f8fa; border:1px solid #d0d7de; border-radius:6px; padding:10px; margin-top:10px;">
                            <strong style="color:#57606a; font-size:0.75em; display:block; margin-bottom:4px;">EVIDENCE:</strong>
                            <code style="font-family:monospace; font-size:0.85em; color:#24292f;">{point.evidence_snippet}</code>
                        </div>
                        """, unsafe_allow_html=True)

    with col_quest:
        st.subheader("Interview Questions")
        for i, q_item in enumerate(questions.questions):
            difficulty_colors = {"beginner": "#dafbe1", "intermediate": "#fff8c5", "expert": "#ffebec"}
            diff_text_colors = {"beginner": "#1a7f37", "intermediate": "#9a6700", "expert": "#cf222e"}
            diff_bg = difficulty_colors.get(q_item.difficulty.lower(), "#f6f8fa")
            diff_text = diff_text_colors.get(q_item.difficulty.lower(), "#656d76")

            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 15px; background: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #d0d7de; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <div style="flex-shrink: 0; width: 32px; height: 32px; background: {diff_bg}; color: {diff_text}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">{i+1}</div>
                <div style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="color: #656d76; font-size: 12px; text-transform: uppercase; font-weight: 600;">Context: {q_item.context}</span>
                        <span style="background: {diff_bg}; color: {diff_text}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; text-transform: uppercase;">{q_item.difficulty}</span>
                    </div>
                    <div style="color: #1f2328; font-size: 15px; line-height: 1.5; font-weight: 600; margin-bottom: 8px;">{q_item.question}</div>
                    <div style="font-size: 11px; color: #656d76;">
                        <strong>Source:</strong> {q_item.source_file}<br>
                        <strong>JD Relation:</strong> {q_item.jd_relation}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Reference Answer", expanded=False):
                st.markdown(f"**Selection Logic:** {q_item.selection_reason}")
                st.info(q_item.reference_answer)
