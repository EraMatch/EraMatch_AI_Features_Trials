"""
Phase 1: Generate and Validate High-Quality Job Descriptions

This script focuses ONLY on generating realistic, correct, and rich job descriptions.
No question generation. Pure JD quality with checkpoint & parallel processing.

Usage:
  # Using Ollama (default)
  python tools/generate_job_descriptions_phase1.py \
    --out data/phase1_job_descriptions.jsonl \
    --positions-file positions.md \
    --model ollama \
    --ollama-model llama3.1:8b

  # Using OpenRouter
  python tools/generate_job_descriptions_phase1.py \
    --out data/phase1_job_descriptions.jsonl \
    --model openrouter \
    --openrouter-key $OPENROUTER_API_KEY

Output: JSONL with {meta, job_description, status, error (if failed)}
"""
import argparse
import json
import os
import time
import threading
from typing import List, Dict, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================================
# VARIANT TEMPLATES (97 variants)
# ============================================================================

VARIANT_TEMPLATES = [
    {"seniority": "Junior", "domain": "Software House", "noise": "low", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "0-2", "tech_focus": "backend", "location": "local", "language": "English", "management_level": "individual contributor", "salary_range": "$40k-$60k", "benefits": "health insurance, flexible hours", "required_certifications": "none", "cultural_fit": "fast-paced startup"},
    {"seniority": "Mid", "domain": "FinTech", "noise": "medium", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "2-5", "tech_focus": "data", "location": "regional", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Telecom", "noise": "high", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "5+", "tech_focus": "infrastructure", "location": "global", "language": "English", "management_level": "team lead"},
    {"seniority": "Mid", "domain": "Marketing", "noise": "low", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "contract", "experience_years": "3-6", "tech_focus": "frontend", "location": "local", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "AI", "noise": "medium", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "7+", "tech_focus": "machine learning", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "DevOps", "noise": "high", "company_size": "mid-size", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "internship", "experience_years": "0-1", "tech_focus": "cloud", "location": "local", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Healthcare", "noise": "medium", "company_size": "enterprise", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "3-5", "tech_focus": "backend", "location": "cross-border", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "E-commerce", "noise": "low", "company_size": "startup", "education": "formal", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "6+", "tech_focus": "product engineering", "location": "regional", "language": "English", "management_level": "staff"},
    {"seniority": "Any", "domain": "Any", "noise": "vague", "company_size": "startup", "education": "none", "code_switch": "none", "work_mode": "remote", "employment_type": "part-time", "experience_years": "any", "tech_focus": "generalist", "location": "any", "language": "any", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Any", "noise": "impossible", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "20+", "tech_focus": "full-stack", "location": "global", "language": "English", "management_level": "head of engineering"},
    {"seniority": "Junior", "domain": "FinTech", "noise": "low", "company_size": "enterprise", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "1-3", "tech_focus": "backend", "location": "local", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Cybersecurity", "noise": "high", "company_size": "mid-size", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "4-7", "tech_focus": "security", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Healthcare", "noise": "medium", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "contract", "experience_years": "8+", "tech_focus": "data", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Mid", "domain": "Gaming", "noise": "low", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "remote", "employment_type": "full-time", "experience_years": "2-4", "tech_focus": "frontend", "location": "cross-border", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "SaaS", "noise": "medium", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "6+", "tech_focus": "platform", "location": "regional", "language": "English", "management_level": "staff"},
    {"seniority": "Entry", "domain": "E-commerce", "noise": "low", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "0-1", "tech_focus": "frontend", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Lead", "domain": "Healthcare", "noise": "medium", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "8+", "tech_focus": "data", "location": "global", "language": "English + Arabish", "management_level": "team lead"},
    {"seniority": "Principal", "domain": "Gaming", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "contract", "experience_years": "10+", "tech_focus": "backend", "location": "regional", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Cybersecurity", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "internship", "experience_years": "0-1", "tech_focus": "security", "location": "local", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "SaaS", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "3-5", "tech_focus": "full-stack", "location": "cross-border", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "FinTech", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "6-8", "tech_focus": "infrastructure", "location": "regional", "language": "English + Arabish", "management_level": "staff"},
    {"seniority": "Lead", "domain": "AI", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "9+", "tech_focus": "machine learning", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Principal", "domain": "Marketing Tech", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "contract", "experience_years": "12+", "tech_focus": "frontend", "location": "local", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Entry", "domain": "Logistics", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "part-time", "experience_years": "0-1", "tech_focus": "backend", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Education", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "full-time", "experience_years": "4-6", "tech_focus": "data", "location": "cross-border", "language": "English + Arabish", "management_level": "team lead"},
    {"seniority": "Senior", "domain": "Automotive", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "7-9", "tech_focus": "embedded", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Lead", "domain": "Retail", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "10+", "tech_focus": "platform", "location": "regional", "language": "Arabic + English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Energy", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "contract", "experience_years": "15+", "tech_focus": "infrastructure", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Media", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "internship", "experience_years": "0-2", "tech_focus": "frontend", "location": "local", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Travel", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "3-6", "tech_focus": "backend", "location": "cross-border", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Insurance", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "full-time", "experience_years": "8-12", "tech_focus": "data", "location": "regional", "language": "Arabic + English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Real Estate", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "contract", "experience_years": "11+", "tech_focus": "full-stack", "location": "local", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Agriculture", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "13+", "tech_focus": "machine learning", "location": "global", "language": "English + Arabish", "management_level": "manager"},
    {"seniority": "Entry", "domain": "Pharmaceutical", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "part-time", "experience_years": "0-1", "tech_focus": "security", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Manufacturing", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "2-4", "tech_focus": "embedded", "location": "local", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Consulting", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "5-7", "tech_focus": "platform", "location": "cross-border", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Non-Profit", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "contract", "experience_years": "9+", "tech_focus": "infrastructure", "location": "global", "language": "English + Arabish", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Government", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "14+", "tech_focus": "data", "location": "local", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Entertainment", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "internship", "experience_years": "0-2", "tech_focus": "frontend", "location": "regional", "language": "Arabic + English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Sports", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "3-5", "tech_focus": "backend", "location": "cross-border", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Music", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "contract", "experience_years": "6-8", "tech_focus": "machine learning", "location": "local", "language": "English + Arabish", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Film", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "10+", "tech_focus": "graphics", "location": "global", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Publishing", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "full-time", "experience_years": "16+", "tech_focus": "platform", "location": "regional", "language": "Arabic + English", "management_level": "manager"},
    {"seniority": "Entry", "domain": "Advertising", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "part-time", "experience_years": "0-1", "tech_focus": "frontend", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Legal Tech", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "4-6", "tech_focus": "security", "location": "cross-border", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "HR Tech", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "contract", "experience_years": "7-10", "tech_focus": "data", "location": "global", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Supply Chain", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "11+", "tech_focus": "backend", "location": "regional", "language": "Arabic + English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Environmental", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "17+", "tech_focus": "infrastructure", "location": "local", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Space", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "arabish", "work_mode": "remote", "employment_type": "internship", "experience_years": "0-2", "tech_focus": "embedded", "location": "global", "language": "English + Arabish", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Defense", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "2-5", "tech_focus": "security", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Research", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "arabish", "work_mode": "hybrid", "employment_type": "contract", "experience_years": "5-8", "tech_focus": "machine learning", "location": "regional", "language": "Arabic + English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Open Source", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "9+", "tech_focus": "platform", "location": "global", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Startup Incubator", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "arabish", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "12+", "tech_focus": "full-stack", "location": "local", "language": "English + Arabish", "management_level": "manager"},
    {"seniority": "Intern", "domain": "Quantum Computing", "noise": "typos", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "internship", "experience_years": "0", "tech_focus": "quantum", "location": "global", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Fellow", "domain": "Metaverse", "noise": "abbreviations", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "hybrid", "employment_type": "contract", "experience_years": "5-10", "tech_focus": "VR/AR", "location": "cross-border", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Staff", "domain": "Web3", "noise": "low", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "8-12", "tech_focus": "blockchain", "location": "regional", "language": "English", "management_level": "staff"},
    {"seniority": "VP", "domain": "Sustainability Tech", "noise": "medium", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "15+", "tech_focus": "green tech", "location": "global", "language": "English", "management_level": "executive"},
    {"seniority": "Junior", "domain": "BioTech", "noise": "high", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "1-3", "tech_focus": "bioinformatics", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Freelance", "noise": "low", "company_size": "freelance", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "freelance", "experience_years": "3-7", "tech_focus": "generalist", "location": "any", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Technical Writing", "noise": "medium", "company_size": "mid-size", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "5-10", "tech_focus": "documentation", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Lead", "domain": "DevRel", "noise": "low", "company_size": "enterprise", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "7-12", "tech_focus": "community", "location": "global", "language": "English", "management_level": "team lead"},
    {"seniority": "Principal", "domain": "Ethics AI", "noise": "high", "company_size": "startup", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "10+", "tech_focus": "AI ethics", "location": "local", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Low-Code", "noise": "medium", "company_size": "mid-size", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "1-4", "tech_focus": "no-code platforms", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "EU Software", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "4-8", "tech_focus": "backend", "location": "EU", "language": "English + German", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "APAC DevOps", "noise": "medium", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "6-10", "tech_focus": "infrastructure", "location": "APAC", "language": "English + Mandarin", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Data + ML", "noise": "low", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "8-12", "tech_focus": "ML engineering", "location": "cross-border", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Full-Stack + DevOps", "noise": "high", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "12+", "tech_focus": "full-stack infra", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Intern", "domain": "IoT Security", "noise": "typos", "company_size": "mid-size", "education": "hands-on", "code_switch": "none", "work_mode": "hybrid", "employment_type": "internship", "experience_years": "0-1", "tech_focus": "IoT security", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Edge AI", "noise": "abbreviations", "company_size": "startup", "education": "balanced", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "3-6", "tech_focus": "edge computing", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "AR/VR Content", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "contract", "experience_years": "5-9", "tech_focus": "VR development", "location": "global", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Robotics Software", "noise": "medium", "company_size": "mid-size", "education": "hands-on", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "7-11", "tech_focus": "robotics", "location": "local", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Climate Tech", "noise": "high", "company_size": "startup", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "10+", "tech_focus": "sustainability", "location": "regional", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Space Tech", "noise": "low", "company_size": "enterprise", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "1-4", "tech_focus": "aerospace", "location": "global", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Defense Cybersecurity", "noise": "medium", "company_size": "government", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "4-8", "tech_focus": "defense security", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Open Source", "noise": "low", "company_size": "community", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "volunteer", "experience_years": "5-10", "tech_focus": "open source", "location": "global", "language": "English", "management_level": "contributor"},
    {"seniority": "Lead", "domain": "Startup Incubator", "noise": "high", "company_size": "incubator", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "8-12", "tech_focus": "mentorship", "location": "local", "language": "English", "management_level": "lead"},
    {"seniority": "Principal", "domain": "Gig Economy", "noise": "medium", "company_size": "platform", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "gig", "experience_years": "10+", "tech_focus": "platform dev", "location": "any", "language": "English", "management_level": "expert"},
    {"seniority": "Junior", "domain": "Hybrid Role", "noise": "low", "company_size": "mid-size", "education": "formal", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "1-3", "tech_focus": "multi-role", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Technical Evangelist", "noise": "high", "company_size": "enterprise", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "4-7", "tech_focus": "evangelism", "location": "global", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Product Security", "noise": "medium", "company_size": "startup", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "6-10", "tech_focus": "security", "location": "local", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Compliance Tech", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "8-12", "tech_focus": "compliance", "location": "regional", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Regulatory Tech", "noise": "high", "company_size": "mid-size", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "contract", "experience_years": "12+", "tech_focus": "regtech", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "HealthTech Data", "noise": "medium", "company_size": "startup", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "1-4", "tech_focus": "health data", "location": "local", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "EdTech Platform", "noise": "low", "company_size": "enterprise", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "3-6", "tech_focus": "education tech", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "AgriTech", "noise": "high", "company_size": "startup", "education": "balanced", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "5-9", "tech_focus": "agritech", "location": "rural", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Smart City", "noise": "medium", "company_size": "government", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "7-11", "tech_focus": "urban tech", "location": "urban", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Environmental Monitoring", "noise": "low", "company_size": "non-profit", "education": "hands-on", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "10+", "tech_focus": "env monitoring", "location": "global", "language": "English", "management_level": "manager"},
    {"seniority": "Junior", "domain": "Wildlife Conservation", "noise": "high", "company_size": "NGO", "education": "balanced", "code_switch": "none", "work_mode": "field", "employment_type": "contract", "experience_years": "1-3", "tech_focus": "conservation tech", "location": "remote", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Humanitarian Aid", "noise": "medium", "company_size": "international", "education": "formal", "code_switch": "none", "work_mode": "deployed", "employment_type": "full-time", "experience_years": "3-7", "tech_focus": "aid tech", "location": "crisis zones", "language": "English + local", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Disaster Response", "noise": "low", "company_size": "agency", "education": "hands-on", "code_switch": "none", "work_mode": "emergency", "employment_type": "full-time", "experience_years": "5-10", "tech_focus": "response tech", "location": "variable", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Social Impact Data", "noise": "high", "company_size": "foundation", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "7-12", "tech_focus": "impact data", "location": "global", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Non-Profit Tech", "noise": "medium", "company_size": "non-profit", "education": "formal", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "12+", "tech_focus": "non-profit dev", "location": "local", "language": "English", "management_level": "director"},
    {"seniority": "Junior", "domain": "Government Digital", "noise": "low", "company_size": "government", "education": "hands-on", "code_switch": "none", "work_mode": "on-site", "employment_type": "full-time", "experience_years": "1-4", "tech_focus": "gov tech", "location": "capital", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Mid", "domain": "Public Sector Cloud", "noise": "high", "company_size": "agency", "education": "balanced", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "4-8", "tech_focus": "cloud gov", "location": "regional", "language": "English", "management_level": "individual contributor"},
    {"seniority": "Senior", "domain": "Election Tech", "noise": "medium", "company_size": "election office", "education": "formal", "code_switch": "none", "work_mode": "on-site", "employment_type": "contract", "experience_years": "6-10", "tech_focus": "election security", "location": "local", "language": "English", "management_level": "team lead"},
    {"seniority": "Lead", "domain": "Civic Tech", "noise": "low", "company_size": "community", "education": "hands-on", "code_switch": "none", "work_mode": "hybrid", "employment_type": "full-time", "experience_years": "7-11", "tech_focus": "civic apps", "location": "urban", "language": "English", "management_level": "staff"},
    {"seniority": "Principal", "domain": "Policy Tech", "noise": "high", "company_size": "think tank", "education": "formal", "code_switch": "none", "work_mode": "remote", "employment_type": "full-time", "experience_years": "15+", "tech_focus": "policy analysis", "location": "global", "language": "English", "management_level": "senior"},
]

DOMAIN_COMPANY_EXAMPLES = {
    "Software House": ["Atlassian", "Toptal", "Turing", "CloudAcademy", "Codementor"],
    "FinTech": ["Stripe", "Affirm", "Wise", "Chime", "SoFi"],
    "Telecom": ["Vodafone", "BT Group", "Telefonica", "Orange", "Deutsche Telekom"],
    "Healthcare": ["Teladoc", "Optum", "CVS Health", "UnitedHealth", "Humana"],
    "E-commerce": ["Farfetch", "Shopify", "Etsy", "Wayfair", "Amplitude"],
    "AI/ML": ["Anthropic", "OpenAI", "Scale AI", "Hugging Face", "Weights & Biases"],
    "DevOps": ["HashiCorp", "Snyk", "JFrog", "Cloudflare", "DataDog"],
    "Marketing Tech": ["Marketo", "HubSpot", "Salesforce", "Segment", "RudderStack"],
    "Logistics": ["Flexport", "Sennder", "Project44", "Shippo", "Flexport"],
    "Real Estate": ["Zillow", "Redfin", "Opendoor", "Knock", "Fundbox"],
    "Advertising": ["The Trade Desk", "Criteo", "MediaMath", "Basis", "Simpli"],
    "Manufacturing": ["Siemens", "GE Digital", "Autodesk", "PTC", "Dassault"],
    "Default": ["TechCorp", "CloudWorks", "DataSystems", "InnovateLabs", "NextGen Solutions"],
}

LOCATION_EXAMPLES = {
    "local": ["New York, NY", "San Francisco, CA", "London, UK", "Berlin, Germany", "Toronto, Canada"],
    "regional": ["Greater London, UK", "Bay Area, CA", "Midwest, USA", "Southeast, USA", "Central EU"],
    "global": ["Remote (Global)", "EU Remote", "APAC Remote", "Americas Remote", "Worldwide"],
    "cross-border": ["Dublin, Ireland", "Singapore", "Amsterdam, Netherlands", "Toronto, Canada", "Sydney, Australia"],
    "any": ["Flexible", "Any Location", "TBD", "Open", "Negotiable"],
}


# ============================================================================
# LLM BACKENDS
# ============================================================================

def call_ollama(prompt: str, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> str:
    """Call Ollama local LLM."""
    import requests
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 2500}}
    resp = requests.post(f"{host}/api/generate", json=payload, timeout=240)
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_openrouter(prompt: str, api_key: str, model: str = "openai/gpt-4o-mini") -> str:
    """Call OpenRouter API."""
    import requests
    url = "https://api.openrouter.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2500}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or choices[0]
        if isinstance(msg, dict):
            return msg.get("content") or msg.get("text") or json.dumps(msg)
        return str(msg)
    return data.get("output") or json.dumps(data)


# ============================================================================
# HELPERS
# ============================================================================

def get_company_name(domain: str) -> str:
    """Get company name by domain."""
    import random
    examples = DOMAIN_COMPANY_EXAMPLES.get(domain, DOMAIN_COMPANY_EXAMPLES["Default"])
    return random.choice(examples)


def get_location_name(location: str) -> str:
    """Get location by category."""
    import random
    examples = LOCATION_EXAMPLES.get(location, LOCATION_EXAMPLES["any"])
    return random.choice(examples)


def is_realistic_variant(variant: Dict[str, str]) -> bool:
    """Filter unrealistic combos."""
    years_str = variant.get("experience_years", "")
    seniority = variant.get("seniority", "")
    try:
        if years_str.startswith("0-") or years_str.startswith("1-"):
            min_years = 0
        elif "+" in years_str:
            min_years = int(years_str.replace("+", ""))
        else:
            parts = years_str.split("-")
            min_years = int(parts[0]) if parts else 0
    except ValueError:
        return True
    if min_years >= 20 and seniority in ["Junior", "Entry", "Intern"]:
        return False
    return True


def build_jd_prompt(position: str, variant: Dict[str, str]) -> str:
    """Build JD generation prompt."""
    company = get_company_name(variant["domain"])
    location = get_location_name(variant.get("location", "local"))
    prompt = (
        f"Write a realistic and well-structured job description for '{position}' at {company} ({location}). "
        f"Seniority: {variant.get('seniority')}. Domain: {variant.get('domain')}. "
        f"Company size: {variant.get('company_size')}. Include responsibilities, required skills, "
        f"preferred skills, and 2-3 example projects. Do NOT use placeholders like [Company Name] or [City]."
    )
    if variant.get("code_switch") == "arabish":
        prompt += " Include some code-switching (Arabish)."
    return prompt


def read_positions(path: str) -> List[str]:
    """Read positions from file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    positions = []
    for l in lines:
        if l and l[0].isdigit():
            parts = l.split(".", 1)
            if len(parts) > 1:
                positions.append(parts[1].strip())
        elif not l.startswith("#"):
            positions.append(l)
    return positions


def build_record_key(position: str, variant: Dict[str, str]) -> str:
    """Create unique key."""
    variant_key = json.dumps(variant, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{position}::{variant_key}"


def load_keys_from_output_jsonl(path: str) -> set:
    """Load completed keys."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                meta = obj.get("meta") or {}
                position = meta.get("position")
                variant = meta.get("variant")
                if isinstance(position, str) and isinstance(variant, dict):
                    keys.add(build_record_key(position, variant))
            except Exception:
                continue
    return keys


def load_checkpoint_keys(path: str) -> set:
    """Load checkpoint."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    completed = data.get("completed_keys")
    return {k for k in completed if isinstance(k, str)} if isinstance(completed, list) else set()


def save_checkpoint(path: str, completed_keys: set):
    """Save checkpoint."""
    tmp_path = f"{path}.tmp"
    payload = {"version": 1, "updated_at": int(time.time()), "completed_keys": sorted(completed_keys)}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def process_jd_variant(
    position: str,
    variant: Dict[str, str],
    record_key: str,
    llm_caller: Callable,
    args
) -> Tuple[Dict[str, Any], str]:
    """Process single variant."""
    meta = {"position": position, "variant": variant}
    jd_prompt = build_jd_prompt(position, variant)
    try:
        jd_text = llm_caller(jd_prompt)
        time.sleep(args.sleep)
    except Exception as e:
        return {"meta": meta, "job_description": "", "status": "failed", "error": f"{type(e).__name__}: {str(e)}"}, record_key
    return {"meta": meta, "job_description": jd_text, "status": "success"}, record_key


# ============================================================================
# MAIN
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Phase 1 JD generation with parallelization")
    parser.add_argument("--positions-file", default=os.path.join(script_dir, "..", "positions.md"))
    parser.add_argument("--out", default=os.path.join(script_dir, "..", "data", "phase1_job_descriptions.jsonl"))
    parser.add_argument("--model", choices=["ollama", "openrouter"], default="ollama")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b"))
    parser.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_API_KEY"))
    parser.add_argument("--openrouter-model", default="openai/gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("VARIANTS_WORKER_COUNT", "1")))
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.checkpoint_file is None:
        args.checkpoint_file = f"{args.out}.checkpoint.json"

    positions = read_positions(args.positions_file)
    plan = [(pos, variant, build_record_key(pos, variant)) for pos in positions for variant in VARIANT_TEMPLATES if is_realistic_variant(variant)]
    planned_keys = {k for _, _, k in plan}

    output_keys = load_keys_from_output_jsonl(args.out) if args.resume else set()
    checkpoint_keys = load_checkpoint_keys(args.checkpoint_file) if args.resume else set()
    completed_keys = (output_keys | checkpoint_keys) & planned_keys

    if args.resume:
        print(f"Resume | completed={len(completed_keys)} pending={len(plan) - len(completed_keys)} total={len(plan)}", flush=True)

    def llm_caller(prompt_text: str) -> str:
        if args.model == "openrouter":
            return call_openrouter(prompt_text, api_key=args.openrouter_key, model=args.openrouter_model)
        return call_ollama(prompt_text, model=args.ollama_model, host=args.ollama_host)

    out_f = open(args.out, "a", encoding="utf-8")
    write_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    newly_completed = 0
    progress = tqdm(total=len(plan), desc="Phase 1 JDs", unit="record", initial=len(completed_keys)) if tqdm else None

    def submit_and_process():
        nonlocal newly_completed
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_item = {}
            for pos, variant, record_key in plan:
                if args.resume and record_key in completed_keys:
                    continue
                future = executor.submit(process_jd_variant, pos, variant, record_key, llm_caller, args)
                future_to_item[future] = (pos, variant, record_key)
            for future in as_completed(future_to_item):
                try:
                    record, rk = future.result()
                except Exception as exc:
                    print(f"Variant failed: {exc}")
                    continue
                with write_lock:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                with checkpoint_lock:
                    completed_keys.add(rk)
                    newly_completed += 1
                    if args.checkpoint_every > 0 and newly_completed % args.checkpoint_every == 0:
                        save_checkpoint(args.checkpoint_file, completed_keys)
                if progress:
                    progress.update(1)

    try:
        submit_and_process()
    finally:
        save_checkpoint(args.checkpoint_file, completed_keys)
        if progress:
            progress.close()
        out_f.close()
        print(f"✓ Phase 1 complete: {len(completed_keys)} JDs")


if __name__ == "__main__":
    main()
