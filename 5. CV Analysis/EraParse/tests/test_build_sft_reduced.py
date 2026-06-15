import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_sft_reduced import clean_target, build_example


def test_clean_target_drops_phantom_github():
    truth = {"github_url": "https://github.com/x", "work_experience": []}
    cleaned = clean_target(truth, doc_text="some cv text", template="T1_classic")
    assert not cleaned.get("github_url")


def test_clean_target_keeps_real_github():
    truth = {"github_url": "https://github.com/x", "work_experience": []}
    cleaned = clean_target(truth, doc_text="github.com/x profile here", template="T1_classic")
    assert cleaned.get("github_url")


def test_clean_target_drops_phantom_linkedin():
    truth = {"linkedin_url": "https://linkedin.com/in/xyz"}
    cleaned = clean_target(truth, doc_text="no social links", template="T1_classic")
    assert not cleaned.get("linkedin_url")


def test_clean_target_drops_nonrendering_projects():
    truth = {"projects": [{"name": "Vector Search API", "technologies": [], "url": ""}]}
    cleaned = clean_target(truth, doc_text="no project section here", template="T3_table")
    assert not cleaned.get("projects")


def test_clean_target_keeps_projects_when_rendered():
    truth = {"projects": [{"name": "Vector Search API", "technologies": [], "url": ""}]}
    cleaned = clean_target(truth, doc_text="vector search api built with python", template="T3_table")
    assert cleaned.get("projects")


def test_clean_target_nonabsent_template_keeps_projects():
    truth = {"projects": [{"name": "My App", "technologies": [], "url": ""}]}
    cleaned = clean_target(truth, doc_text="no project section", template="T1_classic")
    assert cleaned.get("projects")


def test_build_example_shape():
    ex = build_example("cv_1", "MARKDOWN TEXT", {"full_name": "A", "skills": []})
    roles = [m["role"] for m in ex["conversations"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["conversations"][1]["content"] == "MARKDOWN TEXT"
    assert json.loads(ex["conversations"][2]["content"])["full_name"] == "A"
    assert ex["id"] == "cv_1"
