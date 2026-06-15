from pathlib import Path

import pytest


@pytest.fixture
def reduced_target() -> dict[str, object]:
    return {
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "location": "Cairo, Egypt",
        "phone": "+20 100 123 4567 ext. 9",
        "linkedin_url": "https://www.linkedin.com/in/jane-doe/",
        "summary": "Machine learning engineer building document systems.",
        "skills": ["Python", "Machine Learning"],
        "work_experience": [
            {
                "job_title": "ML Engineer",
                "company": "Example AI",
                "start_date": "2022-01",
                "end_date": "Present",
                "duration": "2022-01 - Present",
            }
        ],
        "education": [
            {
                "degree": "B.Sc.",
                "field_of_study": "Computer Science",
                "institution": "Example University",
                "graduation_date": "2021",
            }
        ],
        "projects": None,
        "certifications": None,
    }


@pytest.fixture
def dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "eramatch_benchmark_v4"
