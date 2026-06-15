"""CVSchema v3.0 — Pydantic v2 models for structured CV data.

Extracted from cvs-eramatch.ipynb. All fields are backwards-compatible
with existing ground_truth JSON files.

Models:
    ContactInfo, WorkExperienceEntry, EducationEntry, SkillEntry,
    ProjectEntry, CertificationEntry, MiscItem, CVSchema
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None


class WorkExperienceEntry(BaseModel):
    company: Optional[str] = None
    organization: Optional[str] = None
    job_title: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    duration_months: Optional[int] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    is_remote: Optional[bool] = None
    employment_type: Optional[
        Literal["full-time", "part-time", "contract", "internship", "freelance"]
    ] = None


class EducationEntry(BaseModel):
    institution: Optional[str] = None
    university: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[str] = None
    activities: Optional[str] = None


class SkillEntry(BaseModel):
    skill_name: str
    category: Optional[
        Literal[
            "programming_language",
            "framework",
            "tool",
            "database",
            "cloud",
            "methodology",
            "language",
            "soft_skill",
            "ai_api",
            "ai_ml",
            "ml_ops",
            "infrastructure",
            "container",
            "ci_cd",
            "monitoring",
            "data_tool",
            "data_warehouse",
            "cloud_ml",
            "other",
        ]
    ] = None
    source: Literal["explicit", "inferred"] = "explicit"


class ProjectEntry(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    date: Optional[str] = None


class CertificationEntry(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None
    credential_id: Optional[str] = None


class MiscItem(BaseModel):
    label: str
    raw_text: str
    structured: Optional[Dict[str, Any]] = None


class CVSchema(BaseModel):
    """CVSchema v3.0 — Top-level structured CV representation.

    22 fields covering identity, contact, experience, education,
    skills, projects, certifications, and metadata.
    """

    full_name: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    years_of_experience: Optional[float] = None
    seniority_level: Optional[
        Literal["intern", "junior", "mid", "senior", "lead", "principal", "executive"]
    ] = None
    primary_domain: Optional[str] = None
    has_github: bool = False
    has_linkedin: bool = False

    contact_info: ContactInfo = Field(default_factory=ContactInfo)
    summary: Optional[str] = None
    work_experience: List[WorkExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    skills: List[SkillEntry] = Field(default_factory=list)
    projects: List[ProjectEntry] = Field(default_factory=list)
    certifications: List[CertificationEntry] = Field(default_factory=list)
    misc_data: List[MiscItem] = Field(default_factory=list)

    skills_flat: List[str] = Field(default_factory=list)
    companies: List[str] = Field(default_factory=list)
    job_titles: List[str] = Field(default_factory=list)
    universities: List[str] = Field(default_factory=list)
    degrees: List[str] = Field(default_factory=list)

    parsing_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Return a dict with key fields for manifest / metadata generation."""
        github_projects: List[str] = [
            p.url for p in self.projects if p.url and "github.com" in p.url
        ]
        return {
            "full_name": self.full_name,
            "email": self.email,
            "primary_domain": self.primary_domain,
            "seniority_level": self.seniority_level,
            "source": self.parsing_metadata.get("content_source"),
            "model_used": self.parsing_metadata.get("model_used"),
            "github_projects": github_projects,
        }


__all__ = [
    "ContactInfo",
    "WorkExperienceEntry",
    "EducationEntry",
    "SkillEntry",
    "ProjectEntry",
    "CertificationEntry",
    "MiscItem",
    "CVSchema",
]
