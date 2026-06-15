from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkExperience(StrictModel):
    job_title: str
    company: str
    start_date: str
    end_date: str
    duration: str


class Education(StrictModel):
    degree: str
    field_of_study: str
    institution: str
    graduation_date: str


class Project(StrictModel):
    name: str
    technologies: list[str]
    url: str | None = None


class Certification(StrictModel):
    name: str
    issuer: str
    date: str


class ReducedCVTarget(StrictModel):
    full_name: str
    email: str
    location: str
    phone: str
    linkedin_url: str | None = None
    github_url: str | None = None
    summary: str
    skills: list[str]
    work_experience: list[WorkExperience]
    education: list[Education]
    projects: list[Project] | None = None
    certifications: list[Certification] | None = None


class ArtifactReference(StrictModel):
    kind: str
    path: str
    sha256: str
    size_bytes: int


class ManifestRow(StrictModel):
    cv_id: str
    tier: str
    template: str
    primary_domain: str
    split: str | None = None
    selection_seed: int
    artifacts: dict[str, ArtifactReference]
    page_images: list[ArtifactReference]


class AuditIssue(StrictModel):
    cv_id: str | None = None
    kind: str
    message: str


class AuditReport(StrictModel):
    dataset_root: str
    completed_count: int
    tier_counts: dict[str, int]
    template_counts: dict[str, int]
    issues: list[AuditIssue]
    passed: bool


class EvidenceBundle(StrictModel):
    parser_text: str | None = None
    canonical_text: str | None = None
    ocr_text: str | None = None


class EvidenceUnit(StrictModel):
    evidence_id: str
    text: str
    page: int
    bbox_norm: tuple[int, int, int, int]
    confidence: float = 1.0
    source: str
    reading_order: int
    field_path: str | None = None
    record_index: int | None = None


class EvidenceGraph(StrictModel):
    cv_id: str
    reader: str
    oracle: bool = False
    units: list[EvidenceUnit]
    page_images: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FieldCandidate(StrictModel):
    schema_path: str
    value: Any
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float
    record_index: int | None = None


class RecordLink(StrictModel):
    left_evidence_id: str
    right_evidence_id: str
    confidence: float
    relation: str = "same_record"


class SelectorTrace(StrictModel):
    selector: str
    input_tokens: int
    selected_tokens: int
    selected_evidence_ids: list[str] = Field(default_factory=list)


class GroundedPrediction(StrictModel):
    cv_id: str
    prediction: ReducedCVTarget
    candidates: list[FieldCandidate]
    record_links: list[RecordLink] = Field(default_factory=list)
    selector_trace: SelectorTrace | None = None
    assembly_events: list["ValidationEvent"] = Field(default_factory=list)


class ValidationEvent(StrictModel):
    kind: str
    path: str | None = None
    message: str


class FieldResult(StrictModel):
    path: str
    metric: str
    score: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    jaccard: float | None = None
    supported: bool | None = None
    truth: Any = None
    prediction: Any = None


class DocumentEvaluation(StrictModel):
    json_valid: bool
    schema_valid: bool
    missing_keys: list[str]
    extra_keys: list[str]
    validation_events: list[ValidationEvent]
    field_results: list[FieldResult]
    micro_score: float
    macro_score: float
    unsupported_evidence_rate: float


class AggregateEvaluation(StrictModel):
    document_count: int
    json_valid_rate: float
    schema_valid_rate: float
    micro_score: float
    macro_score: float
    unsupported_evidence_rate: float
    field_scores: dict[str, float]


class RunProvenance(StrictModel):
    code_revision: str | None = None
    manifest_hash: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    seed: int
    resolved_config: dict[str, Any] = Field(default_factory=dict)


class RunRecord(StrictModel):
    run_id: str
    kind: str
    status: Literal["pending", "running", "completed", "failed"]
    provenance: RunProvenance
    model_id: str | None = None
    parser_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class WorkCandidateSpan(StrictModel):
    cv_id: str
    page: int
    field_path: str
    record_index: int
    value: str
    evidence_ids: list[str] = Field(default_factory=list)
    word_start: int
    word_end: int


class WorkRecordTarget(StrictModel):
    record_index: int
    job_title: str
    company: str
    start_date: str
    end_date: str
    duration: str


class WorkDocumentRecordSet(StrictModel):
    cv_id: str
    split: str | None = None
    tier: str | None = None
    reader: str
    oracle: bool = False
    page_count: int
    spans: list[WorkCandidateSpan]
    targets: list[WorkRecordTarget]
