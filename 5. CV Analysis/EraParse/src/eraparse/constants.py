from pathlib import Path

SEED = 20260609
DEFAULT_DATASET_ROOT = Path("../eramatch_benchmark_v4")
DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_MANIFEST_ROOT = DEFAULT_ARTIFACT_ROOT / "manifests"
DEFAULT_AUDIT_REPORT = DEFAULT_ARTIFACT_ROOT / "audit_report.json"
DEFAULT_RUN_DB = DEFAULT_ARTIFACT_ROOT / "runs.duckdb"
DEFAULT_TRIAL_ROOT = DEFAULT_ARTIFACT_ROOT / "trials"
DEFAULT_REPRESENTATION_ROOT = DEFAULT_ARTIFACT_ROOT / "representations"
DEFAULT_ATS_ROOT = DEFAULT_ARTIFACT_ROOT / "ats_baselines"
DEFAULT_EVIDENCE_ROOT = DEFAULT_ARTIFACT_ROOT / "evidence"
DEFAULT_SGE_ROOT = DEFAULT_ARTIFACT_ROOT / "sge"

NUEXTRACT_MODEL_ID = "numind/NuExtract-1.5-tiny"
NUEXTRACT_REVISION = "63e2e80c804d9c97f3f19a4aa25613e7beca83c9"
NUEXTRACT3_MODEL_ID = "numind/NuExtract3"
NUEXTRACT3_REVISION = "acaf70ecff9c3dbbfcbae651b82b66a0d8dbd0c6"
QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DONUT_MODEL_ID = "naver-clova-ix/donut-base"
DONUT_REVISION = "a959cf33c20e09215873e338299c900f57047c61"
LAYOUTLMV3_MODEL_ID = "microsoft/layoutlmv3-base"
LAYOUTLMV3_REVISION = "cfbbbff0762e6aab37086fdd4739ad14fe7d5db4"
PADDLEOCR_VL_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
PADDLEOCR_VL_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
T4_GPU_RATE_PER_SECOND = 0.000164
CURRENT_STAGE_BUDGET_USD = 18.00
SGE_FIELD_PATHS = (
    "full_name",
    "email",
    "location",
    "phone",
    "linkedin_url",
    "github_url",
    "summary",
    "skills",
    "work_experience.*.job_title",
    "work_experience.*.company",
    "work_experience.*.start_date",
    "work_experience.*.end_date",
    "work_experience.*.duration",
    "education.*.degree",
    "education.*.field_of_study",
    "education.*.institution",
    "education.*.graduation_date",
    "projects.*.name",
    "projects.*.technologies",
    "projects.*.url",
    "certifications.*.name",
    "certifications.*.issuer",
    "certifications.*.date",
)
REDUCED_SCHEMA_TEMPLATE = {
    "full_name": "",
    "email": "",
    "location": "",
    "phone": "",
    "linkedin_url": "",
    "github_url": "",
    "summary": "",
    "skills": [],
    "work_experience": [
        {
            "job_title": "",
            "company": "",
            "start_date": "",
            "end_date": "",
            "duration": "",
        }
    ],
    "education": [
        {
            "degree": "",
            "field_of_study": "",
            "institution": "",
            "graduation_date": "",
        }
    ],
    "projects": [{"name": "", "technologies": [], "url": ""}],
    "certifications": [{"name": "", "issuer": "", "date": ""}],
}

EXPECTED_COMPLETED = 4_950
EXPECTED_WORKING = 2_475
EXPECTED_LOCKED = 2_475
EXPECTED_SPLITS = {
    "train": 1_445,
    "validation": 310,
    "id_test": 310,
    "template_ood_test": 410,
}
WORKING_TIER_QUOTAS = {"T1": 625, "T2": 625, "T3": 500, "T4": 375, "T5": 350}
OOD_TEMPLATE_QUOTAS = {
    "T3_nested_tables": 100,
    "T3_europass": 100,
    "T5_infographic": 70,
    "T5_magazine": 70,
    "T5_dark": 70,
}
T4_EXPECTED_COUNT = 750

REQUIRED_ARTIFACT_PATTERNS = {
    "ground_truth": "ground_truth/{cv_id}.json",
    "donut_target": "donut_targets/{cv_id}.json",
    "schema_full": "schema_targets/full/{cv_id}.json",
    "schema_reduced": "schema_targets/reduced/{cv_id}.json",
    "layout": "layout_annotations/{cv_id}_layout.json",
    "field_annotations": "field_annotations/{cv_id}_fields.json",
    "section_annotations": "section_annotations/{cv_id}_sections.json",
    "word_annotations": "word_annotations/{cv_id}_words.json",
    "token_labels": "token_labels/{cv_id}_bio.json",
    "text_ground_truth": "text_ground_truth/{cv_id}.txt",
    "pymupdf_text": "extracted_text/pymupdf/{cv_id}.txt",
    "pdfminer_text": "extracted_text/pdfminer/{cv_id}.txt",
    "pdf": "pdfs/{cv_id}.pdf",
    "source_pdf": "source_pdfs/{cv_id}_source.pdf",
    "template_layout_hint": "template_layout_hints/{cv_id}_template_layout.json",
}

JSON_ARTIFACT_KINDS = {
    "ground_truth",
    "donut_target",
    "schema_full",
    "schema_reduced",
    "layout",
    "field_annotations",
    "section_annotations",
    "word_annotations",
    "token_labels",
    "template_layout_hint",
}
