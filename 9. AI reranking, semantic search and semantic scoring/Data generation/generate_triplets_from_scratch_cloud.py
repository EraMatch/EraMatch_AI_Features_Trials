"""
generate_triplets_from_scratch_cloud.py  (v3 — cloud)
=======================================================
Cloud version of generate_triplets_from_scratch.py.

Identical pipeline and logic; only the Ollama client differs:
  - Hits the Ollama Cloud API (https://ollama.com/v1) instead of localhost
  - Rotates across multiple API keys loaded from ollama_api_keys.txt
  - One concurrent request per key (blocking queue guarantees this)
  - Retired (blacklisted) keys are removed from the pool on auth failure

Output record:
    {"anchor", "positive", "negative", "negative_type", "anchor_domain", "negative_domain"}
"""

import json
import queue
import random
import re
import threading
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


# ═════════════════════════════════════════════════════════════════════════════
# Config — edit before running
# ═════════════════════════════════════════════════════════════════════════════
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"

_KEYS_FILE = Path(__file__).parent / "ollama_api_keys.txt"
if not _KEYS_FILE.exists():
    raise FileNotFoundError(
        f"'{_KEYS_FILE}' not found. Create it with one Ollama API key per line."
    )
API_KEYS: List[str] = [
    line.strip()
    for line in _KEYS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not API_KEYS:
    raise ValueError(f"'{_KEYS_FILE}' is empty. Add at least one Ollama API key.")

GENERATOR_MODEL = "gpt-oss:20b-cloud"   # Ollama Cloud — query + CV generation
JUDGE_MODEL     = "gpt-oss:20b-cloud"  # Ollama Cloud — CV-query matching judge

OUTPUT_DIR   = Path(__file__).parent / "triplets"
NUM_FILES    = 4                 # parallel output part files (merged later)
TARGET_COUNT = 10_000            # total approved samples across all files
MAX_WORKERS  = len(API_KEYS)     # one worker slot per key → all keys run in parallel
MAX_RETRIES  = 3                 # retry attempts per batch before giving up

ENABLE_CROSS_JUDGE = True  # set False to skip intra-domain hard-negative guardrail

SENIORITY_ORDER = ["junior", "mid", "senior", "lead", "staff", "principal"]


# ═════════════════════════════════════════════════════════════════════════════
# Domain & style templates
# ═════════════════════════════════════════════════════════════════════════════
SAFE_DOMAIN_PAIRS = [
    ("Backend (Go, Rust, Python, Django)", "Frontend (React, Next.js, Tailwind)"),
    ("Data Science & AI (PyTorch, LLMs, Pandas)", "Mobile (iOS, Android, Swift)"),
    ("Cybersecurity (Penetration Testing, IAM, SOC)", "Frontend (React, Next.js, Tailwind)"),
    ("Embedded & Systems (C, C++, RTOS, FPGA)", "Full-Stack (MERN, GraphQL)"),
    ("QA & Testing (Selenium, Cypress, JUnit)", "Blockchain & Web3 (Solidity, Ethereum)"),
    ("DevOps & Cloud (AWS, Kubernetes, CI/CD)", "Mobile (iOS, Android, Swift)"),
    ("Site Reliability Engineering (SRE, Observability)", "Frontend (React, Next.js, Tailwind)"),
    ("Platform & Infrastructure (Kafka, Spark, dbt)", "Cybersecurity (Penetration Testing, IAM)"),
]

STYLE_MODES = [
    "Extremely brief and impatient ('need python dev remote 5yrs asap')",
    "Formal corporate job-description with seniority jargon",
    "Conversational ('We are scaling fast and need someone who...')",
    "Bullet-point list of requirements",
    "Casual Slack/chat message with abbreviations and typos",
    "Detailed technical spec with specific tool or version requirements",
]


# ═════════════════════════════════════════════════════════════════════════════
# Arab-world candidate pools
# ═════════════════════════════════════════════════════════════════════════════
_ARAB_MALE_NAMES = [
    "Ahmed", "Mohamed", "Omar", "Ali", "Youssef", "Khalid", "Tariq", "Hassan",
    "Ibrahim", "Samer", "Walid", "Rami", "Karim", "Nasser", "Faris", "Ziad",
    "Mahmoud", "Bilal", "Samir", "Hani", "Wael", "Bassam", "Eyad", "Adnan",
    "Khaled", "Amr", "Tarek", "Sherif", "Mostafa", "Eslam",
]
_ARAB_FEMALE_NAMES = [
    "Sara", "Fatima", "Nour", "Layla", "Rania", "Hana", "Dina", "Yasmine",
    "Mariam", "Aisha", "Salma", "Rana", "Lina", "Nadia", "Mona", "Amira",
    "Ghada", "Reem", "Noura", "Heba", "Maha", "Farah", "Dana", "Nadine",
]
_ARAB_LAST_NAMES = [
    "Hassan", "Al-Rashid", "Ibrahim", "Khalil", "Mansour", "Nasser", "Saleh",
    "Karimi", "Haddad", "Aziz", "Taha", "Qasim", "Al-Sayed", "Daoud",
    "Farouk", "Ghanem", "Hamdan", "Kassem", "Matar", "Nassar", "Othman",
    "Al-Farsi", "Bishara", "Al-Amin", "Idris", "Jabr", "Sharaf", "Zaki",
    "Abdallah", "Al-Shammari", "Al-Ghamdi", "Al-Harbi", "Barakat",
]
_ARAB_CITIES = [
    "Cairo, Egypt", "Alexandria, Egypt", "Dubai, UAE", "Abu Dhabi, UAE",
    "Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia", "Amman, Jordan",
    "Beirut, Lebanon", "Kuwait City, Kuwait", "Doha, Qatar",
    "Muscat, Oman", "Casablanca, Morocco", "Tunis, Tunisia",
    "Baghdad, Iraq", "Manama, Bahrain", "Sharjah, UAE",
]
_ARAB_TECH_COMPANIES = [
    "Careem", "Noon", "Talabat", "Anghami", "Property Finder",
    "Swvl", "Vezeeta", "Instabug", "Fawry", "Paymob", "Halan",
    "stc", "du Telecom", "Etisalat (e&)", "Zain", "Mobily",
    "Saudi Aramco Digital", "ADNOC Digital", "Amazon.ae",
    "Microsoft MENA", "Google Cairo", "IBM Egypt", "Oracle MENA",
    "Huawei MENA", "Vodafone Egypt", "Orange Egypt",
]

_used_names: set = set()
_used_names_lock = threading.Lock()


def _random_arab_name() -> str:
    pool = _ARAB_MALE_NAMES + _ARAB_FEMALE_NAMES
    return f"{random.choice(pool)} {random.choice(_ARAB_LAST_NAMES)}"


def _unique_arab_name() -> str:
    for _ in range(50):
        name = _random_arab_name()
        with _used_names_lock:
            if name not in _used_names:
                _used_names.add(name)
                return name
    name = _random_arab_name() + f" {random.randint(2, 99)}"
    with _used_names_lock:
        _used_names.add(name)
    return name


def _random_arab_city() -> str:
    return random.choice(_ARAB_CITIES)


# ═════════════════════════════════════════════════════════════════════════════
# CV reference example (injected into CV generation prompt)
# ═════════════════════════════════════════════════════════════════════════════
_CV_EXAMPLE = """\
{
  "email": "omar.mansour.dev@gmail.com", "full_name": "Omar Mansour",
  "location": "Cairo, Egypt",
  "summary": "Backend engineer with 6 years building payment microservices.",
  "years_of_experience": 6.0, "seniority_level": "senior",
  "primary_domain": "software_engineering",
  "has_github": true, "has_linkedin": true, "has_portfolio": false,
  "skills": [
    {"category": "Languages",  "skill_name": "Go",         "proficiency": "expert"},
    {"category": "DevOps",     "skill_name": "Kubernetes", "proficiency": "advanced"},
    {"category": "Cloud",      "skill_name": "AWS",        "proficiency": "advanced"},
    {"category": "Databases",  "skill_name": "PostgreSQL", "proficiency": "advanced"}
  ],
  "work_experience": [
    {
      "company": "Fawry", "job_title": "Senior Backend Engineer",
      "start_date": "2021-03", "end_date": null, "is_current": true,
      "location": "Cairo, Egypt",
      "description": "Led microservices migration in Go, handling 2M+ daily transactions.",
      "technologies": ["Go", "Kubernetes", "PostgreSQL", "AWS", "Kafka"]
    },
    {
      "company": "Noon", "job_title": "Backend Engineer",
      "start_date": "2018-06", "end_date": "2021-02", "is_current": false,
      "location": "Cairo, Egypt",
      "description": "Built order management and catalog search services.",
      "technologies": ["Go", "PostgreSQL", "Elasticsearch", "Docker"]
    }
  ],
  "education": [{
    "institution": "Cairo University", "degree": "Bachelor of Science",
    "field_of_study": "Computer Science",
    "start_date": "2014-09", "end_date": "2018-05", "gpa": 3.7, "honors": null
  }],
  "projects": [
    {
      "name": "GoFlow", "description": "Event-driven workflow engine built in Go.",
      "technologies": ["Go", "Kafka", "Redis"],
      "start_date": "2022-01", "end_date": "2022-06", "url": "github.com/omansour/goflow"
    },
    {
      "name": "K8sMonitor", "description": "Kubernetes cluster health dashboard.",
      "technologies": ["Go", "Kubernetes", "Prometheus"],
      "start_date": "2023-03", "end_date": "2023-07", "url": null
    }
  ],
  "certifications": ["AWS Certified Solutions Architect - Associate", "CKA"],
  "languages": ["Arabic", "English"],
  "contact_info": {
    "email": "omar.mansour.dev@gmail.com", "phone": "+20-100-1234567",
    "location": "Cairo, Egypt", "full_name": "Omar Mansour",
    "linkedin_url": "linkedin.com/in/omar-mansour",
    "github_url": "github.com/omansour", "portfolio_url": null, "other_links": []
  },
  "miscellaneous": [], "prescore_v2": null
}"""


# ═════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═════════════════════════════════════════════════════════════════════════════
class SkillEntry(BaseModel):
    category:    str
    skill_name:  str
    proficiency: str = "intermediate"


class WorkEntry(BaseModel):
    company:      str
    job_title:    str
    start_date:   Optional[str] = None
    end_date:     Optional[str] = None
    is_current:   bool          = False
    location:     str           = ""
    description:  str           = ""
    technologies: List[str]     = Field(default_factory=list)


class EducationEntry(BaseModel):
    institution:    str
    degree:         str
    field_of_study: str
    start_date:     Optional[str]   = None
    end_date:       Optional[str]   = None
    gpa:            Optional[float] = None
    honors:         Optional[str]   = None


class ProjectEntry(BaseModel):
    name:         str
    description:  str
    technologies: List[str]     = Field(default_factory=list)
    start_date:   Optional[str] = None
    end_date:     Optional[str] = None
    url:          Optional[str] = None


class ContactInfo(BaseModel):
    email:         str           = ""
    phone:         str           = ""
    location:      str           = ""
    full_name:     str           = ""
    linkedin_url:  Optional[str] = None
    github_url:    Optional[str] = None
    portfolio_url: Optional[str] = None
    other_links:   List[str]     = Field(default_factory=list)


class CandidateCV(BaseModel):
    email:               str
    full_name:           str
    location:            str
    summary:             str
    years_of_experience: float
    seniority_level:     str
    primary_domain:      str
    has_github:          bool              = False
    has_linkedin:        bool              = False
    has_portfolio:       bool              = False
    skills:              List[SkillEntry]
    work_experience:     List[WorkEntry]
    education:           List[EducationEntry]
    projects:            List[ProjectEntry]  = Field(default_factory=list)
    certifications:      List[str]           = Field(default_factory=list)
    languages:           List[str]           = Field(default_factory=lambda: ["Arabic", "English"])
    contact_info:        ContactInfo
    miscellaneous:       List[Any]           = Field(default_factory=list)
    prescore_v2:         Any                 = None


class SampleJudgeVerdict(BaseModel):
    matches: bool
    reason:  str


# ═════════════════════════════════════════════════════════════════════════════
# JSON robustness
# ═════════════════════════════════════════════════════════════════════════════
def _fix_and_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    text = text[start : end + 1]

    text = re.sub(r"\bTrue\b",  "true",  text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b",  "null",  text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"//[^\n\"]*\n", "\n", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc


# ═════════════════════════════════════════════════════════════════════════════
# Type coercion
# ═════════════════════════════════════════════════════════════════════════════
def _coerce_types(cv: Dict[str, Any]) -> Dict[str, Any]:
    for field in ("has_github", "has_linkedin", "has_portfolio"):
        val = cv.get(field)
        if isinstance(val, str):
            cv[field] = val.strip().lower() in ("true", "yes", "1")
        elif val is None:
            cv[field] = False

    yoe = cv.get("years_of_experience")
    if isinstance(yoe, str):
        try:
            cv["years_of_experience"] = float(re.sub(r"[^\d.]", "", yoe) or "0")
        except ValueError:
            cv["years_of_experience"] = 0.0

    for edu in cv.get("education", []):
        gpa = edu.get("gpa")
        if isinstance(gpa, str):
            try:
                edu["gpa"] = float(re.sub(r"[^\d.]", "", gpa)) or None
            except ValueError:
                edu["gpa"] = None

    for job in cv.get("work_experience", []):
        is_curr = job.get("is_current")
        if isinstance(is_curr, str):
            job["is_current"] = is_curr.strip().lower() in ("true", "yes", "1")
        elif is_curr is None:
            job["is_current"] = False

    return cv


# ═════════════════════════════════════════════════════════════════════════════
# CV sanitization
# ═════════════════════════════════════════════════════════════════════════════
_DATE_RE             = re.compile(r"^\d{4}-\d{2}$")
_VALID_PROFICIENCIES = {"beginner", "intermediate", "advanced", "expert"}
_VALID_SENIORITIES   = set(SENIORITY_ORDER)
_FILTER_ONLY_FIELDS  = {
    "employment_type", "contract_duration", "timezone", "remote_ok",
    "visa_sponsorship", "citizenship_required", "security_clearance",
    "no_c2c", "direct_hire", "startup_experience", "min_experience_years",
    "preferred_experience_years", "role", "industry",
}
_LEAKAGE_TOKENS = [
    "same as top-level", "junior|mid|senior", "beginner|intermediate",
    "e.g. software", "3-5 sentences", "2-3 sentence",
]


def _fix_date(d: Any) -> Optional[str]:
    if d is None:
        return None
    return d if _DATE_RE.match(str(d)) else None


def _parse_ym(d: Any) -> Tuple[int, int]:
    try:
        parts = str(d).split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _months_between(start: str, end: str) -> int:
    sy, sm = _parse_ym(start)
    ey, em = _parse_ym(end)
    if sy == 0 or ey == 0:
        return 0
    return max(0, (ey - sy) * 12 + (em - sm))


def _compute_experience_years(work: List[Dict]) -> float:
    now   = datetime.now().strftime("%Y-%m")
    total = sum(
        _months_between(
            j.get("start_date") or "",
            j.get("end_date") or (now if j.get("is_current") else now),
        )
        for j in work
    )
    return round(total / 12, 1)


def sanitize_cv(cv: Dict[str, Any]) -> Dict[str, Any]:
    for f in _FILTER_ONLY_FIELDS:
        cv.pop(f, None)

    for skill in cv.get("skills", []):
        prof = str(skill.get("proficiency", "intermediate"))
        if "|" in prof:
            skill["proficiency"] = next(
                (t.strip().lower() for t in re.split(r"[|/,]", prof)
                 if t.strip().lower() in _VALID_PROFICIENCIES),
                "intermediate",
            )
        else:
            norm = prof.strip().lower()
            skill["proficiency"] = norm if norm in _VALID_PROFICIENCIES else "intermediate"

    sl = str(cv.get("seniority_level", "mid")).strip().lower()
    if "|" in sl or sl not in _VALID_SENIORITIES:
        cv["seniority_level"] = next(
            (t.strip().lower() for t in re.split(r"[|/,]", sl)
             if t.strip().lower() in _VALID_SENIORITIES),
            "mid",
        )
    else:
        cv["seniority_level"] = sl

    pd = str(cv.get("primary_domain", ""))
    if "e.g." in pd or "|" in pd or not pd:
        cv["primary_domain"] = "software_engineering"

    if cv.get("email", "") in {"realistic-fake@example.com", ""}:
        cv["email"] = f"candidate{random.randint(1000, 9999)}@techmail.io"
    if cv.get("full_name", "") in {"First Last", ""}:
        cv["full_name"] = _unique_arab_name()
    if cv.get("location", "") in {"City, Country", ""}:
        cv["location"] = _random_arab_city()

    work = cv.get("work_experience", [])
    for job in work:
        job["start_date"] = _fix_date(job.get("start_date"))
        job["end_date"]   = _fix_date(job.get("end_date"))
        if job.get("is_current"):
            job["end_date"] = None
        elif not job.get("end_date") and job.get("start_date"):
            sy, sm = _parse_ym(job["start_date"])
            if sy:
                em = sm + 24
                ey = sy + (em - 1) // 12
                em = (em - 1) % 12 + 1
                job["end_date"] = f"{ey}-{em:02d}"

    valid_work = sorted(
        [j for j in work if j.get("start_date")], key=lambda j: j["start_date"]
    )
    for i in range(len(valid_work) - 1):
        end_i   = valid_work[i].get("end_date") or ""
        start_n = valid_work[i + 1].get("start_date") or ""
        if end_i and start_n and end_i > start_n:
            sy, sm = _parse_ym(start_n)
            if sy:
                sm -= 1
                if sm == 0:
                    sm, sy = 12, sy - 1
                valid_work[i]["end_date"] = f"{sy}-{sm:02d}"
    cv["work_experience"] = valid_work

    ci = cv.get("contact_info") if isinstance(cv.get("contact_info"), dict) else {}
    ci.update({
        "email":     cv.get("email",     ci.get("email", "")),
        "full_name": cv.get("full_name", ci.get("full_name", "")),
        "location":  cv.get("location",  ci.get("location", "")),
    })
    for field in ("linkedin_url", "github_url", "portfolio_url"):
        if str(ci.get(field, "") or "") in {"linkedin.com/in/handle", "github.com/handle", ""}:
            ci[field] = None
    ci.setdefault("phone", "")
    ci.setdefault("other_links", [])
    cv["contact_info"] = ci

    if valid_work:
        computed = _compute_experience_years(valid_work)
        claimed  = float(cv.get("years_of_experience") or 0)
        if computed > 0 and abs(computed - claimed) > 1.5:
            cv["years_of_experience"] = computed

    cv.setdefault("miscellaneous",  [])
    cv.setdefault("prescore_v2",    None)
    cv.setdefault("certifications", [])
    cv.setdefault("languages",      ["Arabic", "English"])
    cv.setdefault("projects",       [])
    return cv


def _assert_no_leakage(cv: Dict[str, Any]) -> None:
    dump = json.dumps(cv)
    for tok in _LEAKAGE_TOKENS:
        if tok in dump:
            raise ValueError(f"Template leakage: '{tok}'")


# ═════════════════════════════════════════════════════════════════════════════
# System prompts
# ═════════════════════════════════════════════════════════════════════════════
_NAMES_HINT     = ", ".join((_ARAB_MALE_NAMES + _ARAB_FEMALE_NAMES)[:12])
_CITIES_HINT    = ", ".join(_ARAB_CITIES[:6])
_COMPANIES_HINT = ", ".join(_ARAB_TECH_COMPANIES[:8])

_ARAB_CONTEXT = f"""Cultural context for every CV:
  Names     : {_NAMES_HINT} or similar Arab names
  Locations : {_CITIES_HINT} or similar MENA cities
  Companies : {_COMPANIES_HINT} or similar MENA tech companies
  Languages : Arabic always included; English; optionally French
  Phone     : regional (+20 Egypt, +971 UAE, +966 KSA, +962 Jordan, +974 Qatar)"""

QUERY_SYSTEM = """You are an HR data synthesizer.

Generate one realistic tech recruiter search query for the given domain and writing style.
Write freely — the query can mention any combination of: skills, seniority, years of experience,
location, remote policy, employment type (contract/full-time/no C2C), certifications, security
clearance, startup experience, language requirements, start date, or anything a real recruiter
would naturally include.

Style ranges from a single terse line to a detailed multi-sentence paragraph.

Output ONLY the query text — no JSON, no labels, no extra text."""

QUERY_DIFF_SYSTEM = """You are an HR data synthesizer.

Generate a DIFFERENT recruiter search query for the same tech domain.
The previous query is shown below — yours must ask for a notably different role, seniority level,
or technology stack so the two queries clearly represent different hiring needs.

Write freely in the given style — same rules as before (any constraints are fine).

Output ONLY the new query text — no JSON, no labels."""

CV_SYSTEM = f"""You are a synthetic data generator for tech recruiting datasets.

Generate ONE realistic parsed-CV JSON for a candidate who clearly satisfies the given recruiter query.
Read the full query and reflect ALL requirements it expresses — skills, seniority, experience, location,
or anything else mentioned.

{_ARAB_CONTEXT}

RULES:
1. Satisfy the query fully: right domain, appropriate skills at correct proficiency, matching seniority.
2. Internal consistency:
   - years_of_experience ≈ sum of work history durations
   - Skills in the skills array must appear in work experience descriptions/technologies
   - email, full_name, location IDENTICAL between top-level and contact_info
3. Enum values — single lowercase word only:
   proficiency    → beginner | intermediate | advanced | expert
   seniority_level → junior | mid | senior | lead | staff | principal
4. Dates in YYYY-MM format. If is_current=true → end_date must be null.
5. Include 2–4 diverse projects.

OUTPUT: ONLY a raw JSON object — no markdown fences, no explanation.

STRUCTURE TO FOLLOW:
{_CV_EXAMPLE}"""

CV_JUDGE_SYSTEM = """You are a quality-control judge for a recruiting dataset.

Does this candidate CV clearly satisfy this recruiter query?
→ true  if the candidate is in the right domain and looks like a plausible match overall
→ false if there is an obvious mismatch (completely wrong domain, or clearly wrong seniority/skills)

Be lenient on minor gaps. Only reject when the mismatch is obvious.

Output ONLY valid JSON — no markdown:
{"matches": true|false, "reason": "<one short sentence>"}"""


# ═════════════════════════════════════════════════════════════════════════════
# API key pool — one request per key at a time, no exceptions.
#
# _key_pool is a blocking Queue pre-loaded with every key.
# _acquire_key() removes a key (blocks if all are in use).
# _release_key() returns the key, or blacklists it on auth failure.
# ═════════════════════════════════════════════════════════════════════════════
_key_pool:       queue.Queue = queue.Queue()
_key_total:      Dict[str, int] = {k: 0 for k in API_KEYS}
_key_total_lock  = threading.Lock()
_key_blacklist:  set = set()

for _k in API_KEYS:
    _key_pool.put(_k)


def _acquire_key() -> str:
    try:
        return _key_pool.get(timeout=120)
    except queue.Empty:
        active = [k for k in API_KEYS if k not in _key_blacklist]
        if not active:
            raise RuntimeError("All API keys have been retired — cannot continue.")
        raise RuntimeError("Key acquire timed out after 120 s (pool may be exhausted).")


def _release_key(key: str, blacklist: bool = False) -> None:
    with _key_total_lock:
        _key_total[key] += 1
    if blacklist:
        _key_blacklist.add(key)
        tqdm.write(
            f"  [KEY RETIRED] ...{key[-10:]} — auth failure, removed from pool "
            f"({_key_pool.qsize()} keys remaining)"
        )
    else:
        _key_pool.put(key)


def print_key_stats() -> None:
    print("\nAPI key usage summary:")
    for i, key in enumerate(API_KEYS):
        short  = f"...{key[-10:]}"
        status = "RETIRED" if key in _key_blacklist else "active"
        print(f"  Key {i + 1:>2} ({short}): {_key_total[key]:>6} requests  [{status}]")


# ═════════════════════════════════════════════════════════════════════════════
# Ollama Cloud client — acquire a key per request, release in finally
# ═════════════════════════════════════════════════════════════════════════════
def _chat(system: str, user: str, model: str, temperature: float = 0.8) -> str:
    key = _acquire_key()
    released = False
    try:
        client = OpenAI(api_key=key, base_url=OLLAMA_CLOUD_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        blacklist = "401" in str(exc) or "auth" in str(exc).lower()
        _release_key(key, blacklist=blacklist)
        released = True
        raise
    finally:
        if not released:
            _release_key(key)


def _chat_json(system: str, user: str, model: str, temperature: float = 0.8) -> str:
    """Like _chat but forces JSON output via response_format. Use for CV generation only."""
    key = _acquire_key()
    released = False
    try:
        client = OpenAI(api_key=key, base_url=OLLAMA_CLOUD_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        blacklist = "401" in str(exc) or "auth" in str(exc).lower()
        _release_key(key, blacklist=blacklist)
        released = True
        raise
    finally:
        if not released:
            _release_key(key)


# ═════════════════════════════════════════════════════════════════════════════
# Generation functions
# ═════════════════════════════════════════════════════════════════════════════
def generate_query(domain: str, style: str) -> str:
    user = f"Tech domain: {domain}\nWriting style: {style}"
    return _chat(QUERY_SYSTEM, user, GENERATOR_MODEL, temperature=1.0).strip()


def generate_different_query(domain: str, style: str, first_query: str) -> str:
    user = (
        f"Tech domain: {domain}\nWriting style: {style}\n\n"
        f"Previous query (make yours clearly different):\n{first_query}"
    )
    return _chat(QUERY_DIFF_SYSTEM, user, GENERATOR_MODEL, temperature=1.0).strip()


def generate_cv(query: str, name: str) -> Dict[str, Any]:
    user = (
        f"Recruiter query: {query}\n\n"
        f"Candidate name to use: {name}\n\n"
        "Output ONLY the JSON object."
    )
    raw = _chat_json(CV_SYSTEM, user, GENERATOR_MODEL, temperature=0.85)
    cv  = _fix_and_parse_json(raw)
    cv  = _coerce_types(cv)
    cv  = sanitize_cv(cv)
    _assert_no_leakage(cv)
    CandidateCV.model_validate(cv)
    return cv


def judge_cv_matches(query: str, cv: Dict[str, Any]) -> Tuple[bool, str]:
    user = (
        f"Recruiter query:\n{query}\n\n"
        f"Candidate CV:\n{json.dumps(cv, indent=2)}\n\n"
        'Output ONLY: {"matches": true|false, "reason": "..."}'
    )
    raw    = _chat(CV_JUDGE_SYSTEM, user, JUDGE_MODEL, temperature=0.1)
    result = _fix_and_parse_json(raw)
    v      = SampleJudgeVerdict.model_validate(result)
    return v.matches, v.reason


# ═════════════════════════════════════════════════════════════════════════════
# Batch generation — 2 domains × 2 queries × 2 sample types = up to 8 samples
# ═════════════════════════════════════════════════════════════════════════════
def generate_batch() -> List[Dict[str, Any]]:
    """
    Pick 2 domains, generate 4 queries and 4 CVs, judge each CV against its query,
    then assemble samples by cross-referencing within/across domains.
    Returns a list of 0 (batch aborted) or 4–8 approved sample dicts.
    """
    domain_a, domain_b = random.choice(SAFE_DOMAIN_PAIRS)

    # ── Generate 4 queries (q_a2 is explicitly different from q_a1; same for B) ─
    try:
        q_a1 = generate_query(domain_a, random.choice(STYLE_MODES))
        q_a2 = generate_different_query(domain_a, random.choice(STYLE_MODES), q_a1)
        q_b1 = generate_query(domain_b, random.choice(STYLE_MODES))
        q_b2 = generate_different_query(domain_b, random.choice(STYLE_MODES), q_b1)
    except Exception as exc:
        tqdm.write(f"  [QUERY err] {str(exc)[:80]}")
        return []

    # ── Generate one positive CV per query ───────────────────────────────────
    query_map: Dict[str, str] = {"a1": q_a1, "a2": q_a2, "b1": q_b1, "b2": q_b2}
    cvs: Dict[str, Optional[Dict]] = {}
    for key, query in query_map.items():
        try:
            cvs[key] = generate_cv(query, _unique_arab_name())
        except Exception as exc:
            tqdm.write(f"  [CV {key} err] {str(exc)[:80]}")
            cvs[key] = None

    if any(v is None for v in cvs.values()):
        tqdm.write("  [BATCH abort] one or more CV generations failed")
        return []

    # ── Judge: does each CV satisfy its own query? ───────────────────────────
    for key, query in query_map.items():
        try:
            ok, reason = judge_cv_matches(query, cvs[key])
            if not ok:
                tqdm.write(f"  [CV {key} mismatch] {reason[:80]}")
                return []
        except Exception as exc:
            tqdm.write(f"  [JUDGE {key} err] {str(exc)[:80]}")
            return []

    # ── Assemble easy-negative samples (always safe — different domains) ─────
    samples = [
        {"anchor": q_a1, "positive": cvs["a1"], "negative": cvs["b1"],
         "negative_type": "easy", "anchor_domain": domain_a, "negative_domain": domain_b},
        {"anchor": q_a2, "positive": cvs["a2"], "negative": cvs["b2"],
         "negative_type": "easy", "anchor_domain": domain_a, "negative_domain": domain_b},
        {"anchor": q_b1, "positive": cvs["b1"], "negative": cvs["a1"],
         "negative_type": "easy", "anchor_domain": domain_b, "negative_domain": domain_a},
        {"anchor": q_b2, "positive": cvs["b2"], "negative": cvs["a2"],
         "negative_type": "easy", "anchor_domain": domain_b, "negative_domain": domain_a},
    ]

    # ── Guardrail: ensure hard negatives are actually negatives ──────────────
    if ENABLE_CROSS_JUDGE:
        try:
            a2_matches_a1, _ = judge_cv_matches(q_a1, cvs["a2"])
            a1_matches_a2, _ = judge_cv_matches(q_a2, cvs["a1"])
            b2_matches_b1, _ = judge_cv_matches(q_b1, cvs["b2"])
            b1_matches_b2, _ = judge_cv_matches(q_b2, cvs["b1"])

            if not (a2_matches_a1 or a1_matches_a2):
                samples.extend([
                    {"anchor": q_a1, "positive": cvs["a1"], "negative": cvs["a2"],
                     "negative_type": "hard", "anchor_domain": domain_a, "negative_domain": domain_a},
                    {"anchor": q_a2, "positive": cvs["a2"], "negative": cvs["a1"],
                     "negative_type": "hard", "anchor_domain": domain_a, "negative_domain": domain_a},
                ])
            else:
                tqdm.write("  [DOMAIN A SAMPLES REJECTED] False Negative intra-domain overlap detected.")

            if not (b2_matches_b1 or b1_matches_b2):
                samples.extend([
                    {"anchor": q_b1, "positive": cvs["b1"], "negative": cvs["b2"],
                     "negative_type": "hard", "anchor_domain": domain_b, "negative_domain": domain_b},
                    {"anchor": q_b2, "positive": cvs["b2"], "negative": cvs["b1"],
                     "negative_type": "hard", "anchor_domain": domain_b, "negative_domain": domain_b},
                ])
            else:
                tqdm.write("  [DOMAIN B SAMPLES REJECTED] False Negative intra-domain overlap detected.")

        except Exception as exc:
            tqdm.write(f"  [CROSS-JUDGE err] {str(exc)[:80]}")
            return []
    else:
        samples.extend([
            {"anchor": q_a1, "positive": cvs["a1"], "negative": cvs["a2"],
             "negative_type": "hard", "anchor_domain": domain_a, "negative_domain": domain_a},
            {"anchor": q_a2, "positive": cvs["a2"], "negative": cvs["a1"],
             "negative_type": "hard", "anchor_domain": domain_a, "negative_domain": domain_a},
            {"anchor": q_b1, "positive": cvs["b1"], "negative": cvs["b2"],
             "negative_type": "hard", "anchor_domain": domain_b, "negative_domain": domain_b},
            {"anchor": q_b2, "positive": cvs["b2"], "negative": cvs["b1"],
             "negative_type": "hard", "anchor_domain": domain_b, "negative_domain": domain_b},
        ])

    return samples


# ═════════════════════════════════════════════════════════════════════════════
# Multi-file output manager
# ═════════════════════════════════════════════════════════════════════════════
class PartFileWriter:
    """Round-robin writes to NUM_FILES JSONL part files with per-file locks."""

    def __init__(self, output_dir: Path, num_files: int) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._paths   = [output_dir / f"triplets_part_{i:03d}_{ts}.jsonl" for i in range(num_files)]
        self._handles = [open(p, "a", encoding="utf-8") for p in self._paths]
        self._locks   = [threading.Lock() for _ in range(num_files)]
        self._idx      = 0
        self._idx_lock = threading.Lock()

    @property
    def paths(self) -> List[Path]:
        return self._paths

    def write(self, record: Dict[str, Any]) -> None:
        with self._idx_lock:
            idx        = self._idx % len(self._handles)
            self._idx += 1
        with self._locks[idx]:
            self._handles[idx].write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handles[idx].flush()

    def close(self) -> None:
        for fh in self._handles:
            try:
                fh.close()
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 68)
    print("  Triplet Dataset Generator — Ollama Cloud  (v3)")
    print(f"  Generator : {GENERATOR_MODEL}  |  Judge: {JUDGE_MODEL}")
    print(f"  Keys      : {len(API_KEYS)} API keys  |  Workers: {MAX_WORKERS}")
    print(f"  Target    : {TARGET_COUNT:,} samples")
    print(f"  Batch     : 2 domains × 2 queries × 2 types = up to 8 samples/batch")
    print(f"  Output    : {NUM_FILES} part files in {OUTPUT_DIR}/")
    print(f"  Cross-judge guardrail: {'enabled' if ENABLE_CROSS_JUDGE else 'disabled'}")
    print("=" * 68)

    writer        = PartFileWriter(OUTPUT_DIR, NUM_FILES)
    approved      = 0
    batches       = 0
    _counter_lock = threading.Lock()

    print("\nOutput files:")
    for p in writer.paths:
        print(f"  {p.name}")
    print()

    try:
        with tqdm(total=TARGET_COUNT, desc="Approved samples", unit="s") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                pending: set = {pool.submit(generate_batch) for _ in range(MAX_WORKERS)}

                while approved < TARGET_COUNT:
                    done, pending = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        with _counter_lock:
                            batches += 1

                        try:
                            samples = fut.result()
                        except Exception as exc:
                            tqdm.write(f"  [FATAL] {exc}")
                            samples = []

                        n = min(len(samples), TARGET_COUNT - approved)
                        for sample in samples[:n]:
                            writer.write(sample)
                        with _counter_lock:
                            approved += n
                        if n:
                            pbar.update(n)
                            pbar.set_postfix(
                                approved=approved,
                                batches=batches,
                                rate=f"{approved / max(batches, 1):.1f}/batch",
                            )

                        if approved < TARGET_COUNT:
                            pending.add(pool.submit(generate_batch))

    except KeyboardInterrupt:
        tqdm.write("\n[INTERRUPTED] Flushing files and shutting down ...")
    finally:
        writer.close()

    print(f"\nDone.")
    print(f"  Approved : {approved:,} samples")
    print(f"  Batches  : {batches:,}")
    print(f"  Rate     : {approved / max(batches, 1):.1f} samples/batch")
    print(f"\nTo merge all parts:")
    print(f"  cat {OUTPUT_DIR}/triplets_part_*.jsonl > triplets_all.jsonl")

    print_key_stats()


if __name__ == "__main__":
    main()
