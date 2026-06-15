"""
CV Generation Script — Ollama Cloud
=====================================
Identical pipeline to generate_cv_data.py but runs against the Ollama Cloud
inference API with:
  - API key rotation across multiple keys (round-robin, thread-safe)
  - Arab-world names, cities, universities, and companies
  - Post-generation consistency enforcement (email/name/location sync,
    years_of_experience recalculated from work history)

Output: generated_cvs_cloud.jsonl  (one JSON record per line)
"""

import json
import random
import re
import threading
import time
import queue
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"   # adjust if auth keeps failing

# Keys are loaded from ollama_api_keys.txt (one key per line, ignored by git)
_KEYS_FILE = Path(__file__).parent / "ollama_api_keys.txt"
if not _KEYS_FILE.exists():
    raise FileNotFoundError(
        f"'{_KEYS_FILE}' not found. "
        "Create it with one Ollama API key per line."
    )
API_KEYS: List[str] = [
    line.strip()
    for line in _KEYS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not API_KEYS:
    raise ValueError(f"'{_KEYS_FILE}' is empty. Add at least one Ollama API key.")

GENERATOR_MODEL = "gpt-oss:20b-cloud"   # free tier on Ollama Cloud — good JSON
JUDGE_MODEL     = "gpt-oss:120b-cloud"   # larger reasoning model for judging

FILTERS_FILE  = "deepseek_tech_filters_5k.jsonl"
OUTPUT_FILE   = "generated_cvs_cloud.jsonl"

TARGET_PAIRS  = 20000
MAX_WORKERS   = len(API_KEYS)  # one worker slot per key → all keys run in parallel
MAX_RETRIES   = 3

# ─────────────────────────────────────────────────────────────────────────────
# API key pool — one request per key at a time, no exceptions.
#
# _key_pool  is a blocking Queue pre-loaded with every key.
# _acquire_key()  removes a key from the pool (blocks if all are in use).
# _release_key()  returns the key to the pool inside a finally block.
#
# This guarantees at most 1 concurrent request per key regardless of how many
# worker threads are running.
# ─────────────────────────────────────────────────────────────────────────────
_key_pool:  queue.Queue = queue.Queue()
_key_total: Dict[str, int] = {k: 0 for k in API_KEYS}
_key_total_lock = threading.Lock()

for _k in API_KEYS:
    _key_pool.put(_k)


_key_blacklist: set = set()

def _acquire_key() -> str:
    """Block until a free key is available (timeout=120 s to avoid deadlock if pool empties)."""
    key = _key_pool.get(timeout=120)
    return key


def _release_key(key: str, blacklist: bool = False) -> None:
    """Record the completed request. If blacklist=True, retire the key instead of returning it."""
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
        print(f"  Key {i + 1:>2} ({short}): {_key_total[key]:>6} total requests  [{status}]")

# ─────────────────────────────────────────────────────────────────────────────
# Arab-world candidate pools — used in generation prompts
# ─────────────────────────────────────────────────────────────────────────────
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
    "Giza, Egypt", "Mecca, Saudi Arabia", "Medina, Saudi Arabia",
]
_ARAB_UNIVERSITIES = [
    "Cairo University", "Alexandria University", "Ain Shams University",
    "American University in Cairo", "Zewail City of Science and Technology",
    "German University in Cairo", "King Saud University",
    "King Abdulaziz University", "King Abdullah University of Science and Technology (KAUST)",
    "UAE University", "American University of Sharjah", "Khalifa University",
    "Lebanese American University", "American University of Beirut",
    "Jordan University of Science and Technology", "University of Jordan",
    "Qatar University", "Sultan Qaboos University", "University of Bahrain",
]
_ARAB_TECH_COMPANIES = [
    "Careem", "Noon", "Talabat", "Anghami", "Property Finder", "Fetchr",
    "Swvl", "Vezeeta", "Instabug", "Fawry", "Paymob", "Halan", "Sarwa",
    "stc", "du Telecom", "Etisalat (e&)", "Batelco", "Zain", "Mobily",
    "Saudi Aramco Digital", "ADNOC Digital", "Majid Al Futtaim Technology",
    "Amazon.ae", "Microsoft MENA", "Google Cairo", "IBM Egypt",
    "Oracle MENA", "SAP ME", "Huawei MENA", "Ericsson MENA", "Vodafone Egypt",
    "Orange Egypt", "Telecom Egypt", "STC Solutions", "Elm Company",
]

def _random_arab_name() -> str:
    pool = _ARAB_MALE_NAMES + _ARAB_FEMALE_NAMES
    first = random.choice(pool)
    last  = random.choice(_ARAB_LAST_NAMES)
    return f"{first} {last}"

def _random_arab_city() -> str:
    return random.choice(_ARAB_CITIES)

# ─────────────────────────────────────────────────────────────────────────────
# Concrete CV example — Arab world context
# ─────────────────────────────────────────────────────────────────────────────
CV_EXAMPLE = """\
{
  "email": "omar.mansour.dev@gmail.com",
  "full_name": "Omar Mansour",
  "location": "Cairo, Egypt",
  "summary": "Full-stack engineer with 5 years of experience building scalable web platforms across fintech and e-commerce. Experienced in Python/Django backends and React frontends, with a strong focus on API design and cloud deployments.",
  "years_of_experience": 5.0,
  "seniority_level": "mid",
  "primary_domain": "software_engineering",
  "has_github": true,
  "has_linkedin": true,
  "has_portfolio": false,
  "skills": [
    {"category": "Languages",   "skill_name": "Python",     "proficiency": "expert"},
    {"category": "Languages",   "skill_name": "JavaScript", "proficiency": "advanced"},
    {"category": "Frameworks",  "skill_name": "Django",     "proficiency": "expert"},
    {"category": "Frameworks",  "skill_name": "React",      "proficiency": "advanced"},
    {"category": "Databases",   "skill_name": "PostgreSQL", "proficiency": "advanced"},
    {"category": "DevOps",      "skill_name": "Docker",     "proficiency": "intermediate"},
    {"category": "Cloud",       "skill_name": "AWS",        "proficiency": "intermediate"}
  ],
  "work_experience": [
    {
      "company": "Fawry",
      "job_title": "Backend Engineer",
      "start_date": "2022-01",
      "end_date": null,
      "is_current": true,
      "location": "Cairo, Egypt",
      "description": "Developed and maintained payment processing microservices in Python/Django handling 1M+ daily transactions. Designed RESTful APIs consumed by 30+ merchant integrations. Reduced average API response time by 35% through query optimization and Redis caching.",
      "technologies": ["Python", "Django", "PostgreSQL", "Redis", "Docker", "AWS"]
    },
    {
      "company": "Noon",
      "job_title": "Junior Software Engineer",
      "start_date": "2019-07",
      "end_date": "2021-12",
      "is_current": false,
      "location": "Cairo, Egypt",
      "description": "Built product catalog and search features for Egypt's largest e-commerce platform. Contributed to migrating monolith order flow to microservices. Wrote unit and integration tests increasing coverage from 42% to 78%.",
      "technologies": ["Python", "Flask", "PostgreSQL", "Elasticsearch", "React"]
    }
  ],
  "education": [
    {
      "institution": "Cairo University",
      "degree": "Bachelor of Science",
      "field_of_study": "Computer Science",
      "start_date": "2015-09",
      "end_date": "2019-06",
      "gpa": 3.6,
      "honors": null
    }
  ],
  "projects": [
    {
      "name": "OpenBill",
      "description": "Open-source billing library for Django applications supporting multi-currency invoicing and PDF export, 250+ GitHub stars.",
      "technologies": ["Python", "Django", "WeasyPrint"],
      "start_date": "2021-03",
      "end_date": "2021-08",
      "url": "github.com/omansour/openbill"
    },
    {
      "name": "SearchKit",
      "description": "Lightweight Elasticsearch wrapper for Django REST Framework with auto-pagination and field boosting, used in 3 production services.",
      "technologies": ["Python", "Elasticsearch", "Django REST Framework"],
      "start_date": "2020-06",
      "end_date": "2020-11",
      "url": null
    },
    {
      "name": "DevDash",
      "description": "Internal developer dashboard for monitoring microservice health, deployment status, and log aggregation across AWS ECS clusters.",
      "technologies": ["React", "AWS", "Docker"],
      "start_date": "2023-02",
      "end_date": "2023-05",
      "url": null
    }
  ],
  "certifications": ["AWS Certified Developer – Associate"],
  "languages": ["Arabic", "English"],
  "contact_info": {
    "email": "omar.mansour.dev@gmail.com",
    "phone": "+20-100-1234567",
    "location": "Cairo, Egypt",
    "full_name": "Omar Mansour",
    "linkedin_url": "linkedin.com/in/omar-mansour-dev",
    "github_url": "github.com/omansour",
    "portfolio_url": null,
    "other_links": []
  },
  "miscellaneous": [],
  "prescore_v2": null
}"""

# ─────────────────────────────────────────────────────────────────────────────
# Consistency helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ym(date_str: Any) -> Tuple[int, int]:
    """Parse 'YYYY-MM' → (year, month). Returns (0, 0) on failure."""
    try:
        parts = str(date_str).split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


def _months_between(start: str, end: str) -> int:
    sy, sm = _parse_ym(start)
    ey, em = _parse_ym(end)
    if sy == 0 or ey == 0:
        return 0
    return max(0, (ey - sy) * 12 + (em - sm))


def _compute_experience_years(work_experience: List[Dict]) -> float:
    now = datetime.now().strftime("%Y-%m")
    total = 0
    for job in work_experience:
        start = job.get("start_date") or ""
        end   = job.get("end_date") or (now if job.get("is_current") else now)
        total += _months_between(start, end)
    return round(total / 12, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Fields that belong only in a job filter, never in a CV
# ─────────────────────────────────────────────────────────────────────────────
_FILTER_ONLY_FIELDS = {
    "employment_type", "contract_duration", "timezone", "remote_ok",
    "visa_sponsorship", "citizenship_required", "security_clearance",
    "no_c2c", "direct_hire", "startup_experience", "min_experience_years",
    "preferred_experience_years", "role", "industry",
}

_VALID_PROFICIENCIES = {"beginner", "intermediate", "advanced", "expert"}
_VALID_SENIORITIES   = {"junior", "mid", "senior", "lead", "staff", "principal"}
_DATE_RE             = re.compile(r"^\d{4}-\d{2}$")

# Thread-safe set of already-used full_names to prevent duplicates across workers
_used_names:      set = set()
_used_names_lock        = threading.Lock()

def _unique_arab_name() -> str:
    """Return a random Arab name that hasn't been used yet in this run."""
    for _ in range(50):
        name = _random_arab_name()
        with _used_names_lock:
            if name not in _used_names:
                _used_names.add(name)
                return name
    # All combos exhausted — still return something unique via suffix
    name = _random_arab_name() + f" {random.randint(2, 9)}"
    with _used_names_lock:
        _used_names.add(name)
    return name


def _fix_date(d: Any) -> Any:
    """Return d unchanged if it matches YYYY-MM, else None."""
    if d is None:
        return None
    return d if _DATE_RE.match(str(d)) else None


# ─────────────────────────────────────────────────────────────────────────────
# Sanitiser + consistency enforcer
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_cv(cv: Dict[str, Any]) -> Dict[str, Any]:
    # ── Strip filter-only fields ──────────────────────────────────────────────
    for f in _FILTER_ONLY_FIELDS:
        cv.pop(f, None)

    # ── Fix pipe-separated proficiency ────────────────────────────────────────
    for skill in cv.get("skills", []):
        prof = str(skill.get("proficiency", "intermediate"))
        if "|" in prof:
            for token in re.split(r"[|/,]", prof):
                t = token.strip().lower()
                if t in _VALID_PROFICIENCIES:
                    skill["proficiency"] = t
                    break
            else:
                skill["proficiency"] = "intermediate"
        else:
            # normalise casing even without pipes ("Advanced" → "advanced")
            skill["proficiency"] = prof.strip().lower() if prof.strip().lower() in _VALID_PROFICIENCIES else "intermediate"

    # ── Fix seniority_level — normalise casing then validate ──────────────────
    sl = str(cv.get("seniority_level", "mid")).strip().lower()
    if "|" in sl or sl not in _VALID_SENIORITIES:
        for token in re.split(r"[|/,]", sl):
            t = token.strip().lower()
            if t in _VALID_SENIORITIES:
                cv["seniority_level"] = t
                break
        else:
            cv["seniority_level"] = "mid"
    else:
        cv["seniority_level"] = sl   # store lowercase

    # ── Fix primary_domain placeholder ───────────────────────────────────────
    pd = str(cv.get("primary_domain", ""))
    if "e.g." in pd or "|" in pd or not pd:
        cv["primary_domain"] = "software_engineering"

    # ── Replace obvious placeholder values ───────────────────────────────────
    if cv.get("email", "") in {"realistic-fake@example.com", ""}:
        cv["email"] = f"candidate{random.randint(1000, 9999)}@techmail.io"
    if cv.get("full_name", "") in {"First Last", ""}:
        cv["full_name"] = _unique_arab_name()
    if cv.get("location", "") in {"City, Country", ""}:
        cv["location"] = _random_arab_city()

    # ── Fix work_experience dates and is_current/end_date logic ──────────────
    work = cv.get("work_experience", [])
    for job in work:
        job["start_date"] = _fix_date(job.get("start_date"))
        job["end_date"]   = _fix_date(job.get("end_date"))
        # is_current=True must not have an end_date
        if job.get("is_current"):
            job["end_date"] = None
        # is_current=False must have an end_date; derive one if missing
        elif not job.get("end_date") and job.get("start_date"):
            # set end_date = start_date + 24 months as a safe fallback
            sy, sm = _parse_ym(job["start_date"])
            if sy:
                em = sm + 24
                ey = sy + (em - 1) // 12
                em = (em - 1) % 12 + 1
                job["end_date"] = f"{ey}-{em:02d}"

    # ── Fix overlapping jobs — sort by start then push overlapping end dates ──
    valid_work = [j for j in work if j.get("start_date")]
    valid_work.sort(key=lambda j: j["start_date"])
    for i in range(len(valid_work) - 1):
        end_i      = valid_work[i].get("end_date") or ""
        start_next = valid_work[i + 1].get("start_date") or ""
        if end_i and start_next and end_i > start_next:
            # cap end_date of earlier job to one month before the next starts
            sy, sm = _parse_ym(start_next)
            if sy:
                sm -= 1
                if sm == 0:
                    sm, sy = 12, sy - 1
                valid_work[i]["end_date"] = f"{sy}-{sm:02d}"
    cv["work_experience"] = valid_work

    # ── Enforce consistency: contact_info always mirrors top-level ────────────
    ci = cv.get("contact_info")
    if not isinstance(ci, dict):
        ci = {}
    ci["email"]     = cv.get("email", ci.get("email", ""))
    ci["full_name"] = cv.get("full_name", ci.get("full_name", ""))
    ci["location"]  = cv.get("location", ci.get("location", ""))
    for field in ("linkedin_url", "github_url", "portfolio_url"):
        val = str(ci.get(field, "") or "")
        if val in {"linkedin.com/in/handle", "github.com/handle", ""}:
            ci[field] = None
    ci.setdefault("phone", "")
    ci.setdefault("other_links", [])
    cv["contact_info"] = ci

    # ── Enforce consistency: years_of_experience vs work history ─────────────
    if valid_work:
        computed = _compute_experience_years(valid_work)
        claimed  = float(cv.get("years_of_experience") or 0)
        if computed > 0 and abs(computed - claimed) > 1.5:
            cv["years_of_experience"] = computed

    # ── Ensure required fields exist ─────────────────────────────────────────
    cv.setdefault("miscellaneous",  [])
    cv.setdefault("prescore_v2",    None)
    cv.setdefault("certifications", [])
    cv.setdefault("languages",      ["Arabic", "English"])
    cv.setdefault("projects",       [])

    return cv


def _assert_no_leakage(cv: Dict[str, Any]) -> None:
    dump = json.dumps(cv)
    bad_tokens = [
        "same as top-level",
        "junior|mid|senior",
        "beginner|intermediate",
        "e.g. software",
        "3–5 sentences",
        "2–3 sentence",
        "What was built",
    ]
    for tok in bad_tokens:
        if tok in dump:
            raise ValueError(f"Template leakage: '{tok}'")


# ─────────────────────────────────────────────────────────────────────────────
# System prompts — generator
# Built dynamically so the actual pool lists are injected verbatim.
# ─────────────────────────────────────────────────────────────────────────────
_all_first_names  = ", ".join(_ARAB_MALE_NAMES + _ARAB_FEMALE_NAMES)
_all_last_names   = ", ".join(_ARAB_LAST_NAMES)
_all_cities       = ", ".join(_ARAB_CITIES)
_all_universities = ", ".join(_ARAB_UNIVERSITIES)
_all_companies    = ", ".join(_ARAB_TECH_COMPANIES)

_ARAB_CONTEXT_NOTE = f"""\
CULTURAL CONTEXT (apply to every generated CV):
- First names — choose from or like: {_all_first_names}
- Last names  — choose from or like: {_all_last_names}
- Locations   — choose from or like: {_all_cities}
- Universities — choose from (or use a well-known international university) or like: {_all_universities}
- Companies   — choose from (prefer these, mix as needed) or like: {_all_companies}
- Languages spoken: Arabic (always include), English, and optionally French (Maghreb) or others.
- Phone numbers: use regional formats (+20 Egypt, +971 UAE, +966 KSA, +962 Jordan, +965 Kuwait, +974 Qatar, etc.)
"""

GENERATOR_SYSTEM_FULFILLS = f"""You are a synthetic data generator for tech recruiting datasets.

TASK: Generate ONE realistic parsed-CV JSON for a candidate who CLEARLY FULFILLS the given filter criterion.

The filter is a single key-value pair such as:
  {{"skills": ["Python", "Go"]}}   or   {{"min_experience_years": 5}}   or   {{"seniority_level": "Senior"}}

{_ARAB_CONTEXT_NOTE}

RULES:
1. Satisfy the filter completely:
   - skills           → every listed skill in "skills" array with proficiency advanced or expert
   - min_experience_years → years_of_experience >= that number; work history must add up
   - seniority_level / role → CV seniority_level and job titles must match
   - education_level  → correct degree and field
   - certifications   → cert must appear in "certifications" array
2. Internal consistency (MANDATORY):
   - years_of_experience = approximate sum of work history durations
   - skills listed in the skills array must appear in work experience descriptions/technologies
   - email in contact_info must be identical to top-level email
   - full_name in contact_info must be identical to top-level full_name
   - location in contact_info must be identical to top-level location
3. All enum fields must be a SINGLE lowercase word — never pipe-separated:
   - proficiency    → one of: beginner, intermediate, advanced, expert
   - seniority_level → one of: junior, mid, senior, lead, staff, principal
4. Do NOT add filter-specific fields (employment_type, remote_ok, contract_duration, etc.) to the CV.
5. Generate 2–4 projects (never just 1). Each project must differ in domain/tech from the others.
6. Dates must be in YYYY-MM format only. Never write "ASAP", "present", or plain text in date fields.
7. If is_current is true, end_date must be null. If is_current is false, end_date must be set.

OUTPUT: ONLY a raw JSON object — no markdown fences, no explanation text.

EXAMPLE (imitate this structure and level of detail — change ALL values to fit the filter and use Arab world context):
{CV_EXAMPLE}
"""

GENERATOR_SYSTEM_NOT_FULFILLS = f"""You are a synthetic data generator for tech recruiting datasets.

TASK: Generate ONE realistic parsed-CV JSON for a candidate who DOES NOT FULFILL the given filter criterion.

The filter is a single key-value pair such as:
  {{"skills": ["Python", "Go"]}}   or   {{"min_experience_years": 5}}   or   {{"seniority_level": "Senior"}}

{_ARAB_CONTEXT_NOTE}

RULES:
1. Clearly fail the filter:
   - skills           → do NOT include the required skill(s), or only at "beginner" level
   - min_experience_years → years_of_experience at least 1 year BELOW minimum
   - seniority_level  → clearly lower level (e.g. junior when senior required)
   - education_level  → wrong degree type or unrelated field
   - certifications   → do NOT include the required cert
2. The candidate must still be a REAL, PLAUSIBLE tech professional — just not a fit for this criterion.
3. Internal consistency (MANDATORY):
   - years_of_experience = approximate sum of work history durations
   - skills listed in the skills array must appear in work experience descriptions/technologies
   - email in contact_info must be identical to top-level email
   - full_name in contact_info must be identical to top-level full_name
   - location in contact_info must be identical to top-level location
4. All enum fields must be a SINGLE word — never pipe-separated:
   - proficiency    → one of: beginner, intermediate, advanced, expert
   - seniority_level → one of: junior, mid, senior, lead, staff, principal
5. Do NOT add filter-specific fields (employment_type, remote_ok, contract_duration, etc.) to the CV.

OUTPUT: ONLY a raw JSON object — no markdown fences, no explanation text.

EXAMPLE (imitate structure — change ALL values, make candidate NOT fit the filter, use Arab world context):
{CV_EXAMPLE}
"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompts — judge
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM_FULFILLS = """You are a quality-control judge for a tech recruiting dataset.

You receive a single-criterion filter and a generated CV that SHOULD FULFILL it.

Evaluate:
  1. FILTER MATCH — does the CV clearly satisfy the criterion?
  2. CONSISTENCY — are these internally consistent?
     - email/full_name/location match between top-level and contact_info
     - years_of_experience roughly matches work history dates
     - claimed skills appear in work experience descriptions or technologies
  3. QUALITY — realistic, not template placeholders, no pipe-separated enum values.

APPROVE (approved: true)  — all three pass.
REJECT  (approved: false) — any issue found. Give one specific, actionable sentence.

Respond with ONLY valid JSON:
{"approved": true, "reason": "explanation"}"""

JUDGE_SYSTEM_NOT_FULFILLS = """You are a quality-control judge for a tech recruiting dataset.

You receive a single-criterion filter and a generated CV that SHOULD NOT FULFILL it.

Evaluate:
  1. FILTER MISS — does the CV clearly FAIL the criterion? The gap must be unambiguous.
  2. CONSISTENCY — are these internally consistent?
     - email/full_name/location match between top-level and contact_info
     - years_of_experience roughly matches work history dates
     - claimed skills appear in work experience descriptions or technologies
  3. QUALITY — realistic, not template placeholders, no pipe-separated enum values.

APPROVE (approved: true)  — clearly fails filter AND passes consistency and quality.
REJECT  (approved: false) — any issue. Give one specific, actionable sentence.

Respond with ONLY valid JSON:
{"approved": true, "reason": "explanation"}"""

# ─────────────────────────────────────────────────────────────────────────────
# System prompts — fixer
# ─────────────────────────────────────────────────────────────────────────────
FIXER_SYSTEM_FULFILLS = f"""You are a CV repair agent for a tech recruiting dataset.

A generated CV was REJECTED. Fix ONLY the specific issue described by the judge.
Keep the same person and history — make the minimal targeted change.
Ensure contact_info mirrors the top-level email, full_name, and location.

OUTPUT: ONLY the corrected raw JSON object. No markdown, no explanation.

Reference structure:
{CV_EXAMPLE}
"""

FIXER_SYSTEM_NOT_FULFILLS = f"""You are a CV repair agent for a tech recruiting dataset.

A generated CV was REJECTED. Fix ONLY the specific issue described by the judge.
The candidate must still be a realistic tech professional who clearly misses the filter criterion.
Ensure contact_info mirrors the top-level email, full_name, and location.

OUTPUT: ONLY the corrected raw JSON object. No markdown, no explanation.

Reference structure:
{CV_EXAMPLE}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON object found in model output")


def _chat(system: str, user: str, model: str, temperature: float = 0.7) -> str:
    key = _acquire_key()
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
        return response.choices[0].message.content
    finally:
        _release_key(key)


# ─────────────────────────────────────────────────────────────────────────────
# Generation, judging, fixing
# ─────────────────────────────────────────────────────────────────────────────
def generate_cv(atomic_filter: Dict[str, Any], fulfills: bool) -> Dict[str, Any]:
    system  = GENERATOR_SYSTEM_FULFILLS if fulfills else GENERATOR_SYSTEM_NOT_FULFILLS
    outcome = "FULFILL" if fulfills else "NOT FULFILL"
    name_hint = _unique_arab_name()
    user = (
        f"Generate a CV for a candidate who should {outcome} this filter:\n\n"
        f"{json.dumps(atomic_filter, indent=2)}\n\n"
        f"Use this specific candidate name: {name_hint}\n\n"
        "Return ONLY the JSON object."
    )
    raw = _chat(system, user, GENERATOR_MODEL, temperature=0.85)
    cv  = _extract_json(raw)
    cv  = sanitize_cv(cv)
    _assert_no_leakage(cv)
    return cv


def judge_cv(
    atomic_filter: Dict[str, Any], cv: Dict[str, Any], fulfills: bool
) -> Tuple[bool, str]:
    system = JUDGE_SYSTEM_FULFILLS if fulfills else JUDGE_SYSTEM_NOT_FULFILLS
    label  = "SHOULD FULFILL" if fulfills else "SHOULD NOT FULFILL"
    user = (
        f"Filter ({label}):\n{json.dumps(atomic_filter, indent=2)}\n\n"
        f"Candidate CV:\n{json.dumps(cv, indent=2)}\n\n"
        'Respond with ONLY valid JSON: {"approved": true|false, "reason": "..."}'
    )
    raw    = _chat(system, user, JUDGE_MODEL, temperature=0.1)
    result = _extract_json(raw)
    return bool(result["approved"]), str(result.get("reason", ""))


def fix_cv(
    atomic_filter: Dict[str, Any],
    cv: Dict[str, Any],
    fulfills: bool,
    rejection_reason: str,
) -> Dict[str, Any]:
    system  = FIXER_SYSTEM_FULFILLS if fulfills else FIXER_SYSTEM_NOT_FULFILLS
    outcome = "FULFILL" if fulfills else "NOT FULFILL"
    user = (
        f"Filter (candidate should {outcome} it):\n{json.dumps(atomic_filter, indent=2)}\n\n"
        f"Current CV:\n{json.dumps(cv, indent=2)}\n\n"
        f"Judge's rejection reason: {rejection_reason}\n\n"
        "Return ONLY the corrected JSON object."
    )
    raw   = _chat(system, user, GENERATOR_MODEL, temperature=0.5)
    fixed = _extract_json(raw)
    fixed = sanitize_cv(fixed)
    _assert_no_leakage(fixed)
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# Per-filter processing with feedback loop
# ─────────────────────────────────────────────────────────────────────────────
def process_filter(atomic_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    approved: List[Dict[str, Any]] = []

    for fulfills in (True, False):
        label    = "fulfills" if fulfills else "not-fulfills"
        accepted = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Step 1: fresh generation
                cv         = generate_cv(atomic_filter, fulfills)
                ok, reason = judge_cv(atomic_filter, cv, fulfills)

                if not ok:
                    tqdm.write(f"  [REJECTED {label}] attempt {attempt}/{MAX_RETRIES}: {reason}")
                    # Step 2: one fix attempt with feedback
                    try:
                        fixed_cv   = fix_cv(atomic_filter, cv, fulfills, reason)
                        ok, reason = judge_cv(atomic_filter, fixed_cv, fulfills)
                        if ok:
                            cv = fixed_cv
                        else:
                            tqdm.write(f"  [FIX FAILED {label}] attempt {attempt}/{MAX_RETRIES}: {reason}")
                    except Exception as fix_exc:
                        tqdm.write(f"  [FIX ERROR {label}] attempt {attempt}: {fix_exc}")

                if ok:
                    accepted = True
                    approved.append({
                        "parsed_cv":       cv,
                        "filter":          atomic_filter,
                        "fulfills_filter": fulfills,
                        "judge_reason":    reason,
                    })
                    break

            except Exception as exc:
                tqdm.write(f"  [ERROR {label}] attempt {attempt}/{MAX_RETRIES}: {exc}")
                time.sleep(2)

        if not accepted:
            tqdm.write(f"  [GAVE UP {label}] filter={json.dumps(atomic_filter)}")

    return approved


# ─────────────────────────────────────────────────────────────────────────────
# Filter loading
# ─────────────────────────────────────────────────────────────────────────────
def load_atomic_filters(path: str) -> List[Dict[str, Any]]:
    atomic: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            filt = obj.get("target_filters", obj)
            for k, v in filt.items():
                atomic.append({k: v})
    return atomic


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading atomic filters from '{FILTERS_FILE}' ...")
    all_atomic = load_atomic_filters(FILTERS_FILE)
    print(f"  → {len(all_atomic)} atomic filters extracted")

    n       = min(TARGET_PAIRS, len(all_atomic))
    sampled = random.sample(all_atomic, n)
    print(f"  → {n} atomic filters sampled  (TARGET_PAIRS={TARGET_PAIRS})")
    print(f"  → Generator: {GENERATOR_MODEL}  |  Judge: {JUDGE_MODEL}")
    print(f"  → Workers: {MAX_WORKERS}  |  Max retries: {MAX_RETRIES}")
    print(f"  → API keys: {len(API_KEYS)} (load-balanced, all parallel)")
    print(f"  → Output: {OUTPUT_FILE}\n")

    write_lock    = threading.Lock()
    total_written = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_fh:
        with tqdm(total=n, desc="Filters processed") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(process_filter, f): f for f in sampled}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        records = future.result()
                        with write_lock:
                            for rec in records:
                                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                out_fh.flush()
                                total_written += 1
                    except Exception as exc:
                        tqdm.write(f"[FATAL] {exc}")
                    pbar.update(1)
                    pbar.set_postfix(written=total_written)

    print(f"\nDone! {total_written} records written to '{OUTPUT_FILE}'.")
    print(f"Expected up to {n * 2} records ({n} fulfilling + {n} non-fulfilling).")
    print_key_stats()


if __name__ == "__main__":
    main()
