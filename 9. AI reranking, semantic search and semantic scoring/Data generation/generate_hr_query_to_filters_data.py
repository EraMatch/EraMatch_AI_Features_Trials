import os
import json
import random
import threading
import concurrent.futures
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any, Dict
from tqdm import tqdm

# -------------------------------------------------------------
# 0. API Key
# -------------------------------------------------------------
API_KEY_FILENAME = "api_key.txt"

if not os.path.exists(API_KEY_FILENAME):
    raise FileNotFoundError(
        f"Could not find '{API_KEY_FILENAME}'. "
        f"Create a text file with that name containing your DeepSeek API key."
    )

with open(API_KEY_FILENAME, "r", encoding="utf-8") as key_file:
    DEEPSEEK_API_KEY = key_file.read().strip()

if not DEEPSEEK_API_KEY:
    raise ValueError(f"'{API_KEY_FILENAME}' is empty. Paste your API key inside it.")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

GENERATOR_MODEL = "deepseek-v4-flash"
JUDGE_MODEL     = "deepseek-v4-pro"

# -------------------------------------------------------------
# 1. Schemas
# -------------------------------------------------------------
class TrainingSample(BaseModel):
    query: str = Field(..., description="The messy, natural language recruiter prompt.")
    target_filters: Dict[str, Any] = Field(
        ...,
        description=(
            "Dynamic key-value filters extracted from the query. "
            "Include only what is explicitly or strongly implied. "
            "Key count varies by content — could be 1 or 10+."
        )
    )

class JudgeVerdict(BaseModel):
    approved: bool = Field(..., description="True if filters faithfully reflect the query.")
    reason: str = Field(..., description="Brief explanation for approval or rejection.")

# -------------------------------------------------------------
# 2. Domain & Style Config
# -------------------------------------------------------------
TECH_DOMAINS = [
    "Backend (e.g., Go, Rust, Python, Django, Microservices, KAFKA)",
    "Frontend (e.g., React, Next.js, TypeScript, Tailwind, WebAssembly)",
    "DevOps & Cloud (e.g., AWS, Kubernetes, Terraform, CI/CD, Docker)",
    "Data Science & AI (e.g., PyTorch, LLMs, MLOps, LangChain, Pandas)",
    "Cybersecurity (e.g., Penetration Testing, IAM, SOC, SIEM, CISSP)",
    "Mobile (e.g., iOS, Android, Flutter, React Native, Swift, Kotlin)",
    "Embedded & Systems (e.g., C, C++, RTOS, Firmware, FPGA)",
    "QA & Testing (e.g., Selenium, Cypress, JUnit, Load Testing, TDD)",
    "Full-Stack (e.g., MERN, LAMP, Rails, GraphQL, REST APIs)",
    "Platform & Infrastructure (e.g., Kafka, Spark, Flink, Airflow, dbt)",
]

STYLE_MODES = [
    "Extremely brief and impatient (e.g., 'need python guy remote 5yrs')",
    "Corporate job-description copy-paste style with jargon",
    "Conversational and descriptive (e.g., 'We are scaling our team and looking for a rockstar who knows...')",
    "Bullet-point style listing requirements one by one",
    "Casual Slack/chat message with abbreviations and typos",
]

# -------------------------------------------------------------
# 3. Static system prompts — built once at import time.
#    Identical across every call → DeepSeek prefix-caches them.
# -------------------------------------------------------------
_SAMPLE_SCHEMA  = json.dumps(TrainingSample.model_json_schema(), indent=2)
_VERDICT_SCHEMA = json.dumps(JudgeVerdict.model_json_schema(), indent=2)

GENERATOR_SYSTEM = f"""You are an expert HR Data Synthesizer specializing in the Tech industry.
When given a tech domain and writing style, generate a single JSON training sample with:
  "query"          — a messy, realistic recruiter search request
  "target_filters" — the dynamically extracted filters from that query

=== QUERY RULES ===
- Simulate a real tech recruiter or hiring manager pasting a search request.
- Use natural shorthand, abbreviations, and occasional minor typos.
- Let the domain and style naturally drive how many constraints appear (could be 1 or many).

=== FILTER EXTRACTION RULES ===
- Extract ONLY what is explicitly stated or strongly implied in the query.
- Flat key-value map; keys are snake_case strings.
- Values: string, int, bool, or list of strings.
- Include only keys the query warrants. Do NOT pad with nulls or empty lists.
- Represent ambiguity faithfully (e.g., "remote or NYC" → ["Remote", "NYC"]).
- Do NOT invent constraints absent from the query.

=== CRITICAL RULES ===
1. skills vs preferred_skills
   - `skills`           → must-have / required only
   - `preferred_skills` → "a plus", "bonus", "nice to have", "preferred"
   NEVER mix them.
2. "or equivalent" certs → certifications: ["OSCP or equivalent"]
3. Experience phrasing
   - "at least / min / must have X years" → min_experience_years: X
   - "ideally / prefer / looking for ~X years" → preferred_experience_years: X
4. "no c2c / no corp-to-corp / W2 only" → no_c2c: true
5. Contract duration (e.g., "6-month contract") → contract_duration: "6 months"
6. "US citizens only" → citizenship_required: "US citizens only"
7. Candidate's startup background → startup_experience: true  (NOT industry or team_size)
8. LLM examples (e.g., "LLMs like GPT or LLaMA") → extract category only: skills: ["LLMs"]

=== FILTER KEY REFERENCE (non-exhaustive) ===
  role                      string   "Senior Backend Engineer"
  seniority_level           string   "Senior", "Lead", "Staff"
  skills                    list     required skills only
  preferred_skills          list     nice-to-have skills only
  min_experience_years      int      hard minimum (must/at least/min)
  preferred_experience_years int     soft preference (ideally/prefer)
  experience_domain         string   "embedded systems", "ML/DL"
  min_automation_years      int      secondary experience requirement
  location                  str/list "Remote" or ["Remote", "Austin, TX"]
  remote_ok                 bool
  timezone                  string   "PST", "EST", "US timezones"
  employment_type           string   "Full-time", "Contract", "Part-time"
  contract_duration         string   "6 months", "12 months"
  no_c2c                    bool     true when no corp-to-corp
  direct_hire               bool     true when permanent/direct hire
  start_date                string   "ASAP", "Q3 2025"
  visa_sponsorship          bool     false = not available
  citizenship_required      string   "US citizens only"
  security_clearance        string   "TS/SCI", "Secret"
  certifications            list     required certs
  preferred_certifications  list     optional/preferred certs
  education_level           string   "Bachelor's in CS or equivalent"
  languages                 list     ["English", "Spanish"]
  industry                  string   hiring company's industry only
  startup_experience        bool     true = candidate startup background required

Output strictly valid JSON matching this schema:
{_SAMPLE_SCHEMA}
"""

JUDGE_SYSTEM = f"""You are a Quality Control Judge for an AI training dataset.
You receive a recruiter query and dynamically extracted filters.
Determine whether the filters accurately and faithfully represent the query.

=== REJECT (hard failures — any one is enough) ===
  A) A required constraint is missing: role, required skills, strict experience min,
     location, employment type, visa/c2c/citizenship, security clearance, contract duration.
  B) A filter value is factually wrong (wrong number, wrong location, etc.).
  C) A nice-to-have skill is in `skills` instead of `preferred_skills`, or vice versa.
  D) A constraint is hallucinated — not in the query at all.
  E) `industry` is used for candidate background instead of `startup_experience`.
  F) The query sounds obviously machine-generated or unnatural.

=== APPROVE (acceptable omissions — do NOT reject for these) ===
  • Soft-skill preferences ("good communicator", "team player") omitted — not filterable.
  • Minor timezone phrasing simplification ("prefer PST" simplified to timezone: "PST").
  • `seniority_level` omitted when `role` already encodes it ("Senior Backend Engineer").
  • `start_date: ASAP` omitted when urgency was not strongly emphasized.
  • Preferred certifications omitted (only required ones must appear).

Be a fair judge. The goal is accurate, useful training data — not perfection on soft details.

Output strictly valid JSON matching this schema:
{_VERDICT_SCHEMA}
"""

# -------------------------------------------------------------
# 4. Generation & Judge — tiny dynamic user messages only
# -------------------------------------------------------------
def generate_raw_sample(domain: str, style: str) -> TrainingSample:
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM},
            {"role": "user",   "content": f"Domain: {domain}\nStyle: {style}"},
        ],
        response_format={"type": "json_object"},
        temperature=1.0,
    )
    return TrainingSample.model_validate_json(response.choices[0].message.content)


def judge_sample(sample: TrainingSample) -> JudgeVerdict:
    filters_str = json.dumps(sample.target_filters, indent=2)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": f'QUERY:\n"{sample.query}"\n\nFILTERS:\n{filters_str}'},
        ],
        response_format={"type": "json_object"},
    )
    return JudgeVerdict.model_validate_json(response.choices[0].message.content)


def process_one_sample():
    domain    = random.choice(TECH_DOMAINS)
    style     = random.choice(STYLE_MODES)
    candidate = generate_raw_sample(domain, style)
    verdict   = judge_sample(candidate)
    return domain, candidate, verdict

# -------------------------------------------------------------
# 5. Main Pipeline (Concurrent)
# -------------------------------------------------------------
TARGET_COUNT = 5000
OUTPUT_FILE  = "deepseek_tech_filters_5k.jsonl"
MAX_WORKERS  = 25

print(f"🚀 Starting Dataset Generation with DeepSeek API "
      f"(Target: {TARGET_COUNT} samples, Workers: {MAX_WORKERS})...")

approved_count = 0
attempts       = 0
write_lock     = threading.Lock()

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    pbar = tqdm(total=TARGET_COUNT, desc="Approved Samples Generated")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending: set[concurrent.futures.Future] = {
            executor.submit(process_one_sample) for _ in range(MAX_WORKERS)
        }

        while approved_count < TARGET_COUNT:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                attempts += 1
                try:
                    domain, sample, verdict = future.result()

                    if verdict.approved:
                        with write_lock:
                            if approved_count < TARGET_COUNT:
                                f.write(sample.model_dump_json() + "\n")
                                f.flush()
                                approved_count += 1
                                pbar.update(1)
                    else:
                        tqdm.write(f"\n⚠️  [REJECTED] Attempt #{attempts} | Domain: {domain}")
                        tqdm.write(f"   ↳ Query: \"{sample.query}\"")
                        tqdm.write(f"   ↳ Filters: {json.dumps(sample.target_filters)}")
                        tqdm.write(f"   ↳ Reason: {verdict.reason}\n" + "-" * 50)

                except Exception as e:
                    tqdm.write(f"💥 Error on attempt #{attempts}: {str(e)}")

                if approved_count < TARGET_COUNT:
                    pending.add(executor.submit(process_one_sample))

    pbar.close()

print(f"✅ Done! Generated {approved_count} records in '{OUTPUT_FILE}'.")
