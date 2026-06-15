"""Batch 1 Premium CV Generator (Deepseek-Driven).

Generates high-quality premium CVs for all 20 Batch 1 emails using
deepseek-v4-pro:cloud LLM with real GitHub project URLs.

Each CV is built by:
  1. Loading email → specialization mapping from email_specialization_map.json
  2. Fetching real GitHub repos via github_projects.get_top_repos()
  3. Prompting deepseek-v4-pro:cloud to generate a complete CV as JSON
  4. Validating output against CVSchema v3.0

Usage:
    from cv_generator_batch1 import generate_premium_cv, generate_batch1
    cv_data, metadata = generate_premium_cv("anasahdev@gmail.com")
    # Or run full batch:
    results = generate_batch1()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ollama
from tqdm import tqdm

from github_projects import get_top_repos, search_repos
from schema import CVSchema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MAPPING_PATH = BASE_DIR / "email_specialization_map.json"
BATCH_OUTPUT_DIR = BASE_DIR / "batches" / "batch_1_premium"

LLM_MODEL = "deepseek-v4-pro:cloud"
LLM_TIMEOUT = 300  # seconds per attempt
LLM_MAX_RETRIES = 2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tech keyword normalization
# ---------------------------------------------------------------------------

TECH_ALIASES: dict[str, str] = {
    "nextjs": "next.js",
    "next": "next.js",
    "nodejs": "node.js",
    "node": "node.js",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "golang": "go",
    "csharp": "c#",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "ml": "machine-learning",
    "dl": "deep-learning",
}

# Skill categories that represent core technologies worth searching GitHub for
_TECH_CATEGORIES: frozenset[str] = frozenset(
    ("programming_language", "framework", "cloud", "container")
)

# Cache directory for tech-stack-keyed GitHub repo lookups
_TECH_CACHE_DIR = BASE_DIR / ".cache" / "github_repos"
_TECH_CACHE_TTL_SECONDS: int = 24 * 60 * 60  # 24 hours


def _normalize_tech(keyword: str) -> str:
    """Normalize a tech keyword: lowercase, strip, apply aliases."""
    kw = keyword.strip().lower()
    return TECH_ALIASES.get(kw, kw)


def _extract_tech_keywords(cv_data: dict) -> list[str]:
    """Extract top 3-5 tech keywords from CV data for GitHub search.

    Parses skills (programming_language, framework, cloud, container categories)
    and work_experience technologies, normalizes them, and returns the most
    frequently mentioned ones.

    Falls back to the specialization from email_specialization_map if fewer
    than 2 keywords are found.
    """
    counter: Counter[str] = Counter()

    # Parse skills list
    for skill in cv_data.get("skills", []):
        if not isinstance(skill, dict):
            continue
        category = skill.get("category", "")
        if category in _TECH_CATEGORIES:
            name = skill.get("skill_name", "")
            if name:
                normalized = _normalize_tech(name)
                if normalized:
                    counter[normalized] += 1

    # Parse work_experience technologies
    for job in cv_data.get("work_experience", []):
        if not isinstance(job, dict):
            continue
        for tech in job.get("technologies", []):
            if isinstance(tech, str) and tech.strip():
                normalized = _normalize_tech(tech)
                if normalized:
                    counter[normalized] += 1

    # Get top keywords by frequency
    most_common = counter.most_common()
    keywords = [kw for kw, _ in most_common[:5]]

    # If fewer than 2 keywords, add specialization fallback
    if len(keywords) < 2 and "primary_domain" in cv_data:
        spec = cv_data["primary_domain"]
        if spec and spec not in keywords:
            keywords.append(spec)

    return keywords


def _build_github_query(keywords: list[str]) -> str:
    """Build a raw GitHub search query from tech keywords.

    Uses topic: qualifiers and stars:>50 filter.
    Takes top 2-3 keywords for the query.
    """
    if not keywords:
        return "raw:topic:software stars:>50"

    # Use top 2-3 keywords
    top_kws = keywords[:3]
    topic_parts = " ".join(f"topic:{kw}" for kw in top_kws)
    return f"raw:{topic_parts} stars:>50"


def _get_cached_repos(tech_key: str) -> list | None:
    """Check tech-stack-keyed cache for previously fetched repos.

    Args:
        tech_key: Pipe-separated sorted keywords string (e.g. "python|fastapi|docker")

    Returns:
        Cached repo list if fresh (< 24h TTL), None otherwise.
    """
    cache_hash = hashlib.sha256(tech_key.encode()).hexdigest()[:16]
    cache_path = _TECH_CACHE_DIR / f"tech_{cache_hash}.json"

    if not cache_path.exists():
        return None

    mtime = cache_path.stat().st_mtime
    age = time.time() - mtime
    if age > _TECH_CACHE_TTL_SECONDS:
        logger.debug("Tech cache expired for %s (age %.0fs)", tech_key, age)
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.debug(
            "Tech cache hit for %s (%d repos, age %.0fs)", tech_key, len(data), age
        )
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Tech cache read error for %s: %s", tech_key, exc)
        return None


def _save_cached_repos(tech_key: str, repos: list) -> None:
    """Save repos to tech-stack-keyed cache.

    Args:
        tech_key: Pipe-separated sorted keywords string
        repos: List of repo dicts to cache
    """
    _TECH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_hash = hashlib.sha256(tech_key.encode()).hexdigest()[:16]
    cache_path = _TECH_CACHE_DIR / f"tech_{cache_hash}.json"
    cache_path.write_text(
        json.dumps(repos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.debug("Cached %d repos for tech_key=%s", len(repos), tech_key)


# ---------------------------------------------------------------------------
# Specialization map loader
# ---------------------------------------------------------------------------


def load_specialization_map() -> dict[str, dict]:
    """Load email_specialization_map.json.

    Returns:
        Dict mapping email → {specialization, career_stage, ...}
    """
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"email_specialization_map.json not found at {MAPPING_PATH}. "
            "Run specialization_mapper.py first or create the file manually."
        )
    with open(MAPPING_PATH, encoding="utf-8") as f:
        data: dict[str, dict] = json.load(f)
    logger.info(
        "Loaded specialization map with %d entries from %s", len(data), MAPPING_PATH
    )
    return data


# ---------------------------------------------------------------------------
# Deepseek prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a professional CV writer for Egyptian tech professionals. "
    "You generate COMPLETE, REALISTIC CVs as JSON matching the CVSchema v3.0 structure. "
    "Every CV must: (1) include the given email, (2) use the real GitHub repo URLs provided, "
    "(3) have Egyptian education institutions, (4) have 3-5 work experience entries, "
    "(5) have skills matching specialization and repo languages, (6) have a professional summary. "
    "Output ONLY valid JSON — no markdown fences, no commentary."
)


def _build_cv_prompt(
    email: str,
    specialization: str,
    career_stage: str,
    github_repos: list[dict],
) -> str:
    """Build the user prompt for deepseek CV generation."""
    repos_block = ""
    for i, r in enumerate(github_repos, 1):
        topics_str = ", ".join(r.get("topics", [])[:5]) if r.get("topics") else "N/A"
        repos_block += (
            f"  {i}. {r['full_name']} — ★{r['stars']}  Language: {r['language']}\n"
            f"     URL: {r['url']}\n"
            f"     Description: {r['description']}\n"
            f"     Topics: {topics_str}\n"
        )

    # Map career stage to approximate years of experience
    stage_to_years = {
        "intern": "0-1",
        "junior": "1-3",
        "mid": "3-6",
        "senior": "6-10",
        "lead": "10-15",
        "principal": "15+",
    }
    years_range = stage_to_years.get(career_stage, "3-6")

    prompt = f"""Generate a COMPLETE CV as JSON for an Egyptian tech professional with these details:

EMAIL: {email}
SPECIALIZATION: {specialization}
CAREER STAGE: {career_stage} ({years_range} years experience)

REAL GITHUB PROJECTS (use these EXACT URLs as the person's projects):
{repos_block}

REQUIREMENTS:
1. full_name: Generate a realistic Egyptian name (NOT "John Doe" or generic Western names — use common Egyptian first/last names like Ahmed, Mohamed, Youssef, Adham, Anas, Hassan, Omar, Sara, Nour, Mariam combined with Egyptian family names)
2. email: {email} (must match exactly)
3. location: Use Egyptian city (Cairo, Giza, Alexandria, New Cairo, 6th October City, etc.)
4. seniority_level: {career_stage}
5. primary_domain: {specialization}
6. has_github: true (since we have real GitHub repos)
7. summary: Professional summary matching their specialization and career stage
8. work_experience: 3-5 entries at realistic companies (mix of Egyptian tech companies like Instabug, Vezeeta, Swvl, Paymob, Fawry, Breadfast, and international companies with Egypt offices like Microsoft, Amazon, Google, Cisco, IBM). Job titles should match {specialization} and {career_stage} level.
9. education: Use Egyptian universities (Cairo University, Ain Shams University, Alexandria University, Zewail City of Science and Technology, STEM Egypt schools, German University in Cairo, American University in Cairo). Degree should match career stage.
10. skills: List of SkillEntry objects with skill_name and category. Include languages/frameworks from the GitHub repos plus specialization-relevant skills. At least 10 skills.
11. projects: Use the EXACT GitHub repos listed above — copy name, url, description, and include repo languages in technologies list.
12. certifications: 2-3 relevant certifications (AWS, Google Cloud, Azure, PMP, CKA, etc. based on specialization)
13. contact_info: Include email={email}, and generate realistic phone number (+20 prefix), linkedin_url, github_url profiles
14. years_of_experience: Numeric value in the {years_range} range
15. skills_flat: Flat list of all skill names
16. companies: List of company names from work_experience
17. job_titles: List of job titles from work_experience
18. universities: List of university names
19. degrees: List of degree names

JSON STRUCTURE (follow this exact schema):
{{
  "full_name": "...",
  "email": "{email}",
  "location": "...",
  "years_of_experience": ...,
  "seniority_level": "{career_stage}",
  "primary_domain": "{specialization}",
  "has_github": true,
  "has_linkedin": true,
  "contact_info": {{
    "full_name": "...",
    "email": "{email}",
    "phone": "+20...",
    "location": "...",
    "linkedin_url": "...",
    "github_url": "...",
    "portfolio_url": "..."
  }},
  "summary": "...",
  "work_experience": [
    {{
      "company": "...",
      "job_title": "...",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM or Present",
      "description": "...",
      "technologies": ["..."],
      "is_remote": false,
      "employment_type": "full-time"
    }}
  ],
  "education": [
    {{
      "institution": "...",
      "degree": "...",
      "field_of_study": "...",
      "graduation_date": "YYYY-MM"
    }}
  ],
  "skills": [
    {{
      "skill_name": "...",
      "category": "programming_language|framework|tool|database|cloud|methodology|language|soft_skill|ai_api|ai_ml|ml_ops|infrastructure|container|ci_cd|monitoring|data_tool|data_warehouse|cloud_ml|other"
    }}
  ],
  "projects": [
    {{
      "name": "...",
      "description": "...",
      "technologies": ["..."],
      "url": "https://github.com/...",
      "date": "..."
    }}
  ],
  "certifications": [
    {{
      "name": "...",
      "issuer": "...",
      "date": "YYYY-MM"
    }}
  ],
  "misc_data": [],
  "skills_flat": ["..."],
  "companies": ["..."],
  "job_titles": ["..."],
  "universities": ["..."],
  "degrees": ["..."],
  "parsing_metadata": {{}}
}}

Output ONLY the JSON object. No markdown, no explanation."""
    return prompt


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


def _call_deepseek(prompt: str) -> str:
    """Call deepseek-v4-pro:cloud with the given prompt.

    Retries once on failure (max 2 attempts total).

    Returns:
        Raw LLM response content string.
    """
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"timeout": LLM_TIMEOUT},
            )
            content: str = response["message"]["content"].strip()
            return content
        except Exception as exc:
            logger.warning(
                "Deepseek call failed (attempt %d/%d): %s",
                attempt,
                LLM_MAX_RETRIES,
                exc,
            )
            if attempt < LLM_MAX_RETRIES:
                time.sleep(5)

    raise RuntimeError(f"Deepseek LLM call failed after {LLM_MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict:
    """Extract and parse JSON from LLM response.

    Handles markdown code fences and other common LLM output quirks.
    """
    text = raw.strip()
    # Remove markdown fences if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response (first 200 chars): {text[:200]}"
    )


# ---------------------------------------------------------------------------
# CV validation
# ---------------------------------------------------------------------------


def _validate_cv(parsed: dict) -> CVSchema:
    """Validate parsed dict against CVSchema v3.0.

    Returns validated CVSchema object.
    Raises ValueError on validation failure.
    """
    try:
        return CVSchema(**parsed)
    except Exception as exc:
        raise ValueError(f"CVSchema validation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Core: generate_premium_cv
# ---------------------------------------------------------------------------


def generate_premium_cv(email: str, retry_count: int = 0) -> tuple[dict, dict]:
    """Generate a premium CV for the given email using deepseek LLM.

    Args:
        email: Email address from the Batch 1 list.
        retry_count: Internal retry counter (do not set manually).

    Returns:
        Tuple of (cv_data_dict, metadata_dict).
        cv_data_dict: CVSchema-compatible dict with full CV data.
        metadata_dict: Generation metadata (source_email, specialization,
            career_stage, github_projects, model_used, generation_time).

    Raises:
        RuntimeError: If generation fails after max retries.
    """
    start_time = time.time()

    # Load mapping
    spec_map = load_specialization_map()
    if email not in spec_map:
        raise ValueError(f"Email {email} not found in specialization map")
    entry = spec_map[email]
    specialization = entry["specialization"]
    career_stage = entry["career_stage"]

    logger.info(
        "Generating premium CV for %s → %s (%s)", email, specialization, career_stage
    )

    # Fetch real GitHub repos
    github_repos = get_top_repos(specialization, count=5)
    if not github_repos:
        logger.warning(
            "No GitHub repos found for %s, attempting broader search", specialization
        )
        github_repos = get_top_repos(f"raw:topic:software stars:>100", count=5)

    logger.info("Found %d GitHub repos for %s", len(github_repos), specialization)

    # Build prompt
    prompt = _build_cv_prompt(email, specialization, career_stage, github_repos)

    # Call LLM
    try:
        raw_response = _call_deepseek(prompt)
        parsed = _extract_json(raw_response)

        # Force email match
        parsed["email"] = email
        if "contact_info" in parsed and isinstance(parsed["contact_info"], dict):
            parsed["contact_info"]["email"] = email

        # Validate
        cv_schema = _validate_cv(parsed)

    except (ValueError, RuntimeError) as exc:
        if retry_count < 1:
            logger.warning(
                "CV generation failed for %s, retrying (retry %d): %s",
                email,
                retry_count + 1,
                exc,
            )
            return generate_premium_cv(email, retry_count=retry_count + 1)
        else:
            raise RuntimeError(
                f"Premium CV generation failed for {email} after {retry_count + 1} retries: {exc}"
            ) from exc

    generation_time = time.time() - start_time

    # Build cv_data dict from validated schema (use model_dump for Pydantic v2)
    cv_data = cv_schema.model_dump()

    # --- Pass 2: Tech-specific GitHub search and project injection ---
    tech_keywords = _extract_tech_keywords(cv_data)
    logger.info("Extracted tech keywords for %s: %s", email, tech_keywords)

    tech_key = "|".join(sorted(tech_keywords))
    query = _build_github_query(tech_keywords)

    cached_repos = _get_cached_repos(tech_key)
    if cached_repos is not None:
        real_repos = cached_repos
        logger.info(
            "Using cached repos for tech_key=%s (%d repos)", tech_key, len(real_repos)
        )
    else:
        real_repos = search_repos(query, min_stars=50)
        if len(real_repos) < 3:
            logger.info(
                "Only %d repos for query '%s', falling back to specialization '%s'",
                len(real_repos),
                query,
                specialization,
            )
            real_repos = get_top_repos(specialization, count=5)
        _save_cached_repos(tech_key, real_repos)

    # Per-CV shuffling: email hash seeds a deterministic random selection,
    # so each CV gets a different 5-repo subset from the same cache.
    import hashlib as _hashlib
    import random as _random_mod

    _seed = int(_hashlib.md5(email.encode()).hexdigest(), 16) % (2**31)
    _rng = _random_mod.Random(_seed)

    available_repos = list(real_repos)
    _rng.shuffle(available_repos)

    injected_projects = []
    for repo in available_repos[:5]:
        injected_projects.append(
            {
                "name": repo.get("name", ""),
                "description": repo.get("description") or "",
                "technologies": (
                    repo.get("topics", [])[:5]
                    if repo.get("topics")
                    else ([repo.get("language")] if repo.get("language") else [])
                ),
                "url": repo.get("url", ""),
                "date": repo.get("updated_at", "")[:7]
                if repo.get("updated_at")
                else "",
            }
        )
    cv_data["projects"] = injected_projects

    metadata = {
        "source_email": email,
        "specialization": specialization,
        "career_stage": career_stage,
        "github_projects": github_repos,
        "tech_keywords": tech_keywords,
        "model_used": LLM_MODEL,
        "generation_time": round(generation_time, 2),
        "retry_count": retry_count,
    }

    logger.info(
        "Premium CV generated for %s in %.1fs (retries: %d, keywords: %s)",
        email,
        generation_time,
        retry_count,
        tech_keywords,
    )

    return cv_data, metadata


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def generate_batch1(output_dir: Path | None = None) -> list[tuple[str, dict, dict]]:
    """Generate premium CVs for all 20 Batch 1 emails.

    Args:
        output_dir: Directory to save CV JSONs and metadata. If None, no files saved.

    Returns:
        List of (email, cv_data, metadata) tuples.
        Failed emails are logged and skipped (not in the return list).
    """
    spec_map = load_specialization_map()
    emails = list(spec_map.keys())
    results: list[tuple[str, dict, dict]] = []
    failed: list[tuple[str, str]] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

    for idx, email in enumerate(tqdm(emails, desc="Batch 1 Premium CVs", unit="cv")):
        try:
            cv_data, metadata = generate_premium_cv(email)

            results.append((email, cv_data, metadata))

            # Save to files if output_dir provided
            if output_dir is not None:
                safe_name = email.replace("@", "_at_").replace(".", "_")
                cv_path = output_dir / f"{safe_name}_cv.json"
                meta_path = output_dir / f"{safe_name}_metadata.json"

                cv_path.write_text(
                    json.dumps(cv_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                meta_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                logger.info("Saved: %s", safe_name)

        except Exception as exc:
            reason = str(exc)
            failed.append((email, reason))
            logger.error("FAILED: %s — %s", email, reason)

        # Rate-limit: pause between CVs to respect GitHub API limits
        if idx < len(emails) - 1:
            time.sleep(2)

    # Log summary
    logger.info(
        "Batch 1 complete: %d/%d succeeded, %d failed",
        len(results),
        len(emails),
        len(failed),
    )
    if failed:
        for email, reason in failed:
            logger.error("  FAILED %s: %s", email, reason)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    results = generate_batch1(output_dir=BATCH_OUTPUT_DIR)
    print(f"\nGenerated {len(results)} premium CVs")
