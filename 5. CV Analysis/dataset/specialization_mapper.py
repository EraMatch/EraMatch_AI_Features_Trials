"""Email-to-Specialization Mapping Module for EraMatch CV Generation.

Maps Batch 1 email addresses to IT specializations, career stages,
and suggested GitHub search topics using rules-based inference with
LLM fallback for ambiguous cases.

Usage:
    from specialization_mapper import map_all_emails, infer_specialization
    mapping = map_all_emails()
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MAPPING_OUTPUT_PATH = BASE_DIR / "email_specialization_map.json"

LLM_MODEL = "deepseek-v4-pro:cloud"
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 2

VALID_SPECIALIZATIONS = [
    "backend",
    "frontend",
    "data_science",
    "devops",
    "fullstack",
    "mobile",
    "security",
    "cloud",
    "ml_engineering",
    "game_development",
]

VALID_CAREER_STAGES = [
    "intern",
    "junior",
    "mid",
    "senior",
    "lead",
    "principal",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batch 1 Emails
# ---------------------------------------------------------------------------

BATCH1_EMAILS: list[str] = [
    "yousefpublicc@gmail.com",
    "youssefmoustafa220177@gmail.com",
    "youssefmoustafa220188@gmail.com",
    "youssefsaidtower@gmail.com",
    "yousef.1019095@stemoctober.moe.edu.eg",
    "1019095@stemegypt.edu.eg",
    "s-yousef.said@zewailcity.edu.eg",
    "anasbesoanas@gmail.com",
    "passanasredmi5plus@gmail.com",
    "aadhameltholth@gmail.com",
    "adham.eltholth@gmail.com",
    "s-adham.mohamed@zewailcity.edu.eg",
    "asholth2025@gmail.com",
    "prof.eltholth@gmail.com",
    "ne3na3a25@gmail.com",
    "ne3na3agaming@gmail.com",
    "fchatgpt002@gmail.com",
    "perplexityacc77@gmail.com",
    "ahmetkuli701@gmail.com",
    "anasahdev@gmail.com",
]

# ---------------------------------------------------------------------------
# Rules-based inference
# ---------------------------------------------------------------------------


def _apply_rules(email: str) -> dict[str, str] | None:
    """Apply deterministic rules to infer specialization from email patterns.

    Returns None if no confident rule matches (triggers LLM fallback).
    """
    local = email.split("@")[0].lower()
    domain = email.split("@")[1].lower() if "@" in email else ""

    # --- Domain-based rules ---

    # STEM school students (Egypt STEM schools)
    if domain in ("stemoctober.moe.edu.eg", "stemegypt.edu.eg"):
        spec = random.choice(["fullstack", "backend"])
        stage = random.choice(["intern", "junior"])
        return {"specialization": spec, "career_stage": stage}

    # Zewail City students (CS/Engineering university)
    if domain == "zewailcity.edu.eg":
        spec = random.choice(["data_science", "ml_engineering"])
        stage = "junior"
        return {"specialization": spec, "career_stage": stage}

    # --- Local-part pattern rules ---

    # "prof." or "prof" prefix → senior/lead
    if local.startswith("prof.") or local.startswith("prof"):
        return {
            "specialization": "fullstack",
            "career_stage": random.choice(["senior", "lead"]),
        }

    # "gaming" in email → game development
    if "gaming" in local:
        return {"specialization": "game_development", "career_stage": "junior"}

    # "dev" in email → developer
    if "dev" in local:
        return {"specialization": "fullstack", "career_stage": "mid"}

    # "perplexity" or "chatgpt" in email → AI/ML enthusiast
    if "perplexity" in local or "chatgpt" in local:
        return {
            "specialization": "ml_engineering",
            "career_stage": random.choice(["junior", "mid"]),
        }

    # "ahmetkuli" → Turkish name, likely mobile or fullstack
    if "ahmetkuli" in local:
        return {"specialization": "mobile", "career_stage": "mid"}

    # --- Shared ID patterns ---
    # Same student ID (1019095) as STEM emails → treat similarly
    if "1019095" in local:
        return {"specialization": "backend", "career_stage": "intern"}

    # "eltholth" / "tholth" family name pattern
    if "eltholth" in local or "tholth" in local:
        return {"specialization": "fullstack", "career_stage": "junior"}

    # No rule matched
    return None


# ---------------------------------------------------------------------------
# LLM-based inference (fallback for ambiguous emails)
# ---------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = (
    "Given this email address '{email}' belonging to a tech professional, "
    "infer their most likely IT specialization from: "
    "backend, frontend, data_science, devops, fullstack, mobile, security, cloud, ml_engineering, game_development. "
    "Also suggest career_stage from: intern, junior, mid, senior, lead, principal. "
    'Respond in JSON: {{"specialization": "...", "career_stage": "...", "reasoning": "..."}}'
)


def _call_llm(email: str) -> dict[str, Any]:
    """Call Ollama LLM to infer specialization for ambiguous emails.

    Returns parsed dict with specialization, career_stage, reasoning.
    Retries once on failure (max 2 attempts).
    """
    prompt = _LLM_PROMPT_TEMPLATE.format(email=email)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"timeout": LLM_TIMEOUT},
            )
            content = response["message"]["content"].strip()

            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            if "specialization" not in result or "career_stage" not in result:
                raise ValueError(f"Missing required fields in LLM response: {result}")

            spec = result["specialization"].lower().replace(" ", "_")
            if spec not in VALID_SPECIALIZATIONS:
                logger.warning(
                    "LLM returned unknown specialization '%s' for %s, defaulting to 'fullstack'",
                    spec,
                    email,
                )
                result["specialization"] = "fullstack"
            else:
                result["specialization"] = spec

            stage = result["career_stage"].lower()
            if stage not in VALID_CAREER_STAGES:
                logger.warning(
                    "LLM returned unknown career_stage '%s' for %s, defaulting to 'junior'",
                    stage,
                    email,
                )
                result["career_stage"] = "junior"
            else:
                result["career_stage"] = stage

            return result

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLM response parse error for %s (attempt %d/%d): %s",
                email,
                attempt,
                LLM_MAX_RETRIES,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "LLM call failed for %s (attempt %d/%d): %s",
                email,
                attempt,
                LLM_MAX_RETRIES,
                exc,
            )

    logger.error("All LLM attempts failed for %s, using fallback defaults", email)
    return {
        "specialization": "fullstack",
        "career_stage": "junior",
        "reasoning": "LLM fallback — all attempts failed",
    }


# ---------------------------------------------------------------------------
# Core: infer_specialization
# ---------------------------------------------------------------------------


def infer_specialization(email: str) -> dict[str, Any]:
    """Infer IT specialization and career stage from an email address.

    Uses rules-based inference first, falls back to LLM for ambiguous emails.

    Args:
        email: Email address to analyze.

    Returns:
        Dict with keys: email, specialization, career_stage, reasoning,
        suggested_github_topics.
    """
    ruled = _apply_rules(email)
    if ruled is not None:
        ruled["reasoning"] = "Rules-based inference"
        ruled["email"] = email
        ruled["suggested_github_topics"] = infer_github_topics(ruled["specialization"])
        return ruled

    llm_result = _call_llm(email)
    llm_result["email"] = email
    if "reasoning" not in llm_result:
        llm_result["reasoning"] = "LLM-based inference"
    llm_result["suggested_github_topics"] = infer_github_topics(
        llm_result["specialization"]
    )
    return llm_result


# ---------------------------------------------------------------------------
# GitHub topic inference (no API calls — derived from SPECIALIZATION_QUERIES)
# ---------------------------------------------------------------------------

# Local copy of specialization → topic keywords (derived from github_projects.SPECIALIZATION_QUERIES)
# We do NOT import github_projects to avoid circular deps / API calls.
_SPECIALIZATION_TOPICS: dict[str, list[str]] = {
    "backend": ["backend", "python", "api", "server"],
    "frontend": ["frontend", "typescript", "react", "ui"],
    "data_science": ["data-science", "python", "analytics", "visualization"],
    "devops": ["devops", "ci-cd", "kubernetes", "docker"],
    "fullstack": ["fullstack", "web", "javascript", "node"],
    "mobile": ["mobile", "kotlin", "flutter", "ios"],
    "security": ["security", "cryptography", "auth", "pentesting"],
    "cloud": ["cloud", "aws", "gcp", "terraform"],
    "ml_engineering": ["machine-learning", "python", "deep-learning", "nlp"],
    "game_development": ["game-dev", "unity", "unreal", "godot"],
}


def infer_github_topics(specialization: str) -> list[str]:
    """Return suggested GitHub search topics for a specialization.

    Derived from SPECIALIZATION_QUERIES in github_projects module.
    Does NOT call GitHub API.

    Args:
        specialization: One of the 10 valid specialization keys.

    Returns:
        List of topic keyword strings.
    """
    return _SPECIALIZATION_TOPICS.get(specialization, ["programming", "software"])


# ---------------------------------------------------------------------------
# Batch mapping
# ---------------------------------------------------------------------------


def map_all_emails() -> dict[str, dict]:
    """Process all 20 Batch 1 emails and return the mapping dict.

    Returns:
        Dict mapping email → {specialization, career_stage, reasoning,
        suggested_github_topics, email}.
    """
    mapping: dict[str, dict] = {}
    for email in BATCH1_EMAILS:
        result = infer_specialization(email)
        mapping[email] = result
        logger.info(
            "%s → %s (%s)",
            email,
            result["specialization"],
            result["career_stage"],
        )

    MAPPING_OUTPUT_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved mapping to %s (%d entries)", MAPPING_OUTPUT_PATH, len(mapping))

    return mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mapping = map_all_emails()
    print(f"\nMapped {len(mapping)} emails:")
    for email, info in sorted(mapping.items()):
        print(f"  {email} → {info['specialization']} ({info['career_stage']})")

    specs = set(i["specialization"] for i in mapping.values())
    stages = set(i["career_stage"] for i in mapping.values())
    print(f"\nSpecializations: {len(specs)} — {sorted(specs)}")
    print(f"Career stages: {len(stages)} — {sorted(stages)}")
