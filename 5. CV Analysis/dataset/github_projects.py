"""GitHub Project Discovery Module for EraMatch CV Generation.

Discovers popular open-source projects per IT specialization using `gh` CLI.
Provides rate-limit handling and file-based caching.

Usage:
    from github_projects import get_top_repos
    repos = get_top_repos("backend", count=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache" / "github_repos"
CACHE_TTL_SECONDS: int = 24 * 60 * 60  # 24 hours
RATE_LIMIT_THRESHOLD: int = 10  # pause if fewer than this many calls remain
RATE_LIMIT_SEARCH_KEY: str = "search"  # key in /rate_limit response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Specialization → gh search query templates
# ---------------------------------------------------------------------------

SPECIALIZATION_QUERIES: dict[str, str] = {
    "backend": "topic:backend language:python stars:>50",
    "frontend": "topic:frontend language:typescript stars:>50",
    "data_science": "topic:data-science language:python stars:>50",
    "devops": "topic:devops stars:>50",
    "fullstack": "topic:fullstack stars:>50",
    "mobile": "topic:mobile language:kotlin stars:>50",
    "security": "topic:security stars:>50",
    "cloud": "topic:cloud stars:>50",
    "ml_engineering": "topic:machine-learning language:python stars:>50",
    "game_development": "topic:game-dev stars:>50",
}

# gh search repos JSON fields we want back
_GH_FIELDS: str = ",".join(
    [
        "name",
        "fullName",
        "url",
        "description",
        "stargazersCount",
        "language",
        "updatedAt",
        "visibility",
        "isArchived",
        "isFork",
        "defaultBranch",
        "hasWiki",
    ]
)

# Max repos to fetch per search (gh search repos --limit caps at 1000)
_SEARCH_LIMIT: int = 50

# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------


def _run_gh(args: list[str], *, timeout: int = 30) -> str:
    """Run a ``gh`` CLI command and return stdout."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh command failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def check_rate_limit() -> dict[str, Any]:
    """Return the search rate-limit dict from ``gh api rate_limit``."""
    raw = _run_gh(["api", "rate_limit"])
    data = json.loads(raw)
    return data["resources"][RATE_LIMIT_SEARCH_KEY]


def _wait_for_rate_limit() -> None:
    """Block until the search rate-limit has enough remaining calls."""
    info = check_rate_limit()
    remaining = info.get("remaining", 0)
    if remaining >= RATE_LIMIT_THRESHOLD:
        logger.debug("Rate limit OK: %d remaining", remaining)
        return

    reset_epoch: int = info.get("reset", int(time.time()) + 60)
    sleep_seconds = max(reset_epoch - int(time.time()), 5) + 2
    logger.warning(
        "Rate limit low (%d remaining). Sleeping %ds until reset …",
        remaining,
        sleep_seconds,
    )
    time.sleep(sleep_seconds)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _cache_key(specialization: str, min_stars: int) -> str:
    """Deterministic cache key from specialization + min_stars."""
    raw = f"{specialization}:{min_stars}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(specialization: str, min_stars: int) -> Path:
    return CACHE_DIR / f"{_cache_key(specialization, min_stars)}.json"


def _load_cache(specialization: str, min_stars: int) -> list[dict] | None:
    """Load cached results if they exist and are fresh (< TTL)."""
    path = _cache_path(specialization, min_stars)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    age = time.time() - mtime
    if age > CACHE_TTL_SECONDS:
        logger.debug("Cache expired for %s (age %.0fs)", specialization, age)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug(
            "Cache hit for %s (%d repos, age %.0fs)", specialization, len(data), age
        )
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache read error for %s: %s", specialization, exc)
        return None


def _save_cache(specialization: str, min_stars: int, repos: list[dict]) -> None:
    """Persist results to the file cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(specialization, min_stars)
    path.write_text(json.dumps(repos, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Cached %d repos for %s", len(repos), specialization)


# ---------------------------------------------------------------------------
# Topic enrichment (requires per-repo API call — use sparingly)
# ---------------------------------------------------------------------------


def _fetch_topics(full_name: str) -> list[str]:
    """Fetch repository topics via ``gh api repos/{owner}/{repo}``."""
    try:
        raw = _run_gh(["api", f"repos/{full_name}", "--jq", ".topics"], timeout=15)
        # --jq returns JSON array as a string like '["a","b"]'
        return json.loads(raw)
    except Exception as exc:
        logger.debug("Could not fetch topics for %s: %s", full_name, exc)
        return []


# ---------------------------------------------------------------------------
# Core: search_repos
# ---------------------------------------------------------------------------


def search_repos(specialization: str, min_stars: int = 50) -> list[dict]:
    """Search GitHub repos for a given specialization.

    Args:
        specialization: One of the keys in SPECIALIZATION_QUERIES, or a raw
            query string prefixed with ``raw:``.
        min_stars: Minimum star count filter.

    Returns:
        List of dicts with keys:
            name, full_name, url, description, stars, language,
            updated_at, topics
    """
    # Check cache first
    cached = _load_cache(specialization, min_stars)
    if cached is not None:
        return cached

    # Build query
    if specialization.startswith("raw:"):
        query = specialization[4:]
    elif specialization in SPECIALIZATION_QUERIES:
        query = SPECIALIZATION_QUERIES[specialization]
    else:
        # Fallback: treat specialization as topic
        query = f"topic:{specialization} stars:>{min_stars}"

    # Ensure min_stars is in query if not already present
    if f"stars:>" not in query:
        query = f"{query} stars:>{min_stars}"

    # Rate-limit check
    _wait_for_rate_limit()

    logger.info("Searching repos: %s", query)

    # Execute gh search
    raw = _run_gh(
        [
            "search",
            "repos",
            query,
            "--limit",
            str(_SEARCH_LIMIT),
            "--sort",
            "stars",
            "--json",
            _GH_FIELDS,
        ]
    )

    items: list[dict] = json.loads(raw)
    logger.info("gh returned %d results", len(items))

    # Map to canonical schema
    results: list[dict] = []
    for item in items:
        results.append(
            {
                "name": item.get("name", ""),
                "full_name": item.get("fullName", ""),
                "url": item.get("url", ""),
                "description": item.get("description") or "",
                "stars": item.get("stargazersCount", 0),
                "language": item.get("language") or "",
                "updated_at": item.get("updatedAt", ""),
                "topics": [],  # populated lazily if needed
                # extra metadata for filtering (stripped before return)
                "_visibility": item.get("visibility", "public"),
                "_is_archived": item.get("isArchived", False),
                "_is_fork": item.get("isFork", False),
                "_default_branch": item.get("defaultBranch", ""),
            }
        )

    # Save to cache
    _save_cache(specialization, min_stars, results)

    return results


# ---------------------------------------------------------------------------
# Core: get_top_repos
# ---------------------------------------------------------------------------


def _is_active(updated_at: str, months: int = 6) -> bool:
    """Return True if the repo was updated within the last ``months`` months."""
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    cutoff = datetime.now(timezone.utc).timestamp() - months * 30 * 24 * 3600
    return dt.timestamp() >= cutoff


def get_top_repos(specialization: str, count: int = 5) -> list[dict]:
    """Return the top *count* repos for a specialization with quality filters.

    Filters: active (updated < 6 months), public, not archived, not a fork,
    has a README (inferred from default_branch existence).

    Topics are enriched for the returned repos only (minimises API calls).

    Args:
        specialization: Specialization key or ``raw:`` query.
        count: Number of repos to return.

    Returns:
        List of dicts (same schema as search_repos minus internal keys).
    """
    repos = search_repos(specialization, min_stars=50)

    # Apply filters
    filtered: list[dict] = []
    for r in repos:
        if r.get("_is_fork"):
            continue
        if r.get("_is_archived"):
            continue
        if r.get("_visibility") != "public":
            continue
        if not _is_active(r.get("updated_at", "")):
            continue
        if not r.get("_default_branch"):
            continue
        filtered.append(r)

    # Sort by stars descending
    filtered.sort(key=lambda r: r.get("stars", 0), reverse=True)

    # Take top count and strip internal keys
    top = filtered[:count]
    output: list[dict] = []
    for r in top:
        # Enrich topics only for repos we're going to return
        topics = r.get("topics", [])
        if not topics and r.get("full_name"):
            topics = _fetch_topics(r["full_name"])
        output.append(
            {
                "name": r["name"],
                "full_name": r["full_name"],
                "url": r["url"],
                "description": r["description"],
                "stars": r["stars"],
                "language": r["language"],
                "updated_at": r["updated_at"],
                "topics": topics,
            }
        )

    return output


# ---------------------------------------------------------------------------
# CLI convenience (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import sys

    spec = sys.argv[1] if len(sys.argv) > 1 else "backend"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = get_top_repos(spec, count=n)
    for r in results:
        print(
            f"  {r['full_name']}: ★{r['stars']}  ({r['language']})  topics={r['topics'][:3]}"
        )
    print(f"\n{len(results)} repos for '{spec}'")
