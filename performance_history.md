# Performance Optimization History: ACE-Recruiter

This document tracks the evolution of the ACE-Recruiter engine's performance, documenting the methodology and changes that led to specific execution times.

## Trial Summary Table

| Trial | Phase | Total Time | Key Change / Methodology | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Baseline | 4.01m | Search criteria + 5 latest public repos. No deep context. | Stable |
| **2a** | Pillar Search | 11.0m | **Initial JD-Driven Search.** Semantic mapping for all repos. | Slow |
| **2b** | Hyper-Turbo 1 | 6.43m | **Concurrency.** Cloud batching (6) + Pre-filter (50). | Fast |
| **2c** | Hyper-Turbo 2 | 6.34m | **Parallel Fetch.** Concurrent key-file content retrieval. | **Best Result** |
| **2d** | Ultimate Turbo| 7.23m | **Early Break.** `as_completed` + Mastery Break logic. | Regressed |
| **2e** | Lightning Parallelism| 9.91m | **Parallel Search.** 3x parallel LLM Pillar calls. | Significant Regression |

---

## Detailed Methodology

### Trial 1: Baseline (Branch: `trial-1-baseline`)
- **Methodology**: Simple heuristic match of JD keywords against the 5 most recently updated repositories.
- **Analysis**: Fast but lacked depth. High risk of missing high-value "hidden" projects in the candidate's history.

### Trial 2a: The Rubric Shift (Branch: `trial-2a-pillars`)
- **Changes**: Introduced `categorize_profile` (Pillar Search). 
- **Methodology**: LLM analyzes the entire profile to create a technical rubric before scouting.
- **Bottleneck**: Sequential processing of everything made it non-viable for rapid recruiting.

### Trial 2b: Hyper-Turbo (Branch: `trial-2b-hyper`)
- **Changes**: Batch Size 6 for Cloud Models, Pure Heuristic Pre-Filter (Top 50), Superiority Exit.
- **Methodology**: Dropped candidates early if a 80%+ match was found.

### Trial 2c: Parallel Key-Access (Branch: `trial-2c-parallel-fetch`)
- **Changes**: `ThreadPoolExecutor` in the Audit phase to fetch 5 files simultaneously.
- **Result**: Slashing ~40s from the Deep Audit/Mapping phases.

### Trial 2d: Spotlight Efficiency (Branch: `trial-2d-spotlight`)
- **Changes**: `as_completed` scouting logic and `Spotlight Shortcut`.
- **Methodology**: Instead of waiting for batches, process every repo as it returns.
- **Regression Note**: Likely overhead in thread management or UI updates caused a slight slowdown (7.23m).

### Trial 2e: Lightning Parallelism (Branch: `trial-2e-lightning`)
- **Changes**: Parallel Pillar Search (3 chunks of 10), Spotlight-Only mode, `models.py` refactor.
- **Regression Note**: Significant regression (9.91m). Analysis: Cloud provider rate-limit/queuing when hit with 3 simultaneous JD-heavy prompts in Phase 1 (Pillar Search: 207s vs 79s in 2c).

---

## Conclusion & Next Steps
Based on Trial 2c and 2e results, **sequential Pillar Search (Phase 1) is ironically faster** for cloud APIs than parallel chunks, likely due to internal load balancing.
- **Recommendation**: Revert to Trial 2c logic for Phase 1 while keeping the `models.py` refactor and `Spotlight` logic.
