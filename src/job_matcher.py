"""Match a resume against individual job postings in data/job_index.json."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.matching import compute_semantic_similarity, compute_similarity
from src.preprocessing import clean_text


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def default_index_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "job_index.json"


@lru_cache(maxsize=1)
def load_job_index(path: Optional[str] = None) -> List[dict]:
    index_path = Path(path) if path else default_index_path()
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return list(payload.get("jobs") or [])


def match_resume_to_jobs(
    resume_text: str,
    field_ids: Optional[Sequence[str]] = None,
    top_n: int = 8,
    tfidf_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> List[Dict]:
    """
    Rank real job postings for a resume.
    Restricts to selected field_ids when provided (e.g. top field matches).
    """
    jobs = load_job_index()
    if not jobs:
        return []

    allowed = set(field_ids) if field_ids else None
    candidates = [
        job for job in jobs if allowed is None or job.get("field_id") in allowed
    ]
    if not candidates:
        candidates = jobs

    resume_clean = clean_text(resume_text)
    resume_semantic = " ".join(resume_text.lower().split())
    total_w = tfidf_weight + semantic_weight
    results: List[Dict] = []

    for job in candidates:
        desc = job.get("description") or ""
        job_clean = clean_text(f"{job.get('title', '')} {desc}")
        job_semantic = " ".join(f"{job.get('title', '')} {desc}".lower().split())

        tfidf_score, _ = compute_similarity(resume_clean, job_clean)
        semantic_score, _ = compute_semantic_similarity(resume_semantic, job_semantic)
        tfidf_score = _safe_float(tfidf_score)
        semantic_score = _safe_float(semantic_score)
        hybrid = ((tfidf_weight * tfidf_score) + (semantic_weight * semantic_score)) / total_w

        results.append(
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "field_id": job.get("field_id"),
                "field_title": job.get("field_title"),
                "category": job.get("category"),
                "snippet": (desc[:220] + "…") if len(desc) > 220 else desc,
                "score": round(hybrid, 4),
                "tfidf_score": round(tfidf_score, 4),
                "semantic_score": round(semantic_score, 4),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_n]
