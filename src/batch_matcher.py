"""Batch-rank multiple resumes against one field profile or a pasted JD."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.field_catalog import load_job_fields
from src.matching import compute_semantic_similarity, compute_similarity
from src.ner import extract_entities
from src.preprocessing import clean_text


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def _resolve_target(
    field_id: Optional[str],
    job_description: Optional[str],
) -> Dict:
    if job_description and job_description.strip():
        text = job_description.strip()
        return {
            "target_type": "job_description",
            "target_label": "Custom job description",
            "target_id": "custom_jd",
            "match_text": text,
            "skill_weights": None,
            "key_skills": [],
        }

    if not field_id:
        raise ValueError("Provide either field_id or job_description.")

    fields = {f.id: f for f in load_job_fields()}
    field = fields.get(field_id)
    if not field:
        raise ValueError(f"Unknown field_id: {field_id}")

    return {
        "target_type": "field",
        "target_label": field.title,
        "target_id": field.id,
        "match_text": field.match_text,
        "skill_weights": field.skill_weights or None,
        "key_skills": list(field.key_skills),
    }


def score_resume_against_target(
    resume_text: str,
    target_text: str,
    mode: str = "hybrid",
    skill_weights: Optional[Dict[str, float]] = None,
    key_skills: Optional[List[str]] = None,
    tfidf_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> Dict:
    resume_clean = clean_text(resume_text)
    target_clean = clean_text(target_text)
    resume_semantic = " ".join(resume_text.lower().split())
    target_semantic = " ".join(target_text.lower().split())

    tfidf_score, tfidf_df = compute_similarity(
        resume_clean, target_clean, skill_weights=skill_weights
    )
    tfidf_score = _safe_float(tfidf_score)
    keywords = []
    if tfidf_df is not None and not getattr(tfidf_df, "empty", True):
        keywords = tfidf_df.loc[
            :, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)
        ].columns.tolist()
        keywords = [c for c in keywords if len(c) > 2][:15]

    semantic_score, _ = compute_semantic_similarity(resume_semantic, target_semantic)
    semantic_score = _safe_float(semantic_score)
    hybrid = (
        (tfidf_weight * tfidf_score) + (semantic_weight * semantic_score)
    ) / (tfidf_weight + semantic_weight)

    mode = (mode or "hybrid").lower()
    if mode == "tfidf":
        final = tfidf_score
    elif mode == "semantic":
        final = semantic_score
    else:
        final = hybrid

    skills = key_skills or []
    matched_skills = [
        s for s in skills if s.lower() in resume_text.lower() or s.lower() in resume_clean
    ]

    entities = extract_entities(resume_text)

    return {
        "score": round(final, 4),
        "hybrid_score": round(hybrid, 4),
        "tfidf_score": round(tfidf_score, 4),
        "semantic_score": round(semantic_score, 4),
        "matched_keywords": keywords,
        "matched_skills": matched_skills,
        "missing_skills": [s for s in skills if s not in matched_skills][:8],
        "entities": entities,
    }


def batch_rank_resumes(
    resumes: List[Dict],
    field_id: Optional[str] = None,
    job_description: Optional[str] = None,
    mode: str = "hybrid",
) -> Dict:
    """
    resumes: list of {filename, text}
    Rank all resumes against one field profile or a custom JD.
    """
    if not resumes:
        raise ValueError("Upload at least one resume.")

    target = _resolve_target(field_id, job_description)
    ranked: List[Dict] = []
    errors: List[Dict] = []

    for item in resumes:
        filename = item.get("filename") or "resume"
        text = item.get("text") or ""
        if not text.strip():
            errors.append({"filename": filename, "error": "Empty resume text"})
            continue
        try:
            scored = score_resume_against_target(
                text,
                target["match_text"],
                mode=mode,
                skill_weights=target["skill_weights"],
                key_skills=target["key_skills"],
            )
            ranked.append(
                {
                    "filename": filename,
                    "char_count": len(text),
                    **scored,
                }
            )
        except Exception as exc:  # keep batch going
            errors.append({"filename": filename, "error": str(exc)})

    ranked.sort(key=lambda row: row["score"], reverse=True)
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
        if row["score"] > 0.4:
            row["status"] = "Strong Match"
        elif row["score"] > 0.15:
            row["status"] = "Potential Match"
        else:
            row["status"] = "Low Match"

    return {
        "mode": mode,
        "target_type": target["target_type"],
        "target_id": target["target_id"],
        "target_label": target["target_label"],
        "resume_count": len(ranked),
        "error_count": len(errors),
        "rankings": ranked,
        "errors": errors,
    }
