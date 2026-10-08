"""Rank a resume against a catalog of job fields using TF-IDF + semantic similarity."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from src.field_catalog import JobField, load_job_fields
from src.job_matcher import match_resume_to_jobs
from src.matching import compute_semantic_similarity, compute_similarity
from src.ner import extract_entities
from src.preprocessing import clean_text
from src.ranker import extract_features_dict, predict_match_proba

def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def _overlap_keywords(tfidf_df) -> List[str]:
    if tfidf_df is None or getattr(tfidf_df, "empty", True):
        return []
    cols = tfidf_df.loc[:, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)].columns.tolist()
    return [c for c in cols if len(c) > 2][:25]


def match_resume_to_fields(
    resume_text: str,
    fields: Optional[List[JobField]] = None,
    mode: str = "hybrid",
    tfidf_weight: float = 0.4,
    semantic_weight: float = 0.6,
    top_k: Optional[int] = None,
    top_jobs: int = 8,
) -> Dict:
    """
    Score a resume against every job field and recommend real job postings.

    Always computes TF-IDF, semantic, and hybrid scores for comparison.
    `mode` controls which score is used for field ranking order.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    mode = (mode or "hybrid").lower()
    if mode not in {"tfidf", "semantic", "hybrid"}:
        raise ValueError("mode must be one of: tfidf, semantic, hybrid")

    catalog = fields if fields is not None else load_job_fields()
    resume_clean = clean_text(resume_text)
    resume_semantic = " ".join(resume_text.lower().split())

    entities = extract_entities(resume_text)
    results: List[Dict] = []
    total_w = tfidf_weight + semantic_weight

    for field in catalog:
        field_clean = clean_text(field.match_text)
        field_semantic = " ".join(field.match_text.lower().split())

        tfidf_score, tfidf_df = compute_similarity(
            resume_clean,
            field_clean,
            skill_weights=field.skill_weights or None,
        )
        tfidf_score = _safe_float(tfidf_score)
        keywords = _overlap_keywords(tfidf_df)

        semantic_score, _ = compute_semantic_similarity(resume_semantic, field_semantic)
        semantic_score = _safe_float(semantic_score)

        hybrid_score = (
            (tfidf_weight * tfidf_score) + (semantic_weight * semantic_score)
        ) / total_w

        if mode == "tfidf":
            final = tfidf_score
        elif mode == "semantic":
            final = semantic_score
        else:
            final = hybrid_score

        matched_skills = [
            skill
            for skill in field.key_skills
            if skill.lower() in resume_text.lower() or skill.lower() in resume_clean
        ]
        
        all_missing_skills = [s for s in field.key_skills if s not in matched_skills]

        # Extract features and predict with Ranker if available
        feature_dict = extract_features_dict(
            tfidf_score=tfidf_score,
            semantic_score=semantic_score,
            hybrid_score=hybrid_score,
            matched_skills=matched_skills,
            missing_skills=all_missing_skills,
            key_skills=field.key_skills,
            skill_weights=field.skill_weights,
            resume_len=len(resume_clean.split()),
            field_len=len(field_clean.split())
        )
        
        ranker_proba = predict_match_proba(feature_dict)
        alpha = 0.3
        
        # If model is loaded, we blend it or use it to override the final score for 'hybrid'
        if ranker_proba is not None:
            learned_hybrid = alpha * hybrid_score + (1 - alpha) * ranker_proba
            if mode == "hybrid":
                final = learned_hybrid
        else:
            learned_hybrid = None

        results.append(
            {
                "field_id": field.id,
                "title": field.title,
                "category": field.category,
                "summary": field.summary,
                "score": round(final, 4),
                "hybrid_score": round(hybrid_score, 4),
                "tfidf_score": round(tfidf_score, 4),
                "semantic_score": round(semantic_score, 4),
                "ranker_score": round(learned_hybrid, 4) if learned_hybrid is not None else None,
                "matched_keywords": keywords[:15],
                "matched_skills": matched_skills,
                "missing_skills": all_missing_skills[:10],
                "key_skills": field.key_skills,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        results = results[:top_k]

    best = results[0] if results else None

    # Recommend concrete jobs from the top matching fields
    top_field_ids = [row["field_id"] for row in results[:2]]
    recommended_jobs = match_resume_to_jobs(
        resume_text,
        field_ids=top_field_ids or None,
        top_n=top_jobs,
        tfidf_weight=tfidf_weight,
        semantic_weight=semantic_weight,
    )

    comparison = None
    if best:
        comparison = {
            "tfidf": best["tfidf_score"],
            "semantic": best["semantic_score"],
            "hybrid": best["hybrid_score"],
            "ranking_mode": mode,
            "note": (
                "Semantic often beats keywords when resumes use synonyms "
                "(e.g. AI vs Artificial Intelligence)."
            ),
        }

    return {
        "mode": mode,
        "entities": entities,
        "best_match": best,
        "rankings": results,
        "score_comparison": comparison,
        "recommended_jobs": recommended_jobs,
    }
