import os
import json
from typing import Dict, List, Optional
import numpy as np

# Lazy load joblib to avoid slowing down startup if model not loaded
try:
    import joblib
except ImportError:
    joblib = None

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_ranker.joblib")
META_PATH = os.path.join(MODEL_DIR, "ranker_meta.json")

_model = None
_meta = None
_model_loaded = False

def load_model():
    global _model, _meta, _model_loaded
    if _model_loaded:
        return
    _model_loaded = True
    if joblib is None:
        return
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            with open(META_PATH, "r") as f:
                _meta = json.load(f)
        except Exception:
            _model = None
            _meta = None

def extract_features_dict(
    tfidf_score: float,
    semantic_score: float,
    hybrid_score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    key_skills: List[str],
    skill_weights: Optional[Dict[str, float]],
    resume_len: int,
    field_len: int
) -> Dict[str, float]:
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    total_skills = len(key_skills)
    skill_coverage = matched_count / total_skills if total_skills > 0 else 0.0
    
    weighted_skill_score = 0.0
    if skill_weights:
        for skill in matched_skills:
            weighted_skill_score += skill_weights.get(skill, 1.0)
    else:
        weighted_skill_score = float(matched_count)

    return {
        "tfidf_score": float(tfidf_score),
        "semantic_score": float(semantic_score),
        "hybrid_score": float(hybrid_score),
        "matched_skill_count": float(matched_count),
        "missing_skill_count": float(missing_count),
        "skill_coverage": float(skill_coverage),
        "weighted_skill_score": float(weighted_skill_score),
        "resume_length": float(resume_len),
        "field_length": float(field_len),
    }

def predict_match_proba(feature_dict: Dict[str, float]) -> Optional[float]:
    """
    Returns ranker probability if the model is loaded, otherwise None.
    """
    load_model()
    if _model is None or _meta is None:
        return None
    
    features_ordered = []
    feature_names = _meta.get("feature_names", [])
    if not feature_names:
        return None
        
    for name in feature_names:
        features_ordered.append(feature_dict.get(name, 0.0))
        
    X = np.array([features_ordered])
    try:
        # Assuming the positive class is at index 1
        proba = _model.predict_proba(X)[0][1]
        return float(proba)
    except Exception:
        return None
