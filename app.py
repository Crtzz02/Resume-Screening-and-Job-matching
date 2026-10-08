"""
FieldMatch API + web UI

Upload a resume → rank best-fit job fields using TF-IDF + semantic embeddings + NER.
"""
from __future__ import annotations

import math
from pathlib import Path
from threading import Thread
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.extract import SUPPORTED_EXTENSIONS, extract_text_from_bytes
from src.field_catalog import fields_as_dicts, load_job_fields
from src.field_matcher import match_resume_to_fields
from src.batch_matcher import batch_rank_resumes
from src.matching import compute_semantic_similarity, compute_similarity, get_model
from src.ner import extract_entities
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="FieldMatch",
    description="Match resumes to career fields using TF-IDF, semantic embeddings, and NER.",
    version="2.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Request / response models (legacy pair-matching kept for CLI parity) ---

class ResumeInput(BaseModel):
    id: str
    text: str


class MatchRequest(BaseModel):
    job_description: str
    resumes: List[ResumeInput]
    weights: Optional[Dict[str, float]] = None


class MatchResult(BaseModel):
    id: str
    score: float
    matched_keywords: Optional[List[str]] = []


class AnalyzeRequest(BaseModel):
    text: str


class EntityResponse(BaseModel):
    organizations: List[str]
    locations: List[str]
    tech_skills: List[str]


class FieldMatchTextRequest(BaseModel):
    text: str = Field(..., min_length=20)
    mode: str = "hybrid"
    top_k: Optional[int] = None


DEFAULT_SKILL_WEIGHTS = {
    "python": 2.5,
    "django": 2.0,
    "machine": 1.5,
    "learning": 1.5,
    "sql": 1.5,
    "java": 1.2,
}


@app.get("/")
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found. Expected static/index.html")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "FieldMatch", "version": "2.0.0"}


@app.get("/api/v1/fields")
async def list_fields():
    try:
        fields = load_job_fields()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"count": len(fields), "fields": fields_as_dicts(fields)}


@app.get("/api/v1/config/weights")
async def get_weights():
    return {"weights": DEFAULT_SKILL_WEIGHTS}


@app.post("/api/v1/match/fields")
async def match_fields_from_text(request: FieldMatchTextRequest):
    try:
        return match_resume_to_fields(
            request.text,
            mode=request.mode,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Field matching failed: {exc}") from exc


@app.post("/api/v1/match/fields/upload")
async def match_fields_from_upload(
    file: UploadFile = File(...),
    mode: str = Form("hybrid"),
    top_k: Optional[int] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{suffix}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = extract_text_from_bytes(file.filename, raw)
        result = match_resume_to_fields(text, mode=mode, top_k=top_k)
        result["filename"] = file.filename
        result["char_count"] = len(text)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not process resume: {exc}") from exc


@app.post("/api/v1/match/batch")
async def match_batch_resumes(
    files: List[UploadFile] = File(...),
    mode: str = Form("hybrid"),
    field_id: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
):
    """
    Recruiter mode: rank many resumes against one career field or a pasted JD.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one resume.")
    if not field_id and not (job_description and job_description.strip()):
        raise HTTPException(
            status_code=400,
            detail="Choose a field_id or paste a job_description.",
        )

    resumes = []
    for upload in files[:40]:  # soft cap for college demo
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        raw = await upload.read()
        if not raw:
            continue
        try:
            text = extract_text_from_bytes(upload.filename, raw)
            resumes.append({"filename": upload.filename, "text": text})
        except Exception as exc:
            resumes.append({"filename": upload.filename, "text": "", "parse_error": str(exc)})

    # Separate parse failures
    parsed = [r for r in resumes if r.get("text")]
    parse_errors = [
        {"filename": r["filename"], "error": r.get("parse_error") or "Could not extract text"}
        for r in resumes
        if not r.get("text")
    ]

    if not parsed:
        raise HTTPException(status_code=400, detail="No readable resumes in upload.")

    try:
        result = batch_rank_resumes(
            parsed,
            field_id=field_id,
            job_description=job_description,
            mode=mode,
        )
        result["errors"] = (result.get("errors") or []) + parse_errors
        result["error_count"] = len(result["errors"])
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch matching failed: {exc}") from exc


@app.post("/api/v1/match/tfidf", response_model=List[MatchResult])
async def match_tfidf(request: MatchRequest):
    if not request.job_description or not request.resumes:
        raise HTTPException(
            status_code=400,
            detail="Job description and at least one resume must be provided.",
        )

    job_clean = clean_text(request.job_description)
    active_weights = request.weights if request.weights is not None else DEFAULT_SKILL_WEIGHTS
    results = []

    for resume in request.resumes:
        try:
            resume_clean = clean_text(resume.text)
            score, tfidf_df = compute_similarity(
                resume_clean, job_clean, skill_weights=active_weights
            )
            keywords = tfidf_df.loc[
                :, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)
            ].columns.tolist()
            final_score = float(score) if not math.isnan(score) else 0.0
            results.append(
                {
                    "id": resume.id,
                    "score": final_score,
                    "matched_keywords": keywords,
                }
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing resume {resume.id}: {exc}",
            ) from exc

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


@app.post("/api/v1/match/semantic", response_model=List[MatchResult])
async def match_semantic(request: MatchRequest):
    if not request.job_description or not request.resumes:
        raise HTTPException(
            status_code=400,
            detail="Job description and at least one resume must be provided.",
        )

    job_clean = clean_text(request.job_description)
    results = []

    for resume in request.resumes:
        try:
            resume_clean = clean_text(resume.text)
            score, _ = compute_semantic_similarity(resume_clean, job_clean)
            final_score = float(score) if not math.isnan(score) else 0.0
            results.append(
                {
                    "id": resume.id,
                    "score": final_score,
                    "matched_keywords": [],
                }
            )
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="sentence-transformers is not installed/loaded.",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing resume {resume.id}: {exc}",
            ) from exc

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


@app.post("/api/v1/analyze/entities", response_model=EntityResponse)
async def analyze_entities(request: AnalyzeRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided for analysis.")
    try:
        return extract_entities(request.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error during entity extraction: {exc}",
        ) from exc


def preload_model():
    try:
        get_model()
    except Exception:
        # Startup should not crash if model download fails; first request will surface it.
        pass


Thread(target=preload_model, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
