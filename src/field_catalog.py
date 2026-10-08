"""Load and validate job-field profiles used for resume→field matching."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class JobField:
    id: str
    title: str
    category: str
    summary: str
    description: str
    key_skills: List[str]
    skill_weights: Dict[str, float]

    @property
    def match_text(self) -> str:
        """Corpus text used by TF-IDF / semantic engines."""
        skills = " ".join(self.key_skills)
        return f"{self.title}. {self.summary} {self.description} Skills: {skills}"


def default_fields_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "job_fields"


def load_job_fields(fields_dir: Optional[Path] = None) -> List[JobField]:
    """Load every *.json job field profile from the catalog directory."""
    root = fields_dir or default_fields_dir()
    if not root.exists():
        raise FileNotFoundError(
            f"Job field catalog not found at {root}. Add JSON profiles under data/job_fields/."
        )

    fields: List[JobField] = []
    for path in sorted(root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        fields.append(_parse_field(payload, source=path.name))

    if not fields:
        raise ValueError(f"No job field JSON files found in {root}")
    return fields


def _parse_field(payload: dict, source: str) -> JobField:
    required = ("id", "title", "category", "summary", "description", "key_skills")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{source} is missing keys: {', '.join(missing)}")

    skills = payload["key_skills"]
    if not isinstance(skills, list) or not skills:
        raise ValueError(f"{source}: key_skills must be a non-empty list")

    weights = payload.get("skill_weights") or {}
    if not isinstance(weights, dict):
        raise ValueError(f"{source}: skill_weights must be an object/dict")

    return JobField(
        id=str(payload["id"]),
        title=str(payload["title"]),
        category=str(payload["category"]),
        summary=str(payload["summary"]),
        description=str(payload["description"]),
        key_skills=[str(s) for s in skills],
        skill_weights={str(k).lower(): float(v) for k, v in weights.items()},
    )


def fields_as_dicts(fields: List[JobField]) -> List[dict]:
    return [
        {
            "id": f.id,
            "title": f.title,
            "category": f.category,
            "summary": f.summary,
            "key_skills": f.key_skills,
        }
        for f in fields
    ]
