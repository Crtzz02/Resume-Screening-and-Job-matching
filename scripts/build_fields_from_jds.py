"""
Build / refresh data/job_fields/*.json from your JD CSV/XLSX files.

Usage (from project root, venv active):
  python scripts/build_fields_from_jds.py
  python scripts/build_fields_from_jds.py --max-per-field 12
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "job_fields"

# Default paths — change if your files moved
DEFAULT_SOURCES = [
    Path(r"c:\Users\OMKAR KUDAV\Downloads\data.csv"),
    Path(r"c:\Users\OMKAR KUDAV\Downloads\job_title_des.csv\job_title_des.csv"),
    Path(r"c:\Users\OMKAR KUDAV\Downloads\weak_fields_job_descriptions.xlsx"),
    Path(r"c:\Users\OMKAR KUDAV\Downloads\gap_fields_job_descriptions.csv"),
]

# field_id -> (title, category, summary, title_patterns, seed_skills)
FIELD_SPECS: Dict[str, dict] = {
    "software_engineering": {
        "title": "Software Engineering",
        "category": "Technology",
        "summary": "Build and maintain applications, APIs, and systems with modern engineering practices.",
        "patterns": [
            r"software engineer",
            r"java developer",
            r"javascript developer",
            r"node.?js",
            r"php developer",
            r"django developer",
            r"backend developer",
            r"full.?stack",
            r"flutter developer",
            r"wordpress developer",
            r"ios developer",
            r"\bpython developer\b",
        ],
        "skills": [
            "Python", "Java", "JavaScript", "TypeScript", "SQL", "Git",
            "REST API", "Docker", "Django", "Node.js", "Spring", "CI/CD",
        ],
        "weights": {"python": 2.0, "java": 1.8, "javascript": 1.6, "sql": 1.5, "docker": 1.4, "git": 1.4},
    },
    "frontend_engineering": {
        "title": "Frontend / UI Engineering",
        "category": "Technology",
        "summary": "Craft responsive web interfaces with modern JavaScript frameworks.",
        "patterns": [
            r"frontend",
            r"front.?end",
            r"ui engineer",
            r"ui developer",
            r"react developer",
            r"vue developer",
            r"angular developer",
        ],
        "skills": [
            "JavaScript", "TypeScript", "React", "HTML", "CSS", "Vue",
            "Angular", "UI", "Responsive Design", "Jest", "Accessibility",
        ],
        "weights": {"javascript": 2.2, "typescript": 2.0, "react": 2.3, "css": 1.6, "html": 1.5},
    },
    "data_science": {
        "title": "Data Science & Machine Learning",
        "category": "Technology",
        "summary": "Turn data into insights and models using statistics and machine learning.",
        "patterns": [
            r"data scientist",
            r"machine learning",
            r"ml engineer",
            r"deep learning",
        ],
        "skills": [
            "Python", "Machine Learning", "SQL", "Statistics", "pandas",
            "NumPy", "scikit-learn", "Deep Learning", "Feature Engineering",
            "Data Visualization",
        ],
        "weights": {"python": 2.0, "machine": 2.2, "learning": 2.2, "sql": 1.6, "statistics": 1.8},
    },
    "data_analytics": {
        "title": "Data Analytics & Business Intelligence",
        "category": "Business / Data",
        "summary": "Analyze business data, build dashboards, and support decisions with reporting.",
        "patterns": [
            r"data analyst",
            r"business intelligence",
            r"bi analyst",
            r"reporting analyst",
            r"power.?bi",
            r"tableau",
        ],
        "skills": [
            "SQL", "Excel", "Tableau", "Power BI", "Data Visualization",
            "KPI", "Business Intelligence", "Reporting", "Python", "Dashboard",
        ],
        "weights": {"sql": 2.4, "excel": 1.8, "tableau": 2.0, "dashboard": 1.7, "kpi": 1.5},
    },
    "devops_cloud": {
        "title": "DevOps & Cloud Engineering",
        "category": "Technology",
        "summary": "Automate infrastructure, deployments, and reliability on cloud platforms.",
        "patterns": [
            r"devops",
            r"\bsre\b",
            r"cloud engineer",
            r"platform engineer",
            r"site reliability",
        ],
        "skills": [
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "CI/CD", "Linux", "Python", "Monitoring",
        ],
        "weights": {"aws": 2.2, "docker": 2.0, "kubernetes": 2.2, "terraform": 2.0, "linux": 1.6},
    },
    "ai_engineering": {
        "title": "AI / LLM Engineering",
        "category": "Technology",
        "summary": "Build AI products with LLMs, embeddings, retrieval, and neural networks.",
        "patterns": [
            r"\bllm\b",
            r"genai",
            r"generative ai",
            r"nlp engineer",
            r"ai engineer",
            r"prompt engineer",
            r"applied ai",
        ],
        "skills": [
            "Artificial Intelligence", "LLM", "NLP", "Python", "Transformers",
            "Embeddings", "RAG", "PyTorch", "Prompt Engineering", "Hugging Face",
        ],
        "weights": {"llm": 2.4, "nlp": 2.0, "python": 1.8, "rag": 1.8, "pytorch": 1.6},
    },
    "cybersecurity": {
        "title": "Cybersecurity",
        "category": "Technology",
        "summary": "Protect systems through security analysis, monitoring, and risk reduction.",
        "patterns": [
            r"cyber",
            r"security analyst",
            r"security engineer",
            r"penetration",
            r"\bsoc\b",
            r"infosec",
            r"appsec",
            r"\bgrc\b",
            r"incident response",
            r"threat intel",
            r"cloud security",
        ],
        "skills": [
            "Cybersecurity", "Network Security", "Penetration Testing", "SIEM",
            "Vulnerability Assessment", "Linux", "Python", "Incident Response",
            "OWASP", "Cloud Security",
        ],
        "weights": {"security": 2.4, "penetration": 2.0, "siem": 1.8, "vulnerability": 1.8},
    },
    "product_management": {
        "title": "Product Management",
        "category": "Business / Product",
        "summary": "Define product strategy, prioritize roadmaps, and align teams with customer needs.",
        "patterns": [
            r"product manager",
            r"product owner",
            r"technical product",
            r"growth product",
            r"director of product",
        ],
        "skills": [
            "Product Management", "Roadmap", "User Stories", "Agile",
            "Stakeholder Management", "Analytics", "A/B Testing", "Prioritization", "OKRs",
        ],
        "weights": {"product": 2.2, "roadmap": 2.0, "agile": 1.6, "stakeholder": 1.5},
    },
    "digital_marketing": {
        "title": "Digital Marketing",
        "category": "Marketing",
        "summary": "Grow acquisition and engagement through SEO, ads, content, and analytics.",
        "patterns": [
            r"marketing",
            r"\bseo\b",
            r"\bsem\b",
            r"content marketing",
            r"performance marketing",
            r"growth marketer",
            r"social media manager",
        ],
        "skills": [
            "SEO", "SEM", "Google Analytics", "Content Marketing", "Social Media",
            "Email Marketing", "Google Ads", "Copywriting", "CRM", "Campaign Management",
        ],
        "weights": {"seo": 2.2, "marketing": 2.0, "campaign": 1.7, "ads": 1.8},
    },
    "finance_accounting": {
        "title": "Finance & Accounting",
        "category": "Finance",
        "summary": "Manage financial reporting, analysis, budgeting, and compliance.",
        "patterns": [
            r"accountant",
            r"financial analyst",
            r"fp&a",
            r"auditor",
            r"controller",
            r"tax analyst",
            r"finance manager",
        ],
        "skills": [
            "Accounting", "Financial Analysis", "Excel", "Budgeting", "Forecasting",
            "GAAP", "Financial Modeling", "Audit", "FP&A",
        ],
        "weights": {"accounting": 2.2, "finance": 2.0, "excel": 1.8, "budget": 1.6, "gaap": 1.7},
    },
    "hr_people_ops": {
        "title": "Human Resources & People Operations",
        "category": "People / HR",
        "summary": "Support hiring, employee experience, policies, and organizational development.",
        "patterns": [
            r"human resources",
            r"\brecruiter\b",
            r"talent acquisition",
            r"hrbp",
            r"people ops",
            r"hris",
            r"compensation analyst",
            r"hr manager",
            r"hr generalist",
        ],
        "skills": [
            "Recruiting", "Onboarding", "Employee Relations", "HRIS", "ATS",
            "Performance Management", "Talent Acquisition", "Payroll", "Interviewing",
        ],
        "weights": {"recruiting": 2.0, "hr": 2.0, "talent": 1.7, "hris": 1.6, "onboarding": 1.6},
    },
}


def load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  SKIP missing: {path}")
        return pd.DataFrame(columns=["Job Title", "Description"])

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"job title", "title", "job_title"}:
            rename[col] = "Job Title"
        elif key in {"description", "job description", "job_description", "desc"}:
            rename[col] = "Description"
    df = df.rename(columns=rename)

    if "Job Title" not in df.columns or "Description" not in df.columns:
        raise ValueError(f"{path} needs Job Title + Description columns; got {list(df.columns)}")

    out = df[["Job Title", "Description"]].copy()
    out["Job Title"] = out["Job Title"].fillna("").astype(str)
    out["Description"] = out["Description"].fillna("").astype(str)
    out = out[(out["Job Title"].str.strip() != "") & (out["Description"].str.strip() != "")]
    out["source"] = path.name
    print(f"  loaded {len(out):4d} rows from {path.name}")
    return out


def assign_field(title: str) -> str | None:
    text = title.lower()
    # First match wins; order matters for overlapping titles
    priority = [
        "ai_engineering",
        "cybersecurity",
        "frontend_engineering",
        "data_science",
        "data_analytics",
        "devops_cloud",
        "product_management",
        "digital_marketing",
        "finance_accounting",
        "hr_people_ops",
        "software_engineering",
    ]
    for field_id in priority:
        for pat in FIELD_SPECS[field_id]["patterns"]:
            if re.search(pat, text, flags=re.IGNORECASE):
                return field_id
    return None


def clean_desc(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def build_description(rows: pd.DataFrame, summary: str, max_per_field: int) -> str:
    ranked = rows.assign(_len=rows["Description"].str.len()).sort_values("_len", ascending=False)
    picks = ranked.head(max_per_field)
    chunks = [summary]
    for _, row in picks.iterrows():
        chunks.append(clean_desc(row["Description"], max_chars=900))
    # Keep field profile readable for embeddings
    merged = " ".join(chunks)
    return merged[:8000]


def skill_hits(corpus: str, skills: List[str]) -> List[str]:
    low = corpus.lower()
    found = [s for s in skills if s.lower() in low]
    return found or skills


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate JD CSVs into job_fields JSON")
    parser.add_argument("--max-per-field", type=int, default=10, help="JDs sampled per field")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[str(p) for p in DEFAULT_SOURCES],
        help="Paths to CSV/XLSX files",
    )
    args = parser.parse_args()

    print("Loading sources...")
    frames = [load_source(Path(p)) for p in args.sources]
    all_jds = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    print(f"Combined JD rows: {len(all_jds)}")

    all_jds["field_id"] = all_jds["Job Title"].map(assign_field)
    mapped = all_jds.dropna(subset=["field_id"])
    unmapped = len(all_jds) - len(mapped)
    print(f"Mapped to fields: {len(mapped)} | Unmapped skipped: {unmapped}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: List[Tuple[str, int]] = []
    job_index: List[dict] = []
    jobs_per_field = max(args.max_per_field, 20)

    for field_id, spec in FIELD_SPECS.items():
        subset = mapped[mapped["field_id"] == field_id]
        counts.append((field_id, len(subset)))

        if subset.empty:
            print(f"  WARN {field_id}: no rows matched — keeping existing JSON if present")
            continue

        description = build_description(subset, spec["summary"], args.max_per_field)
        skills = skill_hits(description, spec["skills"])
        # Prefer skills that appear often in sampled titles+desc
        blob = " ".join(subset["Job Title"].head(50)) + " " + description
        skills = skill_hits(blob, spec["skills"])

        payload = {
            "id": field_id,
            "title": spec["title"],
            "category": spec["category"],
            "summary": spec["summary"],
            "description": description,
            "key_skills": skills,
            "skill_weights": spec["weights"],
            "meta": {
                "source_jd_count": int(len(subset)),
                "sampled_jds": int(min(args.max_per_field, len(subset))),
            },
        }
        out_path = OUT_DIR / f"{field_id}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out_path.name} from {len(subset)} JDs")

        # Build searchable job postings index (diverse titles, longer descriptions first)
        ranked = subset.assign(_len=subset["Description"].str.len()).sort_values(
            "_len", ascending=False
        )
        # Prefer unique titles
        seen_titles: set = set()
        picked = 0
        for _, row in ranked.iterrows():
            title = str(row["Job Title"]).strip()
            title_key = title.lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            job_index.append(
                {
                    "id": f"{field_id}_{picked}",
                    "title": title,
                    "field_id": field_id,
                    "field_title": spec["title"],
                    "category": spec["category"],
                    "description": clean_desc(row["Description"], max_chars=1800),
                    "source": str(row.get("source", "")),
                }
            )
            picked += 1
            if picked >= jobs_per_field:
                break

    index_path = ROOT / "data" / "job_index.json"
    index_path.write_text(
        json.dumps({"count": len(job_index), "jobs": job_index}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  wrote {index_path.name} with {len(job_index)} job postings")

    print("\nField coverage (mapped rows):")
    for field_id, n in sorted(counts, key=lambda x: -x[1]):
        print(f"  {field_id:22} {n:5}")

    # Dataset advice for thin fields
    print("\nDataset advice (add more JDs if count is low):")
    for field_id, n in sorted(counts, key=lambda x: x[1]):
        if n < 40:
            print(f"  NEED MORE -> {field_id}: only {n} mapped JDs (aim for 40+)")
        elif n < 80:
            print(f"  OK / thin -> {field_id}: {n} JDs")

    print("\nDone. Restart or refresh the app - FieldMatch loads data/job_fields/ automatically.")


if __name__ == "__main__":
    main()
