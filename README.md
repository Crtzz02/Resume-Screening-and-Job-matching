# FieldMatch — Resume → Career Field Matcher

Upload a resume and get a ranked list of **job fields** that fit it best.
Uses the same ML stack as before: **TF-IDF + cosine**, **sentence-transformers (MiniLM)**, and **spaCy NER**.

## What you get

- Web UI at `/` — upload `.txt` / `.pdf` / `.docx` or paste text
- Hybrid ranking across curated fields in `data/job_fields/`
- Explainable signals: matched skills, missing skills, keyword overlap, NER entities
- FastAPI endpoints for integrations
- CLI for classic JD matching and field matching

## Quick start

```bash
cd Semantic-Resume-Screening-System
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if wheel install skipped the model

# First run may download the MiniLM embedding model
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### CLI

```bash
# Rank one resume against all job fields (hybrid)
python main.py --match-fields --resume data/resumes/sample_resume.txt --mode hybrid

# Classic: rank many resumes against one job description
python main.py --resumes data/resumes/ --job data/jobs/sample_job.txt --mode tfidf
python main.py --mode semantic --resumes data/resumes/ --job data/jobs/sample_job.txt
```

## Project layout

```text
├── app.py                 # FastAPI + static UI server
├── main.py                # CLI
├── static/                # FieldMatch web UI
├── data/
│   ├── job_fields/        # JSON field profiles (required for field matching)
│   ├── resumes/           # Sample resumes
│   └── jobs/              # Sample single JDs (classic mode)
└── src/
    ├── preprocessing.py   # NLTK cleaning
    ├── matching.py        # TF-IDF + semantic engines
    ├── ner.py             # spaCy entities + tech skills
    ├── field_catalog.py   # Load job field JSON
    ├── field_matcher.py   # Resume → field ranking
    └── extract.py         # PDF / DOCX / TXT extraction
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/fields` | List job field catalog |
| POST | `/api/v1/match/fields` | JSON `{ "text", "mode" }` → rankings |
| POST | `/api/v1/match/fields/upload` | Multipart file upload → rankings |
| POST | `/api/v1/match/tfidf` | Classic multi-resume vs one JD |
| POST | `/api/v1/match/semantic` | Classic semantic match |
| POST | `/api/v1/analyze/entities` | NER only |

Interactive docs: `/docs`

## Integrate your JD CSV / Excel files

Put (or keep) your files in Downloads, then from the project root:

```bash
pip install openpyxl
python scripts/build_fields_from_jds.py
```

This clusters `data.csv` + `job_title_des.csv` + `weak_fields_job_descriptions.xlsx` into `data/job_fields/*.json`. Edit paths inside `scripts/build_fields_from_jds.py` if your files moved. Then run `python app.py` and refresh the UI.

## Data you need (and how to get it)

See **[DATA.md](DATA.md)** for a full guide. Short version:

1. **Job field profiles** (required) — JSON files in `data/job_fields/` with title, description, skills, optional weights. Seed catalog included (10 fields).
2. **Resumes to test** — your own `.txt` / `.pdf` / `.docx`, or the samples under `data/resumes/`.
3. **Optional: real job postings** — paste into a field’s `description` to tune that field toward a real market role.

You do **not** need labeled training data. Matching is unsupervised similarity (TF-IDF + embeddings), not a supervised classifier.

## Notes

- First semantic/hybrid run downloads `all-MiniLM-L6-v2` from Hugging Face (needs network once).
- Prefer running locally or on a persistent VM/Docker host; the full torch stack is heavy for tiny serverless platforms.
