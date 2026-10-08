# Data guide — what FieldMatch needs

FieldMatch does **not** train a custom neural net on your files. It compares a resume to **job field profiles** using:

- TF-IDF + cosine similarity (keywords, with optional skill boosts)
- Sentence embeddings (`all-MiniLM-L6-v2`) + cosine similarity
- spaCy NER for organizations, locations, and known tech skills

So the data that matters is the **field catalog** and the **resumes you scan**.

---

## 1. Job field profiles (required)

**Where:** `data/job_fields/*.json`  
**Already included:** 10 starter fields (software, data science, AI, analytics, DevOps, frontend, security, product, marketing, HR/finance).

### Schema

```json
{
  "id": "software_engineering",
  "title": "Software Engineering",
  "category": "Technology",
  "summary": "One or two sentences shown in the UI.",
  "description": "Longer job-description style text. The matcher embeds this. Write 80–200 words covering responsibilities and tools.",
  "key_skills": ["Python", "SQL", "Git"],
  "skill_weights": {
    "python": 2.2,
    "sql": 1.5
  }
}
```

| Field | Purpose |
|-------|---------|
| `id` | Stable filename-friendly id |
| `title` | Display name |
| `category` | Grouping label (Technology, Marketing, …) |
| `summary` | Short blurb in UI |
| `description` | Main text for semantic + TF-IDF matching — treat like a JD |
| `key_skills` | Skills checked for “matched / missing” chips |
| `skill_weights` | Optional TF-IDF boosts (lowercase keys that appear in cleaned text) |

### How to get more / better field data

| Source | How to use it | Notes |
|--------|----------------|-------|
| **Write them yourself** | Copy a real JD you care about into `description`, list must-have skills | Best quality control |
| **Company career pages / LinkedIn / Indeed** | Manually copy 1 JD per field into a new JSON | Respect site terms; do not bulk-scrape if forbidden |
| **O*NET / ESCO / NOC** | Public occupation descriptions → paste into `description` | Great for non-tech fields |
| **Kaggle “job descriptions” datasets** | Filter by title, map clusters → one JSON per field | Useful for breadth; clean HTML/noise |
| **Internal HR roles** | Export role family templates from your ATS | Best for company-specific matching |

**Rule of thumb:** 8–20 well-written fields beat 200 noisy ones. Update `description` and `key_skills` when the market changes.

---

## 2. Resumes (required to use the product)

**Formats:** `.txt`, `.pdf`, `.docx`  
**Samples:** `data/resumes/`

### How to get resumes

| Source | Use case |
|--------|----------|
| Your own resume / classmates / consenting peers | Local demos |
| Synthetic resumes you write | Safe public demos (no PII) |
| Open datasets (e.g. Kaggle resume corpora) | Offline evaluation only — check licenses and strip PII before sharing |
| Recruiter ATS export | Real workflow — keep private; do not commit to git |

**Tips for PDFs:** Prefer text-based PDFs. Scanned image PDFs need OCR (not included yet).

---

## 3. Optional: single job description files

**Where:** `data/jobs/*.txt`  
Used by classic CLI/API modes (`main.py` without `--match-fields`, or `/api/v1/match/tfidf`).

Not required for the FieldMatch UI, which uses the JSON field catalog instead.

---

## 4. What you do *not* need

- Labeled “this resume = this field” training sets (unless you later build a supervised classifier)
- GPU (CPU is fine for MiniLM; first download needs internet)
- A vector database (embeddings are computed on the fly for this catalog size)

---

## 5. Improving accuracy

1. Make each field’s `description` specific (tools, duties, domain language).
2. Keep `key_skills` realistic and spelled the way resumes spell them.
3. Raise `skill_weights` only for true must-haves.
4. Prefer **hybrid** mode in the UI for balanced results.
5. Add a new JSON when you care about a new role family — restart not required if you only use the API after reload; restart the server after adding files so the catalog reloads on process start (catalog is read per request today, so new JSON files are picked up automatically).

---

## 6. Privacy

Resumes often contain emails, phones, and addresses. Run locally, do not commit real resumes, and add `uploads/` to `.gitignore` if you later save files to disk.
