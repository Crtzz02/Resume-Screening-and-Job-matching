import argparse
import os
from typing import Dict, List, Optional

from src.field_matcher import match_resume_to_fields
from src.matching import compute_semantic_similarity, compute_similarity
from src.ner import extract_entities
from src.preprocessing import clean_text

DEFAULT_SKILL_WEIGHTS = {
    "python": 2.5,
    "django": 2.0,
    "machine": 1.5,
    "learning": 1.5,
    "sql": 1.5,
    "java": 1.2,
}


def display_results(results: List[Dict], mode: str, top_n: int = 1):
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 60)
    print(f"RESUME SCREENING RANKING (Mode: {mode.upper()})")
    print("=" * 60)
    print(f"{'RANK':<5} | {'RESUME FILENAME':<30} | {'SCORE':<10}")
    print("-" * 60)

    for i, res in enumerate(results, 1):
        print(f"{i:<5} | {res['filename']:<30} | {res['score']:>8.2%}")
    print("=" * 60)

    for i in range(min(top_n, len(results))):
        top = results[i]
        print(f"\n[Rank {i + 1}] Detail: {top['filename']}")

        if mode == "tfidf":
            matched_str = ", ".join(top["keywords"]) if top["keywords"] else "None"
            print(f"Matched Keywords: {matched_str}")
        else:
            print("Note: Semantic mode provides conceptual similarity (no keyword list).")

        if top.get("entities"):
            orgs = top["entities"].get("organizations", [])[:3]
            locs = top["entities"].get("locations", [])[:3]
            skills = top["entities"].get("tech_skills", [])[:5]
            if orgs:
                print(f"Top Orgs: {', '.join(orgs)}")
            if locs:
                print(f"Locations: {', '.join(locs)}")
            if skills:
                print(f"Tech Skills found via NER: {', '.join(skills)}")

        print("Status: ", end="")
        if top["score"] > 0.4:
            print("Strong Match")
        elif top["score"] > 0.15:
            print("Potential Match")
        else:
            print("Low Match")
    print("=" * 60 + "\n")


def display_field_results(payload: Dict):
    rankings = payload.get("rankings") or []
    print("\n" + "=" * 64)
    print(f"FIELD MATCH RANKING (Mode: {payload.get('mode', '').upper()})")
    print("=" * 64)
    print(f"{'RANK':<5} | {'FIELD':<36} | {'SCORE':<10}")
    print("-" * 64)
    for i, row in enumerate(rankings, 1):
        print(f"{i:<5} | {row['title']:<36} | {row['score']:>8.2%}")
    print("=" * 64)

    best = payload.get("best_match")
    if best:
        print(f"\nBest fit: {best['title']} ({best['category']})")
        print(f"Summary: {best['summary']}")
        if best.get("matched_skills"):
            print(f"Matched skills: {', '.join(best['matched_skills'])}")
        if best.get("missing_skills"):
            print(f"Skills to grow: {', '.join(best['missing_skills'][:8])}")

    entities = payload.get("entities") or {}
    if entities.get("organizations") or entities.get("tech_skills"):
        print("\nNER snapshot:")
        if entities.get("organizations"):
            print(f"  Orgs: {', '.join(entities['organizations'][:5])}")
        if entities.get("tech_skills"):
            print(f"  Skills: {', '.join(entities['tech_skills'][:8])}")
    print()


def run_screening(resume_input: str, job_path: str, mode: str, weights: Optional[Dict] = None):
    if not os.path.exists(job_path):
        print(f"Error: Job description not found at {job_path}")
        return

    resume_files = []
    if os.path.isdir(resume_input):
        resume_files = [
            os.path.join(resume_input, f)
            for f in os.listdir(resume_input)
            if f.endswith(".txt")
        ]
    elif os.path.isfile(resume_input):
        resume_files = [resume_input]
    else:
        print(f"Error: Invalid resume path: {resume_input}")
        return

    if not resume_files:
        print("No resumes found to process.")
        return

    try:
        with open(job_path, "r", encoding="utf-8") as handle:
            job_raw = handle.read()
        job_clean = clean_text(job_raw)
    except Exception as exc:
        print(f"Error reading job description: {exc}")
        return

    print(f"Analyzing {len(resume_files)} resume(s) using {mode.upper()} engine...")

    results = []
    for res_path in resume_files:
        try:
            with open(res_path, "r", encoding="utf-8") as handle:
                resume_raw = handle.read()

            resume_clean = clean_text(resume_raw)

            if mode == "semantic":
                score, _ = compute_semantic_similarity(resume_clean, job_clean)
                keywords = []
            else:
                score, tfidf_df = compute_similarity(
                    resume_clean, job_clean, skill_weights=weights
                )
                keywords = tfidf_df.loc[
                    :, (tfidf_df.iloc[0] > 0) & (tfidf_df.iloc[1] > 0)
                ].columns.tolist()

            entities = extract_entities(resume_raw)
            results.append(
                {
                    "filename": os.path.basename(res_path),
                    "score": score,
                    "keywords": keywords,
                    "entities": entities,
                }
            )
        except Exception as exc:
            print(f"Skipping {res_path} due to error: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)
    display_results(results, mode)


def run_field_match(resume_path: str, mode: str):
    if not os.path.isfile(resume_path):
        print(f"Error: Resume file not found: {resume_path}")
        return
    with open(resume_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    print(f"Matching {os.path.basename(resume_path)} to job fields ({mode})...")
    payload = match_resume_to_fields(text, mode=mode)
    display_field_results(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Resume Screening & Field Matching")
    parser.add_argument(
        "--resumes",
        type=str,
        default="data/resumes/",
        help="Resume file/directory for classic JD matching",
    )
    parser.add_argument(
        "--job",
        type=str,
        default="data/jobs/sample_job.txt",
        help="Job description file for classic matching",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tfidf", "semantic", "hybrid"],
        default="tfidf",
        help="Similarity engine",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Disable skill weighting in TF-IDF mode",
    )
    parser.add_argument(
        "--match-fields",
        action="store_true",
        help="Rank resume against data/job_fields catalog instead of a single JD",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="data/resumes/sample_resume.txt",
        help="Single resume path used with --match-fields",
    )

    args = parser.parse_args()

    if args.match_fields:
        mode = "hybrid" if args.mode == "tfidf" else args.mode
        run_field_match(args.resume, mode=mode)
    else:
        if args.mode == "hybrid":
            print("Classic JD matching supports tfidf or semantic. Use --match-fields for hybrid.")
            raise SystemExit(2)
        weights = None if (args.no_weights or args.mode == "semantic") else DEFAULT_SKILL_WEIGHTS
        run_screening(args.resumes, args.job, mode=args.mode, weights=weights)
