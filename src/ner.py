"""Named entity extraction with spaCy + post-filters for resume noise."""
from __future__ import annotations

import re
from typing import Dict, List, Set

import spacy

# --- Tech skills the small NER model often mislabels as ORG/LOC ---
TECH_SKILL_TERMS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "golang", "go",
    "rust", "kotlin", "swift", "php", "ruby", "scala",
    "react", "angular", "vue", "next.js", "nextjs", "node.js", "nodejs", "node",
    "express.js", "expressjs", "express", "django", "flask", "fastapi", "spring",
    "sql", "mysql", "postgresql", "mongodb", "nosql", "redis", "firebase",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab",
    "html", "css", "sass", "tailwind", "bootstrap",
    "machine learning", "deep learning", "data science", "artificial intelligence",
    "nlp", "llm", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "power bi", "tableau", "excel", "linux", "bash",
]

# Places that spaCy often tags as ORG on Indian resumes
KNOWN_PLACES = {
    "maharashtra", "mumbai", "thane", "nerul", "navi mumbai", "pune", "nagpur",
    "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad", "chennai",
    "kolkata", "ahmedabad", "jaipur", "kerala", "karnataka", "gujarat",
    "telangana", "tamil nadu", "uttar pradesh", "west bengal", "goa",
    "rabodi", "thane rabodi",
}

ORG_BLOCKLIST_PHRASES = {
    "computer engineering", "information technology", "information technology graduated",
    "computer science", "electronics", "mechanical engineering", "civil engineering",
    "bachelor of engineering", "bachelor of technology", "master of technology",
    "engineering", "graduated", "graduate",
}

ORG_BLOCKLIST_REGEX = re.compile(
    r"(?i)^("
    r"standard\s*20\d{2}"
    r"|batch\s*20\d{2}"
    r"|class\s*of\s*20\d{2}"
    r"|20\d{2}\s*[-–]\s*20\d{2}"
    r"|b\.?\s*e\.?"
    r"|b\.?\s*tech\.?"
    r"|m\.?\s*tech\.?"
    r"|diploma"
    r"|cgpa"
    r"|sgpa"
    r")$"
)

PERSON_LIKE = re.compile(r"^[A-Z][a-z]{1,20}$")


def _skill_patterns() -> List[dict]:
    patterns: List[dict] = []
    for term in TECH_SKILL_TERMS:
        tokens = term.split()
        if len(tokens) == 1:
            patterns.append({"label": "TECH_SKILL", "pattern": [{"LOWER": tokens[0]}]})
        else:
            patterns.append(
                {"label": "TECH_SKILL", "pattern": [{"LOWER": t} for t in tokens]}
            )
    for phrase in ("node.js", "express.js", "next.js", "c++", "c#"):
        patterns.append({"label": "TECH_SKILL", "pattern": phrase})
    return patterns


try:
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe(
            "entity_ruler",
            before="ner",
            config={"overwrite_ents": True},
        )
        ruler.add_patterns(_skill_patterns())
        place_patterns = [
            {"label": "GPE", "pattern": place}
            for place in sorted(KNOWN_PLACES, key=len, reverse=True)
        ]
        ruler.add_patterns(place_patterns)
except OSError:
    nlp = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _tech_norm_set() -> Set[str]:
    out = {_norm(t) for t in TECH_SKILL_TERMS}
    out.update({"node.js", "nodejs", "express.js", "expressjs", "next.js", "c++", "c#"})
    return out


TECH_NORMS = _tech_norm_set()

INSTITUTION_HINT = re.compile(
    r"(?i)\b(university|college|institute|school|polytechnic|iit|nit|ltd|inc|pvt|"
    r"llc|corp|technologies|solutions|labs|company|hospital|bank|foundation)\b"
)


def _is_bad_location(text: str) -> bool:
    n = _norm(text)
    if len(n) <= 1:
        return True
    if n in TECH_NORMS:
        return True
    if re.search(r"(?i)\.(js|ts|py|net)$", n) or ".js" in n:
        return True
    if re.search(r"(?i)\b(developer|engineer|analyst|intern|student)\b", n):
        return True
    return False


def _is_good_org(text: str) -> bool:
    n = _norm(text)
    if len(n) <= 2:
        return False
    if n in TECH_NORMS or n in KNOWN_PLACES:
        return False
    if n in ORG_BLOCKLIST_PHRASES:
        return False
    if ORG_BLOCKLIST_REGEX.match(n):
        return False
    if re.fullmatch(r"20\d{2}", n):
        return False
    # Lone first names mis-tagged as ORG
    if PERSON_LIKE.match(text.strip()) and " " not in text.strip():
        return False

    # Degree / major phrases without institution words
    if any(
        bad in n
        for bad in (
            "engineering",
            "graduated",
            "information technology",
            "computer science",
            "computer engineering",
        )
    ):
        if not INSTITUTION_HINT.search(text):
            return False

    if INSTITUTION_HINT.search(text):
        return True

    words = text.strip().split()
    if len(words) >= 2 and all(w[:1].isupper() for w in words if w.isalpha()):
        return True
    if text.isupper() and 2 <= len(text) <= 6:
        return True
    # Drop weak single-token ORGs
    if len(words) == 1:
        return False
    return True


def _display_skill(term: str) -> str:
    special = {
        "node.js": "Node.js",
        "express.js": "Express.js",
        "next.js": "Next.js",
        "c++": "C++",
        "c#": "C#",
        "aws": "AWS",
        "sql": "SQL",
        "nlp": "NLP",
        "llm": "LLM",
        "html": "HTML",
        "css": "CSS",
        "gcp": "GCP",
    }
    key = term.lower()
    if key in special:
        return special[key]
    if " " in term:
        return term.title()
    return term.capitalize() if term.islower() else term


def _skill_alias_key(text: str) -> str:
    """Collapse Node / Node.js / nodejs into one key."""
    n = _norm(text).replace(" ", "")
    aliases = {
        "node": "nodejs",
        "nodejs": "nodejs",
        "node.js": "nodejs",
        "express": "expressjs",
        "expressjs": "expressjs",
        "express.js": "expressjs",
        "next": "nextjs",
        "nextjs": "nextjs",
        "next.js": "nextjs",
    }
    return aliases.get(n, n)


def _clean_span(text: str) -> str:
    # spaCy sometimes eats the next line into an ORG span
    text = text.split("\n")[0].strip().strip(",.;:|")
    return text


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract organizations, locations, and tech skills from resume text.
    Uses spaCy NER + EntityRuler, then filters common resume false positives.
    """
    if not nlp:
        return {"organizations": [], "locations": [], "tech_skills": []}

    doc = nlp(text)

    organizations: List[str] = []
    locations: List[str] = []
    tech_skills: List[str] = []

    org_seen: Set[str] = set()
    loc_seen: Set[str] = set()
    skill_seen: Set[str] = set()

    for ent in doc.ents:
        raw = _clean_span(ent.text)
        if not raw:
            continue
        key = _norm(raw)

        if ent.label_ == "TECH_SKILL":
            alias = _skill_alias_key(raw)
            if alias not in skill_seen:
                skill_seen.add(alias)
                tech_skills.append(raw)
            continue

        if ent.label_ in {"GPE", "LOC", "FAC"}:
            if key in TECH_NORMS or _is_bad_location(raw):
                alias = _skill_alias_key(raw)
                if key in TECH_NORMS and alias not in skill_seen:
                    skill_seen.add(alias)
                    tech_skills.append(raw)
                continue
            if key not in loc_seen:
                loc_seen.add(key)
                locations.append(raw)
            continue

        if ent.label_ == "ORG":
            if key in KNOWN_PLACES:
                if key not in loc_seen:
                    loc_seen.add(key)
                    locations.append(raw)
                continue
            if key in TECH_NORMS:
                alias = _skill_alias_key(raw)
                if alias not in skill_seen:
                    skill_seen.add(alias)
                    tech_skills.append(raw)
                continue
            if not _is_good_org(raw):
                continue
            if key not in org_seen:
                org_seen.add(key)
                organizations.append(raw)

    # Gazetteer fallback: known places mentioned in raw text
    for place in sorted(KNOWN_PLACES, key=len, reverse=True):
        if re.search(rf"(?i)(?<!\w){re.escape(place)}(?!\w)", text):
            key = _norm(place)
            if key not in loc_seen:
                loc_seen.add(key)
                locations.append(place.title() if " " in place else place.capitalize())

    # Regex fallback for tech skills missed by the ruler
    for term in TECH_SKILL_TERMS:
        if len(term) < 3:
            continue
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
        if pattern.search(text):
            alias = _skill_alias_key(term)
            if alias not in skill_seen:
                skill_seen.add(alias)
                tech_skills.append(_display_skill(term))

    # Prefer dotted names (Node.js) over bare aliases (Node)
    canonical = {
        "nodejs": "Node.js",
        "expressjs": "Express.js",
        "nextjs": "Next.js",
    }
    final_skills: List[str] = []
    final_seen: Set[str] = set()
    for skill in tech_skills:
        alias = _skill_alias_key(skill)
        if alias in final_seen:
            continue
        final_seen.add(alias)
        final_skills.append(canonical.get(alias, skill))

    # Drop location substrings covered by a longer location (Thane ⊂ Thane Rabodi)
    pruned_locs: List[str] = []
    for loc in locations:
        key = _norm(loc)
        if any(key != _norm(other) and key in _norm(other) for other in locations):
            continue
        pruned_locs.append(loc)

    return {
        "organizations": organizations,
        "locations": pruned_locs,
        "tech_skills": final_skills,
    }


if __name__ == "__main__":
    sample = """
    Omkar Kudav
    Maharashtra, Nerul | Thane Rabodi
    Computer Engineering | Information Technology Graduated | Standard 2022
    Graduate School of Technology | VPMS Polytechnic
    Skills: Python, React, Node.js, Express.js, Machine Learning, TypeScript, Git
    """
    print(extract_entities(sample))
