"""Extract plain text from uploaded resume files."""
from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """Parse resume bytes into UTF-8 text based on file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".txt":
        return _from_txt(data)
    if suffix == ".pdf":
        return _from_pdf(data)
    return _from_docx(data)


def _from_txt(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="ignore")
    text = text.strip()
    if not text:
        raise ValueError("The text file is empty.")
    return text


def _from_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to read PDF resumes: pip install pypdf") from exc

    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("Could not extract text from this PDF (it may be scanned/image-only).")
    return text


def _from_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "Install python-docx to read Word resumes: pip install python-docx"
        ) from exc

    from io import BytesIO

    document = Document(BytesIO(data))
    parts = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("The Word document has no readable paragraph text.")
    return text
