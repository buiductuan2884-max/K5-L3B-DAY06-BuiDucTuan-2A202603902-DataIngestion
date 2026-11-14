"""Read and normalize source data (PyMuPDF text blocks ~ paragraphs)."""
import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from .models import DocumentPage

# Running headers/footers repeated on every page; not content.
_FOOTER = re.compile(
    r"^(\d+\s+)?Chapter \d+ \| .+?(\s+\d+)?$|^This OpenStax book is available for free|^Access for free at",
    re.I,
)
SECTION_HEADING = re.compile(r"^(\d+\.\d+) \| (.+)$")


def normalize_text(text: str) -> str:
    """NFKC, re-join hyphenated line breaks, collapse whitespace inside a paragraph."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: Path, document_id: str, chapters: list[dict], log: list[dict]) -> list[DocumentPage]:
    """Extract counted pages; paragraphs joined by a blank line, footers removed.

    chapters: [{"chapter": 3, "title": "...", "pages": [55, 84]}]  (1-based PDF pages, inclusive)
    """
    doc = fitz.open(path)
    pages: list[DocumentPage] = []
    for ch in chapters:
        first, last = ch["pages"]
        section = None
        for pno in range(first, last + 1):
            try:
                blocks = doc[pno - 1].get_text("blocks")
            except Exception as exc:  # keep going, but record the failure
                log.append({"document_id": document_id, "page": pno, "error": repr(exc)})
                continue
            paras = []
            for b in blocks:
                if b[6] != 0:  # image block
                    continue
                para = normalize_text(b[4])
                if not para or _FOOTER.match(para):
                    continue
                paras.append(para)
            text = "\n\n".join(paras)
            m = next((SECTION_HEADING.match(p) for p in paras if SECTION_HEADING.match(p)), None)
            section_start = section
            if m:
                section = m.group(1)
            if len(text) < 500:
                log.append({"document_id": document_id, "page": pno, "warning": f"short page ({len(text)} chars)"})
            pages.append(DocumentPage(
                document_id=document_id,
                source=str(path.name),
                page=pno,
                text=text,
                metadata={
                    "chapter": ch["chapter"],
                    "chapter_title": ch["title"],
                    "section_at_start": section_start or "intro",
                },
            ))
    return pages
