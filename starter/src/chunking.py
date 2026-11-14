"""Fixed-size (A) and paragraph-based (B) chunking; same metadata fields for both."""
import hashlib
import re

from .ingestion import SECTION_HEADING
from .models import Chunk, DocumentPage


def _make(strategy: str, index: int, page: DocumentPage, page_end: int, text: str, section: str, **extra) -> Chunk:
    key = f"{strategy}|{page.document_id}|{page.page}|{index}|{text}"
    meta = {
        "document_id": page.document_id, "source": page.source, "page": page.page,
        "page_end": page_end, "chapter": page.metadata["chapter"],
        "chapter_title": page.metadata["chapter_title"], "section": section,
        "strategy": strategy, **extra,
    }
    return Chunk(hashlib.sha1(key.encode()).hexdigest()[:16], page.document_id, page.source,
                 page.page, text, meta)


def fixed_size(pages: list[DocumentPage], size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Character windows per page (never cross pages); section tracked across pages of a chapter."""
    chunks, section, chapter = [], None, None
    for page in pages:
        if page.metadata["chapter"] != chapter:
            chapter = page.metadata["chapter"]
            section = f"{chapter}.intro"
        # (offset in collapsed text, section) for each heading on this page
        heads, offset = [], 0
        paras = page.text.split("\n\n")
        for p in paras:
            m = SECTION_HEADING.match(p)
            if m:
                heads.append((offset, m.group(1)))
            offset += len(p) + 1
        text = " ".join(paras)
        for start in range(0, max(len(text), 1), size - overlap):
            piece = text[start:start + size].strip()
            if not piece:
                continue
            sec = section
            for off, s in heads:
                if off <= start:
                    sec = s
            chunks.append(_make("A", len(chunks), page, page.page, piece, sec, char_start=start))
            if start + size >= len(text):
                break
        if heads:
            section = heads[-1][1]
    return chunks


def _split_long(para: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph at '. ' boundaries; hard-cut a sentence only if it alone exceeds max_chars."""
    out, cur = [], ""
    for sent in re.split(r"(?<=\. )", para):
        while len(sent) > max_chars:
            if cur:
                out.append(cur.strip())
                cur = ""
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        if len(cur) + len(sent) > max_chars:
            out.append(cur.strip())
            cur = ""
        cur += sent
    if cur.strip():
        out.append(cur.strip())
    return out


def by_paragraph(pages: list[DocumentPage], max_chars: int = 1000) -> list[Chunk]:
    """Pack paragraphs up to max_chars; a section heading starts a new chunk; chunks may span pages."""
    chunks = []
    by_page = {(p.metadata["chapter"], p.page): p for p in pages}
    for chapter in dict.fromkeys(p.metadata["chapter"] for p in pages):
        section = f"{chapter}.intro"
        buf, buf_pages, buf_sec = [], [], section

        def flush():
            if buf:
                chunks.append(_make("B", len(chunks), by_page[(chapter, buf_pages[0])], buf_pages[-1],
                                    "\n\n".join(buf), buf_sec))
                buf.clear()
                buf_pages.clear()

        for page in (p for p in pages if p.metadata["chapter"] == chapter):
            for para in page.text.split("\n\n"):
                m = SECTION_HEADING.match(para)
                if m:
                    flush()
                    section = m.group(1)
                for piece in (_split_long(para, max_chars) if len(para) > max_chars else [para]):
                    if buf and len("\n\n".join(buf + [piece])) > max_chars:
                        flush()
                    if not buf:
                        buf_sec = section
                    buf.append(piece)
                    buf_pages.append(page.page)
        flush()
    return chunks


if __name__ == "__main__":
    import statistics

    from .run import CHAPTERS, DOC_ID, PDF
    from .ingestion import load_pdf

    pages = load_pdf(PDF, DOC_ID, CHAPTERS, [])
    for name, fn in (("A", fixed_size), ("B", by_paragraph)):
        a, b = fn(pages), fn(pages)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b], "unstable ids"
        assert len({c.chunk_id for c in a}) == len(a), "duplicate ids"
        for c in a:
            assert c.text.strip() and c.page <= c.metadata["page_end"] and c.metadata["section"], c
        lens = [len(c.text) for c in a]
        print(name, len(a), "chunks; chars min/median/mean/max =", min(lens), statistics.median(lens),
              round(statistics.mean(lens)), max(lens), "; max ok:", name == "A" or max(lens) <= 1000)
