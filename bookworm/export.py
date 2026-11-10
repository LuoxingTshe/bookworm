"""Export a corpus to BibTeX / RIS / CSV for reference managers (Zotero, EndNote) and spreadsheets."""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from typing import Literal

from .models import Paper

Format = Literal["bibtex", "ris", "csv"]
EXTENSIONS: dict[str, str] = {"bibtex": "bib", "ris": "ris", "csv": "csv"}


def export(papers: list[Paper], fmt: Format) -> str:
    if fmt == "bibtex":
        return to_bibtex(papers)
    if fmt == "ris":
        return to_ris(papers)
    if fmt == "csv":
        return to_csv(papers)
    raise ValueError(f"unknown format {fmt!r}")


_OA_ID = re.compile(r"^W\d+$")


def _note(p: Paper) -> str:
    parts = [f"OpenAlex {p.id}" if _OA_ID.match(p.id) else p.id]
    if p.relevance:
        parts.append(f"relevance {p.relevance.score:g}: {p.relevance.reason}".rstrip(": "))
    if p.is_seed:
        parts.append("seed")
    if p.versions:
        parts.append("versions: " + "; ".join(f"{v.id} {v.doi or ''}".strip() for v in p.versions))
    if p.retracted:
        parts.append("RETRACTED")
    return "; ".join(parts)


def _url(p: Paper) -> str:
    if p.doi:
        return f"https://doi.org/{p.doi}"
    return f"https://openalex.org/{p.id}" if _OA_ID.match(p.id) else ""


# ---- BibTeX ----
_BIB_ESCAPE = str.maketrans({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": "", "}": ""})


def _ascii_word(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())


def _bib_key(p: Paper, used: set[str]) -> str:
    last = _ascii_word((p.authors[0].split() or [""])[-1]) if p.authors else ""
    word = next((w for w in (_ascii_word(t) for t in p.title.split()) if len(w) > 3), "")
    base = f"{last.lower() or 'anon'}{p.year or ''}{word.lower()}" or p.id
    key, i = base, 0
    while key in used:
        i += 1
        key = f"{base}{chr(ord('a') + i - 1)}" if i <= 26 else f"{base}{i}"
    used.add(key)
    return key


def to_bibtex(papers: list[Paper]) -> str:
    used: set[str] = set()
    entries = []
    for p in papers:
        fields = {
            "title": p.title,
            "author": " and ".join(a for a in p.authors if a),
            "year": str(p.year or ""),
            "journal": p.venue or "",
            "doi": p.doi or "",
            "url": _url(p),
            "abstract": p.abstract or "",
            "keywords": ", ".join(p.keywords),
            "note": _note(p),
        }
        body = ",\n".join(f"  {k} = {{{v.translate(_BIB_ESCAPE)}}}" for k, v in fields.items() if v)
        entries.append(f"@{'article' if p.venue else 'misc'}{{{_bib_key(p, used)},\n{body}\n}}")
    return "\n\n".join(entries) + "\n"


# ---- RIS ----
def to_ris(papers: list[Paper]) -> str:
    out = []
    for p in papers:
        lines = [("TY", "JOUR" if p.venue else "GEN"), ("TI", p.title)]
        lines += [("AU", a) for a in p.authors if a]
        lines += [("PY", str(p.year or "")), ("T2", p.venue or ""), ("DO", p.doi or ""), ("UR", _url(p)),
                  ("AB", (p.abstract or "").replace("\n", " "))]
        lines += [("KW", k) for k in p.keywords]
        lines.append(("N1", _note(p)))
        out.append("\n".join([f"{tag}  - {val}" for tag, val in lines if val] + ["ER  - "]))
    return "\n\n".join(out) + "\n"


# ---- CSV ----
CSV_COLUMNS = ["id", "doi", "title", "authors", "year", "venue", "cited_by_count", "wos_times_cited",
               "relevance", "relevance_reason", "relevance_flag", "origin", "hop", "is_seed", "retracted"]


def to_csv(papers: list[Paper]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for p in papers:
        r = p.relevance
        w.writerow([p.id, p.doi or "", p.title, "; ".join(p.authors), p.year or "", p.venue or "",
                    p.cited_by_count if p.cited_by_count is not None else "",
                    p.wos_times_cited if p.wos_times_cited is not None else "",
                    r.score if r else "", r.reason if r else "", (r.flag or "") if r else "",
                    p.origin, p.hop, p.is_seed, p.retracted])
    return buf.getvalue()
