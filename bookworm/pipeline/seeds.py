"""Seed resolution: DOI / OpenAlex ID / WoS UID / title -> OpenAlex Paper."""

from __future__ import annotations

import re
from pathlib import Path

import httpx

from ..models import Paper
from ..sources.openalex import OpenAlexClient
from ..sources.wos import WosClient
from ..store import normalize_doi

DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/\S+)$", re.I)
OA_RE = re.compile(r"^(?:https?://openalex\.org/)?(W\d+)$", re.I)
WOS_RE = re.compile(r"^WOS:\w+$", re.I)


def classify(s: str) -> tuple[str, str]:
    s = s.strip()
    if m := DOI_RE.match(s):
        return "doi", normalize_doi(m.group(1))
    if m := OA_RE.match(s):
        return "openalex", m.group(1).upper()
    if WOS_RE.match(s):
        return "wos", s.upper()
    return "title", s


def read_seed_file(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def title_from_doi(doi: str, http: httpx.Client | None = None) -> str | None:
    """Get the title via doi.org content negotiation (works for Crossref and DataCite, e.g. arXiv DOIs)."""
    owned = http is None
    http = http or httpx.Client(timeout=20, follow_redirects=True)
    try:
        r = http.get(f"https://doi.org/{doi}", headers={"Accept": "application/vnd.citationstyles.csl+json"})
        if r.status_code != 200:
            return None
        t = r.json().get("title")
        return t[0] if isinstance(t, list) and t else t
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if owned:
            http.close()


def resolve_seeds(
    raw: list[str],
    oa: OpenAlexClient,
    wos: WosClient | None = None,
    http: httpx.Client | None = None,
) -> tuple[list[Paper], list[str]]:
    """Returns (resolved seeds, unresolved inputs)."""
    found: list[Paper] = []
    missing: list[str] = []
    for s in raw:
        kind, val = classify(s)
        p: Paper | None = None
        if kind == "doi":
            p = oa.by_dois([val], origin="seed").get(val)
            if p is None and (title := title_from_doi(val, http)):
                p = oa.by_title(title)
        elif kind == "openalex":
            p = oa.by_id(val, origin="seed")
        elif kind == "wos":
            if wos is not None:
                _, hits = wos.search(f"UT=({val})", limit=1)
                if hits:
                    h = hits[0]
                    p = (oa.by_dois([h.doi], origin="seed").get(h.doi) if h.doi else None) or oa.by_title(h.title)
                    if p is not None:
                        p.wos_uid, p.wos_times_cited = h.wos_uid, h.wos_times_cited
                    else:
                        p = h  # keep the WoS-only record (it will have no citation edges)
        else:
            p = oa.by_title(val)
        if p is None:
            missing.append(s)
            continue
        p.is_seed, p.origin, p.hop = True, "seed", 0
        found.append(p)
    return found, missing
