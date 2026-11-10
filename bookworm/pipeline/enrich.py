"""WoS hits -> OpenAlex records (abstract, referenced_works, cited_by_count), keeping the WoS fields."""

from __future__ import annotations

from ..models import Paper
from ..sources.openalex import OpenAlexClient


def enrich_wos_hits(hits: list[Paper], oa: OpenAlexClient, max_title_lookups: int = 20) -> list[Paper]:
    by_doi = oa.by_dois([h.doi for h in hits if h.doi], origin=hits[0].origin if hits else "wos_search")
    out: list[Paper] = []
    title_budget = max_title_lookups
    for h in hits:
        p = by_doi.get(h.doi) if h.doi else None
        if p is None and title_budget > 0 and h.title:
            title_budget -= 1
            p = oa.by_title(h.title, origin=h.origin)
        if p is None:
            out.append(h)  # no OpenAlex match: keep the WoS record (it will have no citation edges)
            continue
        p.origin, p.hop = h.origin, h.hop
        p.wos_uid, p.wos_times_cited = h.wos_uid, h.wos_times_cited
        p.year = p.year if p.year and (not h.year or abs(p.year - h.year) <= 1) else h.year
        if h.venue:
            p.venue = h.venue
        if h.keywords:
            p.keywords = list(dict.fromkeys(h.keywords + p.keywords))
        out.append(p)
    return out
