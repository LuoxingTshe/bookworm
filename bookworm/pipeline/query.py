"""Research intent -> search queries (WoS advanced query / OpenAlex search) and execution."""

from __future__ import annotations

import json
import math
import re

from ..errors import BudgetError, InputError, UpstreamError
from ..llm import prompts as P
from ..llm.deepseek import LLM
from ..models import Paper, ResearchIntent
from ..sources.openalex import OpenAlexClient
from ..sources.wos import MAX_LIMIT, WosBudgetExceeded, WosClient, WosQueryError
from ..trace import NULL_TRACER, Tracer


# Stable, separately auditable recall branches for garden-heritage searches.
WOS_BRANCHES: dict[str, str] = {
    "vegetation_tls": (
        'TS=("historic garden*" OR "heritage garden*" OR "classical garden*") '
        'AND TS=(tree OR vegetation OR canopy OR "tree structure" OR "carbon storage" OR "ancient tree*") '
        'AND TS=(TLS OR "terrestrial laser scanning" OR LiDAR OR "point cloud*")'
    ),
    "garden_syntax": (
        'TS=("Chinese garden*" OR "historic garden*" OR "heritage garden*" OR "classical garden*") '
        'AND TS=("space syntax" OR wayfinding OR "path network*" OR "spatial configuration" OR "graph theory" OR accessibility)'
    ),
    "garden_reviews": (
        'TS=("garden heritage" OR "heritage garden*" OR "historic garden*" OR "classical garden*") '
        'AND TS=(review OR bibliometric OR "systematic review" OR "scoping review" OR "literature review")'
    ),
    "heritage_pointcloud": (
        'TS=("cultural heritage" OR "built heritage" OR "architectural heritage" OR "heritage site*") '
        'AND TS=("point cloud*" OR LiDAR OR "laser scanning" OR photogrammetry) '
        'AND TS=(segmentation OR classification OR "feature extraction" OR "digital twin" OR HBIM OR "3D documentation")'
    ),
}


def build_wos_queries(intent: ResearchIntent, llm: LLM, n: int = 3) -> list[str]:
    data = llm.chat_json(
        P.fill(P.QUERY_SYSTEM, n=n),
        P.fill(P.QUERY_USER, intent=intent.model_dump_json(indent=1)),
        purpose="wos_query",
    )
    qs = [q.strip() for q in data.get("queries") or [] if isinstance(q, str) and q.strip()]
    qs = [ensure_year_clause(strip_acronym_truncation(q), intent.year_range) for q in qs[:n]]
    return qs or [fallback_wos_query(intent)]


def build_wos_branch_queries(branches: list[str], years: tuple[int | None, int | None]) -> list[str]:
    """Build deterministic, independently named recall branches and apply the requested years to each."""
    unknown = sorted(set(branches) - set(WOS_BRANCHES))
    if unknown:
        raise InputError(f"unknown WoS query branch(es): {', '.join(unknown)}")
    return [ensure_year_clause(WOS_BRANCHES[name], years) for name in dict.fromkeys(branches)]


def ensure_year_clause(q: str, years: tuple[int | None, int | None]) -> str:
    y0, y1 = years
    if (y0 is None and y1 is None) or re.search(r"\bPY\s*=", q, re.I):
        return q
    return f"({q}) AND PY=({y0 or 1900}-{y1 or 2100})"


_ACRONYM_TRUNC = re.compile(r'"([A-Z][A-Z0-9-]{1,5})\*"')


def strip_acronym_truncation(q: str) -> str:
    """Drop * after short all-caps acronyms: in WoS "GAT*" also matches gate/gather (947 vs 52 hits in a live test)."""
    return _ACRONYM_TRUNC.sub(r'"\1"', q)


def fallback_wos_query(intent: ResearchIntent) -> str:
    """Deterministic fallback when the LLM returns nothing usable: AND across concept facets, OR within each facet's synonyms."""
    facets = []
    for c in intent.core_concepts or intent.keywords_en[:2]:
        terms = [c, *intent.synonyms.get(c, [])][:4]
        facets.append("(" + " OR ".join(f'"{t}"' for t in terms) + ")")
    return ensure_year_clause("TS=(" + " AND ".join(facets) + ")", intent.year_range)


def fix_wos_query(query: str, error: str, llm: LLM) -> str | None:
    data = llm.chat_json(
        P.fill(P.QUERY_SYSTEM, n=1), P.fill(P.QUERY_FIX_USER, query=query, error=error), purpose="wos_query_fix"
    )
    q = data.get("query")
    return strip_acronym_truncation(q.strip()) if isinstance(q, str) and q.strip() else None


def run_wos_queries(
    queries: list[str], wos: WosClient, llm: LLM, per_query: int, tracer: Tracer = NULL_TRACER
) -> tuple[list[str], list[Paper], list[str]]:
    """Returns (queries actually executed, hits, warnings). Each query gets one LLM fix-up attempt on a syntax error.

    per_query counts new papers only: queries overlap heavily when all are sorted by times cited, so later queries
    page past hits that earlier ones already returned (up to 3x per_query hits read per query).
    """
    executed: list[str] = []
    hits: list[Paper] = []
    warnings: list[str] = []
    seen: set[str] = set()
    max_pages = math.ceil(3 * per_query / MAX_LIMIT)
    for q in queries:
        for attempt in range(2):
            try:
                total, res = wos.search_all(q, max_results=per_query, seen=seen, max_pages=max_pages)
                executed.append(q)
                hits.extend(res)
                tracer.log("wos_query", query=q, total=total, got=len(res))
                break
            except WosQueryError as e:
                if attempt == 1 or not (fixed := fix_wos_query(q, e.message, llm)):
                    warnings.append(f"query failed: {q} ({e.message})")
                    break
                warnings.append(f"query rewritten: {q} -> {fixed}")
                q = fixed
            except WosBudgetExceeded as e:
                warnings.append(str(e))
                return executed, hits, warnings
    return executed, hits, warnings


def build_openalex_queries(intent: ResearchIntent) -> list[str]:
    """OpenAlex search supports AND/OR/quoted phrases; build one precise query plus one natural-language topic query."""
    facets = []
    for c in intent.core_concepts[:3]:
        terms = [c, *[s.rstrip("*") for s in intent.synonyms.get(c, [])]][:3]
        facets.append("(" + " OR ".join(f'"{t}"' for t in terms) + ")")
    qs = [" AND ".join(facets)] if facets else []
    qs.append(intent.topic)
    return qs


def run_openalex_queries(
    queries: list[str], oa: OpenAlexClient, intent: ResearchIntent, per_query: int
) -> tuple[list[str], list[Paper], list[str]]:
    executed, hits, warnings = [], [], []
    for q in queries:
        try:
            hits += oa.search(q, years=intent.year_range, limit=per_query)
            executed.append(q)
        except BudgetError:
            raise
        except Exception as e:
            warnings.append(f"OpenAlex query failed: {q} ({type(e).__name__})")
    if not executed:
        raise UpstreamError("All OpenAlex queries failed; check network and API configuration")
    return executed, hits, warnings


def dumps_intent(intent: ResearchIntent) -> str:
    return json.dumps(intent.model_dump(), ensure_ascii=False, indent=2)
