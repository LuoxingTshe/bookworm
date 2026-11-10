"""Web of Science Starter API client.

Docs: https://developer.clarivate.com/apis/wos-starter
Starter returns search metadata (title, source, authors, DOI, times cited) but no cited-reference data,
so citation edges and abstracts come from OpenAlex.
"""

from __future__ import annotations

import time

import httpx

from ..config import Settings, get_settings
from ..errors import ConfigError
from ..models import Paper
from ..store import normalize_doi
from ..trace import Tracer
from .cache import NO_CACHE, Cache

MAX_LIMIT = 50


class WosError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"WoS API {status}: {message}")
        self.status = status
        self.message = message


class WosQueryError(WosError):
    """400: query syntax error; can be sent back to the LLM for a fix-up."""


class WosBudgetExceeded(RuntimeError):
    pass


class WosClient:
    def __init__(
        self,
        settings: Settings | None = None,
        cache: Cache = NO_CACHE,
        tracer: Tracer | None = None,
        http: httpx.Client | None = None,
    ):
        self.s = settings or get_settings()
        if not self.s.wos_api_key:
            raise ConfigError("WOS_API_KEY is not set; use --source openalex instead, or configure .env")
        self.cache = cache
        self.tracer = tracer or Tracer()
        self.http = http or httpx.Client(timeout=self.s.request_timeout)
        self._last = 0.0
        self.requests_made = 0

    def close(self):
        self.http.close()

    def _get(self, params: dict) -> dict:
        key = Cache.key("wos", params)
        if (hit := self.cache.get(key)) is not None:
            return hit
        if self.requests_made >= self.s.wos_max_requests:
            raise WosBudgetExceeded(f"WoS request budget for this run used up ({self.s.wos_max_requests})")
        for attempt in range(4):
            if self.requests_made >= self.s.wos_max_requests:
                raise WosBudgetExceeded(f"WoS request budget exhausted ({self.s.wos_max_requests})")
            wait = self.s.wos_min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.http.get(
                f"{self.s.wos_base_url}/documents",
                params=params,
                headers={"X-ApiKey": self.s.wos_api_key, "Accept": "application/json"},
            )
            self.requests_made += 1
            self.tracer.usage.wos_requests += 1
            self.tracer.log("wos", params=params, status=resp.status_code)
            if resp.status_code == 429 and attempt < 3:
                try:
                    delay = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                except ValueError:
                    delay = 2 ** (attempt + 1)
                time.sleep(max(0, min(delay, 30)))
                continue
            break
        if resp.status_code == 400:
            raise WosQueryError(400, _error_message(resp))
        if resp.status_code >= 400:
            raise WosError(resp.status_code, _error_message(resp))
        data = resp.json()
        self.cache.set(key, data)
        return data

    def search(self, query: str, *, limit: int = MAX_LIMIT, page: int = 1, sort: str | None = None) -> tuple[int, list[Paper]]:
        params = {"q": query, "db": "WOS", "limit": min(limit, MAX_LIMIT), "page": page}
        if sort:
            params["sortField"] = sort
        data = self._get(params)
        total = int((data.get("metadata") or {}).get("total") or 0)
        return total, [parse_hit(h) for h in data.get("hits") or []]

    def search_all(
        self,
        query: str,
        max_results: int = 100,
        sort: str | None = "TC+D",
        seen: set[str] | None = None,
        max_pages: int | None = None,
    ) -> tuple[int, list[Paper]]:
        """Page through results until max_results. Sorted by times cited, descending, by default.

        With `seen` (keys from paper_keys() of hits collected earlier), already-seen hits are skipped and do not
        count toward max_results, and new hits are added to it; max_pages then bounds the extra paging.
        """
        out: list[Paper] = []
        # WoS offsets are (page - 1) * limit, so the page size must stay fixed while paging
        limit = MAX_LIMIT if seen is not None else min(MAX_LIMIT, max_results)
        total, page = 0, 1
        while len(out) < max_results and (max_pages is None or page <= max_pages):
            total, hits = self.search(query, limit=limit, page=page, sort=sort)
            for h in hits:
                if seen is not None:
                    if seen & (keys := paper_keys(h)):
                        continue
                    seen |= keys
                out.append(h)
            if not hits or page * limit >= total:
                break
            page += 1
        return total, out[:max_results]


def paper_keys(p: Paper) -> set[str]:
    return {k for k in (p.wos_uid, p.doi) if k}


def parse_hit(h: dict) -> Paper:
    src = h.get("source") or {}
    ids = h.get("identifiers") or {}
    authors = [a.get("displayName") or a.get("wosStandard") or "" for a in (h.get("names") or {}).get("authors") or []]
    tc = next((c.get("count") for c in h.get("citations") or [] if c.get("db") == "WOS"), None)
    year = src.get("publishYear")
    uid = h["uid"]
    return Paper(
        id=uid if uid.startswith("WOS:") else f"WOS:{uid}",
        wos_uid=uid,
        doi=normalize_doi(ids.get("doi")),
        title=h.get("title") or "",
        authors=[a for a in authors if a],
        year=int(year) if year else None,
        venue=src.get("sourceTitle"),
        keywords=(h.get("keywords") or {}).get("authorKeywords") or [],
        wos_times_cited=tc,
        origin="wos_search",
    )


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:500]
    if isinstance(body, dict):
        for k in ("message", "error", "detail", "errors"):
            if k in body:
                v = body[k]
                return v if isinstance(v, str) else str(v)[:500]
    return str(body)[:500]
