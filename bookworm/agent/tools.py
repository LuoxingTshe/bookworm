"""Agent tools: JSON-schema definitions + implementations. Implementations reuse sources/* and pipeline/*; the agent only orchestrates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.deepseek import LLM
from ..dedup import deduplicate, find_duplicate
from ..models import Paper, ResearchIntent
from ..pipeline import enrich, gaps, screen
from ..sources.openalex import OpenAlexClient
from ..sources.wos import WosBudgetExceeded, WosClient, WosError
from ..store import CorpusStore
from ..trace import Tracer
from ..errors import BudgetError


@dataclass
class AgentContext:
    corpus: CorpusStore  # the corpus being built
    llm: LLM
    oa: OpenAlexClient
    wos: WosClient | None
    intent: ResearchIntent
    tracer: Tracer
    threshold: float = 6.0
    max_papers: int | None = None  # hard cap enforced by add_to_corpus
    seen: CorpusStore = field(default_factory=CorpusStore)  # every paper seen via a tool (candidate cache)
    finished: bool = False
    summary: str = ""
    n_added: int = 0
    stop_reason: str | None = None
    checkpoint: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        deduplicate(self.corpus, self.tracer)

    def remember(self, papers: list[Paper]) -> list[Paper]:
        out = []
        for p in papers:
            if (c := self.corpus.get(p.id)) is not None:
                out.append(c)
            else:
                out.append(self.seen.add(p))
        return out

    def lookup(self, pid: str) -> Paper | None:
        return self.corpus.get(pid) or self.seen.get(pid)


def _brief(ctx: AgentContext, p: Paper, abstract_chars: int = 0) -> dict:
    d: dict[str, Any] = {
        "id": p.id, "title": p.title, "year": p.year,
        "cited": p.cited_by_count if p.cited_by_count is not None else p.wos_times_cited,
        "in_corpus": p.id in ctx.corpus,
    }
    if p.relevance:
        d["score"] = p.relevance.score
    if p.retracted:
        d["retracted"] = True
    if abstract_chars and p.abstract:
        d["abstract"] = p.abstract[:abstract_chars]
    return d


def _fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_IDS = {"type": "array", "items": {"type": "string"}, "description": "OpenAlex ids, e.g. W2741809807"}
_LIMIT = {"type": "integer", "description": "max results (default 20, max 50)"}

TOOLS = [
    _fn("search_wos", "Search Web of Science with an advanced query (TS=, TI=, PY=, DT=; AND/OR/NOT; quotes; *). "
        "Results are enriched with OpenAlex ids. Consumes WoS quota.",
        {"query": {"type": "string"}, "limit": _LIMIT}, ["query"]),
    _fn("search_openalex", "Full-text relevance search on OpenAlex (supports AND/OR and quoted phrases). Free.",
        {"query": {"type": "string"}, "year_from": {"type": "integer"}, "year_to": {"type": "integer"}, "limit": _LIMIT},
        ["query"]),
    _fn("get_paper", "Get details (abstract, venue, counts) of a paper by OpenAlex id.", {"id": {"type": "string"}}, ["id"]),
    _fn("get_references", "Backward snowballing: papers cited BY the given paper (most-cited first).",
        {"id": {"type": "string"}, "limit": _LIMIT}, ["id"]),
    _fn("get_citing", "Forward snowballing: papers that CITE the given paper (most-cited first).",
        {"id": {"type": "string"}, "limit": _LIMIT}, ["id"]),
    _fn("screen", "Score relevance (0-10) of candidate papers against the research intent with an LLM judge. "
        "Call before add_to_corpus.", {"ids": _IDS}, ["ids"]),
    _fn("top_missing_refs", "Gap finder: works referenced by many corpus papers but not in the corpus (most co-cited "
        "first). Screen the promising ones, then add them.",
        {"min_count": {"type": "integer", "description": "min corpus papers citing it (default 3)"}, "limit": _LIMIT}, []),
    _fn("add_to_corpus", "Add screened papers (by id) to the corpus. Refuses retracted papers and anything beyond the "
        "hard size cap.", {"ids": _IDS, "note": {"type": "string"}}, ["ids"]),
    _fn("remove_from_corpus", "Remove papers from the corpus.", {"ids": _IDS}, ["ids"]),
    _fn("corpus_status", "Corpus size, score/year distribution, seeds, and the most recent additions.", {}, []),
    _fn("finish", "Finish the session with a brief summary of the corpus and remaining gaps.",
        {"summary": {"type": "string"}}, ["summary"]),
]


def _limit(args: dict, default: int = 20) -> int:
    return max(1, min(50, int(args.get("limit") or default)))


def t_search_wos(ctx: AgentContext, args: dict) -> Any:
    if ctx.wos is None:
        return {"error": "WoS is not configured; use search_openalex."}
    try:
        total, hits = ctx.wos.search(args["query"], limit=_limit(args))
    except WosBudgetExceeded as e:
        return {"error": str(e) + "; use search_openalex from now on."}
    except WosError as e:
        return {"error": e.message, "hint": "fix the query syntax and retry"}
    papers = ctx.remember(enrich.enrich_wos_hits(hits, ctx.oa, max_title_lookups=5))
    ctx.corpus.corpus.meta.queries.append(args["query"])
    return {"total": total, "results": [_brief(ctx, p) for p in papers]}


def t_search_openalex(ctx: AgentContext, args: dict) -> Any:
    years = (args.get("year_from", ctx.intent.year_range[0]), args.get("year_to", ctx.intent.year_range[1]))
    papers = ctx.remember(ctx.oa.search(args["query"], years=years, limit=_limit(args)))
    ctx.corpus.corpus.meta.queries.append(args["query"])
    return {"results": [_brief(ctx, p) for p in papers]}


def t_get_paper(ctx: AgentContext, args: dict) -> Any:
    p = _resolve(ctx, args["id"])
    if p is None:
        return {"error": "not found"}
    d = _brief(ctx, p, abstract_chars=1200)
    d.update(venue=p.venue, authors=p.authors[:6], n_references=len(p.referenced_works), doi=p.doi)
    return d


def _resolve(ctx: AgentContext, pid: str) -> Paper | None:
    p = ctx.lookup(pid)
    if p is None and (fetched := ctx.oa.by_id(pid)):
        p = ctx.remember([fetched])[0]
    return p


def t_get_references(ctx: AgentContext, args: dict) -> Any:
    p = _resolve(ctx, args["id"])
    if p is None:
        return {"error": "not found"}
    refs = ctx.remember(ctx.oa.references(p, limit=_limit(args)))
    return {"of": p.id, "results": [_brief(ctx, r) for r in refs]}


def t_get_citing(ctx: AgentContext, args: dict) -> Any:
    cites = ctx.remember(ctx.oa.citing(args["id"], limit=_limit(args)))
    return {"of": args["id"], "results": [_brief(ctx, c) for c in cites]}


def t_screen(ctx: AgentContext, args: dict) -> Any:
    papers = [p for pid in args["ids"][:60] if (p := _resolve(ctx, pid)) is not None]
    seeds = [p for p in ctx.corpus.papers if p.is_seed]
    scores = screen.llm_screen(papers, ctx.intent, ctx.llm, seeds)
    for p in papers:
        if p.id in scores and not p.is_seed:
            p.relevance = scores[p.id]
    return {"results": [{"id": p.id, "score": scores[p.id].score, "reason": scores[p.id].reason}
                        for p in papers if p.id in scores],
            "threshold": ctx.threshold}


def t_top_missing(ctx: AgentContext, args: dict) -> Any:
    found = gaps.top_missing_refs(ctx.corpus.papers, ctx.corpus, int(args.get("min_count") or 3), _limit(args))
    papers = {p.id: p for p in ctx.remember(ctx.oa.by_ids([rid for rid, _ in found], origin="cocited"))}
    return {"results": [{**_brief(ctx, papers[rid]), "cited_by_corpus": n} for rid, n in found if rid in papers]}


def t_add(ctx: AgentContext, args: dict) -> Any:
    added, low, missing, retracted, over_cap = [], [], [], [], []
    out_of_year = []
    for pid in args["ids"]:
        p = _resolve(ctx, pid)
        if p is None:
            missing.append(pid)
            continue
        start, end = ctx.intent.year_range
        if p.year is not None and ((start is not None and p.year < start) or (end is not None and p.year > end)):
            out_of_year.append(pid)
            continue
        if p.relevance is None or p.relevance.score < ctx.threshold:
            low.append(pid)
            continue
        if p.retracted:
            retracted.append(pid)
            continue
        duplicate = find_duplicate(ctx.corpus, p)
        if duplicate is None and ctx.max_papers is not None and len(ctx.corpus) >= ctx.max_papers:
            over_cap.append(pid)
            continue
        if args.get("note"):
            p.notes.append(args["note"])
        if duplicate is None:
            ctx.n_added += 1
        canonical = ctx.corpus.add(p.model_copy(deep=True))
        if duplicate is not None:
            deduplicate(ctx.corpus, ctx.tracer)
            canonical = ctx.corpus.get(p.id)
        added.append(canonical.id)
    res: dict[str, Any] = {"added": added, "corpus_size": len(ctx.corpus)}
    if out_of_year:
        res["rejected_out_of_year"] = out_of_year
    if low:
        res["rejected_unscreened_or_low_score"] = low
    if retracted:
        res["rejected_retracted"] = retracted
    if over_cap:
        res["rejected_corpus_full"] = over_cap
        res["hint"] = (f"the corpus is at its hard cap of {ctx.max_papers}; remove weaker papers first "
                       "or call finish")
    if missing:
        res["not_found"] = missing
    return res


def t_remove(ctx: AgentContext, args: dict) -> Any:
    removed = [pid for pid in args["ids"] if not (ctx.corpus.get(pid) and ctx.corpus.get(pid).is_seed) and ctx.corpus.remove(pid)]
    return {"removed": removed, "corpus_size": len(ctx.corpus)}


def t_status(ctx: AgentContext, args: dict) -> Any:
    ps = ctx.corpus.papers
    years = sorted(p.year for p in ps if p.year)
    buckets: dict[str, int] = {}
    for p in ps:
        s = p.relevance.score if p.relevance else None
        k = "unscored" if s is None else ("9-10" if s >= 9 else "7-8" if s >= 7 else "<7")
        buckets[k] = buckets.get(k, 0) + 1
    return {
        "size": len(ps),
        "max_papers": ctx.max_papers,
        "seeds": [_brief(ctx, p) for p in ps if p.is_seed],
        "score_buckets": buckets,
        "year_range": [years[0], years[-1]] if years else None,
        "recent_additions": [_brief(ctx, p) for p in ps[-10:]],
        "candidates_seen": len(ctx.seen),
    }


def t_finish(ctx: AgentContext, args: dict) -> Any:
    ctx.finished, ctx.summary = True, str(args.get("summary", ""))
    return {"ok": True}


HANDLERS: dict[str, Callable[[AgentContext, dict], Any]] = {
    "search_wos": t_search_wos, "search_openalex": t_search_openalex, "get_paper": t_get_paper,
    "get_references": t_get_references, "get_citing": t_get_citing, "screen": t_screen,
    "top_missing_refs": t_top_missing, "add_to_corpus": t_add, "remove_from_corpus": t_remove, "corpus_status": t_status, "finish": t_finish,
}


def dispatch(ctx: AgentContext, name: str, raw_args: str) -> str:
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        handler = HANDLERS.get(name)
        result = {"error": f"unknown tool {name}"} if handler is None else handler(ctx, args)
    except BudgetError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        result = {"error": f"bad arguments: {e}"}
    except Exception as e:  # noqa: BLE001 - surface tool errors to the model instead of killing the run
        result = {"error": f"{type(e).__name__}: upstream operation failed"}
        ctx.tracer.warn(f"Tool {name} failed: {type(e).__name__}")
    ctx.tracer.log("tool", name=name, args=raw_args, result=str(result)[:500])
    if ctx.checkpoint is not None:
        ctx.checkpoint()
    return json.dumps(result, ensure_ascii=False)
