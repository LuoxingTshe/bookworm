"""Relevance screening: BM25 prefilter + batched DeepSeek scoring."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from rank_bm25 import BM25Okapi

from ..llm import prompts as P
from ..llm.deepseek import LLM
from ..models import Paper, Relevance, ResearchIntent
from ..errors import BudgetError, UpstreamError

_TOKEN = re.compile(r"[a-z0-9]+|[一-鿿]")
_STOP = set("a an the of in on for and or to with by from as at is are be this that we our using via based".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def intent_query_tokens(intent: ResearchIntent) -> list[str]:
    parts = [intent.topic, *intent.core_concepts, *intent.keywords_en]
    for syns in intent.synonyms.values():
        parts += syns
    return tokenize(" ".join(parts).replace("*", ""))


def bm25_scores(query_tokens: list[str], papers: list[Paper]) -> list[float]:
    if not papers:
        return []
    docs = [tokenize(p.text) or ["_empty_"] for p in papers]
    return list(BM25Okapi(docs).get_scores(query_tokens))


def prefilter(papers: list[Paper], intent: ResearchIntent, keep: int) -> list[Paper]:
    """Keep the top `keep` papers by BM25; papers without an abstract are only matched on title/keywords, which is fine."""
    if len(papers) <= keep:
        return papers
    scores = bm25_scores(intent_query_tokens(intent), papers)
    ranked = sorted(zip(papers, scores), key=lambda x: -x[1])
    return [p for p, _ in ranked[:keep]]


def _paper_block(p: Paper, abstract_chars: int) -> str:
    abs_ = (p.abstract or "(no abstract)")[:abstract_chars]
    return f"- id: {p.id}\n  title: {p.title}\n  year: {p.year}\n  abstract: {abs_}"


def llm_screen(
    papers: list[Paper],
    intent: ResearchIntent,
    llm: LLM,
    seeds: list[Paper] | None = None,
    batch_size: int = 15,
    abstract_chars: int = 900,
    workers: int = 4,
) -> dict[str, Relevance]:
    """Returns id -> Relevance. Papers missing from the LLM output get one retry; if still missing, they get no score."""
    if not papers:
        return {}
    system = P.fill(P.SCREEN_SYSTEM, lang=intent.language)
    seed_txt = "\n".join(f"- {s.title} ({s.year})" for s in seeds or []) or "(none)"
    intent_txt = intent.model_dump_json(include={"topic", "research_questions", "core_concepts", "exclude"})

    def run(batch: list[Paper]) -> dict[str, Relevance]:
        user = P.fill(
            P.SCREEN_USER, intent=intent_txt, seeds=seed_txt,
            papers="\n".join(_paper_block(p, abstract_chars) for p in batch),
        )
        try:
            data = llm.chat_json(system, user, purpose="screen")
        except BudgetError:
            raise
        except Exception as exc:
            if hasattr(llm, "tracer"):
                llm.tracer.warn(f"Screening batch failed: {type(exc).__name__}")
            return {}
        by_id = {p.id: p for p in batch}
        valid = set(by_id)
        out: dict[str, Relevance] = {}
        for r in data.get("results") or []:
            try:
                pid = str(r["id"])
                if pid in valid:
                    flag = "no_abstract" if not by_id[pid].abstract else (str(r["flag"]) if r.get("flag") else None)
                    out[pid] = Relevance(
                        score=max(0.0, min(10.0, float(r["score"]))), reason=str(r.get("reason", "")), flag=flag
                    )
            except (KeyError, TypeError, ValueError):
                continue
        return out

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    results: dict[str, Relevance] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(run, batches):
            results.update(part)
    missing = [p for p in papers if p.id not in results]
    if missing:
        for i in range(0, len(missing), batch_size):
            results.update(run(missing[i : i + batch_size]))
    if not results:
        raise UpstreamError("Screening produced no valid scores; candidates are preserved in the checkpoint")
    if len(results) < len(papers) and hasattr(llm, "tracer"):
        llm.tracer.warn(f"Screening incomplete: {len(papers) - len(results)} candidates have no score")
    return results
