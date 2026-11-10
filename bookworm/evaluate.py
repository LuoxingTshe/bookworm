"""Recall evaluation against a gold set: the reference lists of one or more published surveys on the same topic.

A corpus that covers the topic well should contain most of what an expert survey cites. Only OpenAlex ids are
compared, so WoS-only records (id "WOS:...") can never match.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Corpus, Paper

DEFAULT_KS = (20, 50, 100, 200)


class EvalReport(BaseModel):
    gold_sources: list[str]
    gold_size: int
    corpus_size: int
    found: int
    recall: float
    recall_at: dict[int, float] = Field(default_factory=dict)  # recall within the top-k corpus papers by rank
    gold_share_of_corpus: float  # share of the corpus that is in the gold set (a loose precision proxy)
    missed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def gold_from_surveys(surveys: list[Paper]) -> set[str]:
    return {r for s in surveys for r in s.referenced_works} - {s.id for s in surveys}


def recall_report(corpus: Corpus, surveys: list[Paper], ks: tuple[int, ...] = DEFAULT_KS) -> EvalReport:
    """corpus.papers must be in rank order (CorpusStore.finalize puts seeds first, then by relevance)."""
    aliases = {v.id: p.id for p in corpus.papers for v in p.versions}
    canonical = lambda pid: aliases.get(pid, pid)
    gold = {canonical(pid) for pid in gold_from_surveys(surveys)}
    survey_ids = {canonical(s.id) for s in surveys}
    gold -= survey_ids
    ranked = [p.id for p in corpus.papers if p.id not in survey_ids]
    found = [pid for pid in ranked if pid in gold]
    warnings = [
        f"gold survey {p.id} is a seed of this corpus: its reference list was snowballed directly, recall is inflated"
        for p in corpus.papers if p.id in survey_ids and p.is_seed
    ]
    if not gold:
        warnings.append("gold set is empty (the survey has no references in OpenAlex)")
    n = len(gold) or 1
    return EvalReport(
        gold_sources=sorted(survey_ids),
        gold_size=len(gold),
        corpus_size=len(ranked),
        found=len(found),
        recall=len(found) / n,
        recall_at={k: sum(pid in gold for pid in ranked[:k]) / n for k in ks if k < len(ranked)},
        gold_share_of_corpus=len(found) / (len(ranked) or 1),
        missed=sorted(gold - set(found)),
        warnings=warnings,
    )
