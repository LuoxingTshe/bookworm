"""Data models: the single source of truth for the corpus.json schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Origin = Literal["seed", "wos_search", "openalex_search", "backward", "forward", "cocited", "agent"]
RelationLabel = Literal["extends", "uses_method", "uses_data", "compares", "critiques", "background"]


class ResearchIntent(BaseModel):
    topic: str
    research_questions: list[str] = Field(default_factory=list)
    core_concepts: list[str] = Field(default_factory=list)
    keywords_en: list[str] = Field(default_factory=list)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    exclude: list[str] = Field(default_factory=list)
    year_range: tuple[int | None, int | None] = (None, None)
    doc_types: list[str] = Field(default_factory=list)
    language: str = "en"


class Relevance(BaseModel):
    score: float
    reason: str = ""
    flag: str | None = None  # e.g. "no_abstract": scored from the title only


class PaperRecord(BaseModel):
    id: str  # OpenAlex W-id; when unmatched, "WOS:<uid>"
    doi: str | None = None
    wos_uid: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    cited_by_count: int | None = None
    wos_times_cited: int | None = None
    is_seed: bool = False
    retracted: bool = False  # OpenAlex is_retracted
    origin: Origin = "wos_search"
    hop: int = 0
    relevance: Relevance | None = None
    referenced_works: list[str] = Field(default_factory=list)
    external_refs_count: int = 0
    external_cited_by: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract or ''} {' '.join(self.keywords)}"


class Paper(PaperRecord):
    # Original records, including the representative, before version consolidation.
    # Flat snapshots preserve DOI, year, references, scores and provenance without recursion.
    versions: list[PaperRecord] = Field(default_factory=list)


class EdgeRelation(BaseModel):
    label: RelationLabel
    rationale: str = ""


class Edge(BaseModel):
    source: str  # citing paper
    target: str  # cited paper
    type: Literal["cites"] = "cites"
    provenance: str = "openalex"
    relation: EdgeRelation | None = None
    record_pairs: list[tuple[str, str]] = Field(default_factory=list)  # original citing/cited version ids


class Usage(BaseModel):
    deepseek_prompt_tokens: int = 0
    deepseek_completion_tokens: int = 0
    deepseek_calls: int = 0
    wos_requests: int = 0
    openalex_requests: int = 0


class HopStat(BaseModel):
    hop: int
    candidates: int  # new unique records found in this hop
    screened: int  # sent to the LLM after the BM25 prefilter
    relevant: int  # scored >= threshold


class Prisma(BaseModel):
    """PRISMA-style flow of one build. Counts are unique records after dedup, except search_duplicates."""

    identified: dict[str, int] = Field(default_factory=dict)  # by origin
    search_duplicates: int = 0  # search hits that were already in the pool
    not_screened: int = 0  # cut by the BM25 prefilter (or the LLM returned no score)
    screened: int = 0
    excluded_low_relevance: int = 0
    excluded_retracted: int = 0
    excluded_over_cap: int = 0
    excluded_out_of_year: int = 0
    included: int = 0
    hops: list[HopStat] = Field(default_factory=list)
    stop_reason: str | None = None
    agent_added: int = 0


class Meta(BaseModel):
    version: str = "1"
    complete: bool = True
    warnings: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    prompt: str = ""
    seeds: list[str] = Field(default_factory=list)
    intent: ResearchIntent | None = None
    queries: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    summary: str | None = None
    prisma: Prisma | None = None
    deduplication: dict = Field(default_factory=dict)


class Corpus(BaseModel):
    meta: Meta = Field(default_factory=Meta)
    papers: list[Paper] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
