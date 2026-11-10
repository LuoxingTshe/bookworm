"""Test doubles: a FakeLLM that answers deterministically by purpose, and a FakeOA backed by a small in-memory citation universe."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from bookworm.config import Settings
from bookworm.models import Corpus, Edge, Paper, Relevance
from bookworm.sources.openalex import title_similarity


class FakeLLM:
    def __init__(self, tool_script: list | None = None):
        self.calls: list[str] = []
        self.tool_script = list(tool_script or [])

    def chat_json(self, system: str, user: str, *, purpose: str = "") -> dict:
        self.calls.append(purpose)
        if purpose == "intent":
            return {
                "topic": "Graph neural networks for drug discovery",
                "research_questions": ["How are GNNs used for molecular property prediction?"],
                "core_concepts": ["graph neural network", "drug discovery"],
                "keywords_en": ["graph neural network", "molecular property prediction", "drug discovery"],
                "synonyms": {"graph neural network": ["GNN", "message passing neural network"]},
                "exclude": [],
                "year_range": [None, None],
                "doc_types": [],
                "language": "zh",
            }
        if purpose == "wos_query":
            return {"queries": ['TS=("graph neural network*" AND drug*)', "TS=(GNN AND molecul*)"]}
        if purpose == "wos_query_fix":
            return {"query": 'TS=("graph neural network*")'}
        if purpose == "screen":
            out = []
            for pid, title in re.findall(r"- id: (\S+)\n  title: (.*)", user):
                score = 8 if re.search(r"graph|molecul", title, re.I) else 2
                out.append({"id": pid, "score": score, "reason": "fake"})
            return {"results": out}
        if purpose == "relation":
            n = len(re.findall(r"\[pair \d+\]", user))
            return {"results": [{"pair": i, "label": "extends", "rationale": "fake"} for i in range(n)]}
        raise AssertionError(f"unexpected purpose {purpose}")

    def chat_tools(self, messages, tools):
        self.calls.append("agent")
        if self.tool_script:
            return self.tool_script.pop(0)
        return tool_msg(("finish", '{"summary": "done"}'))


def tool_msg(*calls: tuple[str, str], content: str = ""):
    return SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=n, arguments=a)) for i, (n, a) in enumerate(calls)
        ],
    )


def P(pid, title, year, refs=(), cited=10, doi=None, abstract=None) -> Paper:
    return Paper(
        id=pid, title=title, year=year, referenced_works=list(refs), cited_by_count=cited,
        doi=doi or f"10.1/{pid.lower()}", abstract=abstract or f"Abstract of {title}.", authors=["Ann Author"],
    )


# Small citation universe: W1 is foundational; W5 is the seed; W9 is off-topic
UNIVERSE = {
    p.id: p
    for p in [
        P("W1", "Neural message passing for quantum chemistry", 2017, cited=5000),
        P("W2", "Semi-supervised classification with graph convolutional networks", 2017, cited=9000),
        P("W3", "Molecular graph convolutions: moving beyond fingerprints", 2016, refs=[], cited=1500),
        P("W4", "Analyzing learned molecular representations for property prediction", 2019, refs=["W1", "W3"], cited=1200),
        P("W5", "Graph neural networks for drug discovery: a review", 2021, refs=["W1", "W2", "W4", "W9"], cited=400),
        P("W6", "Geometric graph neural networks for molecular docking", 2022, refs=["W5", "W1"], cited=150),
        P("W7", "Equivariant graph models for molecule generation", 2023, refs=["W6", "W5"], cited=80),
        P("W8", "Large language models meet molecular graphs", 2024, refs=["W7", "W4"], cited=20),
        P("W9", "A history of medieval trade routes", 2010, cited=30),
    ]
}


class FakeOA:
    def __init__(self, universe=None):
        self.u = universe or UNIVERSE
        self.requests = 0

    def _get(self, pid, origin, hop):
        p = self.u[pid].model_copy(deep=True)
        p.origin, p.hop, p.is_seed = origin, hop, origin == "seed"
        return p

    def by_ids(self, ids, origin="backward", hop=0):
        self.requests += 1
        return [self._get(i, origin, hop) for i in dict.fromkeys(ids) if i in self.u]

    def by_id(self, oa_id, origin="agent"):
        r = self.by_ids([oa_id], origin)
        return r[0] if r else None

    def by_dois(self, dois, origin="wos_search", hop=0):
        self.requests += 1
        out = {}
        for p in self.u.values():
            if p.doi in dois:
                out[p.doi] = self._get(p.id, origin, hop)
        return out

    def by_title(self, title, min_sim=0.9, origin="seed"):
        for p in self.u.values():
            if title_similarity(title, p.title) >= min_sim:
                return self._get(p.id, origin, 0)
        return None

    def search(self, query, *, years=(None, None), limit=50, origin="openalex_search"):
        self.requests += 1
        words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3 and w not in ("and",)]
        hits = [p for p in self.u.values() if any(w in p.title.lower() for w in words)]
        return [self._get(p.id, origin, 0) for p in hits[:limit]]

    def citing(self, oa_id, limit=25, hop=1):
        self.requests += 1
        hits = sorted((p for p in self.u.values() if oa_id in p.referenced_works), key=lambda p: -(p.cited_by_count or 0))
        return [self._get(p.id, "forward", hop) for p in hits[:limit]]

    def references(self, paper, limit=25, hop=1):
        refs = self.by_ids(paper.referenced_works, "backward", hop)
        return sorted(refs, key=lambda p: -(p.cited_by_count or 0))[:limit]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None, deepseek_api_key="x", wos_api_key="wos-key", wos_min_interval=0, wos_max_requests=10,
        scib_cache_dir=tmp_path / "cache", scib_corpora_dir=tmp_path / "corpora",
    )


@pytest.fixture
def small_corpus() -> Corpus:
    papers = [UNIVERSE[i].model_copy(deep=True) for i in ["W1", "W2", "W4", "W5", "W6", "W7", "W8"]]
    for p in papers:
        p.relevance = Relevance(score=8)
    papers[3].is_seed = True
    ids = {p.id for p in papers}
    edges = [Edge(source=p.id, target=r) for p in papers for r in p.referenced_works if r in ids]
    return Corpus(papers=papers, edges=edges)
