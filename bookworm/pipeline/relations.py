"""Relations between papers: citation edges within the corpus + optional LLM semantic dependency labels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import get_args

from ..llm import prompts as P
from ..llm.deepseek import LLM
from ..models import Edge, EdgeRelation, RelationLabel
from ..store import CorpusStore
from ..errors import BudgetError

LABELS = set(get_args(RelationLabel))


def build_edges(store: CorpusStore) -> list[Edge]:
    """citing -> cited. Edges come from OpenAlex referenced_works; also fills in each node's external citation counts."""
    old = {(e.source, e.target): e for e in store.corpus.edges}
    edges: list[Edge] = []
    indeg: dict[str, int] = {}
    for p in store.papers:
        refs = {store.canonical_id(r) for r in p.referenced_works} - {p.id}
        internal = sorted(r for r in refs if r in store)
        p.external_refs_count = len(refs) - len(internal)
        original = [(v.id, r) for v in p.versions for r in v.referenced_works]
        covered = {r for _, r in original}
        original += [(p.id, r) for r in p.referenced_works if r not in covered]
        for r in internal:
            prev = old.get((p.id, r))
            raw = list(dict.fromkeys(pair for pair in original if store.canonical_id(pair[1]) == r))
            mapped = any(pair != (p.id, r) for pair in raw)
            edges.append(Edge(source=p.id, target=r, relation=prev.relation if prev else None,
                              provenance="openalex_version_mapped" if mapped else "openalex", record_pairs=raw))
            indeg[r] = indeg.get(r, 0) + 1
    for p in store.papers:
        p.external_cited_by = max(0, (p.cited_by_count or 0) - indeg.get(p.id, 0))
    return edges


def label_edges(
    edges: list[Edge],
    store: CorpusStore,
    llm: LLM,
    lang: str = "en",
    min_score: float = 7.0,
    max_edges: int = 150,
    batch_size: int = 8,
    abstract_chars: int = 500,
    workers: int = 4,
) -> int:
    """Label only edges between high-scoring papers (seeds count as full marks) to control cost. Returns the number labeled."""

    def score(pid: str) -> float:
        p = store.get(pid)
        if p is None:
            return 0
        return 10.0 if p.is_seed else (p.relevance.score if p.relevance else 0)

    todo = [e for e in edges if e.relation is None and score(e.source) >= min_score and score(e.target) >= min_score]
    todo.sort(key=lambda e: -(score(e.source) + score(e.target)))
    todo = todo[:max_edges]
    system = P.fill(P.RELATION_SYSTEM, lang=lang)

    def block(pid: str) -> str:
        p = store.get(pid)
        return f"{p.title} ({p.year}). {(p.abstract or '')[:abstract_chars]}"

    def run(batch: list[Edge]) -> int:
        pairs = "\n\n".join(
            f"[pair {i}]\nCITING: {block(e.source)}\nCITED: {block(e.target)}" for i, e in enumerate(batch)
        )
        try:
            data = llm.chat_json(system, P.fill(P.RELATION_USER, pairs=pairs), purpose="relation")
        except BudgetError:
            raise
        except Exception as exc:
            if hasattr(llm, "tracer"):
                llm.tracer.warn(f"Relation labeling incomplete: {type(exc).__name__}")
            return 0
        n = 0
        for r in data.get("results") or []:
            try:
                i, label = int(r["pair"]), str(r["label"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(batch) and label in LABELS:
                batch[i].relation = EdgeRelation(label=label, rationale=str(r.get("rationale", "")))
                n += 1
        return n

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return sum(ex.map(run, batches))
