"""Citation graph and BFS/DFS traversal (used by the CLI).

Edge direction: citing -> cited (source cites target).
- direction="refs":     follow out-edges, upstream to the foundational works a paper depends on
- direction="cited_by": follow in-edges, downstream to later work that builds on it
- direction="both":     ignore direction
Neighbors are visited in descending priority (relevance, citation count), so results are reproducible.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal

import networkx as nx

from .models import Corpus

Mode = Literal["bfs", "dfs"]
TraverseDirection = Literal["refs", "cited_by", "both"]


def build_graph(corpus: Corpus) -> nx.DiGraph:
    g = nx.DiGraph()
    for p in corpus.papers:
        g.add_node(
            p.id,
            title=p.title,
            year=p.year,
            score=p.relevance.score if p.relevance else 0.0,
            cited=p.cited_by_count or p.wos_times_cited or 0,
            seed=p.is_seed,
        )
    for e in corpus.edges:
        if e.source in g and e.target in g:
            g.add_edge(e.source, e.target, label=e.relation.label if e.relation else None)
    return g


@dataclass
class Visit:
    id: str
    order: int
    depth: int
    parent: str | None
    via: str | None  # "refs" | "cited_by": the edge direction used to reach this node


def _neighbors(g: nx.DiGraph, n: str, direction: TraverseDirection) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if direction in ("refs", "both"):
        out += [(m, "refs") for m in g.successors(n)]
    if direction in ("cited_by", "both"):
        out += [(m, "cited_by") for m in g.predecessors(n)]
    seen: set[str] = set()
    uniq = [(m, v) for m, v in out if not (m in seen or seen.add(m))]
    return sorted(uniq, key=lambda mv: (-g.nodes[mv[0]]["seed"], -g.nodes[mv[0]]["score"], -g.nodes[mv[0]]["cited"], mv[0]))


def traverse(
    g: nx.DiGraph,
    starts: list[str],
    mode: Mode = "bfs",
    direction: TraverseDirection = "both",
    max_depth: int = 3,
    limit: int = 200,
) -> list[Visit]:
    starts = [s for s in dict.fromkeys(starts) if s in g]
    visits: list[Visit] = []
    seen: set[str] = set()

    def visit(n: str, depth: int, parent: str | None, via: str | None) -> bool:
        seen.add(n)
        visits.append(Visit(n, len(visits), depth, parent, via))
        return len(visits) < limit

    if mode == "bfs":
        q: deque[tuple[str, int]] = deque()
        for s in starts:  # multi-source BFS: all start points at depth 0
            if s not in seen:
                if not visit(s, 0, None, None):
                    return visits
                q.append((s, 0))
        while q:
            n, d = q.popleft()
            if d >= max_depth:
                continue
            for m, via in _neighbors(g, n, direction):
                if m not in seen:
                    if not visit(m, d + 1, n, via):
                        return visits
                    q.append((m, d + 1))
    else:
        for s in starts:  # DFS each start point in turn (preorder)
            if s in seen:
                continue
            stack: list[tuple[str, int, str | None, str | None]] = [(s, 0, None, None)]
            while stack:
                n, d, parent, via = stack.pop()
                if n in seen:
                    continue
                if not visit(n, d, parent, via):
                    return visits
                if d < max_depth:
                    # Push in reverse so the highest-priority neighbor is popped first
                    for m, v in reversed(_neighbors(g, n, direction)):
                        if m not in seen:
                            stack.append((m, d + 1, n, v))
    return visits


def visits_to_dicts(visits: list[Visit]) -> list[dict]:
    return [asdict(v) for v in visits]


def path_to(visits: list[Visit], target: str) -> list[str]:
    """Traverse the parent pointers back to the start, returning start -> target."""
    by_id = {v.id: v for v in visits}
    path: list[str] = []
    cur = by_id.get(target)
    while cur is not None:
        path.append(cur.id)
        cur = by_id.get(cur.parent) if cur.parent else None
    return path[::-1]
