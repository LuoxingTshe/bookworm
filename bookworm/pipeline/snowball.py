"""Snowball expansion: backward (references) / forward (citing works)."""

from __future__ import annotations

from typing import Literal

from ..models import Paper
from ..sources.openalex import OpenAlexClient
from ..store import CorpusStore
from ..trace import NULL_TRACER, Tracer

Direction = Literal["backward", "forward", "both"]


def expand(
    pool: CorpusStore,
    oa: OpenAlexClient,
    frontier: list[Paper],
    hop: int,
    per_node: int = 25,
    direction: Direction = "both",
    tracer: Tracer = NULL_TRACER,
) -> list[Paper]:
    """Expand one hop from frontier; returns only new candidates not yet in pool (already added to pool)."""
    new: list[Paper] = []
    # Backward: gather all frontier references and fetch them in batches (one request per 50), cheaper than per-node calls
    if direction in ("backward", "both"):
        ref_ids: list[str] = []
        for p in frontier:
            ref_ids += [r for r in p.referenced_works if pool.get(r) is None]
        refs = oa.by_ids(list(dict.fromkeys(ref_ids)), origin="backward", hop=hop)
        # Cap per node: keep the per_node most-cited references from each frontier node
        keep: set[str] = set()
        by_id = {r.id: r for r in refs}
        for p in frontier:
            mine = sorted((by_id[r] for r in p.referenced_works if r in by_id), key=lambda x: -(x.cited_by_count or 0))
            keep |= {x.id for x in mine[:per_node]}
        for r in refs:
            if r.id in keep and not pool.has(r):
                new.append(pool.add(r))
    if direction in ("forward", "both"):
        for p in frontier:
            if p.id.startswith("WOS:"):
                continue
            for c in oa.citing(p.id, limit=per_node, hop=hop):
                if not pool.has(c):
                    new.append(pool.add(c))
    tracer.log("snowball", hop=hop, frontier=len(frontier), new=len(new))
    return new
