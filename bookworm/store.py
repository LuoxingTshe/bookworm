"""Corpus in-memory container, dedup/merge, and JSON persistence."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .models import Corpus, Edge, Paper, PaperRecord

# Higher priority = closer to the user's intent; on merge, keep the higher-priority origin
_ORIGIN_RANK = {"seed": 5, "wos_search": 4, "openalex_search": 4, "agent": 3, "cocited": 2, "backward": 2, "forward": 2}


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
    return doi or None


def short_oa_id(oa_id: str | None) -> str | None:
    if not oa_id:
        return None
    return oa_id.rstrip("/").split("/")[-1]


class CorpusStore:
    def __init__(self, corpus: Corpus | None = None):
        self.corpus = corpus or Corpus()
        self._by_id: dict[str, Paper] = {}
        self._by_doi: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        papers, self.corpus.papers = self.corpus.papers, []
        for p in papers:
            self.add(p)
        self.remap_edges()

    # ---- queries ----
    def __contains__(self, pid: str) -> bool:
        return pid in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, pid: str) -> Paper | None:
        return self._by_id.get(self.canonical_id(pid))

    def canonical_id(self, pid: str) -> str:
        return self._aliases.get(pid, pid)

    def find_by_doi(self, doi: str | None) -> Paper | None:
        d = normalize_doi(doi)
        return self._by_id.get(self._by_doi[d]) if d and d in self._by_doi else None

    def has(self, paper: Paper) -> bool:
        return self.get(paper.id) is not None or self.find_by_doi(paper.doi) is not None

    @property
    def papers(self) -> list[Paper]:
        return list(self._by_id.values())

    # ---- mutations ----
    def add(self, paper: Paper) -> Paper:
        """Add or merge a paper; returns the canonical instance in the store."""
        paper.doi = normalize_doi(paper.doi)
        existing = self.get(paper.id) or self.find_by_doi(paper.doi)
        if existing is None:
            self._by_id[paper.id] = paper
            if paper.doi:
                self._by_doi[paper.doi] = paper.id
            self._index_versions(paper)
            return paper
        snapshots = self._snapshots(existing, paper) if existing.id != paper.id or existing.versions or paper.versions else []
        old_id = existing.id
        self._merge_into(existing, paper)
        if existing.id != old_id:
            del self._by_id[old_id]
            self._by_id[existing.id] = existing
        if existing.doi:
            self._by_doi[existing.doi] = existing.id
        if snapshots:
            existing.versions = snapshots
            self.rekey()
            self.remap_edges()
        return existing

    @staticmethod
    def _snapshots(*papers: Paper) -> list[PaperRecord]:
        records = {}
        for paper in papers:
            for record in paper.versions or [PaperRecord.model_validate(paper.model_dump(exclude={"versions"}))]:
                records.setdefault((record.id, normalize_doi(record.doi)), record.model_copy(deep=True))
        return [records[key] for key in sorted(records, key=lambda k: (k[0], k[1] or ""))]

    def merge_versions(self, primary: Paper, other: Paper) -> Paper:
        """Consolidate a supported pair, keeping raw snapshots and all identifier aliases."""
        snapshots = self._snapshots(primary, other)
        primary_id = primary.id
        scores = [p.relevance for p in (primary, other) if p.relevance is not None]
        self._merge_into(primary, other)
        primary.id = primary_id
        if scores:
            primary.relevance = max(scores, key=lambda r: r.score).model_copy(deep=True)
        primary.versions = snapshots
        del self._by_id[other.id]
        self.rekey()
        self.remap_edges()
        return primary

    def _index_versions(self, paper: Paper) -> None:
        for v in paper.versions:
            self._aliases[v.id] = paper.id
            if d := normalize_doi(v.doi):
                self._by_doi[d] = paper.id

    def remap_edges(self) -> None:
        edges = {}
        for e in self.corpus.edges:
            source, target = self.canonical_id(e.source), self.canonical_id(e.target)
            if source == target:
                continue
            key = (source, target)
            raw = e.record_pairs or [(e.source, e.target)]
            previous = edges.get(key)
            raw = list(dict.fromkeys((previous.record_pairs if previous else []) + raw))
            mapped = any(pair != key for pair in raw)
            if key not in edges or (edges[key].relation is None and e.relation is not None):
                edges[key] = e.model_copy(update={"source": source, "target": target,
                    "record_pairs": raw,
                    "provenance": "openalex_version_mapped" if mapped else e.provenance})
            else:
                edges[key].record_pairs = raw
                if mapped:
                    edges[key].provenance = "openalex_version_mapped"
        self.corpus.edges = list(edges.values())

    def remove(self, pid: str) -> bool:
        pid = self.canonical_id(pid)
        p = self._by_id.pop(pid, None)
        if p is None:
            return False
        if p.doi and self._by_doi.get(p.doi) == pid:
            del self._by_doi[p.doi]
        self.corpus.edges = [e for e in self.corpus.edges if pid not in (e.source, e.target)]
        self.rekey()
        return True

    def retain(self, keep: set[str]) -> None:
        for pid in [pid for pid in self._by_id if pid not in keep]:
            self.remove(pid)

    @staticmethod
    def _merge_into(dst: Paper, src: Paper) -> None:
        for field in ("doi", "wos_uid", "abstract", "venue", "year", "cited_by_count", "wos_times_cited", "relevance"):
            if getattr(dst, field) in (None, "") and getattr(src, field) not in (None, ""):
                setattr(dst, field, getattr(src, field))
        if not dst.title and src.title:
            dst.title = src.title
        if not dst.authors and src.authors:
            dst.authors = list(src.authors)
        for field in ("keywords", "referenced_works"):
            setattr(dst, field, list(dict.fromkeys(getattr(dst, field) + getattr(src, field))))
        dst.is_seed = dst.is_seed or src.is_seed
        dst.retracted = dst.retracted or src.retracted
        if _ORIGIN_RANK.get(src.origin, 0) > _ORIGIN_RANK.get(dst.origin, 0):
            dst.origin = src.origin
        dst.hop = min(dst.hop, src.hop)
        dst.notes.extend(n for n in src.notes if n not in dst.notes)
        # Upgrade a WoS-only record to its OpenAlex id
        if dst.id.startswith("WOS:") and not src.id.startswith("WOS:"):
            dst.id = src.id

    def rekey(self) -> None:
        """Rebuild indexes after _merge_into may have changed an id."""
        papers = list(self._by_id.values())
        self._by_id = {p.id: p for p in papers}
        self._by_doi = {p.doi: p.id for p in papers if p.doi}
        self._aliases = {}
        for p in papers:
            self._index_versions(p)

    def set_edges(self, edges: list[Edge]) -> None:
        self.corpus.edges = edges

    # ---- persistence ----
    def finalize(self) -> Corpus:
        self.rekey()
        self.corpus.papers = sorted(
            self._by_id.values(),
            key=lambda p: (not p.is_seed, -(p.relevance.score if p.relevance else 0), -(p.cited_by_count or 0)),
        )
        self.corpus.stats = {
            "n_papers": len(self.corpus.papers),
            "n_source_records": sum(len(p.versions) or 1 for p in self.corpus.papers),
            "n_version_groups": sum(bool(p.versions) for p in self.corpus.papers),
            "n_edges": len(self.corpus.edges),
            "n_seeds": sum(p.is_seed for p in self.corpus.papers),
            "by_origin": dict(Counter(p.origin for p in self.corpus.papers)),
            "n_labeled_edges": sum(e.relation is not None for e in self.corpus.edges),
            "n_retracted": sum(p.retracted for p in self.corpus.papers),
            "year_span": _year_span(self.corpus.papers),
        }
        return self.corpus

    def save(self, path: Path) -> Path:
        corpus = self.finalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        from .persistence import atomic_write
        atomic_write(path, corpus.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> "CorpusStore":
        from .errors import InputError
        try:
            return cls(Corpus.model_validate(json.loads(Path(path).read_text(encoding="utf-8"))))
        except (OSError, ValueError) as exc:
            raise InputError(f"Cannot read corpus {path}: {type(exc).__name__}") from exc


def _year_span(papers: list[Paper]) -> list[int] | None:
    years = [p.year for p in papers if p.year]
    return [min(years), max(years)] if years else None


def slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w一-鿿]+", "-", text.lower()).strip("-")
    return s[:maxlen].strip("-") or "corpus"


def new_run_dir(base: Path, prompt: str) -> Path:
    d = base / f"{slugify(prompt)}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    d.mkdir(parents=True, exist_ok=False)
    return d
