"""Conservative, offline version filter. Similarity alone never establishes identity.

Distinct published DOIs and ambiguous metadata remain separate for review. No LLM
is needed for the supported high-confidence rules; every merge keeps raw records.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from itertools import combinations

from .models import Paper, PaperRecord
from .store import CorpusStore, normalize_doi
from .trace import NULL_TRACER, Tracer

POLICY = "versions-v1"


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def author_key(name: str) -> str:
    # WoS uses 'Last, First'; OpenAlex usually uses 'First Last'. Do not infer initials.
    if "," in name:
        last, first = name.split(",", 1)
        name = first + " " + last
    return normalized(name)


def is_preprint(p: PaperRecord) -> bool:
    doi = normalize_doi(p.doi) or ""
    venue = normalized(p.venue or "")
    return (doi.startswith(("10.21203/rs.", "10.1101/", "10.48550/arxiv.", "10.20944/preprints"))
            or venue in {"research square", "arxiv", "biorxiv", "medrxiv", "preprints org"})


def preferred(p: Paper) -> tuple:
    # Prefer a journal/DOI record over its preprint, then an OA id over WoS-only.
    # Citation count does not select the representative; it is version-specific.
    return (is_preprint(p), not bool(p.doi), not bool(re.fullmatch(r"W\d+", p.id)), p.id)


def evidence(a: PaperRecord, b: PaperRecord) -> dict:
    ta, tb = normalized(a.title), normalized(b.title)
    aa, ab = {author_key(x) for x in a.authors if x.strip()}, {author_key(x) for x in b.authors if x.strip()}
    common = aa & ab
    ra, rb = set(a.referenced_works), set(b.referenced_works)
    refs = len(ra & rb)
    edits = SequenceMatcher(None, ta.split(), tb.split(), autojunk=False).get_opcodes()
    return {
        "title_similarity": round(SequenceMatcher(None, ta, tb, autojunk=False).ratio(), 4),
        "title_exact": bool(ta) and ta == tb,
        "title_tokens": min(len(ta.split()), len(tb.split())),
        "title_prefix": bool(ta and tb) and (ta.startswith(tb + " ") or tb.startswith(ta + " ")),
        "title_only_insertions_or_deletions": all(tag != "replace" for tag, *_ in edits),
        "common_authors": len(common),
        "author_coverage": len(common) / max(len(aa), len(ab), 1),
        "first_author_exact": bool(a.authors and b.authors) and author_key(a.authors[0]) == author_key(b.authors[0]),
        "year_gap": abs(a.year - b.year) if a.year is not None and b.year is not None else None,
        "common_references": refs,
        "reference_overlap": refs / max(1, min(len(ra), len(rb))),
    }


def match_rule(a: PaperRecord, b: PaperRecord, e: dict | None = None) -> str | None:
    if a.id == b.id or (normalize_doi(a.doi) and normalize_doi(a.doi) == normalize_doi(b.doi)):
        return "same_identifier"
    e = e or evidence(a, b)
    # A review, correction, sequel or similarly named work must not disappear.
    markers = r"\b(correction|erratum|corrigendum|retraction|book review|part [0-9ivx]+)\b"
    if re.search(markers, normalized(a.title)) or re.search(markers, normalized(b.title)):
        return None
    if e["year_gap"] is None or e["year_gap"] > 3 or not e["first_author_exact"]:
        return None
    if e["title_exact"] and e["title_tokens"] >= 5 and e["author_coverage"] == 1 and e["year_gap"] <= 1:
        if not a.doi or not b.doi:
            return "same_title_authors_year_missing_doi"
    # Two different published DOIs are never collapsed by fuzzy metadata rules.
    if is_preprint(a) == is_preprint(b) or not a.doi or not b.doi:
        return None
    if e["common_authors"] < 2 or e["author_coverage"] < .75 or e["title_tokens"] < 10:
        return None
    if (e["title_similarity"] >= .94 and e["author_coverage"] >= .8
            and e["title_only_insertions_or_deletions"]
            and (e["title_exact"] or (e["common_references"] >= 3 and e["reference_overlap"] >= .2))):
        return "preprint_near_exact_title_authors"
    if (e["title_prefix"]
            and e["common_references"] >= 5 and e["reference_overlap"] >= .6):
        return "preprint_title_authors_references"
    return None


def _records(p: Paper) -> list[PaperRecord]:
    return p.versions or [p]


def group_match(a: Paper, b: Paper) -> tuple[str, dict] | None:
    # Complete linkage prevents A~B~C chains from collapsing incompatible studies.
    matches = [(x, y, evidence(x, y)) for x in _records(a) for y in _records(b)]
    rules = [match_rule(x, y, e) for x, y, e in matches]
    if not all(rules):
        return None
    return str(rules[0]), matches[0][2]


def find_duplicate(store: CorpusStore, paper: Paper) -> Paper | None:
    identity = store.get(paper.id) or store.find_by_doi(paper.doi)
    if identity is not None:
        return identity
    matches = [p for p in store.papers if group_match(p, paper)]
    # Multiple incompatible matches are ambiguous, not permission to merge a chain.
    if len(matches) == 1:
        return matches[0]
    return None


def deduplicate(store: CorpusStore, tracer: Tracer = NULL_TRACER) -> dict:
    before = len(store)
    report = store.corpus.meta.deduplication
    if not report:
        report.update(policy=POLICY, merges=[])
    report["policy"] = POLICY
    # Freeze the candidate graph before merging; only disjoint cliques are safe.
    papers = sorted(store.papers, key=preferred)
    by_id = {p.id: p for p in papers}
    pairs = {}
    neighbors = {p.id: set() for p in papers}
    for a, b in combinations(papers, 2):
        match = group_match(a, b)
        if match:
            pairs[frozenset((a.id, b.id))] = match
            neighbors[a.id].add(b.id)
            neighbors[b.id].add(a.id)
    visited = set()
    for root in papers:
        if root.id in visited:
            continue
        component, todo = set(), [root.id]
        while todo:
            pid = todo.pop()
            if pid in component:
                continue
            component.add(pid)
            todo.extend(neighbors[pid] - component)
        visited.update(component)
        if len(component) < 2 or any(len(neighbors[pid] & component) != len(component) - 1 for pid in component):
            continue
        members = sorted((by_id[pid] for pid in component), key=preferred)
        primary = members[0]
        for other in members[1:]:
            rule, details = pairs[frozenset((primary.id, other.id))]
            event = {"kept_id": primary.id, "removed_id": other.id, "rule": rule, "evidence": details}
            store.merge_versions(primary, other)
            report["merges"].append(event)
            tracer.log("version_merge", **event)
    review = []
    for a, b in combinations(sorted(store.papers, key=lambda p: p.id), 2):
        e = evidence(a, b)
        if e["title_exact"] or e["title_similarity"] >= .8 or (e["title_prefix"] and e["title_tokens"] >= 5):
            review.append({"ids": [a.id, b.id], "titles": [a.title, b.title],
                           "reason": "insufficient_or_conflicting_identity_evidence", "evidence": e})
    report["review_candidates"] = review
    report["last_pass"] = {"before": before, "after": len(store), "merged": before - len(store)}
    return report
