"""Co-citation gap fill: works referenced by many relevant papers but never pulled into the pool.

Snowballing only expands the frontier (seeds + top hits), so a work that most of the corpus cites can still be
missing. Counting references over every relevant paper catches it (LocalCitationNetwork's "top references").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..models import Paper
from ..store import CorpusStore


def top_missing_refs(
    citing: Iterable[Paper], known: CorpusStore, min_count: int = 3, limit: int = 30
) -> list[tuple[str, int]]:
    """(OpenAlex id, number of `citing` papers that reference it) for works not in `known`, most-cited first."""
    counts = Counter(r for p in citing for r in dict.fromkeys(p.referenced_works) if r and known.get(r) is None)
    return [(rid, n) for rid, n in counts.most_common() if n >= min_count][:limit]
