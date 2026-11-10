"""Natural language (+ seed papers) -> structured research intent."""

from __future__ import annotations

from ..llm import prompts as P
from ..llm.deepseek import LLM
from ..models import Paper, ResearchIntent


def seed_block(seeds: list[Paper], abstract_chars: int = 400) -> str:
    if not seeds:
        return ""
    lines = ["Seed papers provided by the researcher:"]
    for i, s in enumerate(seeds, 1):
        abs_ = (s.abstract or "")[:abstract_chars]
        lines.append(f"[{i}] {s.title} ({s.year or 'n.d.'})\n    {abs_}")
    return "\n".join(lines) + "\n"


def parse_intent(
    prompt: str,
    llm: LLM,
    seeds: list[Paper] | None = None,
    years: tuple[int | None, int | None] | None = None,
) -> ResearchIntent:
    data = llm.chat_json(
        P.INTENT_SYSTEM,
        P.fill(P.INTENT_USER, prompt=prompt, seed_block=seed_block(seeds or [])),
        purpose="intent",
    )
    intent = coerce_intent(data, fallback_topic=prompt)
    if years and any(years):
        intent.year_range = years  # explicit CLI arguments take priority
    return intent


def coerce_intent(data: dict, fallback_topic: str) -> ResearchIntent:
    """Tolerate imperfect LLM output: fix field types, drop invalid values."""

    def str_list(v) -> list[str]:
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v or [] if x]

    yr = data.get("year_range") or [None, None]
    if not isinstance(yr, (list, tuple)) or len(yr) != 2:
        yr = [None, None]
    yr = tuple(int(y) if isinstance(y, (int, float)) or (isinstance(y, str) and y.isdigit()) else None for y in yr)
    syn = data.get("synonyms") or {}
    if not isinstance(syn, dict):
        syn = {}
    return ResearchIntent(
        topic=str(data.get("topic") or fallback_topic),
        research_questions=str_list(data.get("research_questions")),
        core_concepts=str_list(data.get("core_concepts")),
        keywords_en=str_list(data.get("keywords_en")),
        synonyms={str(k): str_list(v) for k, v in syn.items()},
        exclude=str_list(data.get("exclude")),
        year_range=yr,
        doc_types=str_list(data.get("doc_types")),
        language=str(data.get("language") or "en"),
    )
