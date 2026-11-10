import json
from pathlib import Path

import httpx
import pytest
import respx

from bookworm.llm.deepseek import parse_json
from bookworm.pipeline.intent import coerce_intent
from bookworm.pipeline.seeds import classify
from bookworm.sources.cache import Cache
from bookworm.sources.openalex import title_similarity, to_paper
from bookworm.sources.wos import WosBudgetExceeded, WosClient, WosQueryError

FIX = Path(__file__).parent / "fixtures"
WOS_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"


def wos_page(n: int, total: int, start: int = 0) -> dict:
    hit = json.loads((FIX / "wos_hit.json").read_text())
    hits = []
    for i in range(n):
        h = json.loads(json.dumps(hit))
        h["uid"] = f"WOS:{start + i:015d}"
        h["identifiers"]["doi"] = f"10.1/w{start + i}"
        hits.append(h)
    return {"metadata": {"total": total, "page": 1, "limit": n}, "hits": hits}


@respx.mock
def test_wos_parse_and_paginate(settings):
    route = respx.get(WOS_URL).mock(
        side_effect=[httpx.Response(200, json=wos_page(50, 70)), httpx.Response(200, json=wos_page(20, 70, 50))]
    )
    c = WosClient(settings)
    total, papers = c.search_all("TS=(x)", max_results=100)
    assert total == 70 and len(papers) == 70 and route.call_count == 2
    p = papers[0]
    assert p.id == p.wos_uid == "WOS:000000000000000"
    assert p.title.startswith("Graph neural networks") and p.year == 2021
    assert p.wos_times_cited == 321 and p.authors == ["Smith, John", "Li, Wei"]
    assert p.keywords == ["graph neural network", "drug discovery"]
    req = route.calls[0].request
    assert req.headers["X-ApiKey"] == "wos-key" and req.url.params["db"] == "WOS"


@respx.mock
def test_wos_400_is_query_error(settings):
    respx.get(WOS_URL).mock(return_value=httpx.Response(400, json={"message": "Invalid query syntax"}))
    with pytest.raises(WosQueryError, match="Invalid query"):
        WosClient(settings).search("TS=((")


@respx.mock
def test_wos_budget_and_cache(settings, tmp_path):
    route = respx.get(WOS_URL).mock(return_value=httpx.Response(200, json=wos_page(1, 1)))
    settings.wos_max_requests = 1
    c = WosClient(settings, cache=Cache(tmp_path / "c.sqlite"))
    c.search("TS=(a)")
    c.search("TS=(a)")  # cache hit, uses no quota
    assert route.call_count == 1
    with pytest.raises(WosBudgetExceeded):
        c.search("TS=(b)")


def test_openalex_to_paper():
    r = json.loads((FIX / "openalex_work.json").read_text())
    p = to_paper(r, "backward", 1)
    assert p.id == "W2919115771" and p.doi == "10.1038/nature14539"
    assert p.abstract == "Deep learning allows computational models"
    assert p.referenced_works == ["W1", "W2"]
    assert p.venue == "Nature" and p.authors == ["Yann LeCun", "Yoshua Bengio"]
    assert p.origin == "backward" and p.hop == 1 and not p.is_seed and not p.retracted
    assert to_paper({**r, "is_retracted": True}).retracted


def test_title_similarity():
    assert title_similarity("Attention Is All You Need", "attention is all you need.") == 1.0
    assert title_similarity("Attention Is All You Need", "Not All Attention Is All You Need") < 0.9


def test_seed_classify():
    assert classify("https://doi.org/10.1038/NATURE14539") == ("doi", "10.1038/nature14539")
    assert classify("W2919115771") == ("openalex", "W2919115771")
    assert classify("https://openalex.org/W1") == ("openalex", "W1")
    assert classify("WOS:000123") == ("wos", "WOS:000123")
    assert classify("Deep learning") == ("title", "Deep learning")


def test_parse_json_tolerant():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_coerce_intent_tolerates_bad_types():
    it = coerce_intent({"keywords_en": "single", "year_range": ["2018", None], "synonyms": [1]}, "fallback")
    assert it.topic == "fallback" and it.keywords_en == ["single"]
    assert it.year_range == (2018, None) and it.synonyms == {}


@respx.mock
def test_wos_paging_keeps_page_size(settings):
    route = respx.get(WOS_URL).mock(
        side_effect=[httpx.Response(200, json=wos_page(50, 120)), httpx.Response(200, json=wos_page(50, 120, 50))]
    )
    total, papers = WosClient(settings).search_all("TS=(x)", max_results=70)
    assert len(papers) == 70 and len({p.id for p in papers}) == 70
    assert [c.request.url.params["limit"] for c in route.calls] == ["50", "50"]
    assert [c.request.url.params["page"] for c in route.calls] == ["1", "2"]


@respx.mock
def test_wos_skips_seen_hits(settings):
    route = respx.get(WOS_URL).mock(
        side_effect=[httpx.Response(200, json=wos_page(50, 120)), httpx.Response(200, json=wos_page(50, 120, 50))]
    )
    seen = {f"10.1/w{i}" for i in range(40)}
    total, papers = WosClient(settings).search_all("TS=(x)", max_results=20, seen=seen, max_pages=3)
    assert [p.doi for p in papers] == [f"10.1/w{i}" for i in range(40, 60)]
    assert route.call_count == 2 and "10.1/w59" in seen
