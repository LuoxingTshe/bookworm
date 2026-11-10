import json

import pytest

from conftest import UNIVERSE, FakeLLM, FakeOA, tool_msg
from bookworm.agent.loop import run_agent
from bookworm.agent.tools import AgentContext
from bookworm.models import Corpus, Paper, Relevance, ResearchIntent
from bookworm.pipeline import query
from bookworm.pipeline.build import BuildParams, build_corpus, params_for_tier
from bookworm.pipeline.intent import parse_intent
from bookworm.sources.wos import WosQueryError
from bookworm.store import CorpusStore
from bookworm.trace import Tracer


class FakeWos:
    def __init__(self, ids=("W5", "W6", "W9")):
        self.ids = ids
        self.queries: list[str] = []

    def search_all(self, q, max_results=50, sort=None, seen=None, max_pages=None):
        self.queries.append(q)
        if "((" in q:
            raise WosQueryError(400, "syntax error near ((")
        hits = [
            Paper(id=f"WOS:{i}", wos_uid=f"WOS:{i}", doi=UNIVERSE[i].doi, title=UNIVERSE[i].title,
                  year=UNIVERSE[i].year, wos_times_cited=99, keywords=["kw"])
            for i in self.ids
        ]
        return len(hits), hits[:max_results]

    def search(self, q, limit=50, page=1, sort=None):
        return self.search_all(q, limit)


def _check_integrity(c: Corpus):
    ids = {p.id for p in c.papers}
    assert len(ids) == len(c.papers)
    assert all(e.source in ids and e.target in ids for e in c.edges)
    Corpus.model_validate(c.model_dump())


def _check_prisma(c: Corpus):
    pr = c.meta.prisma
    n_seeds = sum(p.is_seed for p in c.papers)
    assert pr.included == len(c.papers)
    assert sum(pr.identified.values()) == pr.screened + pr.not_screened + n_seeds
    assert pr.screened == pr.excluded_low_relevance + pr.excluded_retracted + pr.excluded_over_cap + pr.included - n_seeds


def _universe(**changes) -> dict[str, Paper]:
    """UNIVERSE copy; changes maps id -> field overrides (a new id adds a paper)."""
    u = {k: v.model_copy(deep=True) for k, v in UNIVERSE.items()}
    for pid, fields in changes.items():
        if pid in u:
            for k, v in fields.items():
                setattr(u[pid], k, v)
        else:
            u[pid] = Paper(id=pid, doi=f"10.1/{pid.lower()}", authors=["Ann Author"], **fields)
    return u


def test_build_wos_end_to_end(tmp_path):
    out = tmp_path / "run" / "corpus.json"
    tracer = Tracer(tmp_path / "trace.jsonl")
    llm, wos = FakeLLM(), FakeWos()
    params = BuildParams(prompt="图神经网络在药物发现中的应用", source="wos", seeds=["W5"], hops=1, threshold=6, label_edges=True)
    build_corpus(params, llm, FakeOA(), wos, tracer, out, log=lambda m: None)
    c = Corpus.model_validate(json.loads(out.read_text()))
    _check_integrity(c)
    ids = {p.id for p in c.papers}
    assert ids == {"W5", "W6", "W2", "W4", "W7"}  # W9 is off-topic, W1 scores below threshold
    seed = next(p for p in c.papers if p.id == "W5")
    assert seed.is_seed and seed.origin == "seed" and seed.wos_uid == "WOS:W5"  # WoS fields merged in
    w6 = next(p for p in c.papers if p.id == "W6")
    assert w6.origin == "wos_search" and w6.wos_times_cited == 99 and w6.relevance.score == 8
    assert {(e.source, e.target) for e in c.edges} == {("W5", "W2"), ("W5", "W4"), ("W6", "W5"), ("W7", "W6"), ("W7", "W5")}
    assert all(e.relation and e.relation.label == "extends" for e in c.edges)
    assert next(p for p in c.papers if p.id == "W4").external_refs_count == 2
    assert c.meta.intent.topic and c.meta.queries == wos.queries
    assert c.stats["n_edges"] == 5 and c.stats["n_seeds"] == 1
    assert (tmp_path / "trace.jsonl").exists()
    _check_prisma(c)


def test_build_openalex_source(tmp_path):
    out = tmp_path / "corpus.json"
    params = BuildParams(prompt="GNN drug discovery", source="openalex", hops=2, max_papers=4)
    store = build_corpus(params, FakeLLM(), FakeOA(), None, Tracer(), out, log=lambda m: None)
    c = store.corpus
    _check_integrity(c)
    assert 0 < len(c.papers) <= 4
    assert all(p.relevance.score >= 6 for p in c.papers)


def test_wos_query_fixup():
    wos, llm = FakeWos(), FakeLLM()
    executed, hits, warns = query.run_wos_queries(["TS=((broken"], wos, llm, 10)
    assert executed == ['TS=("graph neural network*")'] and hits
    assert any("rewritten" in w for w in warns)


def test_year_clause_added():
    assert query.ensure_year_clause("TS=(a)", (2018, None)) == "(TS=(a)) AND PY=(2018-2100)"
    assert query.ensure_year_clause("TS=(a) AND PY=(2020-2021)", (2018, 2019)) == "TS=(a) AND PY=(2020-2021)"
    assert query.ensure_year_clause("TS=(a)", (None, None)) == "TS=(a)"


def test_intent_years_override():
    it = parse_intent("x", FakeLLM(), years=(2019, 2024))
    assert it.year_range == (2019, 2024) and it.language == "zh"


def test_agent_loop(tmp_path):
    store = CorpusStore()
    seed = UNIVERSE["W5"].model_copy(deep=True)
    seed.is_seed, seed.origin, seed.relevance = True, "seed", Relevance(score=10)
    store.add(seed)
    script = [
        tool_msg(("corpus_status", "{}")),
        tool_msg(("search_openalex", '{"query": "graph docking medieval", "limit": 10}')),
        tool_msg(("screen", '{"ids": ["W6", "W9"]}')),
        tool_msg(("add_to_corpus", '{"ids": ["W6", "W9", "W404"], "note": "from search"}')),
        tool_msg(("get_citing", '{"id": "W6"}'), ("not_a_tool", "{}")),
        tool_msg(("remove_from_corpus", '{"ids": ["W5"]}')),  # seeds cannot be removed
        tool_msg(("finish", '{"summary": "覆盖了 GNN 药物发现主线"}')),
    ]
    llm = FakeLLM(tool_script=script)
    intent = parse_intent("x", llm)
    ctx = AgentContext(corpus=store, llm=llm, oa=FakeOA(), wos=None, intent=intent, tracer=Tracer(tmp_path / "t.jsonl"),
                       max_papers=20)
    run_agent(ctx, "GNN 药物发现", max_steps=10, log=lambda m: None)
    assert ctx.finished and "GNN" in ctx.summary
    assert {p.id for p in store.papers} == {"W5", "W6"}
    assert "from search" in store.get("W6").notes
    trace = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert any(t["name"] == "not_a_tool" and "unknown tool" in t["result"] for t in trace)


def test_gap_fill_adds_cocited_work(tmp_path):
    # W10 is cited by W6, W7, W8 but no search query matches it and there is no snowball hop
    u = _universe(W10=dict(title="MoleculeNet benchmark datasets", year=2018, cited_by_count=3000, abstract="Benchmark."))
    for pid in ("W6", "W7", "W8"):
        u[pid].referenced_works.append("W10")
    params = BuildParams(prompt="GNN drug discovery", source="openalex", hops=0)
    store = build_corpus(params, FakeLLM(), FakeOA(u), None, Tracer(), tmp_path / "c.json", log=lambda m: None)
    w10 = store.get("W10")
    assert w10 is not None and w10.origin == "cocited" and w10.relevance.score == 8
    assert store.corpus.meta.prisma.identified["cocited"] == 1
    assert {(e.source, e.target) for e in store.corpus.edges} >= {("W6", "W10"), ("W7", "W10"), ("W8", "W10")}
    _check_prisma(store.corpus)
    params.gap_fill = False
    store = build_corpus(params, FakeLLM(), FakeOA(u), None, Tracer(), tmp_path / "d.json", log=lambda m: None)
    assert store.get("W10") is None


def test_snowball_stops_on_low_yield(tmp_path):
    # Hop 1 finds W3 (relevant) and W9 (off-topic): yield 50%
    params = BuildParams(prompt="GNN drug discovery", source="openalex", hops=3, min_hop_yield=0.6)
    store = build_corpus(params, FakeLLM(), FakeOA(), None, Tracer(), tmp_path / "c.json", log=lambda m: None)
    pr = store.corpus.meta.prisma
    assert [(h.candidates, h.relevant) for h in pr.hops] == [(2, 1)]
    assert pr.stop_reason == "hop 1: yield 50% < 60%"
    params.min_hop_yield = 0.1  # hop 1 passes; hop 2 has nothing left to find
    store = build_corpus(params, FakeLLM(), FakeOA(), None, Tracer(), tmp_path / "d.json", log=lambda m: None)
    assert store.corpus.meta.prisma.stop_reason == "hop 2: no new candidates"


def test_retracted_papers_excluded(tmp_path):
    u = _universe(W6=dict(retracted=True))
    params = BuildParams(prompt="GNN drug discovery", source="openalex", hops=0)
    store = build_corpus(params, FakeLLM(), FakeOA(u), None, Tracer(), tmp_path / "c.json", log=lambda m: None)
    assert store.get("W6") is None and store.corpus.meta.prisma.excluded_retracted == 1
    _check_prisma(store.corpus)
    params.exclude_retracted = False
    store = build_corpus(params, FakeLLM(), FakeOA(u), None, Tracer(), tmp_path / "d.json", log=lambda m: None)
    assert store.get("W6").retracted and store.corpus.stats["n_retracted"] == 1


def test_tiers_and_overrides():
    quick = params_for_tier("quick", "x", hops=2, threshold=None)
    assert quick.hops == 2 and quick.per_query == 30 and quick.threshold == 6.0
    assert params_for_tier("standard", "x") == BuildParams(prompt="x")
    assert params_for_tier("deep", "x").max_papers == 400
    with pytest.raises(ValueError):
        params_for_tier("huge", "x")
    with pytest.raises(ValueError):
        params_for_tier("quick", "x", not_a_param=1)


def test_agent_hard_cap_retraction_and_gap_tool(tmp_path):
    store = CorpusStore()
    seed = UNIVERSE["W5"].model_copy(deep=True)
    seed.is_seed, seed.origin, seed.relevance = True, "seed", Relevance(score=10)
    store.add(seed)
    u = _universe(W8=dict(retracted=True))
    script = [
        tool_msg(("top_missing_refs", '{"min_count": 1}')),
        tool_msg(("screen", '{"ids": ["W6", "W7", "W8"]}')),
        tool_msg(("add_to_corpus", '{"ids": ["W8", "W6", "W7"]}')),
        tool_msg(("finish", '{"summary": "ok"}')),
    ]
    llm = FakeLLM(tool_script=script)
    ctx = AgentContext(corpus=store, llm=llm, oa=FakeOA(u), wos=None, intent=parse_intent("x", llm),
                       tracer=Tracer(tmp_path / "t.jsonl"), max_papers=2)
    run_agent(ctx, "GNN", max_steps=10, log=lambda m: None)
    assert {p.id for p in store.papers} == {"W5", "W6"} and ctx.n_added == 1
    trace = {t["name"]: t["result"] for t in map(json.loads, (tmp_path / "t.jsonl").read_text().splitlines())}
    assert "rejected_corpus_full" in trace["add_to_corpus"] and "'W7'" in trace["add_to_corpus"]
    assert "rejected_retracted" in trace["add_to_corpus"] and "'W8'" in trace["add_to_corpus"]
    assert "cited_by_corpus" in trace["top_missing_refs"] and "W1" in trace["top_missing_refs"]


def test_screen_flags_missing_abstract():
    from bookworm.pipeline.screen import llm_screen

    papers = [Paper(id="W1", title="Graph models"), Paper(id="W2", title="Graph things", abstract="We study graphs.")]
    res = llm_screen(papers, parse_intent("x", FakeLLM()), FakeLLM())
    assert res["W1"].flag == "no_abstract" and res["W2"].flag is None


def test_strip_acronym_truncation():
    q = 'TS=("GNN*" OR "GAT*" OR "graph attention network*" OR "MPNN*") AND TS=("drug design*")'
    assert query.strip_acronym_truncation(q) == (
        'TS=("GNN" OR "GAT" OR "graph attention network*" OR "MPNN") AND TS=("drug design*")'
    )


def test_named_wos_branches_are_independent_and_year_bounded():
    branches = query.build_wos_branch_queries(
        ["vegetation_tls", "garden_syntax", "garden_reviews", "heritage_pointcloud"], (2018, 2025)
    )
    assert len(branches) == 4
    assert all("PY=(2018-2025)" in branch for branch in branches)
    assert any("tree structure" in branch and "TLS" in branch for branch in branches)
    assert any("space syntax" in branch and "wayfinding" in branch for branch in branches)
    assert any("bibliometric" in branch and "systematic review" in branch for branch in branches)
    assert any("point cloud*" in branch and "segmentation" in branch for branch in branches)
    with pytest.raises(ValueError, match="unknown WoS query branch"):
        query.build_wos_branch_queries(["not-a-branch"], (2018, 2025))


def test_build_enforces_year_range_during_search_and_snowball(tmp_path):
    params = BuildParams(
        prompt="GNN drug discovery", seeds=["W5"], source="openalex", years=(2019, 2023), hops=1,
        gap_fill=False,
    )
    store = build_corpus(params, FakeLLM(), FakeOA(), None, Tracer(), tmp_path / "years.json", log=lambda m: None)
    assert store.get("W5") is not None  # seeds remain even if their publication year is outside the range
    assert all(p.year is None or 2019 <= p.year <= 2023 for p in store.papers if not p.is_seed)
    assert store.corpus.meta.prisma.excluded_out_of_year > 0


def test_fixed_base_queries_bypass_llm_query_generation(tmp_path):
    llm, wos = FakeLLM(), FakeWos()
    params = BuildParams(prompt="GNN", source="wos", wos_queries=['TS=("GNN")'], hops=0, gap_fill=False)
    store = build_corpus(params, llm, FakeOA(), wos, Tracer(), tmp_path / "fixed.json", log=lambda m: None)
    assert "wos_query" not in llm.calls
    assert store.corpus.meta.queries == ['TS=("GNN")']



def test_frozen_intent_skips_intent_parsing(tmp_path):
    llm = FakeLLM()
    frozen = ResearchIntent(topic="frozen topic", core_concepts=["graph neural network"], year_range=(None, None))
    params = BuildParams(prompt="GNN", source="wos", wos_queries=['TS=("GNN")'], years=(2018, 2025),
                         hops=0, gap_fill=False)
    store = build_corpus(params, llm, FakeOA(), FakeWos(), Tracer(), tmp_path / "frozen.json",
                         log=lambda m: None, frozen_intent=frozen)
    assert "intent" not in llm.calls
    assert store.corpus.meta.intent.topic == "frozen topic"
    assert tuple(store.corpus.meta.intent.year_range) == (2018, 2025)
    assert frozen.year_range == (None, None)  # caller's object is not mutated


@pytest.mark.parametrize("source,branches", [("openalex", ["vegetation_tls"]), ("wos", ["unknown"])])
def test_invalid_branch_configuration_fails_before_api_calls(tmp_path, source, branches):
    llm, oa = FakeLLM(), FakeOA()
    with pytest.raises(ValueError):
        build_corpus(BuildParams(prompt="x", source=source, query_branches=branches),
                     llm, oa, FakeWos(), Tracer(), tmp_path / "invalid.json")
    assert not llm.calls and oa.requests == 0
