from bookworm.graph import build_graph, path_to, traverse
from bookworm.models import Corpus, Edge, Paper
from bookworm.store import CorpusStore, normalize_doi


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalize_doi("doi:10.1/x") == "10.1/x"
    assert normalize_doi("") is None


def test_dedup_by_doi_and_upgrade_wos_id():
    s = CorpusStore()
    s.add(Paper(id="WOS:001", wos_uid="WOS:001", doi="10.1/a", title="A", wos_times_cited=5))
    merged = s.add(Paper(id="W10", doi="https://doi.org/10.1/A", title="A", abstract="abs", origin="backward"))
    assert len(s) == 1
    assert merged.id == "W10"  # upgraded to the OpenAlex id
    assert "W10" in s and "WOS:001" not in s
    assert merged.abstract == "abs" and merged.wos_times_cited == 5
    assert merged.origin == "wos_search"  # higher-priority origin is kept


def test_retain_drops_dangling_edges(small_corpus, tmp_path):
    s = CorpusStore(small_corpus)
    s.retain({"W5", "W6", "W1"})
    assert {p.id for p in s.papers} == {"W5", "W6", "W1"}
    assert all(e.source in s and e.target in s for e in s.corpus.edges)
    path = s.save(tmp_path / "c.json")
    loaded = CorpusStore.load(path).finalize()
    assert loaded.stats["n_papers"] == 3


def test_bfs_layers_and_direction(small_corpus):
    g = build_graph(small_corpus)
    v = traverse(g, ["W5"], mode="bfs", direction="refs", max_depth=1)
    assert v[0].id == "W5" and v[0].depth == 0
    assert {x.id for x in v[1:]} == {"W1", "W2", "W4"}  # W9 is not in the corpus
    assert all(x.depth == 1 and x.via == "refs" for x in v[1:])
    down = traverse(g, ["W5"], mode="bfs", direction="cited_by", max_depth=5)
    assert [x.id for x in down] == ["W5", "W6", "W7", "W8"]


def test_dfs_goes_deep_first(small_corpus):
    g = build_graph(small_corpus)
    v = traverse(g, ["W8"], mode="dfs", direction="refs", max_depth=10)
    ids = [x.id for x in v]
    # From W8, the higher-priority neighbor goes deep first (W4 has more citations than W7)
    assert ids[0] == "W8"
    assert ids[1] == "W4"
    assert ids[2] == "W1"  # W4's reference, before returning to W7
    assert path_to(v, "W2")[0] == "W8"
    assert len(set(ids)) == len(ids)


def test_traverse_handles_cycles_and_limit():
    papers = [Paper(id=i, title=i) for i in "ABC"]
    c = Corpus(papers=papers, edges=[Edge(source="A", target="B"), Edge(source="B", target="C"), Edge(source="C", target="A")])
    g = build_graph(c)
    for mode in ("bfs", "dfs"):
        assert sorted(x.id for x in traverse(g, ["A"], mode=mode, direction="both")) == ["A", "B", "C"]
    assert len(traverse(g, ["A"], limit=2)) == 2
