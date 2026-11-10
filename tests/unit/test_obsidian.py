import json
import shutil

import pytest
import yaml

from bookworm.models import EdgeRelation, Paper, Prisma
from bookworm.obsidian import (BASE, CANVAS, NOTES_MARKER, OVERVIEW, RELATION_COLORS, export_vault,
                                 graph_color_groups, network_positions, note_names, separate_bounds)
from bookworm.store import CorpusStore


def _corpus(small_corpus):
    c = CorpusStore(small_corpus).finalize()
    c.edges[0].relation = EdgeRelation(label="extends", rationale="builds on it")
    c.meta.prompt = "GNN 药物发现"
    c.meta.prisma = Prisma(identified={"seed": 1, "openalex_search": 6}, screened=6, included=7)
    return c


def _frontmatter(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("---\n", 2)[1])


def test_vault_layout_and_properties(small_corpus, tmp_path):
    c = _corpus(small_corpus)
    rep = export_vault(c, tmp_path / "Bookworm" / "run1", vault_root=tmp_path)
    out = tmp_path / "Bookworm" / "run1"
    notes = sorted((out / "papers").glob("*.md"))
    assert rep.notes_written == len(notes) == 7 and (out / OVERVIEW).exists()

    e = c.edges[0]
    src = next(p for p in c.papers if p.id == e.source)
    fm = _frontmatter(out / "papers" / f"{note_names(c.papers)[src.id]}.md")
    assert fm["paper_id"] == src.id and fm["bookworm"] is True and fm["year"] == src.year
    assert "bookworm/paper" in fm["tags"]
    # Links carry the vault-relative path, so runs sharing one vault never collide
    assert all(link.startswith("[[Bookworm/run1/papers/") for link in fm["cites"])
    assert len(fm["extends"]) == 1 and fm["extends"][0] in fm["cites"]

    base = yaml.safe_load((out / BASE).read_text(encoding="utf-8"))
    assert base["filters"]["and"][0] == 'file.inFolder("Bookworm/run1/papers")'
    assert [v["name"] for v in base["views"]][0] == "全部文献"
    assert all(v["type"] == "table" and "file.name" in v["order"] for v in base["views"])

    overview = (out / OVERVIEW).read_text(encoding="utf-8")
    assert "```mermaid" in overview and "[[Bookworm/run1/引用图谱.canvas]]" in overview


def _disjoint(a, b, gap=0):
    return (a["x"] + a["width"] + gap <= b["x"] or b["x"] + b["width"] + gap <= a["x"]
            or a["y"] + a["height"] + gap <= b["y"] or b["y"] + b["height"] + gap <= a["y"])


def _check_canvas(c, path):
    canvas = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in canvas["nodes"]}
    assert len(nodes) == len(canvas["nodes"])  # unique ids
    for n in canvas["nodes"]:
        assert {"id", "type", "x", "y", "width", "height"} <= n.keys()
        assert all(isinstance(n[k], int) for k in ("x", "y", "width", "height"))
    cards = [n for n in canvas["nodes"] if n["type"] == "text" and "[[" in n["text"]]
    assert len(cards) == len(c.papers)
    assert all(_disjoint(a, b) for i, a in enumerate(cards) for b in cards[i + 1:])
    legend = next(n for n in canvas["nodes"] if n["type"] == "group")
    assert legend["label"] == "图例" and canvas["nodes"].index(legend) == 0  # groups render beneath cards
    assert all(_disjoint(legend, card) for card in cards)
    assert len(canvas["edges"]) == len(c.edges)
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    for ed in canvas["edges"]:
        src, dst = nodes[ed["fromNode"]], nodes[ed["toNode"]]
        assert ed["toEnd"] == "arrow" and opposite[ed["fromSide"]] == ed["toSide"]
        # Edges leave from the side facing the other card
        if ed["fromSide"] == "right":
            assert src["x"] <= dst["x"]
        elif ed["fromSide"] == "bottom":
            assert src["y"] <= dst["y"]
    labeled = [ed for ed in canvas["edges"] if "label" in ed]
    assert len(labeled) == 1 and labeled[0]["label"] == "扩展/改进" and labeled[0]["color"] == RELATION_COLORS["extends"]
    return canvas


@pytest.mark.skipif(not shutil.which("node"), reason="Node.js not installed")
def test_canvas_uses_renderer_network_layout(small_corpus, tmp_path):
    c = _corpus(small_corpus)
    rep = export_vault(c, tmp_path / "a")
    assert rep.canvas_layout == "fcose"
    canvas = _check_canvas(c, tmp_path / "a" / CANVAS)
    # Seeded layout: re-exporting the same corpus gives the same Canvas
    export_vault(c, tmp_path / "b")
    assert json.loads((tmp_path / "b" / CANVAS).read_text(encoding="utf-8")) == canvas
    # Years only appear in labels: changing them does not move any card
    shifted = c.model_copy(deep=True)
    for p in shifted.papers:
        p.year = None
    assert network_positions(shifted) == network_positions(c)


def test_canvas_falls_back_to_spring_layout_without_node(small_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr("bookworm.obsidian.shutil.which", lambda _: None)
    c = _corpus(small_corpus)
    rep = export_vault(c, tmp_path)
    assert rep.canvas_layout == "spring"
    _check_canvas(c, tmp_path / CANVAS)


def test_separate_bounds_handles_coincident_and_overlapping_cards():
    pos = separate_bounds({"a": (0, 0), "b": (0, 0), "c": (10, 5)}, 320, 150, 40)
    boxes = [{"x": x - 160, "y": y - 75, "width": 320, "height": 150} for x, y in pos.values()]
    assert all(_disjoint(a, b, gap=39) for i, a in enumerate(boxes) for b in boxes[i + 1:])


def test_graph_view_colour_groups_merge_into_existing_settings(small_corpus, tmp_path):
    cfg_path = tmp_path / ".obsidian" / "graph.json"
    cfg_path.parent.mkdir()
    mine = {"query": "path:journal", "color": {"a": 1, "rgb": 1}}
    stale = {"query": "tag:#bookworm/origin/gone", "color": {"a": 1, "rgb": 2}}
    cfg_path.write_text(json.dumps({"showTags": True, "colorGroups": [stale, mine]}), encoding="utf-8")
    rep = export_vault(_corpus(small_corpus), tmp_path / "Bookworm" / "run1", vault_root=tmp_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert rep.graph_colors == cfg_path and cfg["showTags"] is True and "showArrow" not in cfg
    assert cfg["colorGroups"] == graph_color_groups() + [mine]
    # Every paper note is matched by a colour group: queries are built from the same tags the notes carry
    # (graphify#2204: when the two drift apart, groups silently match nothing)
    ours = {g["query"].removeprefix("tag:#") for g in graph_color_groups()}
    for f in (tmp_path / "Bookworm" / "run1" / "papers").glob("*.md"):
        assert ours & set(_frontmatter(f)["tags"]), f.name

    export_vault(_corpus(small_corpus), tmp_path / "Bookworm" / "run1", vault_root=tmp_path)
    assert json.loads(cfg_path.read_text(encoding="utf-8")) == cfg  # idempotent

    cfg_path.write_text("{not json", encoding="utf-8")
    assert export_vault(_corpus(small_corpus), tmp_path / "x", vault_root=tmp_path).graph_colors is None
    assert cfg_path.read_text(encoding="utf-8") == "{not json"  # a config we cannot parse is left alone


def test_standalone_vault_gets_graph_config(small_corpus, tmp_path):
    export_vault(_corpus(small_corpus), tmp_path)
    cfg = json.loads((tmp_path / ".obsidian" / "graph.json").read_text(encoding="utf-8"))
    assert cfg["showArrow"] is True and cfg["colorGroups"] == graph_color_groups()


def test_reexport_keeps_user_notes_and_cleans_stale(small_corpus, tmp_path):
    c = _corpus(small_corpus)
    export_vault(c, tmp_path)
    names = note_names(c.papers)
    keep_id, drop_id = c.papers[1].id, c.papers[2].id
    kept = tmp_path / "papers" / f"{names[keep_id]}.md"
    kept.write_text(kept.read_text(encoding="utf-8") + "我的批注：方法很重要\n", encoding="utf-8")
    overview = tmp_path / OVERVIEW
    overview.write_text(overview.read_text(encoding="utf-8") + "综述提纲\n", encoding="utf-8")
    (tmp_path / "papers" / "随手记.md").write_text("not generated", encoding="utf-8")

    store = CorpusStore(c.model_copy(deep=True))
    store.remove(keep_id)
    store.remove(drop_id)
    rep = export_vault(store.finalize(), tmp_path)
    assert rep.stale_kept == [kept.name] and kept.exists() and "方法很重要" in kept.read_text(encoding="utf-8")
    assert rep.stale_removed == [f"{names[drop_id]}.md"]
    assert (tmp_path / "papers" / "随手记.md").exists()  # files we didn't generate are never touched
    assert "综述提纲" in overview.read_text(encoding="utf-8")

    # Re-adding the paper regenerates the note above the marker and keeps the user's text below it
    rep = export_vault(c, tmp_path)
    text = kept.read_text(encoding="utf-8")
    assert rep.notes_kept_user_text == 1 and text.count(NOTES_MARKER) == 1 and text.endswith("我的批注：方法很重要\n")


def test_note_names_are_safe_and_unique():
    ps = [Paper(id="W1", title='A/B: "test"? [x]#1', authors=["Ann Lee"], year=2020),
          Paper(id="W2", title='A/B: "test"? [x]#1', authors=["Ann Lee"], year=2020),
          Paper(id="W3", title="", authors=[], year=None)]
    names = note_names(ps)
    assert names["W1"] == "Lee 2020 - A B test x 1"
    assert names["W2"] == "Lee 2020 - A B test x 1 (W2)" and names["W3"] == "Anon n.d. - W3"
    assert not any(ch in n for n in names.values() for ch in '\\/:*?"<>|#^[]')
