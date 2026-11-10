"""Obsidian vault export: one note per paper + a Base (table views) + a Canvas (citation network) + an overview.

Layout of the output folder (open it as a vault, or write it into an existing vault with vault_root):
    总览.md            overview: need, intent, queries, PRISMA flow (mermaid), embedded Base, link to the Canvas
    文献库.base         Obsidian Bases table views over the paper notes' properties (Obsidian >= 1.9)
    引用图谱.canvas     JSON Canvas 1.0: fCoSE network layout (bookworm/layout/layout.cjs), edges
                       citing -> cited with relation labels/colours, plus a colour legend
    papers/*.md        properties (frontmatter) + abstract + typed citation links
    <vault>/.obsidian/graph.json   colour groups for the core Graph view (origin / seed / retracted tags), merged
                       into the vault's existing settings; only groups querying tag:#bookworm/ are ours

Links are written as vault-relative paths ([[sub/folder/papers/Name|Alias]]) so they stay unambiguous when several runs
share one vault. Everything after NOTES_MARKER in a note is user-owned and survives re-export.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .models import Corpus, Paper, Prisma

NOTES_MARKER = "%% bookworm:notes — 此行以下的内容在重新导出时保留 %%"
DEFAULT_NOTES = "\n## 我的笔记\n\n"
PAPERS_DIR = "papers"
OVERVIEW, BASE, CANVAS = "总览.md", "文献库.base", "引用图谱.canvas"

ORIGIN_NAMES = {
    "seed": "种子文献", "wos_search": "WoS 检索", "openalex_search": "OpenAlex 检索", "backward": "后向扩展（参考文献）",
    "forward": "前向扩展（施引文献）", "cocited": "共被引补缺", "agent": "Agent 添加",
}
RELATION_NAMES = {
    "extends": "扩展/改进", "uses_method": "使用方法", "uses_data": "使用数据", "compares": "对比", "critiques": "质疑",
    "background": "背景引用",
}
# Node colours: the former dev renderer's --o-* tokens, so the Canvas and the
# Graph view colour a paper the same way. Retracted papers use Canvas preset "1" (the theme's red).
ORIGIN_COLORS = {"seed": "#d99a00", "openalex_search": "#2f6fdb", "wos_search": "#2f6fdb", "backward": "#2a9d8f",
                 "forward": "#8e5bd6", "cocited": "#c2410c", "agent": "#e07a2f"}
RETRACTED_COLOR = "1"
RETRACTED_RGB = "#e03131"
RELATION_COLORS = {"extends": "#e0457b", "uses_method": "#2a9d8f", "uses_data": "#3a86ff", "compares": "#f4a261",
                   "critiques": "#d62828"}

# Canvas geometry (px)
NODE_W, NODE_H, H_GAP = 320, 150, 40
LEGEND_GAP = 160
# Layout: fCoSE network (bookworm/layout/layout.cjs) under Node.js, seeded so re-exports are stable
LAYOUT_JS = Path(__file__).resolve().parent / "layout" / "layout.cjs"
LAYOUT_SEED, LAYOUT_TIMEOUT = 42, 120
GRAPH_TAG_PREFIX = "tag:#bookworm/"


@dataclass
class VaultReport:
    out_dir: Path
    notes_written: int = 0
    notes_kept_user_text: int = 0  # existing notes whose user section was carried over
    stale_removed: list[str] = field(default_factory=list)
    stale_kept: list[str] = field(default_factory=list)  # no longer in the corpus but holding user notes
    canvas_layout: str = ""  # fcose / cose (layout.cjs under Node.js) or spring (networkx fallback)
    graph_colors: Path | None = None  # .obsidian/graph.json we merged colour groups into (None: left untouched)


# ---------- naming / links ----------
_BAD_CHARS = re.compile(r'[\\/:*?"<>|#^\[\]]+')


def _first_author_last(p: Paper) -> str:
    if not p.authors or not p.authors[0].strip():
        return "Anon"
    return p.authors[0].split()[-1]


def short_label(p: Paper) -> str:
    return f"{_first_author_last(p)} {p.year or 'n.d.'}"


def note_names(papers: list[Paper], maxlen: int = 90) -> dict[str, str]:
    """paper id -> note basename (no extension), unique within the export."""
    names: dict[str, str] = {}
    used: set[str] = set()
    for p in papers:
        base = _BAD_CHARS.sub(" ", f"{short_label(p)} - {p.title or p.id}")
        base = re.sub(r"\s+", " ", base).strip(" .")[:maxlen].rstrip(" .")
        name = base if base.lower() not in used else f"{base} ({p.id})"
        used.add(name.lower())
        names[p.id] = name
    return names


def _yaml_str(s: str) -> str:
    # JSON string literals are valid YAML double-quoted scalars
    return json.dumps(s, ensure_ascii=False)


def _yaml_sq(s: str) -> str:
    """YAML single-quoted scalar: the form Bases filters use, since expressions contain double quotes."""
    return "'" + s.replace("'", "''") + "'"


def _yaml_list(key: str, items: list[str]) -> list[str]:
    if not items:
        return []
    return [f"{key}:"] + [f"  - {_yaml_str(i)}" for i in items]


# ---------- export ----------
def export_vault(corpus: Corpus, out_dir: Path, vault_root: Path | None = None) -> VaultReport:
    out_dir = Path(out_dir)
    vault_root = Path(vault_root) if vault_root else out_dir
    rel = out_dir.resolve().relative_to(vault_root.resolve()).as_posix()
    prefix = "" if rel == "." else rel + "/"
    papers_dir = out_dir / PAPERS_DIR
    papers_dir.mkdir(parents=True, exist_ok=True)

    names = note_names(corpus.papers)
    by_id = {p.id: p for p in corpus.papers}

    def link(pid: str) -> str:
        return f"[[{prefix}{PAPERS_DIR}/{names[pid]}|{short_label(by_id[pid])}]]"

    out_edges: dict[str, list] = defaultdict(list)
    indeg: Counter[str] = Counter()
    for e in corpus.edges:
        if e.source in by_id and e.target in by_id:
            out_edges[e.source].append(e)
            indeg[e.target] += 1

    report = VaultReport(out_dir=out_dir)
    for p in corpus.papers:
        path = papers_dir / f"{names[p.id]}.md"
        user_text = _user_section(path)
        if user_text is not None and user_text.strip() not in ("", DEFAULT_NOTES.strip()):
            report.notes_kept_user_text += 1
        path.write_text(_paper_note(p, out_edges[p.id], indeg[p.id], link) + (user_text or DEFAULT_NOTES),
                        encoding="utf-8")
        report.notes_written += 1

    current = {f"{n}.md" for n in names.values()}
    for f in sorted(papers_dir.glob("*.md")):
        if f.name in current or not _is_generated(f):
            continue
        user_text = _user_section(f) or ""
        if user_text.strip() in ("", DEFAULT_NOTES.strip()):
            f.unlink()
            report.stale_removed.append(f.name)
        else:
            report.stale_kept.append(f.name)

    (out_dir / BASE).write_text(_base_file(prefix), encoding="utf-8")
    canvas, report.canvas_layout = _canvas(corpus, names, indeg, prefix)
    (out_dir / CANVAS).write_text(json.dumps(canvas, ensure_ascii=False, indent=1), encoding="utf-8")
    report.graph_colors = write_graph_colors(vault_root)
    overview = out_dir / OVERVIEW
    overview_user = _user_section(overview)
    overview.write_text(_overview(corpus, indeg, link, prefix) + (overview_user or "\n## 笔记\n\n"), encoding="utf-8")
    return report


def _user_section(path: Path) -> str | None:
    """Text after NOTES_MARKER (including the leading newline), or None if the file/marker is absent."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for marker in (NOTES_MARKER, NOTES_MARKER.replace("bookworm", "scibooster")):
        i = text.find(marker)
        if i >= 0:
            return text[i + len(marker):]
    return None


def _is_generated(path: Path) -> bool:
    head = path.read_text(encoding="utf-8")[:400]
    return head.startswith("---\n") and any(marker in head for marker in ("\nbookworm: true\n", "\nscibooster: true\n"))


# ---------- paper note ----------
def _paper_note(p: Paper, edges: list, corpus_cites: int, link) -> str:
    r = p.relevance
    fm = ["---", "bookworm: true", f"title: {_yaml_str(p.title)}"]
    fm += _yaml_list("authors", p.authors)
    if p.year:
        fm.append(f"year: {p.year}")
    if p.venue:
        fm.append(f"venue: {_yaml_str(p.venue)}")
    if p.doi:
        fm += [f"doi: {_yaml_str(p.doi)}", f"url: {_yaml_str('https://doi.org/' + p.doi)}"]
    fm.append(f"paper_id: {_yaml_str(p.id)}")
    if p.versions:
        fm.append(f"version_count: {len(p.versions)}")
        fm += _yaml_list("version_dois", list(dict.fromkeys(v.doi for v in p.versions if v.doi)))
    if r:
        fm += [f"relevance: {r.score:g}", f"relevance_reason: {_yaml_str(r.reason)}"]
        if r.flag:
            fm.append(f"relevance_flag: {_yaml_str(r.flag)}")
    fm += [f"origin: {p.origin}", f"hop: {p.hop}", f"seed: {str(p.is_seed).lower()}",
           f"retracted: {str(p.retracted).lower()}"]
    if p.cited_by_count is not None:
        fm.append(f"cited_by_count: {p.cited_by_count}")
    if p.wos_times_cited is not None:
        fm.append(f"wos_times_cited: {p.wos_times_cited}")
    fm.append(f"corpus_cites: {corpus_cites}")
    tags = ["bookworm/paper", f"bookworm/origin/{p.origin}"]
    if p.is_seed:
        tags.append("bookworm/seed")
    if p.retracted:
        tags.append("bookworm/retracted")
    fm += _yaml_list("tags", tags)
    fm += _yaml_list("cites", [link(e.target) for e in edges])
    # Typed citation links (one list property per relation label) for Bases columns and typed-link graph plugins
    typed: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.relation:
            typed[e.relation.label].append(link(e.target))
    for label in RELATION_NAMES:
        fm += _yaml_list(label, typed.get(label, []))
    fm.append("---")

    body = [f"# {p.title or p.id}", ""]
    if p.retracted:
        body += ["> [!danger] 已撤稿", "> OpenAlex 将这篇文献标记为已撤稿。", ""]
    meta = ", ".join(p.authors[:8]) + (" 等" if len(p.authors) > 8 else "")
    links = []
    if p.doi:
        links.append(f"[DOI](https://doi.org/{p.doi})")
    if re.fullmatch(r"W\d+", p.id):
        links.append(f"[OpenAlex](https://openalex.org/{p.id})")
    body += [meta, f"*{p.venue or '—'}* · {p.year or 'n.d.'}" + (" · " + " · ".join(links) if links else ""), ""]
    if r:
        flag = "（无摘要，仅凭标题评分）" if r.flag == "no_abstract" else ""
        body += [f"**相关度 {r.score:g}/10**{flag}：{r.reason}", ""]
    body += ["> [!abstract] 摘要"]
    body += [f"> {line}" if line else ">" for line in (p.abstract or "（无摘要）").splitlines()]
    body.append("")
    if p.versions:
        body += ["## 文献版本", "", "以下原始记录合并展示；引用图汇总各版本引用，相关度取已记录评分的最高值。", ""]
        for v in p.versions:
            url = f"[DOI](https://doi.org/{v.doi})" if v.doi else v.id
            body.append(f"- {v.year or 'n.d.'} · {v.title} · {v.venue or '—'} · {url}（{v.id}）")
        body.append("")
    if edges:
        body += [f"## 引用了库内 {len(edges)} 篇", ""]
        for e in edges:
            rel = f" — {RELATION_NAMES.get(e.relation.label, e.relation.label)}：{e.relation.rationale}" if e.relation else ""
            version_note = "（含版本映射；原始引用见 corpus.json）" if e.provenance == "openalex_version_mapped" else ""
            body.append(f"- {link(e.target)}{rel}{version_note}")
        body.append("")
    body.append(f"被库内 {corpus_cites} 篇引用（见反向链接面板）；外部另有 {p.external_cited_by} 次被引。")
    if p.notes:
        body += ["", "## 检索备注", ""] + [f"- {n}" for n in p.notes]
    body += ["", NOTES_MARKER]
    return "\n".join(fm) + "\n\n" + "\n".join(body)


# ---------- Base ----------
def _base_file(prefix: str) -> str:
    folder = f"{prefix}{PAPERS_DIR}"
    cols = ["file.name", "year", "relevance", "origin", "corpus_cites", "cited_by_count", "venue", "seed"]
    order = "\n".join(f"      - {c}" for c in cols)
    sort = "    sort:\n      - property: relevance\n        direction: DESC\n      - property: corpus_cites\n        direction: DESC"

    def view(name: str, filt: str | None = None, extra_cols: tuple[str, ...] = ()) -> str:
        v = ["  - type: table", f"    name: {_yaml_str(name)}"]
        if filt:
            v.append(f"    filters: {filt}")
        v.append("    order:")
        v.append(order + "".join(f"\n      - {c}" for c in extra_cols))
        v.append(sort)
        return "\n".join(v)

    return "\n".join([
        "filters:",
        "  and:",
        "    - " + _yaml_sq(f"file.inFolder({json.dumps(folder, ensure_ascii=False)})"),
        "    - " + _yaml_sq('file.ext == "md"'),
        "properties:",
        "  relevance:\n    displayName: 相关度",
        "  origin:\n    displayName: 来源",
        "  corpus_cites:\n    displayName: 库内被引",
        "  cited_by_count:\n    displayName: 总被引",
        "  year:\n    displayName: 年份",
        "  venue:\n    displayName: 期刊/会议",
        "  seed:\n    displayName: 种子",
        "  relevance_reason:\n    displayName: 相关性理由",
        "views:",
        view("全部文献"),
        view("核心（相关度 ≥ 8）", _yaml_sq("relevance >= 8"), ("relevance_reason",)),
        view("共被引补缺", _yaml_sq('origin == "cocited"'), ("relevance_reason",)),
        view("仅凭标题评分", _yaml_sq('relevance_flag == "no_abstract"'), ("relevance_reason",)),
        view("已撤稿", _yaml_sq("retracted == true")),
    ]) + "\n"


# ---------- Canvas ----------
def _cid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _run_layout_js(payload: dict) -> tuple[dict[str, tuple[float, float]], str] | None:
    """Card centres from the fCoSE layout (bookworm/layout/layout.cjs under Node); None if unavailable."""
    node = shutil.which("node")
    if not node or not LAYOUT_JS.exists():
        return None
    try:
        r = subprocess.run([node, str(LAYOUT_JS)], input=json.dumps(payload), capture_output=True, text=True,
                           timeout=LAYOUT_TIMEOUT, check=True)
        out = json.loads(r.stdout)
        pos = {k: (float(v["x"]), float(v["y"])) for k, v in out["positions"].items()}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    if set(pos) != {n["id"] for n in payload["nodes"]} or not all(map(math.isfinite, (c for xy in pos.values() for c in xy))):
        return None
    return pos, out.get("layout", "fcose")


def separate_bounds(pos: dict[str, tuple[float, float]], w: float, h: float, gutter: float) -> dict:
    """Port of NetworkLayout's separateBounds for equal-sized cards: the smallest uniform scale that leaves a gutter
    between every pair of boxes (angles, and so edge crossings, are preserved)."""
    seen: set[tuple[float, float]] = set()
    pos = dict(pos)
    for k, (x, y) in pos.items():
        while (x, y) in seen:
            x += w + gutter
        seen.add((x, y))
        pos[k] = (x, y)
    pts = list(pos.values())
    scale = 1.0
    for i, (ax, ay) in enumerate(pts):
        for bx, by in pts[i + 1:]:
            dx, dy = abs(bx - ax), abs(by - ay)
            sx = (w + gutter) / dx if dx else math.inf
            sy = (h + gutter) / dy if dy else math.inf
            scale = max(scale, min(sx, sy))
    return {k: (x * scale, y * scale) for k, (x, y) in pos.items()}


def network_positions(corpus: Corpus, spacing: float = 1.0) -> tuple[dict[str, tuple[float, float]], str]:
    """Card centres for the citation network, plus the engine used ("fcose" / "cose" via Node, else "spring").

    fCoSE network layout: positions come from citation links only, years are not a constraint.
    Without Node.js, networkx's Fruchterman-Reingold spring layout + the same bounding-box separation stand in.
    """
    ids = [p.id for p in corpus.papers]
    known = set(ids)
    edges = [(e.source, e.target) for e in corpus.edges
             if e.source in known and e.target in known and e.source != e.target]
    if not ids:
        return {}, "none"
    payload = {"nodes": [{"id": i, "width": NODE_W, "height": NODE_H} for i in ids],
               "edges": [{"source": s, "target": t} for s, t in edges], "spacing": spacing, "seed": LAYOUT_SEED}
    got = _run_layout_js(payload)
    if got:
        return got
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(ids)
    g.add_edges_from(edges)
    raw = nx.spring_layout(g, seed=LAYOUT_SEED, iterations=200)
    pos = {k: (float(x), float(y)) for k, (x, y) in raw.items()}
    return separate_bounds(pos, NODE_W, NODE_H, 2 * H_GAP * spacing), "spring"


def _sides(a: tuple[float, float], b: tuple[float, float]) -> tuple[str, str]:
    """Facing sides of two cards, judged against the card's aspect ratio so edges leave from the nearest side."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) * NODE_H >= abs(dy) * NODE_W:
        return ("right", "left") if dx >= 0 else ("left", "right")
    return ("bottom", "top") if dy >= 0 else ("top", "bottom")


def _canvas(corpus: Corpus, names: dict[str, str], indeg: Counter, prefix: str) -> tuple[dict, str]:
    by_id = {p.id: p for p in corpus.papers}
    centres, engine = network_positions(corpus)
    # Top-left corners on an integer grid, shifted so the network starts at (0, 0)
    x0 = min((x for x, _ in centres.values()), default=0) - NODE_W / 2
    y0 = min((y for _, y in centres.values()), default=0) - NODE_H / 2
    corner = {k: (round(x - NODE_W / 2 - x0), round(y - NODE_H / 2 - y0)) for k, (x, y) in centres.items()}
    cards = [_card(p, names[p.id], indeg[p.id], prefix, *corner[p.id]) for p in corpus.papers]
    edges = []
    for e in corpus.edges:
        if e.source not in by_id or e.target not in by_id or e.source == e.target:
            continue
        from_side, to_side = _sides(centres[e.source], centres[e.target])
        ed = {"id": _cid("edge", e.source, e.target), "fromNode": _cid("paper", e.source), "fromSide": from_side,
              "toNode": _cid("paper", e.target), "toSide": to_side, "toEnd": "arrow"}
        if e.relation and e.relation.label != "background":
            ed["label"] = RELATION_NAMES[e.relation.label]
            ed["color"] = RELATION_COLORS[e.relation.label]
        edges.append(ed)
    # Groups first so they render beneath the cards
    return {"nodes": _legend(corpus) + cards, "edges": edges}, engine


def _legend(corpus: Corpus) -> list[dict]:
    """Colour key in a group to the left of the network: one swatch card per origin / relation present."""
    if not corpus.papers:
        return []
    origins = list(dict.fromkeys(p.origin for p in corpus.papers if p.origin in ORIGIN_COLORS))
    rels = list(dict.fromkeys(e.relation.label for e in corpus.edges
                              if e.relation and e.relation.label in RELATION_COLORS))
    items = [(ORIGIN_NAMES.get(o, o), ORIGIN_COLORS[o]) for o in origins]
    if any(p.retracted for p in corpus.papers):
        items.append(("已撤稿", RETRACTED_COLOR))
    items += [(f"→ {RELATION_NAMES[r]}（引用边）", RELATION_COLORS[r]) for r in rels]
    w, h, gap, pad = 240, 50, 12, 30
    head = 110
    x = -(w + 2 * pad) - LEGEND_GAP
    nodes = [{"id": _cid("legend"), "type": "group", "label": "图例", "x": x, "y": 0, "width": w + 2 * pad,
              "height": head + len(items) * (h + gap) + pad}]
    nodes.append({"id": _cid("legend", "note"), "type": "text", "x": x + pad, "y": pad, "width": w, "height": head - pad,
                  "text": "箭头：施引 → 被引\n位置按引用关系排布，年份见卡片\n★ 种子文献"})
    for i, (label, color) in enumerate(items):
        nodes.append({"id": _cid("legend", label), "type": "text", "x": x + pad, "y": head + i * (h + gap),
                      "width": w, "height": h, "text": label, "color": color})
    return nodes


def _card(p: Paper, name: str, corpus_cites: int, prefix: str, x: int, y: int) -> dict:
    star = " ★" if p.is_seed else ""
    warn = " ⚠️已撤稿" if p.retracted else ""
    title = p.title if len(p.title) <= 110 else p.title[:107] + "…"
    stats = [f"相关度 {p.relevance.score:g}" if p.relevance else None,
             f"被引 {p.cited_by_count}" if p.cited_by_count is not None else None, f"库内 {corpus_cites}"]
    text = (f"**[[{prefix}{PAPERS_DIR}/{name}|{short_label(p)}]]**{star}{warn}\n{title}\n\n"
            + " · ".join(s for s in stats if s))
    node = {"id": _cid("paper", p.id), "type": "text", "x": x, "y": y, "width": NODE_W, "height": NODE_H, "text": text}
    color = RETRACTED_COLOR if p.retracted else ORIGIN_COLORS.get(p.origin)
    if color:
        node["color"] = color
    return node


# ---------- Graph view ----------
def _rgb(color: str) -> int:
    return int(color.lstrip("#"), 16)


def graph_color_groups() -> list[dict]:
    """Core Graph view colour groups, first match wins: retracted, then seed, then each origin."""
    groups = [("bookworm/retracted", RETRACTED_RGB), ("bookworm/seed", ORIGIN_COLORS["seed"])]
    groups += [(f"bookworm/origin/{o}", c) for o, c in ORIGIN_COLORS.items() if o != "seed"]
    return [{"query": f"tag:#{tag}", "color": {"a": 1, "rgb": _rgb(c)}} for tag, c in groups]


def write_graph_colors(vault_root: Path) -> Path | None:
    """Merge our colour groups into <vault>/.obsidian/graph.json, keeping every other setting and the user's groups.

    Groups whose query starts with tag:#bookworm/ are replaced; a fresh config also turns on arrows (citations are
    directed). An unreadable graph.json is left alone. Obsidian reads the file when it opens the vault.
    """
    path = Path(vault_root) / ".obsidian" / "graph.json"
    cfg: dict = {"showArrow": True}
    if path.exists():
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(cfg, dict) or not isinstance(cfg.get("colorGroups", []), list):
            return None
    theirs = [g for g in cfg.get("colorGroups", [])
              if not (isinstance(g, dict) and str(g.get("query", "")).startswith((GRAPH_TAG_PREFIX, "tag:#scibooster/")))]
    cfg["colorGroups"] = graph_color_groups() + theirs
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------- overview ----------
def _prisma_mermaid(pr: Prisma) -> list[str]:
    ident = " · ".join(f"{ORIGIN_NAMES.get(k, k)} {n}" for k, n in pr.identified.items())
    excl = pr.excluded_low_relevance + pr.excluded_retracted + pr.excluded_over_cap
    q = lambda s: s.replace('"', "'")  # noqa: E731 - mermaid labels are double-quoted
    lines = ["```mermaid", "flowchart TD",
             f'  A["识别（去重后）{sum(pr.identified.values())}<br>{q(ident)}"] --> B["LLM 筛选 {pr.screened}"]',
             f'  A --> N["BM25 预筛未送审 {pr.not_screened}"]',
             f'  B --> X["排除 {excl}<br>相关度不足 {pr.excluded_low_relevance} · 已撤稿 {pr.excluded_retracted}'
             f' · 超出规模上限 {pr.excluded_over_cap}"]',
             f'  B --> I["纳入 {pr.included}' + (f'<br>含 Agent 新增 {pr.agent_added}' if pr.agent_added else "") + '"]',
             "```"]
    if pr.hops:
        lines += [""] + [f"- 第 {h.hop} 跳：新增 {h.candidates}，送筛 {h.screened}，相关 {h.relevant}" for h in pr.hops]
    if pr.stop_reason:
        lines.append(f"- 滚雪球提前停止：{pr.stop_reason}")
    return lines


def _overview(corpus: Corpus, indeg: Counter, link, prefix: str) -> str:
    m, st = corpus.meta, corpus.stats
    it = m.intent
    lines = ["---", "bookworm: true", "tags:", "  - bookworm/overview", "---", "",
             f"# {m.prompt or '文献图谱'}", ""]
    lines += [f"{st.get('n_papers', len(corpus.papers))} 篇文献（种子 {st.get('n_seeds', 0)}）· "
              f"{st.get('n_edges', len(corpus.edges))} 条库内引用 · 年份 {st.get('year_span')}", "",
              f"- 引用图谱（网状布局，箭头指向被引文献）：[[{prefix}{CANVAS}]]；"
              "也可打开 Graph view，节点按来源着色",
              f"- 文献表：[[{prefix}{BASE}]]（需要 Obsidian 1.9 及以上）", ""]
    if it:
        lines += ["## 研究意图", "", it.topic, ""] + [f"- {q}" for q in it.research_questions] + [""]
    if m.queries:
        lines += ["## 检索式", ""] + [f"- `{q}`" for q in m.queries] + [""]
    if m.summary:
        lines += ["## Agent 总结", "", m.summary, ""]
    if m.prisma:
        lines += ["## 筛选流程（PRISMA）", ""] + _prisma_mermaid(m.prisma) + [""]
    top = sorted(corpus.papers, key=lambda p: -indeg[p.id])[:10]
    if top and indeg[top[0].id]:
        lines += ["## 库内被引最多", ""]
        lines += [f"{i}. {link(p.id)} {p.title} — 库内被引 {indeg[p.id]}" for i, p in enumerate(top, 1) if indeg[p.id]]
        lines.append("")
    lines += ["## 文献表", "", f"![[{prefix}{BASE}#全部文献]]", ""]
    u = m.usage
    lines += [f"*生成于 {m.created_at} · DeepSeek {u.deepseek_calls} 次调用 / "
              f"{u.deepseek_prompt_tokens + u.deepseek_completion_tokens} tokens · OpenAlex {u.openalex_requests} 次请求*",
              "", NOTES_MARKER]
    return "\n".join(lines)
