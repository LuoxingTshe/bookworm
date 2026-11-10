"""Application layer: one function per task, each returning a JSON-serialisable result model.

The CLI is a thin shell over these functions; scripts and calling agents can import them directly.
Progress goes to the `log` callback, failures raise `errors.BookwormError` subclasses, and nothing is
printed here, so results can be rendered for people or dumped as JSON for another agent.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .config import Settings, get_settings, resolve_source
from .errors import ConfigError, InputError
from .evaluate import EvalReport
from .models import Corpus, Paper, Prisma, ResearchIntent, Usage
from .store import CorpusStore
from .requests import ResearchRequest
from .persistence import atomic_write
from .diagnostics import describe_error

Log = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


# ---------- result models (the --json contract; `bookworm schema` prints them) ----------
class PaperBrief(BaseModel):
    id: str
    doi: str | None = None
    title: str = ""
    year: int | None = None
    venue: str | None = None
    first_author: str | None = None
    score: float | None = None
    origin: str = ""
    is_seed: bool = False
    corpus_cites: int = 0  # cited by this many papers inside the corpus
    cited_by_count: int | None = None


class CorpusSummary(BaseModel):
    """Compact digest of a corpus, small enough to hand back to a calling agent instead of corpus.json."""

    topic: str | None = None
    n_papers: int = 0
    n_seeds: int = 0
    n_edges: int = 0
    n_labeled_edges: int = 0
    n_retracted: int = 0
    year_span: list[int] | None = None
    by_origin: dict[str, int] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    prisma: Prisma | None = None
    agent_summary: str | None = None
    top_papers: list[PaperBrief] = Field(default_factory=list)  # most cited within the corpus, then by score


class ObsidianInfo(BaseModel):
    dir: str
    overview: str
    vault_root: str
    notes_written: int = 0
    notes_kept_user_text: int = 0
    stale_removed: list[str] = Field(default_factory=list)
    stale_kept: list[str] = Field(default_factory=list)
    canvas_layout: str = ""


class Result(BaseModel):
    ok: bool = True
    command: str
    warnings: list[str] = Field(default_factory=list)
    protocol_version: str = "1"


class RunResult(Result):
    """build / agent / deduplicate: one run directory holding corpus.json and its derived outputs."""

    complete: bool = True  # False: the agent stopped before `finish` or was interrupted; a partial corpus is saved
    run_dir: str
    corpus: str
    outputs: dict[str, str] = Field(default_factory=dict)  # name -> path (trace, bibtex, report, ...)
    obsidian: ObsidianInfo | None = None
    summary: CorpusSummary
    usage: Usage | None = None
    cleanup: dict | None = None  # retention report; None when earlier results were deliberately preserved
    stop_reason: str | None = None
    error: dict[str, str] | None = None
    source: str | None = None


class DoctorResult(Result):
    ready: bool = False
    source: str = "openalex"
    version: str = ""
    capabilities: dict[str, bool] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)


class IntentResult(Result):
    intent: ResearchIntent
    source: str
    queries: list[str]
    seeds: list[PaperBrief] = Field(default_factory=list)
    usage: Usage


class StatsResult(Result):
    corpus: str
    summary: CorpusSummary


class GoldSurvey(PaperBrief):
    n_references: int = 0


class EvalResult(Result):
    corpus: str
    report_path: str
    gold: list[GoldSurvey]
    report: EvalReport
    missed_top: list[PaperBrief] = Field(default_factory=list)


class ExportResult(Result):
    corpus: str
    format: str
    path: str
    n_papers: int


class ObsidianResult(Result):
    corpus: str
    obsidian: ObsidianInfo


class TraverseVisit(BaseModel):
    order: int
    depth: int
    via: str | None
    parent: str | None
    id: str
    year: int | None = None
    title: str = ""


class TraverseResult(Result):
    corpus: str
    mode: str
    direction: str
    visits: list[TraverseVisit]


RESULT_MODELS: dict[str, type[Result]] = {
    "run": RunResult, "task": RunResult, "doctor": DoctorResult,
    "intent": IntentResult, "build": RunResult, "agent": RunResult, "deduplicate": RunResult,
    "stats": StatsResult, "eval": EvalResult, "export": ExportResult, "obsidian": ObsidianResult,
    "traverse": TraverseResult,
}


def result_schemas() -> dict:
    error = {"type": "object", "properties": {
        "ok": {"const": False}, "command": {"type": "string"},
        "error": {"type": "object", "properties": {"code": {"type": "string"}, "message": {"type": "string"}}},
    }}
    return {"protocol_version": "1", "request": ResearchRequest.model_json_schema(),
            "results": {name: model.model_json_schema() for name, model in RESULT_MODELS.items()}, "error": error}


# ---------- helpers ----------
def parse_years(s: str | None) -> tuple[int | None, int | None]:
    if not s:
        return (None, None)
    m = re.fullmatch(r"\s*(\d{4})?\s*-\s*(\d{4})?\s*", s)
    if not m:
        if re.fullmatch(r"\d{4}", s.strip()):
            if int(s) == 0:
                raise InputError("year must be between 1 and 9999")
            return (int(s), int(s))
        raise InputError("year range format: 2015-2025, 2015-, -2020, or 2020")
    years = (int(m.group(1)) if m.group(1) else None, int(m.group(2)) if m.group(2) else None)
    try:
        return ResearchRequest(prompt="validate", years=years).years
    except ValueError:
        raise InputError("year range must be ordered and between 1 and 9999") from None


def collect_seeds(seed: list[str], seed_file: Path | None) -> list[str]:
    from .pipeline.seeds import read_seed_file

    out = list(seed or [])
    if seed_file:
        out += read_seed_file(seed_file)
    return list(dict.fromkeys(out))


def check_config(settings: Settings, source: str) -> None:
    """Fail before creating a run directory when a needed key is missing."""
    if not settings.deepseek_api_key:
        raise ConfigError("DEEPSEEK_API_KEY is not set (see .env.example)")
    if source == "wos" and not settings.wos_api_key:
        raise ConfigError("WOS_API_KEY is not set. Configure .env, or use --source openalex")


def make_clients(settings: Settings, tracer, source: str, run=None):
    """(llm, openalex, wos-or-None). Raises ConfigError when a needed key is missing."""
    from .llm.deepseek import DeepSeek
    from .sources.cache import Cache
    from .sources.openalex import OpenAlexClient
    from .sources.wos import WosClient

    cache = Cache(settings.scib_cache_dir / "http.sqlite")
    if run is not None:
        run.cache(cache)
    llm = DeepSeek(settings, tracer)
    oa = OpenAlexClient(settings, cache, tracer)
    wos = None
    if settings.wos_api_key and source == "wos":
        wos = WosClient(settings, cache, tracer)
    elif source == "wos":
        raise ConfigError("WOS_API_KEY is not set. Configure .env, or use --source openalex")
    if run is not None:
        for client in (llm, oa, wos):
            if client is not None:
                run.client(client)
    return llm, oa, wos


def _openalex(settings: Settings):
    from .sources.cache import Cache
    from .sources.openalex import OpenAlexClient
    from .trace import Tracer

    return OpenAlexClient(settings, Cache(settings.scib_cache_dir / "http.sqlite"), Tracer())


def brief(p: Paper, corpus_cites: int = 0) -> PaperBrief:
    return PaperBrief(
        id=p.id, doi=p.doi, title=p.title, year=p.year, venue=p.venue,
        first_author=p.authors[0] if p.authors else None,
        score=p.relevance.score if p.relevance else None, origin=p.origin, is_seed=p.is_seed,
        corpus_cites=corpus_cites, cited_by_count=p.cited_by_count if p.cited_by_count is not None else p.wos_times_cited,
    )


def summarize(corpus: Corpus, top: int = 10) -> CorpusSummary:
    st = corpus.stats
    indeg = Counter(e.target for e in corpus.edges)
    ranked = sorted(corpus.papers, key=lambda p: (-indeg[p.id], -(p.relevance.score if p.relevance else 0)))
    m = corpus.meta
    return CorpusSummary(
        topic=m.intent.topic if m.intent else None,
        n_papers=st.get("n_papers", len(corpus.papers)), n_seeds=st.get("n_seeds", 0),
        n_edges=st.get("n_edges", len(corpus.edges)), n_labeled_edges=st.get("n_labeled_edges", 0),
        n_retracted=st.get("n_retracted", 0), year_span=st.get("year_span"), by_origin=st.get("by_origin", {}),
        queries=list(m.queries), prisma=m.prisma, agent_summary=m.summary,
        top_papers=[brief(p, indeg[p.id]) for p in ranked[:top]],
    )


def load_corpus(path: Path) -> Corpus:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"corpus not found: {path}")
    return CorpusStore.load(path).finalize()


def obsidian_target(corpus_path: Path, vault: Path | None = None, folder: str | None = None,
                    settings: Settings | None = None) -> tuple[Path, Path]:
    """(output dir, vault root). Default: <run>/obsidian as its own vault; with a vault: <vault>/Bookworm/<run>."""
    vault = vault or (settings or get_settings()).scib_obsidian_vault
    if not vault:
        out = corpus_path.parent / "obsidian"
        return out, out
    vault = vault.expanduser()
    return vault / (folder or f"Bookworm/{corpus_path.parent.name}"), vault


def write_obsidian(corpus: Corpus, corpus_path: Path, vault: Path | None = None, folder: str | None = None,
                   out_dir: Path | None = None) -> ObsidianInfo:
    from .obsidian import OVERVIEW, export_vault

    out, root = (out_dir, out_dir) if out_dir else obsidian_target(corpus_path, vault, folder)
    rep = export_vault(corpus, out, root)
    return ObsidianInfo(
        dir=str(out), overview=str(out / OVERVIEW), vault_root=str(root), notes_written=rep.notes_written,
        notes_kept_user_text=rep.notes_kept_user_text, stale_removed=rep.stale_removed, stale_kept=rep.stale_kept,
        canvas_layout=rep.canvas_layout,
    )


def _run_paths(settings: Settings, prompt: str, out: Path | None) -> tuple[Path, Path]:
    from .store import new_run_dir

    if out is not None:
        out = Path(out).expanduser()
        if out.suffix.lower() != ".json" or out.name in {"request.json", "result.json", "checkpoint.json", "cleanup.json"}:
            raise InputError("out must be a corpus .json file, not a reserved run metadata filename")
    run_dir = out.parent if out else new_run_dir(settings.scib_corpora_dir, prompt)
    return run_dir.resolve(), (out or run_dir / "corpus.json").resolve()


# ---------- tasks ----------
def doctor(source: str = "auto") -> DoctorResult:
    """Offline readiness check; never returns credential values or calls an API."""
    import shutil
    from . import __version__
    from .config import workspace

    settings = get_settings()
    selected = resolve_source(source, settings)
    ready = bool(settings.deepseek_api_key) and (selected != "wos" or bool(settings.wos_api_key))
    warnings = []
    if not settings.deepseek_api_key:
        warnings.append("DEEPSEEK_API_KEY is not configured")
    if selected == "wos" and not settings.wos_api_key:
        warnings.append("WOS_API_KEY is not configured")
    return DoctorResult(command="doctor", ready=ready, source=selected, version=__version__, warnings=warnings,
        capabilities={"llm": bool(settings.deepseek_api_key), "wos": bool(settings.wos_api_key),
                      "openalex_key": bool(settings.openalex_api_key), "node": shutil.which("node") is not None,
                      "offline_commands": True},
        paths={"workspace": str(workspace()), "runs": str(settings.scib_corpora_dir.resolve()),
               "cache": str(settings.scib_cache_dir.resolve())})


def _close_clients(llm, oa, wos):
    for client in (llm, oa, wos):
        if client is not None and hasattr(client, "close"):
            client.close()
    if hasattr(oa, "cache"):
        oa.cache.close()


def parse_intent(prompt: str, seeds: list[str] = (), years: tuple[int | None, int | None] = (None, None),
                 source: Literal["auto", "wos", "openalex"] = "auto", log: Log = _quiet) -> IntentResult:
    """Intent and candidate queries only, with optional seed resolution."""
    from .pipeline import intent as intent_mod, query
    from .pipeline.seeds import resolve_seeds
    from .trace import Tracer

    ResearchRequest(prompt=prompt, seeds=list(seeds), years=years, source=source)
    settings = get_settings()
    source = resolve_source(source, settings)
    check_config(settings, source)
    tracer = Tracer()
    llm, oa, wos = make_clients(settings, tracer, source)
    try:
        found, missing = resolve_seeds(list(seeds), oa, wos) if seeds else ([], [])
        warnings = [f"could not resolve seed: {m}" for m in missing]
        for warning in warnings:
            log(warning)
        it = intent_mod.parse_intent(prompt, llm, found, years)
        qs = query.build_wos_queries(it, llm) if source == "wos" else query.build_openalex_queries(it)
        return IntentResult(command="intent", intent=it, source=source, queries=qs,
                            seeds=[brief(p) for p in found], usage=tracer.usage, warnings=warnings)
    finally:
        _close_clients(llm, oa, wos)


def _save_run(store, out_path, tracer, run, *, command, source, top, obsidian, error=None):
    """Persist outputs before marking a run complete; every path returned is absolute."""
    from .export import EXTENSIONS, export as render
    from .pipeline.relations import build_edges

    meta = store.corpus.meta
    meta.warnings = list(dict.fromkeys([*meta.warnings, *tracer.warnings]))
    if tracer.warnings:
        meta.complete = False
        meta.stop_reason = meta.stop_reason if error else "upstream_incomplete"
    if not len(store):
        meta.warnings.append("No papers included; this is not evidence that no relevant literature exists")
    if any(p.relevance is None for p in store.papers):
        meta.warnings.append("Partial corpus contains unscored candidates; check relevance before citing")
    if any(p.year is None for p in store.papers):
        meta.warnings.append("Some papers have unknown publication years; verify them manually")
    store.set_edges(build_edges(store))
    store.save(out_path)
    outputs = {"trace": str(run.directory / "trace.jsonl"), "request": str(run.directory / "request.json"),
               "result": str(run.directory / "result.json"), "checkpoint": str(run.directory / "checkpoint.json")}
    for fmt, extension in EXTENSIONS.items():
        target = run.directory / f"corpus.{extension}"
        atomic_write(target, render(store.corpus.papers, fmt))
        outputs[fmt] = str(target)
    result = RunResult(command=command, complete=meta.complete, run_dir=str(run.directory), corpus=str(out_path),
                       source=source, outputs=outputs, summary=summarize(store.corpus, top),
                       usage=tracer.usage, stop_reason=meta.stop_reason, error=error, warnings=list(meta.warnings))
    if obsidian:
        try:
            result.obsidian = write_obsidian(store.corpus, out_path)
        except Exception as exc:
            code, message, _ = describe_error(exc)
            result.complete = meta.complete = False
            result.stop_reason = meta.stop_reason = "output_failed"
            result.error = {"code": code, "message": message}
            result.warnings.append("Corpus and reference exports saved; Obsidian output failed")
            meta.warnings = list(result.warnings)
            store.save(out_path)
    if not result.complete:
        result.warnings.append("partial result saved; previous results and cache preserved")
    atomic_write(run.directory / "result.json", result.model_dump_json(indent=2))
    if result.complete:
        result.cleanup = run.complete([out_path])
        result.warnings.extend(result.cleanup["errors"])
        atomic_write(run.directory / "result.json", result.model_dump_json(indent=2))
    return result


def build(params, *, out: Path | None = None, obsidian: bool = True, keep_history: bool = False,
          keep_cache: bool = False, top: int = 10, log: Log = _quiet) -> RunResult:
    """Run the bounded pipeline, preserving checkpoints and explicit partial results."""
    from dataclasses import asdict, replace
    from .pipeline.build import build_corpus
    from .runtime import RetrievalRun
    from .trace import Tracer

    settings = get_settings()
    params = replace(params, source=resolve_source(params.source, settings))
    if not 0 <= top <= 100:
        raise InputError("top must be between 0 and 100")
    check_config(settings, params.source)
    run_dir, out_path = _run_paths(settings, params.prompt, out)
    with RetrievalRun(settings, run_dir, kind="build", keep_history=keep_history, keep_cache=keep_cache) as run:
        tracer = Tracer(run_dir / "trace.jsonl")
        tracer.max_tokens = params.max_tokens
        tracer.log("start", command="build")
        atomic_write(run_dir / "request.json", json.dumps(asdict(params), ensure_ascii=False, indent=2))
        error = None
        try:
            llm, oa, wos = make_clients(settings, tracer, params.source, run)
            store = build_corpus(params, llm, oa, wos, tracer, out_path, log=log)
        except (Exception, KeyboardInterrupt) as exc:
            checkpoint = run_dir / "checkpoint.json"
            if not checkpoint.exists():
                raise
            store = CorpusStore.load(checkpoint)
            code, message, _ = describe_error(exc)
            error = {"code": code, "message": message}
            store.corpus.meta.complete = False
            store.corpus.meta.stop_reason = code
            store.corpus.meta.warnings.append(message)
            store.corpus.meta.usage = tracer.usage
            tracer.log("stopped", **error)
        return _save_run(store, out_path, tracer, run, command="build", source=params.source,
                         top=top, obsidian=obsidian, error=error)


def extend(prompt: str, *, corpus: Path | None = None, seeds: list[str] = (),
           years: tuple[int | None, int | None] = (None, None), threshold: float = 6.0, max_papers: int = 80,
           max_steps: int = 12, max_tokens: int = 100000, label_edges: bool = False, obsidian: bool = True,
           out: Path | None = None, keep_history: bool = False, keep_cache: bool = False, top: int = 10,
           source: str = "auto", log: Log = _quiet) -> RunResult:
    """Explore autonomously; continuation starts a fresh budget and never overwrites its input."""
    from .agent import loop
    from .agent.tools import AgentContext
    from .models import Relevance
    from .pipeline import intent as intent_mod, relations
    from .pipeline.seeds import resolve_seeds
    from .runtime import RetrievalRun
    from .trace import Tracer

    request = ResearchRequest(prompt=prompt, corpus=corpus, seeds=list(seeds), years=years, threshold=threshold,
                              max_papers=max_papers, max_steps=max_steps, max_tokens=max_tokens, out=out,
                              obsidian=obsidian, top=top, source=source)
    settings = get_settings()
    source = resolve_source(source, settings)
    check_config(settings, source)
    store = CorpusStore.load(corpus) if corpus else CorpusStore()
    if len(store) > max_papers:
        raise InputError(f"Input corpus already contains {len(store)} papers; increase max_papers")
    if corpus and out and Path(out).resolve() == Path(corpus).resolve():
        raise InputError("out must differ from the input corpus")
    run_dir, out_path = _run_paths(settings, prompt, out)
    with RetrievalRun(settings, run_dir, kind="agent", keep_history=keep_history, keep_cache=keep_cache) as run:
        tracer = Tracer(run_dir / "trace.jsonl")  # usage is per invocation, not inherited
        tracer.max_tokens = max_tokens
        tracer.log("start", command="agent", source=source)
        atomic_write(run_dir / "request.json", request.model_dump_json(indent=2))
        meta = store.corpus.meta
        if corpus:
            meta.params.setdefault("previous_runs", []).append({"corpus": str(Path(corpus).resolve()),
                                                               "usage": meta.usage.model_dump(), "prompt": meta.prompt})
        meta.usage = tracer.usage
        meta.complete, meta.stop_reason, meta.summary = False, "starting", None
        meta.warnings = []
        meta.prompt = prompt
        meta.seeds = list(dict.fromkeys(meta.seeds + list(seeds)))
        meta.params.setdefault("agent_runs", []).append(request.model_dump(mode="json"))
        checkpoint = lambda: store.save(run_dir / "checkpoint.json")
        checkpoint()
        ctx = None
        error = None
        try:
            llm, oa, wos = make_clients(settings, tracer, source, run)
            if seeds:
                found, missing = resolve_seeds(list(seeds), oa, wos)
                meta.warnings.extend(f"could not resolve seed: {m}" for m in missing)
                for paper in found:
                    if not store.has(paper) and len(store) >= max_papers:
                        raise InputError("Resolved seeds exceed max_papers")
                    paper.relevance = Relevance(score=10, reason="seed")
                    store.add(paper)
                checkpoint()
            # The new task, including an explicit new year range, drives continuation screening.
            meta.intent = intent_mod.parse_intent(prompt, llm, [p for p in store.papers if p.is_seed], years)
            checkpoint()
            log(f"Intent: {meta.intent.topic}")
            ctx = AgentContext(corpus=store, llm=llm, oa=oa, wos=wos, intent=meta.intent, tracer=tracer,
                               threshold=threshold, max_papers=max_papers, checkpoint=checkpoint)
            loop.run_agent(ctx, prompt, max_steps, max_tokens, log=log)
            meta.complete = ctx.finished
            meta.stop_reason = "finished" if ctx.finished else (ctx.stop_reason or "unfinished")
            if label_edges:
                edges = relations.build_edges(store)
                store.set_edges(edges)
                relations.label_edges(edges, store, llm, meta.intent.language)
        except (Exception, KeyboardInterrupt) as exc:
            code, message, _ = describe_error(exc)
            error = {"code": code, "message": message}
            meta.complete, meta.stop_reason = False, code
            meta.warnings.append(message)
            tracer.log("stopped", **error)
        if ctx is not None:
            meta.summary = ctx.summary or None
            if meta.prisma is not None:
                meta.prisma.agent_added += ctx.n_added
                meta.prisma.included = len(store)
        checkpoint()
        return _save_run(store, out_path, tracer, run, command="agent", source=source,
                         top=top, obsidian=obsidian, error=error)


def run(request: ResearchRequest | dict, *, log: Log = _quiet) -> RunResult:
    """The primary public API for people and delegating agents."""
    from .pipeline.build import params_for_tier

    request = ResearchRequest.model_validate(request)
    params = params_for_tier(request.tier, request.prompt, source=request.source, seeds=request.seeds,
                            years=request.years, max_papers=request.max_papers,
                            threshold=request.threshold, max_tokens=request.max_tokens)
    if request.mode == "pipeline":
        result = build(params, out=request.out, obsidian=request.obsidian, keep_cache=True, top=request.top, log=log)
    else:
        result = extend(request.prompt, corpus=request.corpus, seeds=request.seeds, years=request.years,
                        source=request.source, max_papers=params.max_papers, max_steps=request.max_steps,
                        max_tokens=request.max_tokens, threshold=request.threshold, out=request.out,
                        obsidian=request.obsidian, keep_cache=True, top=request.top, log=log)
    result.command = "run"
    atomic_write(Path(result.run_dir) / "request.json", request.model_dump_json(indent=2))
    atomic_write(Path(result.run_dir) / "result.json", result.model_dump_json(indent=2))
    return result


def deduplicate(corpus: Path, out: Path, *, obsidian: bool = True, top: int = 10) -> RunResult:
    """Merge well-supported publication versions offline into a new directory, preserving the source corpus."""
    from .dedup import deduplicate as filter_versions
    from .export import EXTENSIONS, export
    from .pipeline.relations import build_edges

    if out.exists():
        raise InputError("--out must be a new directory")
    if not Path(corpus).is_file():
        raise InputError(f"corpus not found: {corpus}")
    store = CorpusStore.load(corpus)
    report = filter_versions(store)
    store.set_edges(build_edges(store))
    out.mkdir(parents=True, exist_ok=False)
    store.save(out / "corpus.json")
    outputs = {"report_json": str(out / "deduplication.json"), "report_md": str(out / "deduplication.md")}
    (out / "deduplication.json").write_text(json.dumps({
        "source": str(corpus.resolve()), "stats": store.corpus.stats, **report,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 文献版本去重报告", "", f"来源：`{corpus.resolve()}`", "",
             f"本次 {report['last_pass']['before']} → {len(store)} 条主记录；"
             f"合并 {report['last_pass']['merged']} 条，保留 {store.corpus.stats['n_source_records']} 条原始版本记录。", "",
             "按题名、作者、年份、预印本来源及引用信息作保守匹配；未查询外部版本关系，也未调用 LLM。",
             "主记录优先选正式发表版本，评分取已记录最高值，种子与撤稿标记保留。",
             "原始版本保存在 corpus.json 的 versions 中；引用边保留 record_pairs 原始端点。",
             "既有 PRISMA 保留原检索运行口径，去重后的数量见本报告和 stats。", "", "## 合并记录", ""]
    for m in report["merges"]:
        p = store.get(m["kept_id"])
        lines.append(f"- `{m['removed_id']}` → `{m['kept_id']}`：{p.title if p else ''}（{m['rule']}）")
    lines += ["", "## 待核对", ""]
    lines += [f"- {' / '.join(r['ids'])}：{' / '.join(r['titles'])}" for r in report["review_candidates"]] or ["无相似题名候选；这不代表已排除所有潜在重复。"]
    (out / "deduplication.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for fmt, ext in EXTENSIONS.items():
        (out / f"corpus.{ext}").write_text(export(store.corpus.papers, fmt), encoding="utf-8")
        outputs[fmt] = str(out / f"corpus.{ext}")
    result = RunResult(command="deduplicate", run_dir=str(out), corpus=str(out / "corpus.json"), outputs=outputs,
                       summary=summarize(store.corpus, top))
    if obsidian:
        result.obsidian = write_obsidian(store.corpus, out / "corpus.json", out_dir=out / "obsidian")
    result.warnings += [f"needs review: {' / '.join(r['ids'])}" for r in report["review_candidates"]]
    return result


def stats(corpus: Path, top: int = 10) -> StatsResult:
    return StatsResult(command="stats", corpus=str(corpus), summary=summarize(load_corpus(corpus), top))


def evaluate(corpus: Path, gold: list[str], *, show_missed: int = 15, out: Path | None = None) -> EvalResult:
    """Recall against the reference lists of published surveys (OpenAlex only, no LLM)."""
    from .evaluate import recall_report
    from .pipeline.seeds import resolve_seeds

    c = load_corpus(corpus)
    oa = _openalex(get_settings())
    try:
        surveys, missing = resolve_seeds(list(gold), oa)
        warnings = [f"could not resolve gold survey: {m}" for m in missing]
        if not surveys:
            raise InputError("; ".join(warnings) or "no gold survey given")
        rep = recall_report(c, surveys)
        missed = sorted(oa.by_ids(rep.missed), key=lambda p: -(p.cited_by_count or 0))[:show_missed] if show_missed and rep.missed else []
        out_path = out or Path(corpus).parent / "eval.json"
        if Path(out_path).resolve() == Path(corpus).resolve():
            raise InputError("evaluation output must not overwrite the input corpus")
        atomic_write(out_path, rep.model_dump_json(indent=2))
        gold_briefs = [GoldSurvey(**brief(sv).model_dump(), n_references=len(sv.referenced_works)) for sv in surveys]
        return EvalResult(command="eval", corpus=str(corpus), report_path=str(out_path), gold=gold_briefs, report=rep,
                          missed_top=[brief(p) for p in missed], warnings=warnings + rep.warnings)
    finally:
        _close_clients(None, oa, None)


def export(corpus: Path, fmt: str = "bibtex", *, min_score: float = 0.0, out: Path | None = None) -> ExportResult:
    from .export import EXTENSIONS, export as render

    if fmt not in EXTENSIONS:
        raise InputError(f"format must be one of {', '.join(EXTENSIONS)}")
    c = load_corpus(corpus)
    papers = [p for p in c.papers if p.is_seed or (p.relevance.score if p.relevance else 0) >= min_score]
    out_path = out or Path(corpus).with_suffix(f".{EXTENSIONS[fmt]}")
    if Path(out_path).resolve() == Path(corpus).resolve():
        raise InputError("export output must not overwrite the input corpus")
    atomic_write(out_path, render(papers, fmt))
    return ExportResult(command="export", corpus=str(Path(corpus).resolve()), format=fmt,
                        path=str(Path(out_path).resolve()), n_papers=len(papers))


def obsidian(corpus: Path, *, vault: Path | None = None, folder: str | None = None) -> ObsidianResult:
    if folder and not (vault or get_settings().scib_obsidian_vault):
        raise InputError("--folder needs --vault (or SCIB_OBSIDIAN_VAULT)")
    info = write_obsidian(load_corpus(corpus), Path(corpus), vault, folder)
    return ObsidianResult(command="obsidian", corpus=str(corpus), obsidian=info)


def traverse(corpus: Path, start: list[str], *, mode: str = "bfs", direction: str = "both", depth: int = 3,
             limit: int = 100) -> TraverseResult:
    from .graph import build_graph, traverse as trav

    if mode not in ("bfs", "dfs") or direction not in ("refs", "cited_by", "both") or depth < 0 or limit < 1:
        raise InputError("traverse requires bfs/dfs, refs/cited_by/both, depth >= 0 and limit >= 1")

    g = build_graph(load_corpus(corpus))
    bad = [s for s in start if s not in g]
    if bad:
        raise InputError(f"Not in corpus: {', '.join(bad)}")
    visits = trav(g, start, mode=mode, direction=direction, max_depth=depth, limit=limit)
    return TraverseResult(command="traverse", corpus=str(corpus), mode=mode, direction=direction, visits=[
        TraverseVisit(order=v.order, depth=v.depth, via=v.via, parent=v.parent, id=v.id,
                      year=g.nodes[v.id]["year"], title=g.nodes[v.id]["title"] or "")
        for v in visits
    ])
