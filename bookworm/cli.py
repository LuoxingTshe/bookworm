"""Bookworm CLI: a thin shell over `service`.

People get rich console output. Calling agents pass `--json` (or set SCIB_JSON=1): progress goes to
stderr, and stdout carries exactly one JSON object per command — the `service` result model, or
{"ok": false, "command", "error": {"code", "message"}} with the exit code from `errors`.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import typer
try:  # Typer 0.26+ vendors Click; older supported releases use the installed package.
    from typer import _click as click
except ImportError:
    import click
from rich.console import Console
from rich.table import Table

from . import __release__, __version__, service
from .config import configure
from .errors import BookwormError, InputError
from .diagnostics import describe_error
from .requests import ResearchRequest
from typer.core import TyperGroup
from .models import Usage

console = Console()
_mode = {"json": False, "quiet": False}


class ProtocolGroup(TyperGroup):
    """Include parser errors in the JSON protocol, even before a callback runs."""
    def main(self, args=None, **kwargs):
        args = list(args if args is not None else sys.argv[1:])
        machine = "--json" in args or "task" in args or any(
            os.environ.get(name, "").lower() in {"1", "true", "yes"} for name in ("BOOKWORM_JSON", "SCIB_JSON"))
        _mode.update(json=machine, quiet=False)
        standalone = kwargs.pop("standalone_mode", True)
        previous = signal.getsignal(signal.SIGTERM)
        try:
            result = super().main(args=args, standalone_mode=False, **kwargs)
            if standalone and isinstance(result, int):
                raise SystemExit(result)
            return result
        except click.ClickException as exc:
            if machine:
                command = next((arg for arg in args if arg in service.RESULT_MODELS), "cli")
                typer.echo(json.dumps({"ok": False, "command": command, "protocol_version": "1",
                    "error": {"code": "invalid_input", "message": exc.format_message()}}, ensure_ascii=False))
            else:
                exc.show()
            if standalone:
                raise SystemExit(exc.exit_code)
            raise typer.Exit(exc.exit_code) from None
        except typer.Exit as exc:
            if standalone:
                raise SystemExit(exc.exit_code)
            raise
        finally:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, previous)


app = typer.Typer(cls=ProtocolGroup, add_completion=False,
                  help="Bookworm 蠹: autonomous literature research and structured sub-agent tasks")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(json.dumps({"name": "Bookworm 蠹", "version": __version__}) if _mode["json"] else f"{__release__} {__version__}")
        raise typer.Exit()


def _sigterm(signum, frame):
    # A caller's timeout usually sends SIGTERM: treat it like Ctrl+C so runs are marked interrupted
    # and the agent saves its partial corpus.
    raise KeyboardInterrupt


@app.callback()
def options(
    ctx: typer.Context,
    home: Optional[Path] = typer.Option(None, "--home", help="Workspace holding .env and runtime data"),
    version: bool = typer.Option(False, "--version", callback=_show_version,
                                 is_eager=True, help="Show the release version and exit."),
    json_out: bool = typer.Option(False, "--json", envvar=["BOOKWORM_JSON", "SCIB_JSON"],
                                  help="Machine-readable mode: one JSON result on stdout, progress on stderr."),
    quiet: bool = typer.Option(False, "--quiet", "-q", envvar=["BOOKWORM_QUIET", "SCIB_QUIET"], help="Suppress progress messages."),
) -> None:
    """Retrieve, evaluate and export literature corpora."""
    global console
    configure(home)
    _mode.update(json=json_out or ctx.invoked_subcommand == "task", quiet=quiet)
    console = Console(stderr=_mode["json"])
    if ctx.invoked_subcommand in ("build", "agent", "run", "task") and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sigterm)


def _log(msg: str) -> None:
    if not _mode["quiet"]:
        console.print(msg)


def _emit(result: service.Result) -> bool:
    """In JSON mode print the result and return True; otherwise the caller renders it for people."""
    if _mode["json"]:
        typer.echo(result.model_dump_json())
        if isinstance(result, service.RunResult) and not result.complete:
            raise typer.Exit(130 if result.stop_reason == "interrupted" else 3)
    return _mode["json"]


def _fail(command: str, code: str, message: str, exit_code: int):
    if _mode["json"]:
        typer.echo(json.dumps({"ok": False, "command": command, "protocol_version": "1", "error": {"code": code, "message": message}},
                              ensure_ascii=False))
    else:
        console.print(f"[red]{message}[/]")
    raise typer.Exit(exit_code)


@contextmanager
def _task(command: str):
    try:
        yield
    except (typer.Exit, typer.Abort, click.ClickException):
        raise
    except BookwormError as e:
        _fail(command, e.code, str(e), e.exit_code)
    except (Exception, KeyboardInterrupt) as exc:
        code, message, exit_code = describe_error(exc)
        _fail(command, code, message, exit_code)


def _parse_years(s: str | None) -> tuple[int | None, int | None]:
    try:
        return service.parse_years(s)
    except BookwormError as e:
        raise typer.BadParameter(str(e)) from None


def _report_cleanup(report: dict | None) -> None:
    if report is None:
        console.print("[yellow]Partial result saved; previous results and cache are preserved.[/]")
        return
    console.print(f"[dim]Cleanup: {len(report['removed_runs'])} earlier results removed; "
                  f"{len(report['cleared_caches'])} caches cleared.[/]")
    for error in report['errors']:
        console.print(f"[yellow]Cleanup incomplete: {error}[/]")


def _print_obsidian(info: service.ObsidianInfo | None) -> None:
    if info is None:
        return
    kept = f" (kept your text in {info.notes_kept_user_text})" if info.notes_kept_user_text else ""
    console.print(f"[green]✓ Obsidian:[/] {info.notes_written} notes{kept} → {info.overview}")
    if info.canvas_layout == "spring":
        console.print("  [yellow]Canvas laid out with the networkx fallback: install Node.js to use the fCoSE layout[/]")
    if info.stale_removed:
        console.print(f"  removed {len(info.stale_removed)} notes no longer in the corpus")
    for name in info.stale_kept:
        console.print(f"  [yellow]kept {name}: no longer in the corpus but has your notes[/]")
    if info.vault_root == info.dir:
        console.print(f"  Open the folder [bold]{info.dir}[/] as a vault in Obsidian (Bases need Obsidian ≥ 1.9)")


def _print_usage(u: Usage) -> None:
    console.print(
        f"[dim]Usage: DeepSeek {u.deepseek_calls} calls / {u.deepseek_prompt_tokens}+{u.deepseek_completion_tokens} tokens · "
        f"WoS {u.wos_requests} requests · OpenAlex {u.openalex_requests} requests[/]"
    )


def _print_run(r: service.RunResult) -> None:
    from .store import CorpusStore

    if r.summary.agent_summary and r.command == "agent":
        console.print(f"\n[bold]Agent summary:[/] {r.summary.agent_summary}")
    _print_stats(CorpusStore.load(Path(r.corpus)).finalize())
    if r.usage:
        _print_usage(r.usage)
    console.print(f"\n[green]✓ Corpus written to[/] {r.corpus}")
    _print_obsidian(r.obsidian)
    for warning in r.warnings:
        console.print(f"Warning: {warning}", markup=False)
    if not r.complete:
        raise typer.Exit(130 if r.stop_reason == "interrupted" else 3)


@app.command()
def schema():
    """Print the JSON Schema of every --json result (and of the error object)."""
    typer.echo(json.dumps(service.result_schemas(), ensure_ascii=False, indent=None if _mode["json"] else 2))


@app.command()
def intent(
    prompt: str = typer.Argument(..., help="Natural-language research need"),
    seed: list[str] = typer.Option([], "--seed", "-s", help="Seed paper: DOI / OpenAlex ID / WOS:UID / title; repeatable"),
    years: Optional[str] = typer.Option(None, help="Year range, e.g. 2018-2025"),
    source: str = typer.Option("auto", help="auto | wos | openalex: query source"),
):
    """Parse the research intent and generate candidate queries only (no search), for debugging prompts."""
    with _task("intent"):
        r = service.parse_intent(prompt, seed, _parse_years(years), source, log=_log)
        if _emit(r):
            return
        console.print_json(r.intent.model_dump_json())
        console.print(f"\n[bold]{source} queries:[/]")
        for q in r.queries:
            console.print(f"  · {q}")
        _print_usage(r.usage)


@app.command()
def build(
    max_tokens: int = typer.Option(100000, min=1, help="Observed token budget"),
    prompt: str = typer.Argument(..., help="Natural-language research need (Chinese or English)"),
    seed: list[str] = typer.Option([], "--seed", "-s", help="Seed paper: DOI / OpenAlex ID / WOS:UID / title; repeatable"),
    seed_file: Optional[Path] = typer.Option(None, help="Seed file, one per line"),
    years: Optional[str] = typer.Option(None, help="Year range, e.g. 2018-2025"),
    source: str = typer.Option("auto", help="Primary source: auto | wos | openalex"),
    tier: str = typer.Option("quick", help="Preset depth: quick | standard | deep (options below override it)"),
    n_queries: Optional[int] = typer.Option(None, help="Number of WoS queries the LLM generates [tier]"),
    query_branch: list[str] = typer.Option([], "--branch", help="Additional WoS recall branch: vegetation_tls, garden_syntax, garden_reviews, heritage_pointcloud; repeatable"),
    per_query: Optional[int] = typer.Option(None, help="Max results per query [tier]"),
    hops: Optional[int] = typer.Option(None, help="Max snowball hops; stops early on low yield [tier]"),
    direction: str = typer.Option("both", help="Snowball direction: backward | forward | both"),
    per_node: Optional[int] = typer.Option(None, help="Max references/citing works expanded per node [tier]"),
    frontier: Optional[int] = typer.Option(None, help="Max frontier nodes per hop [tier]"),
    prefilter: Optional[int] = typer.Option(None, help="Max candidates sent to LLM screening per round [tier]"),
    threshold: Optional[float] = typer.Option(None, help="Relevance threshold 0-10 [default 6]"),
    max_papers: Optional[int] = typer.Option(None, help="Max corpus size [tier]"),
    min_hop_yield: Optional[float] = typer.Option(None, help="Stop snowballing when a hop's relevant share is below this [0.1]"),
    gap_fill: bool = typer.Option(True, "--gap-fill/--no-gap-fill", help="Fetch works cited by many relevant papers but missing"),
    gap_min_count: Optional[int] = typer.Option(None, help="Gap fill: min relevant papers citing a missing work [3]"),
    keep_retracted: bool = typer.Option(False, "--keep-retracted", help="Keep papers OpenAlex marks as retracted"),
    label_edges: bool = typer.Option(False, "--label-edges", help="Have the LLM label semantic dependencies on citation edges"),
    obsidian: bool = typer.Option(True, "--obsidian/--no-obsidian", help="Also write the Obsidian vault output"),
    out: Optional[Path] = typer.Option(None, help="Output corpus.json path (default artifacts/runs/<slug>-<time>/corpus.json)"),
    keep_history: bool = typer.Option(False, "--keep-history", help="Keep earlier completed results"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep HTTP cache for repeat experiments"),
    top: int = typer.Option(10, help="Papers listed in the --json summary"),
):
    """Run the full pipeline and build a corpus with citation relations."""
    from .pipeline.build import TIERS, params_for_tier

    if source not in ("auto", "wos", "openalex"):
        raise typer.BadParameter("--source must be auto, wos or openalex")
    if direction not in ("backward", "forward", "both"):
        raise typer.BadParameter("--direction must be backward / forward / both")
    if tier not in TIERS:
        raise typer.BadParameter(f"--tier must be one of {', '.join(TIERS)}")
    with _task("build"):
        params = params_for_tier(
            tier, prompt, seeds=service.collect_seeds(seed, seed_file), years=_parse_years(years), source=source,
            n_queries=n_queries, query_branches=query_branch, per_query=per_query, hops=hops, direction=direction, per_node=per_node,
            frontier_size=frontier, prefilter_keep=prefilter, threshold=threshold, max_papers=max_papers,
            min_hop_yield=min_hop_yield, gap_fill=gap_fill, gap_min_count=gap_min_count,
            exclude_retracted=not keep_retracted, label_edges=label_edges, max_tokens=max_tokens,
        )
        r = service.build(params, out=out, obsidian=obsidian, keep_history=keep_history, keep_cache=keep_cache,
                          top=top, log=_log)
        if not _emit(r):
            _print_run(r)
            _report_cleanup(r.cleanup)


@app.command()
def agent(
    source: str = typer.Option("auto", help="auto | wos | openalex"),
    prompt: str = typer.Argument(..., help="Natural-language research need"),
    corpus: Optional[Path] = typer.Option(None, help="Continue extending an existing corpus.json"),
    seed: list[str] = typer.Option([], "--seed", "-s", help="Seed paper (when starting fresh)"),
    years: Optional[str] = typer.Option(None, help="Year range"),
    threshold: float = typer.Option(6.0, help="Relevance threshold for adding to the corpus"),
    max_papers: int = typer.Option(80, help="Hard cap on corpus size (add_to_corpus refuses beyond it)"),
    max_steps: int = typer.Option(12, min=1, max=200, help="Max agent steps"),
    max_tokens: int = typer.Option(100000, min=1, help="Observed token budget; in-flight calls may exceed it"),
    label_edges: bool = typer.Option(False, "--label-edges"),
    obsidian: bool = typer.Option(True, "--obsidian/--no-obsidian", help="Also update the Obsidian vault output"),
    out: Optional[Path] = typer.Option(None, help="Output path (default: a new run directory, including when continuing --corpus)"),
    keep_history: bool = typer.Option(False, "--keep-history", help="Keep earlier completed results"),
    keep_cache: bool = typer.Option(False, "--keep-cache", help="Keep HTTP cache for repeat experiments"),
    top: int = typer.Option(10, help="Papers listed in the --json summary"),
):
    """DeepSeek-driven tool-calling agent that explores and extends the corpus on its own."""
    with _task("agent"):
        r = service.extend(prompt, source=source, corpus=corpus, seeds=seed, years=_parse_years(years), threshold=threshold,
                           max_papers=max_papers, max_steps=max_steps, max_tokens=max_tokens, label_edges=label_edges,
                           obsidian=obsidian, out=out, keep_history=keep_history, keep_cache=keep_cache, top=top, log=_log)
        if not _emit(r):
            _print_run(r)
            _report_cleanup(r.cleanup)


@app.command()
def deduplicate(
    corpus: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(..., help="New output directory; the source corpus is preserved"),
    obsidian: bool = typer.Option(True, "--obsidian/--no-obsidian", help="Write a local Obsidian vault"),
):
    """Merge well-supported publication versions offline, preserving originals and a review report."""
    with _task("deduplicate"):
        r = service.deduplicate(corpus, out, obsidian=obsidian)
        if _emit(r):
            return
        _print_stats(service.load_corpus(Path(r.corpus)))
        n_review = sum(w.startswith("needs review") for w in r.warnings)
        merged = json.loads(Path(r.outputs["report_json"]).read_text(encoding="utf-8"))["last_pass"]["merged"]
        console.print(f"[green]✓ Version filter:[/] {merged} merged, {n_review} pairs need review → {out}")


@app.command()
def stats(
    corpus: Path = typer.Argument(..., exists=True),
    top: int = typer.Option(15, help="Most-connected papers to list"),
):
    """Show corpus statistics and the most-connected papers."""
    with _task("stats"):
        if not _emit(service.stats(corpus, top)):
            _print_stats(service.load_corpus(corpus), top=top)


@app.command()
def traverse(
    corpus: Path = typer.Argument(..., exists=True),
    start: list[str] = typer.Option(..., "--start", help="Start paper id; repeatable"),
    mode: str = typer.Option("bfs", help="bfs | dfs"),
    direction: str = typer.Option("both", help="refs (upstream to references) | cited_by (downstream to citing works) | both"),
    depth: int = typer.Option(3),
    limit: int = typer.Option(100),
):
    """BFS/DFS traversal over the citation graph in the terminal."""
    with _task("traverse"):
        r = service.traverse(corpus, start, mode=mode, direction=direction, depth=depth, limit=limit)
        if _emit(r):
            return
        t = Table(title=f"{mode.upper()} · {direction} · depth≤{depth}")
        for col in ("#", "depth", "via", "id", "year", "title"):
            t.add_column(col)
        for v in r.visits:
            t.add_row(str(v.order + 1), "  " * v.depth + str(v.depth), {"refs": "↑ref", "cited_by": "↓cited"}.get(v.via, "●"),
                      v.id, str(v.year or ""), v.title[:80])
        console.print(t)


@app.command("obsidian")
def obsidian_cmd(
    corpus: Path = typer.Argument(..., exists=True),
    vault: Optional[Path] = typer.Option(None, help="Existing vault to write into (default: SCIB_OBSIDIAN_VAULT, else <run>/obsidian)"),
    folder: Optional[str] = typer.Option(None, help="Folder inside the vault (default Bookworm/<run>)"),
):
    """Write (or refresh) the Obsidian vault output: paper notes, a Base, a citation Canvas, an overview note."""
    with _task("obsidian"):
        r = service.obsidian(corpus, vault=vault, folder=folder)
        if not _emit(r):
            _print_obsidian(r.obsidian)


@app.command("eval")
def eval_cmd(
    corpus: Path = typer.Argument(..., exists=True),
    gold: list[str] = typer.Option(..., "--gold", "-g", help="Survey whose reference list is the gold set: DOI / OpenAlex ID / title; repeatable"),
    show_missed: int = typer.Option(15, help="List this many missed gold papers (most cited first)"),
    out: Optional[Path] = typer.Option(None, help="Report path (default: eval.json next to the corpus)"),
):
    """Recall of the corpus against the references of published surveys (needs OpenAlex only, no LLM)."""
    with _task("eval"):
        r = service.evaluate(corpus, gold, show_missed=show_missed, out=out)
        if _emit(r):
            return
        rep = r.report
        for w in r.warnings:
            console.print(f"[yellow]⚠ {w}[/]")
        for sv in r.gold:
            console.print(f"Gold: {sv.id} {sv.title[:80]} ({sv.year}) · {sv.n_references} references")
        console.print(
            f"\n[bold]Recall {rep.recall:.1%}[/] ({rep.found}/{rep.gold_size} gold papers in a corpus of {rep.corpus_size})"
            f" · gold share of corpus {rep.gold_share_of_corpus:.1%}"
        )
        if rep.recall_at:
            console.print("  " + " · ".join(f"recall@{k} {v:.1%}" for k, v in rep.recall_at.items()))
        if r.missed_top:
            t = Table(title="Most-cited gold papers missing from the corpus")
            for col in ("id", "year", "cited", "title"):
                t.add_column(col)
            for p in r.missed_top:
                t.add_row(p.id, str(p.year or ""), str(p.cited_by_count or ""), p.title[:80])
            console.print(t)
        console.print(f"[green]✓ Report written to[/] {r.report_path}")


@app.command("export")
def export_cmd(
    corpus: Path = typer.Argument(..., exists=True),
    fmt: str = typer.Option("bibtex", "--format", "-f", help="bibtex | ris | csv"),
    min_score: float = typer.Option(0.0, help="Only papers scoring at least this (seeds always included)"),
    out: Optional[Path] = typer.Option(None, help="Output path (default: corpus.<ext> next to the corpus)"),
):
    """Export the corpus for Zotero / EndNote (BibTeX, RIS) or spreadsheets (CSV)."""
    with _task("export"):
        r = service.export(corpus, fmt, min_score=min_score, out=out)
        if not _emit(r):
            console.print(f"[green]✓ {r.n_papers} papers exported to[/] {r.path}")


def _print_stats(corpus, top: int = 8) -> None:
    st = corpus.stats
    console.print(
        f"\n[bold]Corpus:[/] {st.get('n_papers')} papers ({st.get('n_seeds')} seeds) · {st.get('n_edges')} citation edges"
        f" ({st.get('n_labeled_edges')} labeled) · years {st.get('year_span')}"
    )
    console.print(f"  origins: {st.get('by_origin')}")
    if st.get("n_retracted"):
        console.print(f"  [red]{st['n_retracted']} retracted paper(s) in the corpus[/]")
    if (pr := corpus.meta.prisma) is not None:
        console.print(
            f"  flow: identified {sum(pr.identified.values())} (+{pr.search_duplicates} duplicate search hits) → "
            f"screened {pr.screened} (not screened {pr.not_screened}) → excluded low {pr.excluded_low_relevance} · "
            f"retracted {pr.excluded_retracted} · over cap {pr.excluded_over_cap} → included {pr.included}"
            + (f" · agent +{pr.agent_added}" if pr.agent_added else "")
        )
        if pr.stop_reason:
            console.print(f"  snowball stopped early: {pr.stop_reason}")
    indeg: dict[str, int] = {}
    for e in corpus.edges:
        indeg[e.target] = indeg.get(e.target, 0) + 1
    if indeg:
        by_id = {p.id: p for p in corpus.papers}
        t = Table(title="Most cited within the corpus")
        for col in ("in-corpus cites", "id", "year", "score", "title"):
            t.add_column(col)
        for pid, k in sorted(indeg.items(), key=lambda x: -x[1])[:top]:
            p = by_id[pid]
            t.add_row(str(k), pid, str(p.year or ""), str(p.relevance.score if p.relevance else ""), p.title[:70])
        console.print(t)


@app.command()
def doctor(source: str = typer.Option("auto")):
    """Check local configuration without network access or exposing secrets."""
    with _task("doctor"):
        result = service.doctor(source)
        if not _emit(result):
            console.print(f"{__release__} {result.version}: {'ready' if result.ready else 'configuration needed'}")
            for warning in result.warnings:
                console.print(warning, markup=False)
        if not result.ready:
            raise typer.Exit(2)


@app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Research need in Chinese or English"),
    mode: str = typer.Option("agent", help="agent: autonomous tool loop; pipeline: fixed retrieval stages"),
    source: str = typer.Option("auto", help="auto | wos | openalex"),
    tier: str = typer.Option("quick", help="quick | standard | deep"),
    seed: list[str] = typer.Option([], "--seed", "-s"),
    seed_file: Optional[Path] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    corpus: Optional[Path] = typer.Option(None, help="Continue a corpus or checkpoint in agent mode"),
    out: Optional[Path] = typer.Option(None, help="Output corpus.json in a new directory"),
    max_papers: Optional[int] = typer.Option(None, min=1),
    max_steps: int = typer.Option(12, min=1, max=200),
    max_tokens: int = typer.Option(100000, min=1),
    threshold: float = typer.Option(6, min=0, max=10),
    obsidian: bool = typer.Option(False, "--obsidian/--no-obsidian"),
    top: int = typer.Option(10, min=0, max=100),
):
    """Research autonomously and save corpus, references, trace and a structured result."""
    with _task("run"):
        request = ResearchRequest(prompt=prompt, mode=mode, source=source, tier=tier,
            seeds=service.collect_seeds(seed, seed_file), years=_parse_years(years), corpus=corpus, out=out,
            max_papers=max_papers, max_steps=max_steps, max_tokens=max_tokens, threshold=threshold,
            obsidian=obsidian, top=top)
        result = service.run(request, log=_log)
        if not _emit(result):
            _print_run(result)


@app.command("task")
def task_cmd(request: str = typer.Argument("-", help="JSON request file, or - for stdin")):
    """Execute one JSON task; always emit one JSON result on stdout."""
    with _task("task"):
        from .persistence import atomic_write
        try:
            raw = sys.stdin.read() if request == "-" else Path(request).read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise InputError("request must contain exactly one JSON object") from None
        result = service.run(ResearchRequest.model_validate(data), log=_log)
        result.command = "task"
        atomic_write(Path(result.run_dir) / "result.json", result.model_dump_json(indent=2))
        _emit(result)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
